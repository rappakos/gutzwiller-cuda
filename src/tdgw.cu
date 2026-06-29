// ===========================================================================
//  tdgw.cu  --  Time-dependent Gutzwiller dynamics of the Bose-Hubbard model
//               on a GPU.  Square lattice; single precision; RK4.
//
//  Physics: Rapp, PRA 90, 053607 (2014), Eq.(14)-(15). Mirrors the verified
//  Python reference in ../reference/tdgw_reference.py one-to-one.
//
//      i d/dt f_m(j) = eps_m(j) f_m(j)
//                      - conj(Phi_j) sqrt(m+1) f_{m+1}(j)
//                      -      Phi_j  sqrt(m)   f_{m-1}(j)
//      eps_m(j) = (U/2) m(m-1) + (V0 r_j^2 - mu0) m
//      Phi_j    = sum_{k in nbr(j)} J_{jk} <b_k>,  <b_j> = sum_m sqrt(m+1) conj(f_m) f_{m+1}
//
//  -------------------------------------------------------------------------
//  WHY RK4 (and NOT split-step):
//    A Strang split-step that freezes Phi during the hop sub-step is *not*
//    number-conserving, because the mean-field hopping generator -(Phi b^dag
//    + Phi* b) changes particle number on a site (unlike the GPE's diagonal
//    |psi|^2 nonlinearity). Numerically the freeze-Phi scheme leaks N_tot at
//    O(dt) -- measured ~10% over T=2 on a 6x6 test. RK4 on the *instantaneous*
//    coupled RHS conserves N_tot to ~1e-7 and E_tot to ~1e-5 (matches the
//    tight-tolerance Python RK45 reference). See ../tests/host_split_step_check.cpp.
//    => integrator = explicit RK on the full RHS. On a 6 GB card the only cost
//       is buffer count; for 192^3 use the 2N-storage low-storage RK noted below.
//
//  Design (the "architecture"):
//    * SoA, component-planar layout f[m*N + j]: for fixed m, adjacent sites
//      are adjacent in memory -> fully coalesced warp loads/stores.
//    * Lattice = neighbour list (nbr[d*N+j], Jdir[d]). Square (z=4), triangular
//      (z=6), cubic (z=6) differ ONLY in make_*(); kernels never change.
//    * RHS = three kernels: k_psi (reduce f->psi), k_phi (neighbour gather),
//      k_rhs (local tridiagonal update). RK4 calls the RHS four times/step.
//    * Single precision; conservation monitors are the live accuracy gauge.
//
//  Build:  see ../CMakeLists.txt   (targets sm_75 = RTX 2060)
//  Run:    ./tdgw --L 192 --D 12 --steps 20000 --dt 0.002
//          ./tdgw --selftest          (host-vs-device diff + conservation)
// ===========================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <complex>
#include <string>
#include <algorithm>

#include <thrust/complex.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>

using real = float;
using cplx = thrust::complex<float>;

#define DMAX 32                       // max Fock states (compile-time, for safety checks)
#define TPB  128                      // threads per block

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
      fprintf(stderr, "CUDA error %s at %s:%d\n",                              \
              cudaGetErrorString(_e), __FILE__, __LINE__);                     \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

// ---------------------------------------------------------------------------
//  Lattice as a neighbour list (host side; uploaded once).
// ---------------------------------------------------------------------------
struct Lattice {
  int L, N, Z;
  std::vector<int>  nbr;    // size Z*N : nbr[d*N+j] = neighbour of j in dir d, or -1
  std::vector<real> Jdir;   // size Z   : hopping amplitude per direction
  std::vector<real> r2;     // size N   : (r_j)^2 from trap centre, lattice units
};

// Square lattice, open boundaries. Directions +x,-x,+y,-y (z=4).
// Anisotropic hopping J1=Jx along x, J2=Jy along y (isotropic when Jx==Jy).
Lattice make_square(int L, real Jx, real Jy) {
  Lattice lat;
  lat.L = L; lat.N = L * L; lat.Z = 4;
  lat.nbr.assign(lat.Z * lat.N, -1);
  lat.Jdir = {Jx, Jx, Jy, Jy};
  lat.r2.resize(lat.N);
  const real cx = (L - 1) * 0.5f, cy = (L - 1) * 0.5f;
  auto id = [L](int x, int y) { return x + L * y; };
  for (int y = 0; y < L; ++y)
    for (int x = 0; x < L; ++x) {
      int j = id(x, y);
      if (x + 1 < L)  lat.nbr[0 * lat.N + j] = id(x + 1, y);
      if (x - 1 >= 0) lat.nbr[1 * lat.N + j] = id(x - 1, y);
      if (y + 1 < L)  lat.nbr[2 * lat.N + j] = id(x, y + 1);
      if (y - 1 >= 0) lat.nbr[3 * lat.N + j] = id(x, y - 1);
      lat.r2[j] = (x - cx) * (x - cx) + (y - cy) * (y - cy);
    }
  return lat;
}
// triangular: make_triangular() with Z=6 axial offsets + Jdir {J1,J1,J2,J2,J3,J3}.
// 3D:         make_cubic()      with Z=6: +-x,+-y,+-z.  Kernels unchanged.

// ---------------------------------------------------------------------------
//  Kernels.  One thread == one lattice site throughout.
// ---------------------------------------------------------------------------

// Reduce each site's Fock vector to the order parameter psi_j = <b_j>.
__global__ void k_psi(const cplx* __restrict__ f, cplx* __restrict__ psi,
                      int N, int D) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= N) return;
  cplx s(0.f, 0.f);
  for (int m = 0; m < D - 1; ++m)
    s += sqrtf((float)(m + 1)) * thrust::conj(f[m * N + j]) * f[(m + 1) * N + j];
  psi[j] = s;
}

// Gather neighbours into the mean field Phi_j (the only non-local step).
__global__ void k_phi(const cplx* __restrict__ psi, const int* __restrict__ nbr,
                      const real* __restrict__ Jdir, cplx* __restrict__ phi,
                      int N, int Z) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= N) return;
  cplx s(0.f, 0.f);
  for (int d = 0; d < Z; ++d) {
    int k = nbr[d * N + j];
    if (k >= 0) s += Jdir[d] * psi[k];
  }
  phi[j] = s;
}

// Local tridiagonal RHS: out = d/dt f = -i ( eps f - conj(Phi) b f - Phi b^dag f ).
__global__ void k_rhs(const cplx* __restrict__ f, const cplx* __restrict__ phi,
                      const real* __restrict__ r2, real U, real V0, real mu0,
                      cplx* __restrict__ out, int N, int D) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= N) return;
  real base = V0 * r2[j] - mu0;
  cplx ph = phi[j], cph = thrust::conj(ph);
  for (int m = 0; m < D; ++m) {
    real eps = 0.5f * U * m * (m - 1) + base * m;
    cplx v = eps * f[m * N + j];
    if (m + 1 < D) v += -cph * sqrtf((float)(m + 1)) * f[(m + 1) * N + j];
    if (m - 1 >= 0) v += -ph  * sqrtf((float)m)       * f[(m - 1) * N + j];
    out[m * N + j] = cplx(0.f, -1.f) * v;            // multiply by -i
  }
}

// out = base + coeff * k   (build an RK stage argument).
__global__ void k_axpy(cplx* __restrict__ out, const cplx* __restrict__ base,
                       const cplx* __restrict__ k, real coeff, long n) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i < n) out[i] = base[i] + coeff * k[i];
}

// f += (h/6) (k1 + 2 k2 + 2 k3 + k4)   (RK4 combine).
__global__ void k_rk4_combine(cplx* __restrict__ f, const cplx* __restrict__ k1,
                              const cplx* __restrict__ k2, const cplx* __restrict__ k3,
                              const cplx* __restrict__ k4, real h, long n) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i < n) f[i] += (h / 6.f) * (k1[i] + 2.f * k2[i] + 2.f * k3[i] + k4[i]);
}

// Per-site diagnostics: number n_j, norm, local energy, bond energy.
__global__ void k_diag(const cplx* __restrict__ f, const cplx* __restrict__ psi,
                       const cplx* __restrict__ phi, const real* __restrict__ r2,
                       real U, real V0, real* n_out, real* nrm_out,
                       real* eloc_out, real* ebond_out, int N, int D) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= N) return;
  real n = 0.f, nrm = 0.f, el = 0.f;
  for (int m = 0; m < D; ++m) {
    real p = thrust::norm(f[m * N + j]);              // |f_m|^2
    nrm += p; n += m * p;
    el  += (0.5f * U * m * (m - 1) + V0 * r2[j] * m) * p;
  }
  n_out[j] = n; nrm_out[j] = nrm; eloc_out[j] = el;
  ebond_out[j] = -(thrust::conj(psi[j]) * phi[j]).real();   // -Re conj(psi)Phi
}

// ---------------------------------------------------------------------------
//  Device state + RK4 driver.
// ---------------------------------------------------------------------------
struct Device {
  int N, D, Z; long n;                                // n = N*D total amplitudes
  cplx *f=0, *ftmp=0, *k1=0, *k2=0, *k3=0, *k4=0, *psi=0, *phi=0;
  int  *nbr=0; real *Jdir=0, *r2=0, *nd=0, *nrm=0, *eloc=0, *ebond=0;

  void alloc(const Lattice& lat, int D_) {
    N = lat.N; D = D_; Z = lat.Z; n = (long)N * D;
    for (cplx** p : {&f,&ftmp,&k1,&k2,&k3,&k4})        // 6 full-state buffers
      CUDA_CHECK(cudaMalloc(p, sizeof(cplx) * n));
    CUDA_CHECK(cudaMalloc(&psi, sizeof(cplx) * N));
    CUDA_CHECK(cudaMalloc(&phi, sizeof(cplx) * N));
    CUDA_CHECK(cudaMalloc(&nbr, sizeof(int) * Z * N));
    CUDA_CHECK(cudaMalloc(&Jdir, sizeof(real) * Z));
    for (real** p : {&r2,&nd,&nrm,&eloc,&ebond})
      CUDA_CHECK(cudaMalloc(p, sizeof(real) * N));
    CUDA_CHECK(cudaMemcpy(nbr,  lat.nbr.data(),  sizeof(int) * Z * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(Jdir, lat.Jdir.data(), sizeof(real) * Z,    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(r2,   lat.r2.data(),   sizeof(real) * N,    cudaMemcpyHostToDevice));
  }
  void free_all() {
    for (void* p : {(void*)f,(void*)ftmp,(void*)k1,(void*)k2,(void*)k3,(void*)k4,
                    (void*)psi,(void*)phi,(void*)nbr,(void*)Jdir,(void*)r2,
                    (void*)nd,(void*)nrm,(void*)eloc,(void*)ebond}) cudaFree(p);
  }
};

// Evaluate k = d/dt(src):  psi <- src, phi <- gather(psi), k <- rhs(src,phi).
static void eval_rhs(Device& d, const cplx* src, cplx* k,
                     real U, real V0, real mu0) {
  int gN = (d.N + TPB - 1) / TPB;
  k_psi<<<gN, TPB>>>(src, d.psi, d.N, d.D);
  k_phi<<<gN, TPB>>>(d.psi, d.nbr, d.Jdir, d.phi, d.N, d.Z);
  k_rhs<<<gN, TPB>>>(src, d.phi, d.r2, U, V0, mu0, k, d.N, d.D);
}

// One classic RK4 step. Clear and recognisable; uses 6 state buffers.
// For 192^3 on 6 GB, swap this for a 2N-storage low-storage RK (Williamson /
// Carpenter-Kennedy): same 4th order, only 2 full buffers. Same RHS kernels.
void rk4_step(Device& d, real U, real V0, real mu0, real h) {
  long gn = (d.n + TPB - 1) / TPB;
  eval_rhs(d, d.f, d.k1, U, V0, mu0);
  k_axpy<<<gn, TPB>>>(d.ftmp, d.f, d.k1, 0.5f * h, d.n); eval_rhs(d, d.ftmp, d.k2, U, V0, mu0);
  k_axpy<<<gn, TPB>>>(d.ftmp, d.f, d.k2, 0.5f * h, d.n); eval_rhs(d, d.ftmp, d.k3, U, V0, mu0);
  k_axpy<<<gn, TPB>>>(d.ftmp, d.f, d.k3,        h, d.n); eval_rhs(d, d.ftmp, d.k4, U, V0, mu0);
  k_rk4_combine<<<gn, TPB>>>(d.f, d.k1, d.k2, d.k3, d.k4, h, d.n);
}

// Named functor: an extended __device__ lambda cannot have its return type
// queried from host code, which thrust::transform_reduce requires -> use a functor.
struct AbsDevOne { __host__ __device__ float operator()(float x) const { return fabsf(x - 1.0f); } };
struct Diag { double N, E, max_norm_dev; };
Diag diagnostics(Device& d, real U, real V0) {
  int gN = (d.N + TPB - 1) / TPB;
  k_psi <<<gN, TPB>>>(d.f, d.psi, d.N, d.D);
  k_phi <<<gN, TPB>>>(d.psi, d.nbr, d.Jdir, d.phi, d.N, d.Z);
  k_diag<<<gN, TPB>>>(d.f, d.psi, d.phi, d.r2, U, V0,
                      d.nd, d.nrm, d.eloc, d.ebond, d.N, d.D);
  CUDA_CHECK(cudaDeviceSynchronize());
  thrust::device_ptr<real> n(d.nd), eloc(d.eloc), ebond(d.ebond), nrm(d.nrm);
  double Ntot = thrust::reduce(n, n + d.N, 0.0);
  double Eloc = thrust::reduce(eloc, eloc + d.N, 0.0);
  double Ebnd = thrust::reduce(ebond, ebond + d.N, 0.0);
  float  mdev = thrust::transform_reduce(
      nrm, nrm + d.N, AbsDevOne(),
      0.0f, thrust::maximum<real>());
  return {Ntot, Eloc + Ebnd, (double)mdev};
}

// ---------------------------------------------------------------------------
//  Initial state: coherent bump (placeholder for the equilibrium-GA ground
//  state). For real runs, replace with imaginary-time / self-consistent GA.
// ---------------------------------------------------------------------------
std::vector<cplx> coherent_bump(const Lattice& lat, int D, real amp, real width) {
  std::vector<cplx> f((size_t)lat.N * D, cplx(0.f, 0.f));
  const real cx = (lat.L - 1) * 0.5f, cy = (lat.L - 1) * 0.5f;
  for (int y = 0; y < lat.L; ++y)
    for (int x = 0; x < lat.L; ++x) {
      int j = x + lat.L * y;
      real a = amp * expf(-((x - cx) * (x - cx) + (y - cy) * (y - cy)) / (width * width));
      double fact = 1.0, nrm = 0.0; std::vector<double> c(D);
      for (int m = 0; m < D; ++m) {
        if (m > 0) fact *= m;
        c[m] = exp(-0.5 * a * a) * pow((double)a, m) / sqrt(fact);
        nrm += c[m] * c[m];
      }
      for (int m = 0; m < D; ++m)
        f[(size_t)m * lat.N + j] = cplx((float)(c[m] / sqrt(nrm)), 0.f);
    }
  return f;
}

// ===========================================================================
//  SELF-TEST: host RK4 mirror, diffed against the device.  Isolates
//  parallelisation bugs (the host code is obviously serial-correct).
// ===========================================================================
namespace host {
using hc = std::complex<float>;
static inline int FI(int m, int j, int N) { return m * N + j; }
void psi(const std::vector<hc>& f, std::vector<hc>& p, int N, int D) {
  for (int j = 0; j < N; ++j) { hc s(0,0);
    for (int m = 0; m < D-1; ++m) s += sqrtf((float)(m+1))*std::conj(f[FI(m,j,N)])*f[FI(m+1,j,N)];
    p[j] = s; }
}
void rhs(const std::vector<hc>& f, const Lattice& lat, real U, real V0, real mu0,
         int D, std::vector<hc>& out) {
  int N = lat.N; std::vector<hc> p(N), ph(N); psi(f, p, N, D);
  for (int j = 0; j < N; ++j) { hc s(0,0);
    for (int d = 0; d < lat.Z; ++d) { int k = lat.nbr[d*N+j]; if (k>=0) s += lat.Jdir[d]*p[k]; }
    ph[j] = s; }
  for (int j = 0; j < N; ++j) { real base = V0*lat.r2[j]-mu0;
    for (int m = 0; m < D; ++m) { hc v(0,0); real eps = 0.5f*U*m*(m-1)+base*m;
      v += eps*f[FI(m,j,N)];
      if (m+1<D) v += -std::conj(ph[j])*sqrtf((float)(m+1))*f[FI(m+1,j,N)];
      if (m-1>=0) v += -ph[j]*sqrtf((float)m)*f[FI(m-1,j,N)];
      out[FI(m,j,N)] = hc(0,-1)*v; } }
}
void rk4(std::vector<hc>& f, const Lattice& lat, real U, real V0, real mu0, real h, int D) {
  size_t n = f.size(); std::vector<hc> k1(n),k2(n),k3(n),k4(n),t(n);
  rhs(f, lat, U, V0, mu0, D, k1);
  for (size_t i=0;i<n;++i) t[i]=f[i]+0.5f*h*k1[i]; rhs(t,lat,U,V0,mu0,D,k2);
  for (size_t i=0;i<n;++i) t[i]=f[i]+0.5f*h*k2[i]; rhs(t,lat,U,V0,mu0,D,k3);
  for (size_t i=0;i<n;++i) t[i]=f[i]+h*k3[i];      rhs(t,lat,U,V0,mu0,D,k4);
  for (size_t i=0;i<n;++i) f[i]+=(h/6.f)*(k1[i]+2.f*k2[i]+2.f*k3[i]+k4[i]);
}
} // namespace host

int run_selftest() {
  printf("SELF-TEST  (host RK4 vs device RK4)\n");
  const int L = 8, D = 8, steps = 200;
  const real U = -0.5f, V0 = -2.5e-3f, mu0 = 0.f, dt = 2e-3f;
  Lattice lat = make_square(L, 1.0f, 1.0f);

  std::vector<cplx> f0 = coherent_bump(lat, D, 0.8f, 3.0f);
  std::vector<host::hc> fh((size_t)lat.N * D);
  std::memcpy(fh.data(), f0.data(), sizeof(cplx) * lat.N * D);     // identical layout

  Device dev; dev.alloc(lat, D);
  CUDA_CHECK(cudaMemcpy(dev.f, f0.data(), sizeof(cplx) * lat.N * D, cudaMemcpyHostToDevice));

  Diag d0 = diagnostics(dev, U, V0);
  for (int s = 0; s < steps; ++s) {
    rk4_step(dev, U, V0, mu0, dt);
    host::rk4(fh, lat, U, V0, mu0, dt, D);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  Diag d1 = diagnostics(dev, U, V0);

  std::vector<cplx> fd((size_t)lat.N * D);
  CUDA_CHECK(cudaMemcpy(fd.data(), dev.f, sizeof(cplx) * lat.N * D, cudaMemcpyDeviceToHost));

  double maxdiff = 0.0;
  for (size_t i = 0; i < fd.size(); ++i)
    maxdiff = std::max(maxdiff,
        (double)std::abs(std::complex<float>(fd[i].real(), fd[i].imag()) - fh[i]));

  printf("  %dx%d, D=%d, %d steps, dt=%g\n", L, L, D, steps, dt);
  printf("  host vs device   max|df|   = %.3e   (expect ~1e-6, FP32 reorder)\n", maxdiff);
  printf("  N_tot drift      |dN|      = %.3e\n", std::abs(d1.N - d0.N));
  printf("  E_tot drift      |dE|      = %.3e\n", std::abs(d1.E - d0.E));
  printf("  max |site-norm-1|          = %.3e\n", d1.max_norm_dev);
  dev.free_all();
  bool ok = maxdiff < 1e-4 && std::abs(d1.N - d0.N) < 1e-3 && d1.max_norm_dev < 1e-4;
  printf("  --> %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
  int L = 192, D = 12, steps = 2000, diag_every = 200;
  real dt = 2e-3f, U = -0.5f, V0 = -2.5e-3f, mu0 = 0.f, Jx = 1.f, Jy = 1.f;
  bool selftest = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto nf = [&](real& v) { v = atof(argv[++i]); };
    auto ni = [&](int& v) { v = atoi(argv[++i]); };
    if (a == "--selftest") selftest = true;
    else if (a == "--L") ni(L);           else if (a == "--D") ni(D);
    else if (a == "--steps") ni(steps);   else if (a == "--dt") nf(dt);
    else if (a == "--U") nf(U);           else if (a == "--V0") nf(V0);
    else if (a == "--Jx") nf(Jx);         else if (a == "--Jy") nf(Jy);
    else if (a == "--diag") ni(diag_every);
  }
  if (D > DMAX) { fprintf(stderr, "D > DMAX (%d)\n", DMAX); return 1; }
  if (selftest) return run_selftest();

  Lattice lat = make_square(L, Jx, Jy);
  printf("square %dx%d  N=%d  D=%d  Jx=%g Jy=%g  U=%g V0=%g  dt=%g  steps=%d\n",
         L, L, lat.N, D, Jx, Jy, U, V0, dt, steps);
  printf("state buffers: 6 x %.1f MB = %.2f GB (complex float, RK4)\n",
         sizeof(cplx) * (double)lat.N * D / 1e6,
         6.0 * sizeof(cplx) * (double)lat.N * D / 1e9);

  Device dev; dev.alloc(lat, D);
  std::vector<cplx> f0 = coherent_bump(lat, D, 0.8f, 0.15f * L);
  CUDA_CHECK(cudaMemcpy(dev.f, f0.data(), sizeof(cplx) * lat.N * D, cudaMemcpyHostToDevice));

  Diag d0 = diagnostics(dev, U, V0);
  printf("step %6d   N=%.6f   E=%.6f   |norm-1|=%.2e\n", 0, d0.N, d0.E, d0.max_norm_dev);
  for (int s = 1; s <= steps; ++s) {
    rk4_step(dev, U, V0, mu0, dt);
    if (s % diag_every == 0) {
      Diag d = diagnostics(dev, U, V0);
      printf("step %6d   N=%.6f   E=%.6f   |norm-1|=%.2e   dN=%.2e\n",
             s, d.N, d.E, d.max_norm_dev, std::abs(d.N - d0.N));
    }
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  dev.free_all();
  return 0;
}
