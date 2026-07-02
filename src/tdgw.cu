// ===========================================================================
//  tdgw.cu  --  Time-dependent Gutzwiller dynamics of the Bose-Hubbard model
//               on a GPU.  Square lattice; single precision.
//
//  Physics: Rapp, PRA 87, 043611 (2013) [square / negative-T], Eq.(5);
//  same EOM as PRA 90, 053607 (2014). Mirrors ../reference/tdgw_reference.py.
//
//      i d/dt f_m(j) = eps_m(j) f_m(j)
//                      - conj(Phi_j) sqrt(m+1) f_{m+1}(j)
//                      -      Phi_j  sqrt(m)   f_{m-1}(j)
//      eps_m(j) = (U/2) m(m-1) + (V0 r_j^2 - mu0) m
//      Phi_j    = sum_{k in nbr(j)} J_{jk} <b_k>,  <b_j> = sum_m sqrt(m+1) conj(f_m) f_{m+1}
//
//  -------------------------------------------------------------------------
//  INTEGRATORS (two; pick with --integrator):
//    splitstep (default, production):
//      exact-diagonal Strang split-step + midpoint-Phi predictor-corrector
//      + per-site renormalize. The diagonal on-site phase is applied EXACTLY,
//      so the stiff regime J/U ~ 0.0023 (U/J ~ 435) costs nothing -- where
//      plain RK4 must resolve the fast U-phase and goes unstable (NaN).
//      Recomputing Phi at the step midpoint drops the frozen-Phi N_tot leak
//      from O(dt) (~7% at dt=1e-2) to ~1e-5. Uses f, ftmp, psi, phi.
//    rk4 (cross-check):
//      classic RK4 on the full coupled RHS. Conserves N to ~1e-7 but only in
//      the NON-stiff limit; kept because it must agree with split-step there
//      (an independent check). Uses 6 full buffers.
//    Both verified in tests/split_step_prototype.cpp and tests/host_split_step_check.cpp.
//
//  Design: SoA component-planar layout f[m*N + j] (coalesced); lattice as a
//  neighbour list (square/triangular/cubic differ only in make_*()); the RHS /
//  sub-steps are per-site kernels. Single precision; conservation monitors gauge accuracy.
//
//  Build:  see ../CMakeLists.txt   (targets sm_75 = RTX 2060)
//  Run:    ./tdgw --L 192 --D 12 --steps 20000 --dt 0.002 [--integrator splitstep|rk4]
//          ./tdgw --selftest          (host-vs-device diff + conservation, both integrators)
// ===========================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <complex>
#include <string>
#include <algorithm>
#include <cstdint>

#include <thrust/complex.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>

using real = float;
using cplx = thrust::complex<float>;

#define DMAX 32                       // max Fock states (sizes per-thread local arrays)
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
// Triangular lattice (z=6), open boundaries. Sites live on a sheared grid with
// lattice vectors a1=(1,0), a2=(-1/2, sqrt3/2) (unit NN spacing, 120 deg); the
// six neighbours are +-a1 (J1), +-a2 (J2), +-(a1+a2) (J3). r^2 is the *physical*
// (oblique) distance from the trap centre, so the trap stays isotropic in real
// space. Kernels/integrator/diagnostics are unchanged -- only the neighbour list
// and Jdir differ from make_square. Matches reference/tdgw_reference.py
// triangular_lattice(). PRA 90, 053607 (2014): J1=J3>J2 is the frustration knob.
Lattice make_triangular(int L, real J1, real J2, real J3) {
  Lattice lat;
  lat.L = L; lat.N = L * L; lat.Z = 6;
  lat.nbr.assign(lat.Z * lat.N, -1);
  lat.Jdir = {J1, J1, J2, J2, J3, J3};
  lat.r2.resize(lat.N);
  const real c = (L - 1) * 0.5f, s3 = 0.86602540378f;   // sqrt(3)/2
  auto id = [L](int x, int y) { return x + L * y; };
  auto put = [&](int d, int j, int x, int y) {
    if (x >= 0 && x < L && y >= 0 && y < L) lat.nbr[d * lat.N + j] = id(x, y);
  };
  for (int y = 0; y < L; ++y)
    for (int x = 0; x < L; ++x) {
      int j = id(x, y);
      put(0, j, x + 1, y);   put(1, j, x - 1, y);       // +-a1  (J1)
      put(2, j, x,     y + 1); put(3, j, x, y - 1);      // +-a2  (J2)
      put(4, j, x + 1, y + 1); put(5, j, x - 1, y - 1);  // +-(a1+a2) (J3)
      real Rx = (x - c) - 0.5f * (y - c), Ry = s3 * (y - c);
      lat.r2[j] = Rx * Rx + Ry * Ry;
    }
  return lat;
}
// 3D: make_cubic() with Z=6: +-x,+-y,+-z.  Kernels unchanged.

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

// Local tridiagonal RHS (RK4 path): out = -i ( eps f - conj(Phi) b f - Phi b^dag f ).
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

// out = base + coeff * k   (RK stage argument).
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

// --- split-step sub-steps --------------------------------------------------

// Exact unitary rotation of the diagonal on-site part for time dt: eps_m is real
// -> pure phase, norm-preserving, no dt penalty from large U.
__global__ void k_phase(cplx* __restrict__ f, const real* __restrict__ r2,
                        real U, real V0, real mu0, real dt, int N, int D) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= N) return;
  real base = V0 * r2[j] - mu0;
  for (int m = 0; m < D; ++m) {
    real eps = 0.5f * U * m * (m - 1) + base * m;
    real ph  = -eps * dt;
    f[m * N + j] *= cplx(cosf(ph), sinf(ph));
  }
}

// Hop sub-step: apply exp(-i H_hop dt) with Phi frozen, H_hop the local (DxD)
// Hermitian tridiagonal -(conj(Phi) b + Phi b^dag). Short Taylor series; ||H||dt
// is small (Phi ~ z J <b>). Per-thread local Fock vectors.
__global__ void k_hop(cplx* __restrict__ f, const cplx* __restrict__ phi,
                      real dt, int N, int D, int order) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= N) return;
  cplx term[DMAX], acc[DMAX];
  for (int m = 0; m < D; ++m) { cplx v = f[m * N + j]; term[m] = v; acc[m] = v; }
  cplx ph = phi[j], cph = thrust::conj(ph);
  for (int k = 1; k <= order; ++k) {
    cplx Ht[DMAX];
    for (int m = 0; m < D; ++m) {
      cplx v(0.f, 0.f);
      if (m + 1 < D) v += -cph * sqrtf((float)(m + 1)) * term[m + 1];
      if (m - 1 >= 0) v += -ph  * sqrtf((float)m)       * term[m - 1];
      Ht[m] = v;
    }
    cplx coef(0.f, -dt / (float)k);                  // (-i dt)/k
    for (int m = 0; m < D; ++m) { term[m] = coef * Ht[m]; acc[m] += term[m]; }
  }
  for (int m = 0; m < D; ++m) f[m * N + j] = acc[m];
}

// Per-site renormalization: enforce sum_m |f_m(j)|^2 = 1 (the GA constraint).
__global__ void k_renorm(cplx* __restrict__ f, int N, int D) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= N) return;
  real s = 0.f;
  for (int m = 0; m < D; ++m) s += thrust::norm(f[m * N + j]);
  real inv = rsqrtf(s);
  for (int m = 0; m < D; ++m) f[m * N + j] *= inv;
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
//  Device state + integrator drivers.
// ---------------------------------------------------------------------------
struct Device {
  int N, D, Z; long n;                                // n = N*D total amplitudes
  cplx *f=0, *ftmp=0, *k1=0, *k2=0, *k3=0, *k4=0, *psi=0, *phi=0;
  int  *nbr=0; real *Jdir=0, *r2=0, *nd=0, *nrm=0, *eloc=0, *ebond=0;

  void alloc(const Lattice& lat, int D_) {
    N = lat.N; D = D_; Z = lat.Z; n = (long)N * D;
    for (cplx** p : {&f,&ftmp,&k1,&k2,&k3,&k4})        // split-step uses only f,ftmp
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

// Classic RK4 step (cross-check integrator). 6 buffers.
void rk4_step(Device& d, real U, real V0, real mu0, real h) {
  long gn = (d.n + TPB - 1) / TPB;
  eval_rhs(d, d.f, d.k1, U, V0, mu0);
  k_axpy<<<gn, TPB>>>(d.ftmp, d.f, d.k1, 0.5f * h, d.n); eval_rhs(d, d.ftmp, d.k2, U, V0, mu0);
  k_axpy<<<gn, TPB>>>(d.ftmp, d.f, d.k2, 0.5f * h, d.n); eval_rhs(d, d.ftmp, d.k3, U, V0, mu0);
  k_axpy<<<gn, TPB>>>(d.ftmp, d.f, d.k3,        h, d.n); eval_rhs(d, d.ftmp, d.k4, U, V0, mu0);
  k_rk4_combine<<<gn, TPB>>>(d.f, d.k1, d.k2, d.k3, d.k4, h, d.n);
}

// Exact-diagonal Strang split-step with midpoint-Phi predictor-corrector and
// per-site renormalize (production). Stable for U/J >> 1; N_tot ~1e-5 at dt=1e-2.
void splitstep_step(Device& d, real U, real V0, real mu0, real dt, int order = 10) {
  int gN = (d.N + TPB - 1) / TPB;
  k_phase<<<gN, TPB>>>(d.f, d.r2, U, V0, mu0, 0.5f * dt, d.N, d.D);
  k_psi  <<<gN, TPB>>>(d.f, d.psi, d.N, d.D);
  k_phi  <<<gN, TPB>>>(d.psi, d.nbr, d.Jdir, d.phi, d.N, d.Z);
  // predictor: half-hop a copy to get Phi at the step midpoint
  CUDA_CHECK(cudaMemcpy(d.ftmp, d.f, sizeof(cplx) * d.n, cudaMemcpyDeviceToDevice));
  k_hop<<<gN, TPB>>>(d.ftmp, d.phi, 0.5f * dt, d.N, d.D, order);
  k_psi<<<gN, TPB>>>(d.ftmp, d.psi, d.N, d.D);
  k_phi<<<gN, TPB>>>(d.psi, d.nbr, d.Jdir, d.phi, d.N, d.Z);   // corrected Phi
  // full hop with midpoint Phi, second half phase, renormalize
  k_hop  <<<gN, TPB>>>(d.f, d.phi, dt, d.N, d.D, order);
  k_phase<<<gN, TPB>>>(d.f, d.r2, U, V0, mu0, 0.5f * dt, d.N, d.D);
  k_renorm<<<gN, TPB>>>(d.f, d.N, d.D);
}

// Generic one-step dispatch.
void step(Device& d, const std::string& integ, real U, real V0, real mu0, real dt) {
  if (integ == "rk4") rk4_step(d, U, V0, mu0, dt);
  else                splitstep_step(d, U, V0, mu0, dt);
}

// Named functor: an extended __device__ lambda cannot have its return type
// queried from host code, which thrust::transform_reduce requires -> use a functor.
struct AbsDevOne { __host__ __device__ float operator()(float x) const { return fabsf(x - 1.0f); } };
struct NormC   { __host__ __device__ double operator()(const cplx& z) const { return (double)thrust::norm(z); } };
// N0 = condensate occupation Sum|<b_j>|^2; Ebond = <H_hop> (sign: >0 = inverse population / T<0).
struct Diag { double N, E, max_norm_dev, N0, Ebond, R; };  // R = rms cloud radius (sites)
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
      nrm, nrm + d.N, AbsDevOne(), 0.0f, thrust::maximum<real>());
  thrust::device_ptr<cplx> psip(d.psi);
  double N0 = thrust::transform_reduce(psip, psip + d.N, NormC(), 0.0, thrust::plus<double>());
  thrust::device_ptr<real> r2p(d.r2);
  double Rsum = thrust::inner_product(n, n + d.N, r2p, 0.0);   // Sum n_j r_j^2
  double R = (Ntot > 0.0) ? sqrt(Rsum / Ntot) : 0.0;
  return {Ntot, Eloc + Ebnd, (double)mdev, N0, Ebnd, R};
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

// ---------------------------------------------------------------------------
//  Field I/O. Shared binary format with reference/fieldio.py and analysis/tof.py:
//    int32 L, int32 D, then N*D complex<float> as f[m*N + j] (re,im interleaved).
// ---------------------------------------------------------------------------
static void write_field(const std::string& path, const std::vector<cplx>& f, int L, int D) {
  FILE* fp = fopen(path.c_str(), "wb");
  if (!fp) { fprintf(stderr, "cannot open %s for writing\n", path.c_str()); std::exit(1); }
  int32_t hdr[2] = {L, D};
  fwrite(hdr, sizeof(int32_t), 2, fp);
  fwrite(f.data(), sizeof(cplx), (size_t)L * L * D, fp);
  fclose(fp);
}
static std::vector<cplx> read_field(const std::string& path, int& L, int& D) {
  FILE* fp = fopen(path.c_str(), "rb");
  if (!fp) { fprintf(stderr, "cannot open %s for reading\n", path.c_str()); std::exit(1); }
  int32_t hdr[2];
  if (fread(hdr, sizeof(int32_t), 2, fp) != 2) { fprintf(stderr, "bad header %s\n", path.c_str()); std::exit(1); }
  L = hdr[0]; D = hdr[1];
  std::vector<cplx> f((size_t)L * L * D);
  if (fread(f.data(), sizeof(cplx), f.size(), fp) != f.size()) { fprintf(stderr, "short read %s\n", path.c_str()); std::exit(1); }
  fclose(fp);
  return f;
}

// ===========================================================================
//  SELF-TEST: host mirror of BOTH integrators, diffed against the device.
// ===========================================================================
namespace host {
using hc = std::complex<float>;
static inline int FI(int m, int j, int N) { return m * N + j; }
void psi(const std::vector<hc>& f, std::vector<hc>& p, int N, int D) {
  for (int j = 0; j < N; ++j) { hc s(0,0);
    for (int m = 0; m < D-1; ++m) s += sqrtf((float)(m+1))*std::conj(f[FI(m,j,N)])*f[FI(m+1,j,N)];
    p[j] = s; }
}
void phi(const std::vector<hc>& p, const Lattice& lat, std::vector<hc>& ph) {
  for (int j = 0; j < lat.N; ++j) { hc s(0,0);
    for (int d = 0; d < lat.Z; ++d) { int k = lat.nbr[d*lat.N+j]; if (k>=0) s += lat.Jdir[d]*p[k]; }
    ph[j] = s; }
}
void rhs(const std::vector<hc>& f, const Lattice& lat, real U, real V0, real mu0,
         int D, std::vector<hc>& out) {
  int N = lat.N; std::vector<hc> p(N), ph(N); psi(f, p, N, D); phi(p, lat, ph);
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
void phase(std::vector<hc>& f, const Lattice& lat, real U, real V0, real mu0, real dt, int D) {
  for (int j = 0; j < lat.N; ++j) { real base = V0*lat.r2[j]-mu0;
    for (int m = 0; m < D; ++m) { real e = 0.5f*U*m*(m-1)+base*m, p=-e*dt;
      f[FI(m,j,lat.N)] *= hc(cosf(p), sinf(p)); } }
}
void hop(std::vector<hc>& f, const std::vector<hc>& ph, real dt, int N, int D, int order) {
  for (int j = 0; j < N; ++j) { std::vector<hc> t(D), a(D), H(D);
    for (int m=0;m<D;++m){ t[m]=f[FI(m,j,N)]; a[m]=t[m]; } hc P=ph[j], cP=std::conj(P);
    for (int k=1;k<=order;++k){ for(int m=0;m<D;++m){ hc v(0,0);
        if(m+1<D)v+=-cP*sqrtf((float)(m+1))*t[m+1];
        if(m-1>=0)v+=-P*sqrtf((float)m)*t[m-1]; H[m]=v; }
      hc cf(0.f,-dt/(float)k); for(int m=0;m<D;++m){ t[m]=cf*H[m]; a[m]+=t[m]; } }
    for(int m=0;m<D;++m) f[FI(m,j,N)]=a[m]; }
}
void renorm(std::vector<hc>& f, int N, int D) {
  for (int j=0;j<N;++j){ double s=0; for(int m=0;m<D;++m)s+=std::norm(f[FI(m,j,N)]);
    float inv=1.0f/sqrtf((float)s); for(int m=0;m<D;++m)f[FI(m,j,N)]*=inv; }
}
void ss(std::vector<hc>& f, const Lattice& lat, real U, real V0, real mu0, real dt, int D, int order=10) {
  std::vector<hc> p(lat.N), ph(lat.N);
  phase(f,lat,U,V0,mu0,0.5f*dt,D);
  psi(f,p,lat.N,D); phi(p,lat,ph);
  std::vector<hc> fh=f; hop(fh,ph,0.5f*dt,lat.N,D,order);   // predictor
  psi(fh,p,lat.N,D); phi(p,lat,ph);                        // midpoint Phi
  hop(f,ph,dt,lat.N,D,order);
  phase(f,lat,U,V0,mu0,0.5f*dt,D);
  renorm(f,lat.N,D);
}
} // namespace host

int run_selftest() {
  printf("SELF-TEST  (host vs device; both integrators; non-stiff)\n");
  const int L = 8, D = 8, steps = 200;
  const real U = -0.5f, V0 = -2.5e-3f, mu0 = 0.f, dt = 2e-3f;
  Lattice lat = make_square(L, 1.0f, 1.0f);
  std::vector<cplx> f0 = coherent_bump(lat, D, 0.8f, 3.0f);
  size_t sz = (size_t)lat.N * D;

  Device dev; dev.alloc(lat, D);
  int rc = 0;
  std::vector<host::hc> dev_res[2];
  const char* names[2] = {"split-step", "RK4       "};

  for (int which = 0; which < 2; ++which) {
    std::vector<host::hc> fh(sz);
    std::memcpy(fh.data(), f0.data(), sizeof(cplx) * sz);
    CUDA_CHECK(cudaMemcpy(dev.f, f0.data(), sizeof(cplx) * sz, cudaMemcpyHostToDevice));
    Diag d0 = diagnostics(dev, U, V0);
    for (int s = 0; s < steps; ++s) {
      if (which == 0) { splitstep_step(dev, U, V0, mu0, dt); host::ss(fh, lat, U, V0, mu0, dt, D); }
      else            { rk4_step(dev, U, V0, mu0, dt);       host::rk4(fh, lat, U, V0, mu0, dt, D); }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    Diag d1 = diagnostics(dev, U, V0);
    std::vector<cplx> fd(sz);
    CUDA_CHECK(cudaMemcpy(fd.data(), dev.f, sizeof(cplx) * sz, cudaMemcpyDeviceToHost));
    std::vector<host::hc> fdv(sz); double md = 0.0;
    for (size_t i = 0; i < sz; ++i) {
      fdv[i] = std::complex<float>(fd[i].real(), fd[i].imag());
      md = std::max(md, (double)std::abs(fdv[i] - fh[i]));
    }
    dev_res[which] = fdv;
    printf("  [%s] host-vs-device max|df|=%.3e   |dN|=%.3e   |norm-1|=%.3e\n",
           names[which], md, std::abs(d1.N - d0.N), d1.max_norm_dev);
    rc |= (md < 1e-4 && std::abs(d1.N - d0.N) < 1e-3 && d1.max_norm_dev < 1e-4) ? 0 : 1;
  }
  double cross = 0.0;
  for (size_t i = 0; i < sz; ++i) cross = std::max(cross, (double)std::abs(dev_res[0][i] - dev_res[1][i]));
  printf("  split-step vs RK4 (device) max|df|=%.3e   (must agree in non-stiff limit)\n", cross);
  rc |= (cross < 1e-3) ? 0 : 1;
  dev.free_all();
  printf("  --> %s\n", rc == 0 ? "PASS" : "FAIL");
  return rc;
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
  int L = 192, D = 12, steps = 2000, diag_every = 200;
  real dt = 2e-3f, U = -0.5f, V0 = -2.5e-3f, mu0 = 0.f, Jx = 1.f, Jy = 1.f;
  real J1 = 1.f, J2 = 1.f, J3 = 1.f;               // triangular hoppings
  std::string lattice = "square";                  // square | triangular
  bool selftest = false;
  std::string integ = "splitstep";
  std::string loadpath, dumppath, dumpPrefix = "frame";
  int dumpEvery = 0;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto nf = [&](real& v) { v = atof(argv[++i]); };
    auto ni = [&](int& v) { v = atoi(argv[++i]); };
    if (a == "--selftest") selftest = true;
    else if (a == "--L") ni(L);           else if (a == "--D") ni(D);
    else if (a == "--steps") ni(steps);   else if (a == "--dt") nf(dt);
    else if (a == "--U") nf(U);           else if (a == "--V0") nf(V0);
    else if (a == "--Jx") nf(Jx);         else if (a == "--Jy") nf(Jy);
    else if (a == "--lattice") lattice = argv[++i];
    else if (a == "--J1") nf(J1);         else if (a == "--J2") nf(J2);
    else if (a == "--J3") nf(J3);
    else if (a == "--diag") ni(diag_every);
    else if (a == "--integrator") integ = argv[++i];
    else if (a == "--load") loadpath = argv[++i];
    else if (a == "--dump") dumppath = argv[++i];
    else if (a == "--dump-every") dumpEvery = atoi(argv[++i]);
    else if (a == "--dump-prefix") dumpPrefix = argv[++i];
  }
  if (D > DMAX) { fprintf(stderr, "D > DMAX (%d)\n", DMAX); return 1; }
  if (integ != "splitstep" && integ != "rk4") {
    fprintf(stderr, "--integrator must be splitstep or rk4\n"); return 1;
  }
  if (lattice != "square" && lattice != "triangular") {
    fprintf(stderr, "--lattice must be square or triangular\n"); return 1;
  }
  if (selftest) return run_selftest();

  std::vector<cplx> f0;
  if (!loadpath.empty()) {                       // load initial state (sets L, D)
    int Lf, Df; f0 = read_field(loadpath, Lf, Df);
    L = Lf; D = Df;
    printf("loaded initial state from %s (L=%d, D=%d)\n", loadpath.c_str(), L, D);
    if (D > DMAX) { fprintf(stderr, "loaded D > DMAX (%d)\n", DMAX); return 1; }
  }

  Lattice lat = (lattice == "triangular") ? make_triangular(L, J1, J2, J3)
                                          : make_square(L, Jx, Jy);
  if (lattice == "triangular")
    printf("triangular %dx%d  N=%d  D=%d  J1=%g J2=%g J3=%g  U=%g V0=%g  dt=%g  steps=%d\n",
           L, L, lat.N, D, J1, J2, J3, U, V0, dt, steps);
  else
    printf("square %dx%d  N=%d  D=%d  Jx=%g Jy=%g  U=%g V0=%g  dt=%g  steps=%d\n",
           L, L, lat.N, D, Jx, Jy, U, V0, dt, steps);
  printf("integrator: %s   state buffer: %.1f MB each (RK4 holds 6; split-step ~2)\n",
         integ.c_str(), sizeof(cplx) * (double)lat.N * D / 1e6);

  Device dev; dev.alloc(lat, D);
  if (loadpath.empty()) f0 = coherent_bump(lat, D, 0.8f, 0.15f * L);
  CUDA_CHECK(cudaMemcpy(dev.f, f0.data(), sizeof(cplx) * lat.N * D, cudaMemcpyHostToDevice));

  Diag d0 = diagnostics(dev, U, V0);
  printf("step %6d   N=%.2f  N0/N=%.4f  K=%+.3f  R=%.1f  |norm-1|=%.1e\n",
         0, d0.N, d0.N0 / d0.N, d0.Ebond, d0.R, d0.max_norm_dev);
  auto dump_frame = [&](int step_) {
    std::vector<cplx> fo((size_t)lat.N * D);
    CUDA_CHECK(cudaMemcpy(fo.data(), dev.f, sizeof(cplx) * lat.N * D, cudaMemcpyDeviceToHost));
    char fn[512]; snprintf(fn, sizeof fn, "%s_%05d.bin", dumpPrefix.c_str(), step_);
    write_field(fn, fo, L, D);
  };
  if (dumpEvery > 0) dump_frame(0);
  for (int s = 1; s <= steps; ++s) {
    step(dev, integ, U, V0, mu0, dt);
    if (s % diag_every == 0) {
      Diag d = diagnostics(dev, U, V0);
      printf("step %6d   N=%.2f  N0/N=%.4f  K=%+.3f  R=%.1f  |norm-1|=%.1e  dN=%.1e\n",
             s, d.N, d.N0 / d.N, d.Ebond, d.R, d.max_norm_dev, std::abs(d.N - d0.N));
    }
    if (dumpEvery > 0 && s % dumpEvery == 0) dump_frame(s);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  if (!dumppath.empty()) {
    std::vector<cplx> fout((size_t)lat.N * D);
    CUDA_CHECK(cudaMemcpy(fout.data(), dev.f, sizeof(cplx) * lat.N * D, cudaMemcpyDeviceToHost));
    write_field(dumppath, fout, L, D);
    printf("dumped final state -> %s\n", dumppath.c_str());
  }
  dev.free_all();
  return 0;
}
