// Standalone verification of the split-step ALGORITHM used by the CUDA kernels.
// Pure C++ (no CUDA) -> compiles with g++. It replicates the host:: mirror from
// tdgw.cu and checks (a) conservation of per-site norm / N_tot / E_tot, and
// (b) that the Strang split-step converges to the trusted RK45 Python reference
// as dt -> 0 (here we just confirm 2nd-order-ish error and tight conservation).
#include <cstdio>
#include <cmath>
#include <vector>
#include <complex>
using hc = std::complex<float>;
using real = float;

struct Lattice { int L,N,Z; std::vector<int> nbr; std::vector<real> Jdir, r2; };
Lattice make_square(int L, real Jx, real Jy){
  Lattice lat; lat.L=L; lat.N=L*L; lat.Z=4; lat.nbr.assign(4*lat.N,-1);
  lat.Jdir={Jx,Jx,Jy,Jy}; lat.r2.resize(lat.N);
  real cx=(L-1)*0.5f, cy=(L-1)*0.5f; auto id=[L](int x,int y){return x+L*y;};
  for(int y=0;y<L;++y)for(int x=0;x<L;++x){int j=id(x,y);
    if(x+1<L)lat.nbr[0*lat.N+j]=id(x+1,y); if(x-1>=0)lat.nbr[1*lat.N+j]=id(x-1,y);
    if(y+1<L)lat.nbr[2*lat.N+j]=id(x,y+1); if(y-1>=0)lat.nbr[3*lat.N+j]=id(x,y-1);
    lat.r2[j]=(x-cx)*(x-cx)+(y-cy)*(y-cy);} return lat;
}
static inline int FI(int m,int j,int N){return m*N+j;}
void psi(const std::vector<hc>&f,std::vector<hc>&p,int N,int D){
  for(int j=0;j<N;++j){hc s(0,0);for(int m=0;m<D-1;++m)
    s+=sqrtf((float)(m+1))*std::conj(f[FI(m,j,N)])*f[FI(m+1,j,N)]; p[j]=s;}}
void phi(const std::vector<hc>&p,const Lattice&lat,std::vector<hc>&ph){
  for(int j=0;j<lat.N;++j){hc s(0,0);for(int d=0;d<lat.Z;++d){int k=lat.nbr[d*lat.N+j];
    if(k>=0)s+=lat.Jdir[d]*p[k];} ph[j]=s;}}
void phase(std::vector<hc>&f,const Lattice&lat,real U,real V0,real mu0,real dt,int D){
  for(int j=0;j<lat.N;++j){real base=V0*lat.r2[j]-mu0;for(int m=0;m<D;++m){
    real eps=0.5f*U*m*(m-1)+base*m,p=-eps*dt; f[FI(m,j,lat.N)]*=hc(cosf(p),sinf(p));}}}
void hop(std::vector<hc>&f,const std::vector<hc>&ph,real dt,int N,int D,int order){
  for(int j=0;j<N;++j){std::vector<hc> term(D),acc(D),Ht(D);
    for(int m=0;m<D;++m){term[m]=f[FI(m,j,N)];acc[m]=term[m];}
    hc P=ph[j],cP=std::conj(P);
    for(int k=1;k<=order;++k){for(int m=0;m<D;++m){hc v(0,0);
        if(m+1<D)v+=-cP*sqrtf((float)(m+1))*term[m+1];
        if(m-1>=0)v+=-P*sqrtf((float)m)*term[m-1]; Ht[m]=v;}
      hc coef(0.f,-dt/(float)k); for(int m=0;m<D;++m){term[m]=coef*Ht[m];acc[m]+=term[m];}}
    for(int m=0;m<D;++m)f[FI(m,j,N)]=acc[m];}}
void step(std::vector<hc>&f,const Lattice&lat,real U,real V0,real mu0,real dt,int D,int order){
  std::vector<hc> p(lat.N),ph(lat.N);
  phase(f,lat,U,V0,mu0,0.5f*dt,D); psi(f,p,lat.N,D); phi(p,lat,ph);
  hop(f,ph,dt,lat.N,D,order); phase(f,lat,U,V0,mu0,0.5f*dt,D);}

// --- full coupled RHS (Phi computed instantaneously) + classic RK4 ---------
void rhs(const std::vector<hc>&f,const Lattice&lat,real U,real V0,real mu0,int D,std::vector<hc>&out){
  int N=lat.N; std::vector<hc> p(N),ph(N); psi(f,p,N,D); phi(p,lat,ph);
  for(int j=0;j<N;++j){real base=V0*lat.r2[j]-mu0;
    for(int m=0;m<D;++m){hc val=hc(0,0); real eps=0.5f*U*m*(m-1)+base*m;
      val+=eps*f[FI(m,j,N)];
      if(m+1<D)val+=-std::conj(ph[j])*sqrtf((float)(m+1))*f[FI(m+1,j,N)];
      if(m-1>=0)val+=-ph[j]*sqrtf((float)m)*f[FI(m-1,j,N)];
      out[FI(m,j,N)]=hc(0,-1)*val;}}}
void rk4(std::vector<hc>&f,const Lattice&lat,real U,real V0,real mu0,real dt,int D){
  size_t n=f.size(); std::vector<hc> k1(n),k2(n),k3(n),k4(n),t(n);
  rhs(f,lat,U,V0,mu0,D,k1);
  for(size_t i=0;i<n;++i)t[i]=f[i]+0.5f*dt*k1[i]; rhs(t,lat,U,V0,mu0,D,k2);
  for(size_t i=0;i<n;++i)t[i]=f[i]+0.5f*dt*k2[i]; rhs(t,lat,U,V0,mu0,D,k3);
  for(size_t i=0;i<n;++i)t[i]=f[i]+dt*k3[i];      rhs(t,lat,U,V0,mu0,D,k4);
  for(size_t i=0;i<n;++i)f[i]+=(dt/6.f)*(k1[i]+2.f*k2[i]+2.f*k3[i]+k4[i]);}

double Ntot(const std::vector<hc>&f,int N,int D){double n=0;for(int j=0;j<N;++j)
  for(int m=0;m<D;++m)n+=m*std::norm(f[FI(m,j,N)]); return n;}
double Etot(const std::vector<hc>&f,const Lattice&lat,real U,real V0,int D){
  std::vector<hc> p(lat.N),ph(lat.N); psi(f,p,lat.N,D); phi(p,lat,ph);
  double el=0; for(int j=0;j<lat.N;++j)for(int m=0;m<D;++m)
    el+=(0.5*U*m*(m-1)+V0*lat.r2[j]*m)*std::norm(f[FI(m,j,lat.N)]);
  double eb=0; for(int j=0;j<lat.N;++j) eb+=-std::real(std::conj(p[j])*ph[j]);
  return el+eb;}
double maxnormdev(const std::vector<hc>&f,int N,int D){double mx=0;for(int j=0;j<N;++j){
  double s=0;for(int m=0;m<D;++m)s+=std::norm(f[FI(m,j,N)]); mx=std::max(mx,fabs(s-1.0));}return mx;}

int main(){
  const int L=6,D=7,order=8; const real U=-0.5f,V0=-2.5e-3f,mu0=0.f;
  Lattice lat=make_square(L,1.f,1.f);
  auto init=[&](){std::vector<hc> f((size_t)lat.N*D,hc(0,0)); real cx=(L-1)*0.5f,cy=(L-1)*0.5f;
    for(int y=0;y<L;++y)for(int x=0;x<L;++x){int j=x+L*y;
      real a=0.8f*expf(-((x-cx)*(x-cx)+(y-cy)*(y-cy))/9.0f);
      double fact=1,nrm=0; std::vector<double> c(D);
      for(int m=0;m<D;++m){if(m>0)fact*=m; c[m]=exp(-0.5*a*a)*pow((double)a,m)/sqrt(fact); nrm+=c[m]*c[m];}
      for(int m=0;m<D;++m)f[(size_t)m*lat.N+j]=hc((float)(c[m]/sqrt(nrm)),0.f);} return f;};

  double N0ref=Ntot(init(),lat.N,D);
  printf("Conservation check (g++, FP32), %dx%d D=%d, N_tot0=%.4f, T=2.0\n",L,L,D,N0ref);
  double T=2.0;
  printf("\n[A] Strang split-step (freeze Phi):\n");
  for(real dt : {4e-3f,2e-3f,1e-3f}){
    auto f=init(); double N0=Ntot(f,lat.N,D),E0=Etot(f,lat,U,V0,D);
    int steps=(int)lround(T/dt);
    for(int s=0;s<steps;++s) step(f,lat,U,V0,mu0,dt,D,order);
    printf("  dt=%.1e  |dN|=%.2e  |dE|=%.2e  max|norm-1|=%.2e\n",
           dt,fabs(Ntot(f,lat.N,D)-N0),fabs(Etot(f,lat,U,V0,D)-E0),maxnormdev(f,lat.N,D));
  }
  printf("\n[B] classic RK4 on full coupled RHS:\n");
  for(real dt : {4e-3f,2e-3f,1e-3f}){
    auto f=init(); double N0=Ntot(f,lat.N,D),E0=Etot(f,lat,U,V0,D);
    int steps=(int)lround(T/dt);
    for(int s=0;s<steps;++s) rk4(f,lat,U,V0,mu0,dt,D);
    printf("  dt=%.1e  |dN|=%.2e  |dE|=%.2e  max|norm-1|=%.2e\n",
           dt,fabs(Ntot(f,lat.N,D)-N0),fabs(Etot(f,lat,U,V0,D)-E0),maxnormdev(f,lat.N,D));
  }
  return 0;
}
