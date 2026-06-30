// Host prototype of the EXACT-DIAGONAL Strang split-step + per-site renormalize.
// Goal: confirm it (a) matches RK4 in the non-stiff limit, (b) stays stable and
// conserves N_tot in the stiff U/J regime where RK4 is unusable, (c) converges as
// dt -> 0. Pure C++/g++; this is the numerics check before porting to CUDA.
#include <cstdio>
#include <cmath>
#include <vector>
#include <complex>
using hc = std::complex<float>;
using real = float;

struct Lat { int L,N,Z; std::vector<int> nbr; std::vector<real> J,r2; };
Lat sq(int L, real Jx, real Jy){ Lat l; l.L=L; l.N=L*L; l.Z=4; l.nbr.assign(4*l.N,-1);
  l.J={Jx,Jx,Jy,Jy}; l.r2.resize(l.N); real c=(L-1)*0.5f; auto id=[L](int x,int y){return x+L*y;};
  for(int y=0;y<L;++y)for(int x=0;x<L;++x){int j=id(x,y);
    if(x+1<L)l.nbr[0*l.N+j]=id(x+1,y); if(x-1>=0)l.nbr[1*l.N+j]=id(x-1,y);
    if(y+1<L)l.nbr[2*l.N+j]=id(x,y+1); if(y-1>=0)l.nbr[3*l.N+j]=id(x,y-1);
    l.r2[j]=(x-c)*(x-c)+(y-c)*(y-c);} return l; }
static inline int FI(int m,int j,int N){return m*N+j;}

void psi(const std::vector<hc>&f,std::vector<hc>&p,int N,int D){for(int j=0;j<N;++j){hc s(0,0);
  for(int m=0;m<D-1;++m)s+=sqrtf((float)(m+1))*std::conj(f[FI(m,j,N)])*f[FI(m+1,j,N)]; p[j]=s;}}
void phiv(const std::vector<hc>&p,const Lat&l,std::vector<hc>&ph){for(int j=0;j<l.N;++j){hc s(0,0);
  for(int d=0;d<l.Z;++d){int k=l.nbr[d*l.N+j]; if(k>=0)s+=l.J[d]*p[k];} ph[j]=s;}}

// exact unitary rotation of the diagonal on-site part (no dt limit from large U)
void phase(std::vector<hc>&f,const Lat&l,real U,real V0,real mu0,real dt,int D){
  for(int j=0;j<l.N;++j){real b=V0*l.r2[j]-mu0; for(int m=0;m<D;++m){
    real e=0.5f*U*m*(m-1)+b*m, p=-e*dt; f[FI(m,j,l.N)]*=hc(cosf(p),sinf(p));}}}
// hop sub-step: exp(-i H dt) with Phi frozen, H tridiagonal; Taylor to `ord`
void hop(std::vector<hc>&f,const std::vector<hc>&ph,real dt,int N,int D,int ord){
  for(int j=0;j<N;++j){std::vector<hc> t(D),a(D),H(D);
    for(int m=0;m<D;++m){t[m]=f[FI(m,j,N)];a[m]=t[m];} hc P=ph[j],cP=std::conj(P);
    for(int k=1;k<=ord;++k){for(int m=0;m<D;++m){hc v(0,0);
        if(m+1<D)v+=-cP*sqrtf((float)(m+1))*t[m+1];
        if(m-1>=0)v+=-P*sqrtf((float)m)*t[m-1]; H[m]=v;}
      hc cf(0.f,-dt/(float)k); for(int m=0;m<D;++m){t[m]=cf*H[m];a[m]+=t[m];}}
    for(int m=0;m<D;++m)f[FI(m,j,N)]=a[m];}}
// per-site renormalize: enforce sum_m |f_m|^2 = 1
void renorm(std::vector<hc>&f,int N,int D){for(int j=0;j<N;++j){double s=0;
  for(int m=0;m<D;++m)s+=std::norm(f[FI(m,j,N)]); float inv=1.0f/sqrtf((float)s);
  for(int m=0;m<D;++m)f[FI(m,j,N)]*=inv;}}

// One Strang split-step. midphi=true recomputes Phi at the half-hop (predictor-corrector).
void ss(std::vector<hc>&f,const Lat&l,real U,real V0,real mu0,real dt,int D,int ord,bool midphi){
  std::vector<hc> p(l.N),ph(l.N);
  phase(f,l,U,V0,mu0,0.5f*dt,D);
  psi(f,p,l.N,D); phiv(p,l,ph);
  if(midphi){ std::vector<hc> fh=f; hop(fh,ph,0.5f*dt,l.N,D,ord);
              psi(fh,p,l.N,D); phiv(p,l,ph); }
  hop(f,ph,dt,l.N,D,ord);
  phase(f,l,U,V0,mu0,0.5f*dt,D);
  renorm(f,l.N,D);
}

// reference RHS + classic RK4 (the independently validated path)
void rhs(const std::vector<hc>&f,const Lat&l,real U,real V0,real mu0,int D,std::vector<hc>&o){int N=l.N;
  std::vector<hc> p(N),ph(N); psi(f,p,N,D); phiv(p,l,ph);
  for(int j=0;j<N;++j){real b=V0*l.r2[j]-mu0; for(int m=0;m<D;++m){hc v(0,0); real e=0.5f*U*m*(m-1)+b*m;
    v+=e*f[FI(m,j,N)];
    if(m+1<D)v+=-std::conj(ph[j])*sqrtf((float)(m+1))*f[FI(m+1,j,N)];
    if(m-1>=0)v+=-ph[j]*sqrtf((float)m)*f[FI(m-1,j,N)]; o[FI(m,j,N)]=hc(0,-1)*v;}}}
void rk4(std::vector<hc>&f,const Lat&l,real U,real V0,real mu0,real h,int D){size_t n=f.size();
  std::vector<hc> k1(n),k2(n),k3(n),k4(n),t(n); rhs(f,l,U,V0,mu0,D,k1);
  for(size_t i=0;i<n;++i)t[i]=f[i]+0.5f*h*k1[i]; rhs(t,l,U,V0,mu0,D,k2);
  for(size_t i=0;i<n;++i)t[i]=f[i]+0.5f*h*k2[i]; rhs(t,l,U,V0,mu0,D,k3);
  for(size_t i=0;i<n;++i)t[i]=f[i]+h*k3[i]; rhs(t,l,U,V0,mu0,D,k4);
  for(size_t i=0;i<n;++i)f[i]+=(h/6.f)*(k1[i]+2.f*k2[i]+2.f*k3[i]+k4[i]);}

double Ntot(const std::vector<hc>&f,int N,int D){double n=0;for(int j=0;j<N;++j)
  for(int m=0;m<D;++m)n+=m*std::norm(f[FI(m,j,N)]); return n;}
double maxdiff(const std::vector<hc>&a,const std::vector<hc>&b){double m=0;
  for(size_t i=0;i<a.size();++i)m=std::max(m,(double)std::abs(a[i]-b[i])); return m;}
double maxnormdev(const std::vector<hc>&f,int N,int D){double mx=0;for(int j=0;j<N;++j){double s=0;
  for(int m=0;m<D;++m)s+=std::norm(f[FI(m,j,N)]); mx=std::max(mx,fabs(s-1.0));} return mx;}

const int L=6, D=7, ORD=10; const real V0=-2.5e-3f, mu0=0.f, J=1.f;
std::vector<hc> init(){ std::vector<hc> f((size_t)L*L*D,hc(0,0)); real c=(L-1)*0.5f;
  for(int y=0;y<L;++y)for(int x=0;x<L;++x){int j=x+L*y; real a=0.8f*expf(-((x-c)*(x-c)+(y-c)*(y-c))/9.f);
    double fc=1,nr=0; std::vector<double> cc(D);
    for(int m=0;m<D;++m){if(m>0)fc*=m; cc[m]=exp(-0.5*a*a)*pow((double)a,m)/sqrt(fc); nr+=cc[m]*cc[m];}
    for(int m=0;m<D;++m)f[(size_t)m*L*L+j]=hc((float)(cc[m]/sqrt(nr)),0.f);} return f; }

int main(){
  Lat l=sq(L,J,J);
  printf("=== [A] non-stiff anchor: split-step vs RK4 (U/J=2, T=1) ===\n");
  { real U=-2.f, T=1.0f, dt=2e-3f; int steps=(int)(T/dt);
    auto a=init(); for(int s=0;s<steps;++s) ss(a,l,U,V0,mu0,dt,D,ORD,false);
    auto b=init(); for(int s=0;s<steps;++s) rk4(b,l,U,V0,mu0,dt,D);
    printf("  split vs RK4  max|df|=%.2e   (split N drift %.2e, RK4 N drift %.2e)\n",
           maxdiff(a,b), fabs(Ntot(a,l.N,D)-Ntot(init(),l.N,D)), fabs(Ntot(b,l.N,D)-Ntot(init(),l.N,D)));
  }
  printf("\n=== [B] stiff regime: split-step stability + N conservation (T=1) ===\n");
  printf("    (RK4 is NaN-unstable here at any of these dt)\n");
  for(real U : {-50.f,-200.f,-435.f}){
    printf("  U/J=%-5.0f\n", U/J);
    double N0=Ntot(init(),l.N,D);
    for(real dt : {1e-2f,5e-3f,2e-3f,1e-3f}){
      int steps=(int)(1.0f/dt);
      auto f=init(); for(int s=0;s<steps;++s) ss(f,l,U,V0,mu0,dt,D,ORD,false);
      double dN=fabs(Ntot(f,l.N,D)-N0)/N0;
      printf("     dt=%.0e  |dN|/N=%.2e  max|norm-1|=%.2e\n", dt, dN, maxnormdev(f,l.N,D));
    }
  }
  printf("\n=== [C] stiff convergence: split-step self-consistency (U/J=200, T=1) ===\n");
  { real U=-200.f, T=1.0f; std::vector<hc> ref=init();
    int rs=(int)(T/2.5e-4f); for(int s=0;s<rs;++s) ss(ref,l,U,V0,mu0,2.5e-4f,D,ORD,false);
    for(real dt : {1e-2f,5e-3f,2e-3f}){ int steps=(int)(T/dt);
      auto f=init(); for(int s=0;s<steps;++s) ss(f,l,U,V0,mu0,dt,D,ORD,false);
      printf("     dt=%.0e  max|df vs fine ref|=%.2e\n", dt, maxdiff(f,ref)); }
  }
  printf("\n=== [D] frozen-Phi vs midpoint-Phi (predictor-corrector), N drift ===\n");
  for(real U : {-2.f,-200.f}){ double N0=Ntot(init(),l.N,D);
    printf("  U/J=%-5.0f (T=1)\n", U/J);
    for(real dt : {1e-2f,5e-3f,2e-3f,1e-3f}){ int steps=(int)(1.0f/dt);
      auto a=init(); for(int s=0;s<steps;++s) ss(a,l,U,V0,mu0,dt,D,ORD,false);
      auto b=init(); for(int s=0;s<steps;++s) ss(b,l,U,V0,mu0,dt,D,ORD,true);
      printf("     dt=%.0e  frozen |dN|/N=%.2e   midpoint |dN|/N=%.2e\n",
             dt, fabs(Ntot(a,l.N,D)-N0)/N0, fabs(Ntot(b,l.N,D)-N0)/N0); }
  }
  printf("\n=== [E] correctness anchor: midpoint split-step vs trusted RK4 (U/J=2) ===\n");
  { real U=-2.f;
    for(real dt : {2e-3f,1e-3f}){ int steps=(int)(1.0f/dt);
      auto a=init(); for(int s=0;s<steps;++s) ss(a,l,U,V0,mu0,dt,D,ORD,true);
      auto b=init(); for(int s=0;s<steps;++s) rk4(b,l,U,V0,mu0,dt,D);
      printf("     dt=%.0e  max|df vs RK4|=%.2e\n", dt, maxdiff(a,b)); }
  }
  return 0;
}
