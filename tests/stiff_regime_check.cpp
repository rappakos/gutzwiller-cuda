// Does the integrator choice survive the REAL regime U/J >> 1 (deep lattice,
// J/U ~ 0.0023 in Rapp 2013)? Compares, at a fixed dt where the on-site phase
// is under-resolved, (A) exact-diagonal Strang split-step vs (B) plain RK4,
// each against a well-resolved fine-dt RK4 reference. Pure C++ / g++.
#include <cstdio>
#include <cmath>
#include <vector>
#include <complex>
using hc=std::complex<float>; using real=float;
struct Lat{int L,N,Z;std::vector<int>nbr;std::vector<real>J,r2;};
Lat sq(int L,real Jx,real Jy){Lat l;l.L=L;l.N=L*L;l.Z=4;l.nbr.assign(4*l.N,-1);
 l.J={Jx,Jx,Jy,Jy};l.r2.resize(l.N);real c=(L-1)*0.5f;auto id=[L](int x,int y){return x+L*y;};
 for(int y=0;y<L;++y)for(int x=0;x<L;++x){int j=id(x,y);
  if(x+1<L)l.nbr[0*l.N+j]=id(x+1,y);if(x-1>=0)l.nbr[1*l.N+j]=id(x-1,y);
  if(y+1<L)l.nbr[2*l.N+j]=id(x,y+1);if(y-1>=0)l.nbr[3*l.N+j]=id(x,y-1);
  l.r2[j]=(x-c)*(x-c)+(y-c)*(y-c);}return l;}
static inline int FI(int m,int j,int N){return m*N+j;}
void psi(const std::vector<hc>&f,std::vector<hc>&p,int N,int D){for(int j=0;j<N;++j){hc s(0,0);
 for(int m=0;m<D-1;++m)s+=sqrtf((float)(m+1))*std::conj(f[FI(m,j,N)])*f[FI(m+1,j,N)];p[j]=s;}}
void phiv(const std::vector<hc>&p,const Lat&l,std::vector<hc>&ph){for(int j=0;j<l.N;++j){hc s(0,0);
 for(int d=0;d<l.Z;++d){int k=l.nbr[d*l.N+j];if(k>=0)s+=l.J[d]*p[k];}ph[j]=s;}}
// exact diagonal phase
void phase(std::vector<hc>&f,const Lat&l,real U,real V0,real mu0,real dt,int D){for(int j=0;j<l.N;++j){
 real b=V0*l.r2[j]-mu0;for(int m=0;m<D;++m){real e=0.5f*U*m*(m-1)+b*m,p=-e*dt;f[FI(m,j,l.N)]*=hc(cosf(p),sinf(p));}}}
// hop via Taylor exp(-iH dt), Phi frozen
void hop(std::vector<hc>&f,const std::vector<hc>&ph,real dt,int N,int D,int ord){for(int j=0;j<N;++j){
 std::vector<hc>t(D),a(D),H(D);for(int m=0;m<D;++m){t[m]=f[FI(m,j,N)];a[m]=t[m];}hc P=ph[j],cP=std::conj(P);
 for(int k=1;k<=ord;++k){for(int m=0;m<D;++m){hc v(0,0);if(m+1<D)v+=-cP*sqrtf((float)(m+1))*t[m+1];
  if(m-1>=0)v+=-P*sqrtf((float)m)*t[m-1];H[m]=v;}hc cf(0.f,-dt/(float)k);for(int m=0;m<D;++m){t[m]=cf*H[m];a[m]+=t[m];}}
 for(int m=0;m<D;++m)f[FI(m,j,N)]=a[m];}}
void sstep(std::vector<hc>&f,const Lat&l,real U,real V0,real mu0,real dt,int D){std::vector<hc>p(l.N),ph(l.N);
 phase(f,l,U,V0,mu0,0.5f*dt,D);psi(f,p,l.N,D);phiv(p,l,ph);hop(f,ph,dt,l.N,D,10);phase(f,l,U,V0,mu0,0.5f*dt,D);}
void rhs(const std::vector<hc>&f,const Lat&l,real U,real V0,real mu0,int D,std::vector<hc>&o){int N=l.N;
 std::vector<hc>p(N),ph(N);psi(f,p,N,D);phiv(p,l,ph);for(int j=0;j<N;++j){real b=V0*l.r2[j]-mu0;
 for(int m=0;m<D;++m){hc v(0,0);real e=0.5f*U*m*(m-1)+b*m;v+=e*f[FI(m,j,N)];
  if(m+1<D)v+=-std::conj(ph[j])*sqrtf((float)(m+1))*f[FI(m+1,j,N)];
  if(m-1>=0)v+=-ph[j]*sqrtf((float)m)*f[FI(m-1,j,N)];o[FI(m,j,N)]=hc(0,-1)*v;}}}
void rk4(std::vector<hc>&f,const Lat&l,real U,real V0,real mu0,real h,int D){size_t n=f.size();
 std::vector<hc>k1(n),k2(n),k3(n),k4(n),t(n);rhs(f,l,U,V0,mu0,D,k1);
 for(size_t i=0;i<n;++i)t[i]=f[i]+0.5f*h*k1[i];rhs(t,l,U,V0,mu0,D,k2);
 for(size_t i=0;i<n;++i)t[i]=f[i]+0.5f*h*k2[i];rhs(t,l,U,V0,mu0,D,k3);
 for(size_t i=0;i<n;++i)t[i]=f[i]+h*k3[i];rhs(t,l,U,V0,mu0,D,k4);
 for(size_t i=0;i<n;++i)f[i]+=(h/6.f)*(k1[i]+2.f*k2[i]+2.f*k3[i]+k4[i]);}
double diff(const std::vector<hc>&a,const std::vector<hc>&b){double m=0;for(size_t i=0;i<a.size();++i)m=std::max(m,(double)std::abs(a[i]-b[i]));return m;}
double Ntot(const std::vector<hc>&f,int N,int D){double n=0;for(int j=0;j<N;++j)for(int m=0;m<D;++m)n+=m*std::norm(f[FI(m,j,N)]);return n;}
int main(){const int L=4,D=7;const real V0=0,mu0=0,J=1.f;
 for(real U : {-2.f,-50.f,-200.f}){
  Lat l=sq(L,J,J);
  auto init=[&](){std::vector<hc>f((size_t)l.N*D,hc(0,0));real c=(L-1)*0.5f;
   for(int y=0;y<L;++y)for(int x=0;x<L;++x){int j=x+L*y;real a=0.8f*expf(-((x-c)*(x-c)+(y-c)*(y-c))/4.f);
   double fc=1,nr=0;std::vector<double>cc(D);for(int m=0;m<D;++m){if(m>0)fc*=m;cc[m]=exp(-0.5*a*a)*pow((double)a,m)/sqrt(fc);nr+=cc[m]*cc[m];}
   for(int m=0;m<D;++m)f[(size_t)m*l.N+j]=hc((float)(cc[m]/sqrt(nr)),0.f);}return f;};
  double T=0.1; real dt=2e-3f; int steps=(int)(T/dt);
  // fine reference: RK4 at dt/200 (on-site phase well resolved)
  auto ref=init(); int fsteps=steps*200; real fdt=dt/200;
  for(int s=0;s<fsteps;++s) rk4(ref,l,U,V0,mu0,fdt,D);
  auto fa=init(); for(int s=0;s<steps;++s) sstep(fa,l,U,V0,mu0,dt,D);   // split-step
  auto fb=init(); for(int s=0;s<steps;++s) rk4(fb,l,U,V0,mu0,dt,D);     // plain RK4
  double Uo=fabs(U)* (D-1) *dt;  // ~ max on-site phase per step (rad)
  printf("U/J=%6.1f  |U|*m_max*dt=%.2f rad/step   split-step err=%.2e  RK4 err=%.2e   (RK4 |dN|=%.2e)\n",
         U/J, Uo, diff(fa,ref), diff(fb,ref), fabs(Ntot(fb,l.N,D)-Ntot(init(),l.N,D)));
 }
 return 0;}
