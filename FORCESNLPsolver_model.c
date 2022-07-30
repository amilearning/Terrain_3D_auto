/* This file was automatically generated by CasADi.
   The CasADi copyright holders make no ownership claim of its contents. */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) FORCESNLPsolver_model_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real FORCESNLPsolver_float
#endif

#ifndef casadi_int
#define casadi_int solver_int32_default
#endif

/* Add prefix to internal symbols */
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_f1 CASADI_PREFIX(f1)
#define casadi_f2 CASADI_PREFIX(f2)
#define casadi_f3 CASADI_PREFIX(f3)
#define casadi_f4 CASADI_PREFIX(f4)
#define casadi_f5 CASADI_PREFIX(f5)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_s4 CASADI_PREFIX(s4)
#define casadi_s5 CASADI_PREFIX(s5)
#define casadi_sign CASADI_PREFIX(sign)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

casadi_real casadi_sign(casadi_real x) { return x<0 ? -1 : x>0 ? 1 : x;}

static const casadi_int casadi_s0[13] = {9, 1, 0, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8};
static const casadi_int casadi_s1[7] = {3, 1, 0, 3, 0, 1, 2};
static const casadi_int casadi_s2[5] = {1, 1, 0, 1, 0};
static const casadi_int casadi_s3[15] = {1, 9, 0, 0, 0, 1, 2, 3, 3, 3, 3, 3, 0, 0, 0};
static const casadi_int casadi_s4[11] = {7, 1, 0, 7, 0, 1, 2, 3, 4, 5, 6};
static const casadi_int casadi_s5[48] = {7, 9, 0, 6, 12, 13, 14, 17, 23, 26, 30, 36, 0, 1, 2, 3, 4, 5, 0, 1, 2, 4, 5, 6, 0, 1, 0, 1, 2, 0, 1, 2, 3, 4, 5, 0, 1, 4, 0, 1, 2, 5, 0, 1, 2, 4, 5, 6};

/* FORCESNLPsolver_objective_0:(i0[9],i1[3])->(o0) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a2, a3;
  a0=2.;
  a1=arg[0]? arg[0][2] : 0;
  a2=arg[1]? arg[1][0] : 0;
  a1=(a1-a2);
  a1=fabs(a1);
  a1=(a0*a1);
  a2=arg[0]? arg[0][3] : 0;
  a3=arg[1]? arg[1][1] : 0;
  a2=(a2-a3);
  a2=fabs(a2);
  a0=(a0*a2);
  a1=(a1+a0);
  a0=20.;
  a2=arg[0]? arg[0][4] : 0;
  a3=arg[1]? arg[1][2] : 0;
  a2=(a2-a3);
  a2=fabs(a2);
  a0=(a0*a2);
  a1=(a1+a0);
  if (res[0]!=0) res[0][0]=a1;
  return 0;
}

CASADI_SYMBOL_EXPORT int FORCESNLPsolver_objective_0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int FORCESNLPsolver_objective_0_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int FORCESNLPsolver_objective_0_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void FORCESNLPsolver_objective_0_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int FORCESNLPsolver_objective_0_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void FORCESNLPsolver_objective_0_release(int mem) {
}

CASADI_SYMBOL_EXPORT void FORCESNLPsolver_objective_0_incref(void) {
}

CASADI_SYMBOL_EXPORT void FORCESNLPsolver_objective_0_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int FORCESNLPsolver_objective_0_n_in(void) { return 2;}

CASADI_SYMBOL_EXPORT casadi_int FORCESNLPsolver_objective_0_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real FORCESNLPsolver_objective_0_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* FORCESNLPsolver_objective_0_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* FORCESNLPsolver_objective_0_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* FORCESNLPsolver_objective_0_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* FORCESNLPsolver_objective_0_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int FORCESNLPsolver_objective_0_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 2;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}

/* FORCESNLPsolver_dobjective_0:(i0[9],i1[3])->(o0[1x9,3nz]) */
static int casadi_f1(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a2;
  a0=2.;
  a1=arg[0]? arg[0][2] : 0;
  a2=arg[1]? arg[1][0] : 0;
  a1=(a1-a2);
  a1=casadi_sign(a1);
  a1=(a0*a1);
  if (res[0]!=0) res[0][0]=a1;
  a1=arg[0]? arg[0][3] : 0;
  a2=arg[1]? arg[1][1] : 0;
  a1=(a1-a2);
  a1=casadi_sign(a1);
  a0=(a0*a1);
  if (res[0]!=0) res[0][1]=a0;
  a0=20.;
  a1=arg[0]? arg[0][4] : 0;
  a2=arg[1]? arg[1][2] : 0;
  a1=(a1-a2);
  a1=casadi_sign(a1);
  a0=(a0*a1);
  if (res[0]!=0) res[0][2]=a0;
  return 0;
}

CASADI_SYMBOL_EXPORT int FORCESNLPsolver_dobjective_0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f1(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int FORCESNLPsolver_dobjective_0_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int FORCESNLPsolver_dobjective_0_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void FORCESNLPsolver_dobjective_0_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int FORCESNLPsolver_dobjective_0_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void FORCESNLPsolver_dobjective_0_release(int mem) {
}

CASADI_SYMBOL_EXPORT void FORCESNLPsolver_dobjective_0_incref(void) {
}

CASADI_SYMBOL_EXPORT void FORCESNLPsolver_dobjective_0_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int FORCESNLPsolver_dobjective_0_n_in(void) { return 2;}

CASADI_SYMBOL_EXPORT casadi_int FORCESNLPsolver_dobjective_0_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real FORCESNLPsolver_dobjective_0_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* FORCESNLPsolver_dobjective_0_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* FORCESNLPsolver_dobjective_0_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* FORCESNLPsolver_dobjective_0_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* FORCESNLPsolver_dobjective_0_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s3;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int FORCESNLPsolver_dobjective_0_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 2;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}

/* FORCESNLPsolver_dynamics_0:(i0[9],i1[3])->(o0[7]) */
static int casadi_f2(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a32, a4, a5, a6, a7, a8, a9;
  a0=arg[0]? arg[0][2] : 0;
  a1=8.3333333333333332e-03;
  a2=arg[0]? arg[0][5] : 0;
  a3=arg[0]? arg[0][4] : 0;
  a4=cos(a3);
  a4=(a2*a4);
  a5=arg[0]? arg[0][6] : 0;
  a6=sin(a3);
  a6=(a5*a6);
  a4=(a4-a6);
  a6=2.;
  a7=2.5000000000000001e-02;
  a8=arg[0]? arg[0][0] : 0;
  a9=(a7*a8);
  a9=(a2+a9);
  a10=arg[0]? arg[0][7] : 0;
  a11=(a7*a10);
  a11=(a3+a11);
  a12=cos(a11);
  a12=(a9*a12);
  a13=4.0350877192982454e-01;
  a14=arg[0]? arg[0][1] : 0;
  a15=(a14*a2);
  a16=arg[0]? arg[0][8] : 0;
  a17=(a16*a8);
  a15=(a15+a17);
  a15=(a13*a15);
  a17=(a7*a15);
  a17=(a5+a17);
  a18=sin(a11);
  a18=(a17*a18);
  a12=(a12-a18);
  a12=(a6*a12);
  a4=(a4+a12);
  a12=(a7*a8);
  a12=(a2+a12);
  a18=1.7543859649122806e+00;
  a19=(a14*a2);
  a20=(a16*a8);
  a19=(a19+a20);
  a19=(a18*a19);
  a20=(a7*a19);
  a20=(a10+a20);
  a21=(a7*a20);
  a21=(a3+a21);
  a22=cos(a21);
  a22=(a12*a22);
  a23=(a14*a9);
  a24=(a7*a14);
  a24=(a16+a24);
  a25=(a24*a8);
  a23=(a23+a25);
  a23=(a13*a23);
  a25=(a7*a23);
  a25=(a5+a25);
  a26=sin(a21);
  a26=(a25*a26);
  a22=(a22-a26);
  a22=(a6*a22);
  a4=(a4+a22);
  a22=5.0000000000000003e-02;
  a26=(a22*a8);
  a26=(a2+a26);
  a27=(a14*a9);
  a24=(a24*a8);
  a27=(a27+a24);
  a27=(a18*a27);
  a24=(a7*a27);
  a24=(a10+a24);
  a28=(a22*a24);
  a28=(a3+a28);
  a29=cos(a28);
  a29=(a26*a29);
  a30=(a14*a12);
  a7=(a7*a14);
  a7=(a16+a7);
  a31=(a7*a8);
  a30=(a30+a31);
  a30=(a13*a30);
  a31=(a22*a30);
  a31=(a5+a31);
  a32=sin(a28);
  a32=(a31*a32);
  a29=(a29-a32);
  a4=(a4+a29);
  a4=(a1*a4);
  a0=(a0+a4);
  if (res[0]!=0) res[0][0]=a0;
  a0=arg[0]? arg[0][3] : 0;
  a4=sin(a3);
  a4=(a2*a4);
  a29=cos(a3);
  a29=(a5*a29);
  a4=(a4+a29);
  a29=sin(a11);
  a9=(a9*a29);
  a11=cos(a11);
  a17=(a17*a11);
  a9=(a9+a17);
  a9=(a6*a9);
  a4=(a4+a9);
  a9=sin(a21);
  a9=(a12*a9);
  a21=cos(a21);
  a25=(a25*a21);
  a9=(a9+a25);
  a9=(a6*a9);
  a4=(a4+a9);
  a9=sin(a28);
  a9=(a26*a9);
  a28=cos(a28);
  a31=(a31*a28);
  a9=(a9+a31);
  a4=(a4+a9);
  a4=(a1*a4);
  a0=(a0+a4);
  if (res[0]!=0) res[0][1]=a0;
  a20=(a6*a20);
  a20=(a10+a20);
  a24=(a6*a24);
  a20=(a20+a24);
  a12=(a14*a12);
  a7=(a7*a8);
  a12=(a12+a7);
  a12=(a18*a12);
  a7=(a22*a12);
  a7=(a10+a7);
  a20=(a20+a7);
  a20=(a1*a20);
  a3=(a3+a20);
  if (res[0]!=0) res[0][2]=a3;
  a3=(a6*a8);
  a3=(a8+a3);
  a20=(a6*a8);
  a3=(a3+a20);
  a3=(a3+a8);
  a3=(a1*a3);
  a2=(a2+a3);
  if (res[0]!=0) res[0][3]=a2;
  a23=(a6*a23);
  a15=(a15+a23);
  a30=(a6*a30);
  a15=(a15+a30);
  a30=(a14*a26);
  a22=(a22*a14);
  a22=(a16+a22);
  a23=(a22*a8);
  a30=(a30+a23);
  a13=(a13*a30);
  a15=(a15+a13);
  a15=(a1*a15);
  a5=(a5+a15);
  if (res[0]!=0) res[0][4]=a5;
  a27=(a6*a27);
  a19=(a19+a27);
  a12=(a6*a12);
  a19=(a19+a12);
  a26=(a14*a26);
  a22=(a22*a8);
  a26=(a26+a22);
  a18=(a18*a26);
  a19=(a19+a18);
  a19=(a1*a19);
  a10=(a10+a19);
  if (res[0]!=0) res[0][5]=a10;
  a10=(a6*a14);
  a10=(a14+a10);
  a6=(a6*a14);
  a10=(a10+a6);
  a10=(a10+a14);
  a1=(a1*a10);
  a16=(a16+a1);
  if (res[0]!=0) res[0][6]=a16;
  return 0;
}

CASADI_SYMBOL_EXPORT int FORCESNLPsolver_dynamics_0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f2(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int FORCESNLPsolver_dynamics_0_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int FORCESNLPsolver_dynamics_0_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void FORCESNLPsolver_dynamics_0_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int FORCESNLPsolver_dynamics_0_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void FORCESNLPsolver_dynamics_0_release(int mem) {
}

CASADI_SYMBOL_EXPORT void FORCESNLPsolver_dynamics_0_incref(void) {
}

CASADI_SYMBOL_EXPORT void FORCESNLPsolver_dynamics_0_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int FORCESNLPsolver_dynamics_0_n_in(void) { return 2;}

CASADI_SYMBOL_EXPORT casadi_int FORCESNLPsolver_dynamics_0_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real FORCESNLPsolver_dynamics_0_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* FORCESNLPsolver_dynamics_0_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* FORCESNLPsolver_dynamics_0_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* FORCESNLPsolver_dynamics_0_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* FORCESNLPsolver_dynamics_0_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s4;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int FORCESNLPsolver_dynamics_0_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 2;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}

/* FORCESNLPsolver_ddynamics_0:(i0[9],i1[3])->(o0[7x9,36nz]) */
static int casadi_f3(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a4, a40, a41, a42, a43, a44, a45, a46, a47, a48, a49, a5, a50, a51, a52, a53, a6, a7, a8, a9;
  a0=8.3333333333333332e-03;
  a1=2.;
  a2=2.5000000000000001e-02;
  a3=arg[0]? arg[0][4] : 0;
  a4=arg[0]? arg[0][7] : 0;
  a5=(a2*a4);
  a5=(a3+a5);
  a6=cos(a5);
  a7=(a2*a6);
  a8=sin(a5);
  a9=4.0350877192982454e-01;
  a10=arg[0]? arg[0][8] : 0;
  a11=(a9*a10);
  a12=(a2*a11);
  a13=(a8*a12);
  a7=(a7-a13);
  a7=(a1*a7);
  a13=1.7543859649122806e+00;
  a14=arg[0]? arg[0][1] : 0;
  a15=arg[0]? arg[0][5] : 0;
  a16=(a14*a15);
  a17=arg[0]? arg[0][0] : 0;
  a18=(a10*a17);
  a16=(a16+a18);
  a16=(a13*a16);
  a16=(a2*a16);
  a16=(a4+a16);
  a16=(a2*a16);
  a16=(a3+a16);
  a18=cos(a16);
  a19=(a2*a18);
  a20=(a2*a17);
  a20=(a15+a20);
  a21=sin(a16);
  a22=(a13*a10);
  a23=(a2*a22);
  a24=(a2*a23);
  a25=(a21*a24);
  a25=(a20*a25);
  a19=(a19-a25);
  a25=sin(a16);
  a26=(a2*a14);
  a27=(a2*a14);
  a27=(a10+a27);
  a26=(a26+a27);
  a26=(a9*a26);
  a28=(a2*a26);
  a29=(a25*a28);
  a30=arg[0]? arg[0][6] : 0;
  a31=(a2*a17);
  a31=(a15+a31);
  a32=(a14*a31);
  a33=(a27*a17);
  a32=(a32+a33);
  a32=(a9*a32);
  a32=(a2*a32);
  a32=(a30+a32);
  a33=cos(a16);
  a34=(a33*a24);
  a34=(a32*a34);
  a29=(a29+a34);
  a19=(a19-a29);
  a19=(a1*a19);
  a7=(a7+a19);
  a19=5.0000000000000003e-02;
  a29=(a14*a31);
  a34=(a27*a17);
  a29=(a29+a34);
  a29=(a13*a29);
  a29=(a2*a29);
  a4=(a4+a29);
  a4=(a19*a4);
  a4=(a3+a4);
  a29=cos(a4);
  a34=(a19*a29);
  a35=(a19*a17);
  a35=(a15+a35);
  a36=sin(a4);
  a37=(a2*a14);
  a37=(a37+a27);
  a37=(a13*a37);
  a27=(a2*a37);
  a38=(a19*a27);
  a39=(a36*a38);
  a39=(a35*a39);
  a34=(a34-a39);
  a39=sin(a4);
  a40=(a2*a14);
  a41=(a2*a14);
  a41=(a10+a41);
  a40=(a40+a41);
  a40=(a9*a40);
  a42=(a19*a40);
  a43=(a39*a42);
  a44=(a14*a20);
  a45=(a41*a17);
  a44=(a44+a45);
  a44=(a9*a44);
  a44=(a19*a44);
  a44=(a30+a44);
  a45=cos(a4);
  a46=(a45*a38);
  a46=(a44*a46);
  a43=(a43+a46);
  a34=(a34-a43);
  a7=(a7+a34);
  a7=(a0*a7);
  if (res[0]!=0) res[0][0]=a7;
  a7=sin(a5);
  a34=(a2*a7);
  a43=cos(a5);
  a12=(a43*a12);
  a34=(a34+a12);
  a34=(a1*a34);
  a12=sin(a16);
  a46=(a2*a12);
  a47=cos(a16);
  a48=(a47*a24);
  a48=(a20*a48);
  a46=(a46+a48);
  a48=cos(a16);
  a28=(a48*a28);
  a16=sin(a16);
  a24=(a16*a24);
  a24=(a32*a24);
  a28=(a28-a24);
  a46=(a46+a28);
  a46=(a1*a46);
  a34=(a34+a46);
  a46=sin(a4);
  a28=(a19*a46);
  a24=cos(a4);
  a49=(a24*a38);
  a49=(a35*a49);
  a28=(a28+a49);
  a49=cos(a4);
  a42=(a49*a42);
  a4=sin(a4);
  a38=(a4*a38);
  a38=(a44*a38);
  a42=(a42-a38);
  a28=(a28+a42);
  a34=(a34+a28);
  a34=(a0*a34);
  if (res[0]!=0) res[0][1]=a34;
  a23=(a1*a23);
  a27=(a1*a27);
  a23=(a23+a27);
  a27=(a2*a14);
  a27=(a27+a41);
  a27=(a13*a27);
  a41=(a19*a27);
  a23=(a23+a41);
  a23=(a0*a23);
  if (res[0]!=0) res[0][2]=a23;
  if (res[0]!=0) res[0][3]=a19;
  a26=(a1*a26);
  a11=(a11+a26);
  a40=(a1*a40);
  a11=(a11+a40);
  a40=(a19*a14);
  a26=(a19*a14);
  a26=(a10+a26);
  a40=(a40+a26);
  a40=(a9*a40);
  a11=(a11+a40);
  a11=(a0*a11);
  if (res[0]!=0) res[0][4]=a11;
  a37=(a1*a37);
  a22=(a22+a37);
  a27=(a1*a27);
  a22=(a22+a27);
  a27=(a19*a14);
  a27=(a27+a26);
  a27=(a13*a27);
  a22=(a22+a27);
  a22=(a0*a22);
  if (res[0]!=0) res[0][5]=a22;
  a22=(a9*a15);
  a27=(a2*a22);
  a26=(a8*a27);
  a26=(a1*a26);
  a37=(a13*a15);
  a11=(a2*a37);
  a40=(a2*a11);
  a23=(a21*a40);
  a23=(a20*a23);
  a41=(a2*a17);
  a41=(a31+a41);
  a41=(a9*a41);
  a34=(a2*a41);
  a28=(a25*a34);
  a42=(a33*a40);
  a42=(a32*a42);
  a28=(a28+a42);
  a23=(a23+a28);
  a23=(a1*a23);
  a26=(a26+a23);
  a23=(a2*a17);
  a23=(a31+a23);
  a23=(a13*a23);
  a28=(a2*a23);
  a42=(a19*a28);
  a38=(a36*a42);
  a38=(a35*a38);
  a50=(a2*a17);
  a50=(a20+a50);
  a50=(a9*a50);
  a51=(a19*a50);
  a52=(a39*a51);
  a53=(a45*a42);
  a53=(a44*a53);
  a52=(a52+a53);
  a38=(a38+a52);
  a26=(a26+a38);
  a26=(a0*a26);
  a26=(-a26);
  if (res[0]!=0) res[0][6]=a26;
  a27=(a43*a27);
  a27=(a1*a27);
  a26=(a47*a40);
  a26=(a20*a26);
  a34=(a48*a34);
  a40=(a16*a40);
  a40=(a32*a40);
  a34=(a34-a40);
  a26=(a26+a34);
  a26=(a1*a26);
  a27=(a27+a26);
  a26=(a24*a42);
  a26=(a35*a26);
  a51=(a49*a51);
  a42=(a4*a42);
  a42=(a44*a42);
  a51=(a51-a42);
  a26=(a26+a51);
  a27=(a27+a26);
  a27=(a0*a27);
  if (res[0]!=0) res[0][7]=a27;
  a11=(a1*a11);
  a28=(a1*a28);
  a11=(a11+a28);
  a28=(a2*a17);
  a28=(a20+a28);
  a28=(a13*a28);
  a27=(a19*a28);
  a11=(a11+a27);
  a11=(a0*a11);
  if (res[0]!=0) res[0][8]=a11;
  a41=(a1*a41);
  a22=(a22+a41);
  a50=(a1*a50);
  a22=(a22+a50);
  a50=(a19*a17);
  a50=(a35+a50);
  a50=(a9*a50);
  a22=(a22+a50);
  a22=(a0*a22);
  if (res[0]!=0) res[0][9]=a22;
  a23=(a1*a23);
  a37=(a37+a23);
  a28=(a1*a28);
  a37=(a37+a28);
  a28=(a19*a17);
  a28=(a35+a28);
  a28=(a13*a28);
  a37=(a37+a28);
  a37=(a0*a37);
  if (res[0]!=0) res[0][10]=a37;
  if (res[0]!=0) res[0][11]=a19;
  a37=1.;
  if (res[0]!=0) res[0][12]=a37;
  if (res[0]!=0) res[0][13]=a37;
  a28=sin(a3);
  a28=(a15*a28);
  a23=cos(a3);
  a23=(a30*a23);
  a28=(a28+a23);
  a23=sin(a5);
  a22=(a31*a23);
  a50=(a14*a15);
  a10=(a10*a17);
  a50=(a50+a10);
  a50=(a9*a50);
  a50=(a2*a50);
  a50=(a30+a50);
  a10=cos(a5);
  a41=(a50*a10);
  a22=(a22+a41);
  a22=(a1*a22);
  a28=(a28+a22);
  a22=(a20*a21);
  a41=(a32*a33);
  a22=(a22+a41);
  a22=(a1*a22);
  a28=(a28+a22);
  a22=(a35*a36);
  a41=(a44*a45);
  a22=(a22+a41);
  a28=(a28+a22);
  a28=(a0*a28);
  a28=(-a28);
  if (res[0]!=0) res[0][14]=a28;
  a28=cos(a3);
  a15=(a15*a28);
  a28=sin(a3);
  a30=(a30*a28);
  a15=(a15-a30);
  a30=cos(a5);
  a28=(a31*a30);
  a5=sin(a5);
  a22=(a50*a5);
  a28=(a28-a22);
  a28=(a1*a28);
  a15=(a15+a28);
  a28=(a20*a47);
  a22=(a32*a16);
  a28=(a28-a22);
  a28=(a1*a28);
  a15=(a15+a28);
  a28=(a35*a24);
  a22=(a44*a4);
  a28=(a28-a22);
  a15=(a15+a28);
  a15=(a0*a15);
  if (res[0]!=0) res[0][15]=a15;
  if (res[0]!=0) res[0][16]=a37;
  a15=cos(a3);
  a28=(a9*a14);
  a22=(a2*a28);
  a41=(a8*a22);
  a6=(a6-a41);
  a6=(a1*a6);
  a15=(a15+a6);
  a6=(a13*a14);
  a41=(a2*a6);
  a11=(a2*a41);
  a27=(a21*a11);
  a27=(a20*a27);
  a18=(a18-a27);
  a27=(a9*a14);
  a26=(a2*a27);
  a51=(a25*a26);
  a42=(a33*a11);
  a42=(a32*a42);
  a51=(a51+a42);
  a18=(a18-a51);
  a18=(a1*a18);
  a15=(a15+a18);
  a18=(a13*a14);
  a51=(a2*a18);
  a42=(a19*a51);
  a34=(a36*a42);
  a34=(a35*a34);
  a29=(a29-a34);
  a34=(a9*a14);
  a40=(a19*a34);
  a38=(a39*a40);
  a52=(a45*a42);
  a52=(a44*a52);
  a38=(a38+a52);
  a29=(a29-a38);
  a15=(a15+a29);
  a15=(a0*a15);
  if (res[0]!=0) res[0][17]=a15;
  a15=sin(a3);
  a22=(a43*a22);
  a7=(a7+a22);
  a7=(a1*a7);
  a15=(a15+a7);
  a7=(a47*a11);
  a7=(a20*a7);
  a12=(a12+a7);
  a26=(a48*a26);
  a11=(a16*a11);
  a11=(a32*a11);
  a26=(a26-a11);
  a12=(a12+a26);
  a12=(a1*a12);
  a15=(a15+a12);
  a12=(a24*a42);
  a12=(a35*a12);
  a46=(a46+a12);
  a40=(a49*a40);
  a42=(a4*a42);
  a42=(a44*a42);
  a40=(a40-a42);
  a46=(a46+a40);
  a15=(a15+a46);
  a15=(a0*a15);
  if (res[0]!=0) res[0][18]=a15;
  a41=(a1*a41);
  a51=(a1*a51);
  a41=(a41+a51);
  a51=(a13*a14);
  a15=(a19*a51);
  a41=(a41+a15);
  a41=(a0*a41);
  if (res[0]!=0) res[0][19]=a41;
  if (res[0]!=0) res[0][20]=a37;
  a27=(a1*a27);
  a28=(a28+a27);
  a34=(a1*a34);
  a28=(a28+a34);
  a34=(a9*a14);
  a28=(a28+a34);
  a28=(a0*a28);
  if (res[0]!=0) res[0][21]=a28;
  a18=(a1*a18);
  a6=(a6+a18);
  a51=(a1*a51);
  a6=(a6+a51);
  a14=(a13*a14);
  a6=(a6+a14);
  a6=(a0*a6);
  if (res[0]!=0) res[0][22]=a6;
  a6=sin(a3);
  a14=(a1*a8);
  a6=(a6+a14);
  a14=(a1*a25);
  a6=(a6+a14);
  a6=(a6+a39);
  a6=(a0*a6);
  a6=(-a6);
  if (res[0]!=0) res[0][23]=a6;
  a3=cos(a3);
  a6=(a1*a43);
  a3=(a3+a6);
  a6=(a1*a48);
  a3=(a3+a6);
  a3=(a3+a49);
  a3=(a0*a3);
  if (res[0]!=0) res[0][24]=a3;
  if (res[0]!=0) res[0][25]=a37;
  a23=(a2*a23);
  a23=(a31*a23);
  a10=(a2*a10);
  a10=(a50*a10);
  a23=(a23+a10);
  a23=(a1*a23);
  a10=(a2*a21);
  a10=(a20*a10);
  a3=(a2*a33);
  a3=(a32*a3);
  a10=(a10+a3);
  a10=(a1*a10);
  a23=(a23+a10);
  a10=(a19*a36);
  a10=(a35*a10);
  a3=(a19*a45);
  a3=(a44*a3);
  a10=(a10+a3);
  a23=(a23+a10);
  a23=(a0*a23);
  a23=(-a23);
  if (res[0]!=0) res[0][26]=a23;
  a30=(a2*a30);
  a31=(a31*a30);
  a5=(a2*a5);
  a50=(a50*a5);
  a31=(a31-a50);
  a31=(a1*a31);
  a50=(a2*a47);
  a50=(a20*a50);
  a5=(a2*a16);
  a5=(a32*a5);
  a50=(a50-a5);
  a50=(a1*a50);
  a31=(a31+a50);
  a50=(a19*a24);
  a50=(a35*a50);
  a5=(a19*a4);
  a5=(a44*a5);
  a50=(a50-a5);
  a31=(a31+a50);
  a31=(a0*a31);
  if (res[0]!=0) res[0][27]=a31;
  if (res[0]!=0) res[0][28]=a19;
  if (res[0]!=0) res[0][29]=a37;
  a31=(a9*a17);
  a50=(a2*a31);
  a8=(a8*a50);
  a8=(a1*a8);
  a5=(a13*a17);
  a30=(a2*a5);
  a23=(a2*a30);
  a21=(a21*a23);
  a21=(a20*a21);
  a10=(a9*a17);
  a3=(a2*a10);
  a25=(a25*a3);
  a33=(a33*a23);
  a33=(a32*a33);
  a25=(a25+a33);
  a21=(a21+a25);
  a21=(a1*a21);
  a8=(a8+a21);
  a21=(a13*a17);
  a2=(a2*a21);
  a25=(a19*a2);
  a36=(a36*a25);
  a36=(a35*a36);
  a33=(a9*a17);
  a6=(a19*a33);
  a39=(a39*a6);
  a45=(a45*a25);
  a45=(a44*a45);
  a39=(a39+a45);
  a36=(a36+a39);
  a8=(a8+a36);
  a8=(a0*a8);
  a8=(-a8);
  if (res[0]!=0) res[0][30]=a8;
  a43=(a43*a50);
  a43=(a1*a43);
  a47=(a47*a23);
  a20=(a20*a47);
  a48=(a48*a3);
  a16=(a16*a23);
  a32=(a32*a16);
  a48=(a48-a32);
  a20=(a20+a48);
  a20=(a1*a20);
  a43=(a43+a20);
  a24=(a24*a25);
  a35=(a35*a24);
  a49=(a49*a6);
  a4=(a4*a25);
  a44=(a44*a4);
  a49=(a49-a44);
  a35=(a35+a49);
  a43=(a43+a35);
  a43=(a0*a43);
  if (res[0]!=0) res[0][31]=a43;
  a30=(a1*a30);
  a2=(a1*a2);
  a30=(a30+a2);
  a2=(a13*a17);
  a19=(a19*a2);
  a30=(a30+a19);
  a30=(a0*a30);
  if (res[0]!=0) res[0][32]=a30;
  a10=(a1*a10);
  a31=(a31+a10);
  a33=(a1*a33);
  a31=(a31+a33);
  a9=(a9*a17);
  a31=(a31+a9);
  a31=(a0*a31);
  if (res[0]!=0) res[0][33]=a31;
  a21=(a1*a21);
  a5=(a5+a21);
  a1=(a1*a2);
  a5=(a5+a1);
  a13=(a13*a17);
  a5=(a5+a13);
  a0=(a0*a5);
  if (res[0]!=0) res[0][34]=a0;
  if (res[0]!=0) res[0][35]=a37;
  return 0;
}

CASADI_SYMBOL_EXPORT int FORCESNLPsolver_ddynamics_0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f3(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int FORCESNLPsolver_ddynamics_0_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int FORCESNLPsolver_ddynamics_0_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void FORCESNLPsolver_ddynamics_0_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int FORCESNLPsolver_ddynamics_0_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void FORCESNLPsolver_ddynamics_0_release(int mem) {
}

CASADI_SYMBOL_EXPORT void FORCESNLPsolver_ddynamics_0_incref(void) {
}

CASADI_SYMBOL_EXPORT void FORCESNLPsolver_ddynamics_0_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int FORCESNLPsolver_ddynamics_0_n_in(void) { return 2;}

CASADI_SYMBOL_EXPORT casadi_int FORCESNLPsolver_ddynamics_0_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real FORCESNLPsolver_ddynamics_0_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* FORCESNLPsolver_ddynamics_0_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* FORCESNLPsolver_ddynamics_0_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* FORCESNLPsolver_ddynamics_0_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* FORCESNLPsolver_ddynamics_0_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s5;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int FORCESNLPsolver_ddynamics_0_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 2;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}

/* FORCESNLPsolver_objective_1:(i0[9],i1[3])->(o0) */
static int casadi_f4(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a2, a3;
  a0=2.;
  a1=arg[0]? arg[0][2] : 0;
  a2=arg[1]? arg[1][0] : 0;
  a1=(a1-a2);
  a1=fabs(a1);
  a1=(a0*a1);
  a2=arg[0]? arg[0][3] : 0;
  a3=arg[1]? arg[1][1] : 0;
  a2=(a2-a3);
  a2=fabs(a2);
  a0=(a0*a2);
  a1=(a1+a0);
  a0=20.;
  a2=arg[0]? arg[0][4] : 0;
  a3=arg[1]? arg[1][2] : 0;
  a2=(a2-a3);
  a2=fabs(a2);
  a0=(a0*a2);
  a1=(a1+a0);
  if (res[0]!=0) res[0][0]=a1;
  return 0;
}

CASADI_SYMBOL_EXPORT int FORCESNLPsolver_objective_1(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f4(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int FORCESNLPsolver_objective_1_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int FORCESNLPsolver_objective_1_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void FORCESNLPsolver_objective_1_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int FORCESNLPsolver_objective_1_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void FORCESNLPsolver_objective_1_release(int mem) {
}

CASADI_SYMBOL_EXPORT void FORCESNLPsolver_objective_1_incref(void) {
}

CASADI_SYMBOL_EXPORT void FORCESNLPsolver_objective_1_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int FORCESNLPsolver_objective_1_n_in(void) { return 2;}

CASADI_SYMBOL_EXPORT casadi_int FORCESNLPsolver_objective_1_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real FORCESNLPsolver_objective_1_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* FORCESNLPsolver_objective_1_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* FORCESNLPsolver_objective_1_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* FORCESNLPsolver_objective_1_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* FORCESNLPsolver_objective_1_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int FORCESNLPsolver_objective_1_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 2;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}

/* FORCESNLPsolver_dobjective_1:(i0[9],i1[3])->(o0[1x9,3nz]) */
static int casadi_f5(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a2;
  a0=2.;
  a1=arg[0]? arg[0][2] : 0;
  a2=arg[1]? arg[1][0] : 0;
  a1=(a1-a2);
  a1=casadi_sign(a1);
  a1=(a0*a1);
  if (res[0]!=0) res[0][0]=a1;
  a1=arg[0]? arg[0][3] : 0;
  a2=arg[1]? arg[1][1] : 0;
  a1=(a1-a2);
  a1=casadi_sign(a1);
  a0=(a0*a1);
  if (res[0]!=0) res[0][1]=a0;
  a0=20.;
  a1=arg[0]? arg[0][4] : 0;
  a2=arg[1]? arg[1][2] : 0;
  a1=(a1-a2);
  a1=casadi_sign(a1);
  a0=(a0*a1);
  if (res[0]!=0) res[0][2]=a0;
  return 0;
}

CASADI_SYMBOL_EXPORT int FORCESNLPsolver_dobjective_1(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f5(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int FORCESNLPsolver_dobjective_1_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int FORCESNLPsolver_dobjective_1_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void FORCESNLPsolver_dobjective_1_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int FORCESNLPsolver_dobjective_1_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void FORCESNLPsolver_dobjective_1_release(int mem) {
}

CASADI_SYMBOL_EXPORT void FORCESNLPsolver_dobjective_1_incref(void) {
}

CASADI_SYMBOL_EXPORT void FORCESNLPsolver_dobjective_1_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int FORCESNLPsolver_dobjective_1_n_in(void) { return 2;}

CASADI_SYMBOL_EXPORT casadi_int FORCESNLPsolver_dobjective_1_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real FORCESNLPsolver_dobjective_1_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* FORCESNLPsolver_dobjective_1_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* FORCESNLPsolver_dobjective_1_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* FORCESNLPsolver_dobjective_1_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* FORCESNLPsolver_dobjective_1_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s3;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int FORCESNLPsolver_dobjective_1_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 2;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
