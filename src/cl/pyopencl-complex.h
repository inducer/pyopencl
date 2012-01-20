/*
 * Copyright (c) 1999
 * Silicon Graphics Computer Systems, Inc.
 *
 * Copyright (c) 1999
 * Boris Fomitchev
 *
 * Copyright (c) 2012
 * Andreas Kloeckner
 *
 * This material is provided "as is", with absolutely no warranty expressed
 * or implied. Any use is at your own risk.
 *
 * Permission to use or copy this software for any purpose is hereby granted
 * without fee, provided the above notices are retained on all copies.
 * Permission to modify the code and to distribute modified code is granted,
 * provided the above notices are retained, and a notice that the code was
 * modified is included with the above copyright notice.
 *
 */

// This file is available in pyopencl kernels and provides complex types
// 'cfloat_t' and 'cdouble_t', along with a number of special functions
// as visible below, e.g. cdouble_log(z).
//
// Under the hood, the complex types are simply float2 and double2.

#define PYOPENCL_DECLARE_COMPLEX_TYPE_INT(REAL_TP, TPROOT, TP) \
  \
  REAL_TP TPROOT##_real(TP a) { return a.x; } \
  REAL_TP TPROOT##_imag(TP a) { return a.y; } \
  \
  TP TPROOT##_conj(TP a) { return (TP)(a.x, -a.y); } \
  \
  TP to_##TPROOT(REAL_TP a) \
  { return (TP)(a, 0); } \
  \
  TP TPROOT##_mul(TP a, TP b) \
  { \
    return (TP)( \
        a.x*b.x - a.y*b.y, \
        a.x*b.y + a.y*b.x); \
  } \
  \
  TP TPROOT##_rdivide(REAL_TP z1, TP z2) \
  { \
    REAL_TP ar = z2.x >= 0 ? z2.x : -z2.x; \
    REAL_TP ai = z2.y >= 0 ? z2.y : -z2.y; \
    \
    if (ar <= ai) { \
      REAL_TP ratio = z2.x / z2.y; \
      REAL_TP denom = z2.y * (1 + ratio * ratio); \
      return (TP)((z1 * ratio) / denom, - z1 / denom); \
    } \
    else { \
      REAL_TP ratio = z2.y / z2.x; \
      REAL_TP denom = z2.x * (1 + ratio * ratio); \
      return (TP)(z1 / denom, - (z1 * ratio) / denom); \
    } \
  } \
  \
  TP TPROOT##_divide(TP z1, TP z2) \
  { \
    REAL_TP ar = z2.x >= 0 ? z2.x : -z2.x; \
    REAL_TP ai = z2.y >= 0 ? z2.y : -z2.y; \
    \
    if (ar <= ai) { \
      REAL_TP ratio = z2.x / z2.y; \
      REAL_TP denom = z2.y * (1 + ratio * ratio); \
      return (TP)( \
         (z1.x * ratio + z1.y) / denom, \
         (z1.y * ratio - z1.x) / denom); \
    } \
    else { \
      REAL_TP ratio = z2.y / z2.x; \
      REAL_TP denom = z2.x * (1 + ratio * ratio); \
      return (TP)( \
         (z1.x + z1.y * ratio) / denom, \
         (z1.y - z1.x * ratio) / denom); \
    } \
  } \
  \
  TP TPROOT##_exp(TP a) \
  { \
    REAL_TP expr = exp(a.x); \
    REAL_TP cosi; \
    REAL_TP sini = sincos(a.y, &cosi); \
    return (TP)(expr * cosi, expr * sini); \
  } \
  \
  TP TPROOT##_log(TP a) \
  { return (TP)(log(hypot(a.x, a.y)), atan2(a.y, a.x)); } \
  \
  TP TPROOT##_rpower(TP a, REAL_TP b) \
  { \
    REAL_TP logr = log(hypot(a.x, a.y)); \
    REAL_TP logi = atan2(a.y, a.x); \
    REAL_TP x = exp(logr * b); \
    REAL_TP y = logi * b; \
    \
    REAL_TP cosy; \
    REAL_TP siny = sincos(y, &cosy); \
    \
    return (TP)(x * cosy, x*siny); \
  } \
  \
  TP TPROOT##_sqrt(TP a) \
  { \
    REAL_TP re = a.x; \
    REAL_TP im = a.y; \
    REAL_TP mag = hypot(re, im); \
    TP result; \
    \
    if (mag == 0.f) { \
      result.x = result.y = 0.f; \
    } else if (re > 0.f) { \
      result.x = sqrt(0.5f * (mag + re)); \
      result.y = im/result.x/2.f; \
    } else { \
      result.y = sqrt(0.5f * (mag - re)); \
      if (im < 0.f) \
        result.y = - result.y; \
      result.x = im/result.y/2.f; \
    } \
    return result; \
  }

#define PYOPENCL_DECLARE_COMPLEX_TYPE(BASE) \
  typedef BASE##2 c##BASE##_t; \
  \
  PYOPENCL_DECLARE_COMPLEX_TYPE_INT(BASE, c##BASE, c##BASE##_t)

PYOPENCL_DECLARE_COMPLEX_TYPE(float);
PYOPENCL_DECLARE_COMPLEX_TYPE(double);
