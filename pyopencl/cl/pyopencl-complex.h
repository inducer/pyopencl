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

// This file is available for inclusion in pyopencl kernels and provides
// complex types 'cfloat_t' and 'cdouble_t', along with a number of special
// functions as visible below, e.g. cdouble_log(z).
//
// Under the hood, the complex types are simply float2 and double2.
// Note that native (operator-based) addition (float + float2) and
// multiplication (float2*float1) is defined for these types,
// but do not match the rules of complex arithmetic.

#pragma once

#define PYOPENCL_DECLARE_COMPLEX_TYPE_INT(REAL_TP, REAL_3LTR, TPROOT, TP) \
  \
  REAL_TP TPROOT##_real(TP a) { return a.real; } \
  REAL_TP TPROOT##_imag(TP a) { return a.imag; } \
  REAL_TP TPROOT##_abs(TP a) { return hypot(a.real, a.imag); } \
  \
  TP TPROOT##_new(REAL_TP real, REAL_TP imag) \
  { \
    TP result; \
    result.real = real; \
    result.imag = imag; \
    return result; \
  } \
  \
  TP TPROOT##_fromreal(REAL_TP real) \
  { \
    TP result; \
    result.real = real; \
    result.imag = 0; \
    return result; \
  } \
  \
  \
  TP TPROOT##_neg(TP a) { return TPROOT##_new(-a.real, -a.imag); } \
  TP TPROOT##_conj(TP a) { return TPROOT##_new(a.real, -a.imag); } \
  \
  TP TPROOT##_add(TP a, TP b) \
  { \
    return TPROOT##_new(a.real + b.real, a.imag + b.imag); \
    ; \
  } \
  TP TPROOT##_addr(TP a, REAL_TP b) \
  { \
    return TPROOT##_new(b+a.real, a.imag); \
  } \
  TP TPROOT##_radd(REAL_TP a, TP b) \
  { \
    return TPROOT##_new(a+b.real, b.imag); \
  } \
  \
  TP TPROOT##_sub(TP a, TP b) \
  { \
    return TPROOT##_new(a.real - b.real, a.imag - b.imag); \
    ; \
  } \
  \
  TP TPROOT##_mul(TP a, TP b) \
  { \
    return TPROOT##_new( \
        a.real*b.real - a.imag*b.imag, \
        a.real*b.imag + a.imag*b.real); \
  } \
  \
  TP TPROOT##_mulr(TP a, REAL_TP b) \
  { \
    return TPROOT##_new(a.real*b, a.imag*b); \
  } \
  \
  TP TPROOT##_rmul(REAL_TP a, TP b) \
  { \
    return TPROOT##_new(a*b.real, a*b.imag); \
  } \
  \
  TP TPROOT##_rdivide(REAL_TP z1, TP z2) \
  { \
    if (fabs(z2.real) <= fabs(z2.imag)) { \
      REAL_TP ratio = z2.real / z2.imag; \
      REAL_TP denom = z2.imag * (1 + ratio * ratio); \
      return TPROOT##_new((z1 * ratio) / denom, - z1 / denom); \
    } \
    else { \
      REAL_TP ratio = z2.imag / z2.real; \
      REAL_TP denom = z2.real * (1 + ratio * ratio); \
      return TPROOT##_new(z1 / denom, - (z1 * ratio) / denom); \
    } \
  } \
  \
  TP TPROOT##_divide(TP z1, TP z2) \
  { \
    REAL_TP ratio, denom, a, b, c, d; \
    \
    if (fabs(z2.real) <= fabs(z2.imag)) { \
      ratio = z2.real / z2.imag; \
      denom = z2.imag; \
      a = z1.imag; \
      b = z1.real; \
      c = -z1.real; \
      d = z1.imag; \
    } \
    else { \
      ratio = z2.imag / z2.real; \
      denom = z2.real; \
      a = z1.real; \
      b = z1.imag; \
      c = z1.imag; \
      d = -z1.real; \
    } \
    denom *= (1 + ratio * ratio); \
    return TPROOT##_new( \
       (a + b * ratio) / denom, \
       (c + d * ratio) / denom); \
  } \
  \
  TP TPROOT##_divider(TP a, REAL_TP b) \
  { \
    return TPROOT##_new(a.real/b, a.imag/b); \
  } \
  \
  TP TPROOT##_pow(TP a, TP b) \
  { \
    REAL_TP logr = log(hypot(a.real, a.imag)); \
    REAL_TP logi = atan2(a.imag, a.real); \
    REAL_TP x = exp(logr * b.real - logi * b.imag); \
    REAL_TP y = logr * b.imag + logi * b.real; \
    \
    REAL_TP cosy; \
    REAL_TP siny = sincos(y, &cosy); \
    return TPROOT##_new(x*cosy, x*siny); \
  } \
  \
  TP TPROOT##_powr(TP a, REAL_TP b) \
  { \
    REAL_TP logr = log(hypot(a.real, a.imag)); \
    REAL_TP logi = atan2(a.imag, a.real); \
    REAL_TP x = exp(logr * b); \
    REAL_TP y = logi * b; \
    \
    REAL_TP cosy; \
    REAL_TP siny = sincos(y, &cosy); \
    \
    return TPROOT##_new(x * cosy, x*siny); \
  } \
  \
  TP TPROOT##_rpow(REAL_TP a, TP b) \
  { \
    REAL_TP logr = log(a); \
    REAL_TP x = exp(logr * b.real); \
    REAL_TP y = logr * b.imag; \
    \
    REAL_TP cosy; \
    REAL_TP siny = sincos(y, &cosy); \
    return TPROOT##_new(x * cosy, x * siny); \
  } \
  \
  TP TPROOT##_sqrt(TP a) \
  { \
    REAL_TP re = a.real; \
    REAL_TP im = a.imag; \
    REAL_TP mag = hypot(re, im); \
    TP result; \
    \
    if (mag == 0.f) { \
      result.real = result.imag = 0.f; \
    } else if (re > 0.f) { \
      result.real = sqrt(0.5f * (mag + re)); \
      result.imag = im/result.real/2.f; \
    } else { \
      result.imag = sqrt(0.5f * (mag - re)); \
      if (im < 0.f) \
        result.imag = - result.imag; \
      result.real = im/result.imag/2.f; \
    } \
    return result; \
  } \
  \
  TP TPROOT##_exp(TP a) \
  { \
    REAL_TP expr = exp(a.real); \
    REAL_TP cosi; \
    REAL_TP sini = sincos(a.imag, &cosi); \
    return TPROOT##_new(expr * cosi, expr * sini); \
  } \
  \
  TP TPROOT##_log(TP a) \
  { return TPROOT##_new(log(hypot(a.real, a.imag)), atan2(a.imag, a.real)); } \
  \
  TP TPROOT##_sin(TP a) \
  { \
    REAL_TP cosr; \
    REAL_TP sinr = sincos(a.real, &cosr); \
    return TPROOT##_new(sinr*cosh(a.imag), cosr*sinh(a.imag)); \
  } \
  \
  TP TPROOT##_cos(TP a) \
  { \
    REAL_TP cosr; \
    REAL_TP sinr = sincos(a.real, &cosr); \
    return TPROOT##_new(cosr*cosh(a.imag), -sinr*sinh(a.imag)); \
  } \
  \
  TP TPROOT##_tan(TP a) \
  { \
    REAL_TP re2 = 2.f * a.real; \
    REAL_TP im2 = 2.f * a.imag; \
    \
    const REAL_TP limit = log(REAL_3LTR##_MAX); \
    \
    if (fabs(im2) > limit) \
      return TPROOT##_new(0.f, (im2 > 0 ? 1.f : -1.f)); \
    else \
    { \
      REAL_TP den = cos(re2) + cosh(im2); \
      return TPROOT##_new(sin(re2) / den, sinh(im2) / den); \
    } \
  } \
  \
  TP TPROOT##_sinh(TP a) \
  { \
    REAL_TP cosi; \
    REAL_TP sini = sincos(a.imag, &cosi); \
    return TPROOT##_new(sinh(a.real)*cosi, cosh(a.real)*sini); \
  } \
  \
  TP TPROOT##_cosh(TP a) \
  { \
    REAL_TP cosi; \
    REAL_TP sini = sincos(a.imag, &cosi); \
    return TPROOT##_new(cosh(a.real)*cosi, sinh(a.real)*sini); \
  } \
  \
  TP TPROOT##_tanh(TP a) \
  { \
    REAL_TP re2 = 2.f * a.real; \
    REAL_TP im2 = 2.f * a.imag; \
    \
    const REAL_TP limit = log(REAL_3LTR##_MAX); \
    \
    if (fabs(re2) > limit) \
      return TPROOT##_new((re2 > 0 ? 1.f : -1.f), 0.f); \
    else \
    { \
      REAL_TP den = cosh(re2) + cos(im2); \
      return TPROOT##_new(sinh(re2) / den, sin(im2) / den); \
    } \
  } \

#define PYOPENCL_DECLARE_COMPLEX_TYPE(BASE, BASE_3LTR) \
  typedef union \
  { \
    struct { BASE x, y; }; \
    struct { BASE real, imag; }; \
  } c##BASE##_t; \
  \
  PYOPENCL_DECLARE_COMPLEX_TYPE_INT(BASE, BASE_3LTR, c##BASE, c##BASE##_t)

PYOPENCL_DECLARE_COMPLEX_TYPE(float, FLT);
#define cfloat_cast(a) cfloat_new((a).real, (a).imag)

#ifdef PYOPENCL_DEFINE_CDOUBLE
PYOPENCL_DECLARE_COMPLEX_TYPE(double, DBL);
#define cdouble_cast(a) cdouble_new((a).real, (a).imag)
#endif
