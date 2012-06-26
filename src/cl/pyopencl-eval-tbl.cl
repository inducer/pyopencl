//  Pieced together from Boost C++ and Cephes by
//  Andreas Kloeckner (C) 2012
//
//  Pieces from:
//
//  Copyright (c) 2006 Xiaogang Zhang, John Maddock
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See
//  http://www.boost.org/LICENSE_1_0.txt)
//
// Cephes Math Library Release 2.8:  June, 2000
// Copyright 1984, 1987, 1989, 1992, 2000 by Stephen L. Moshier
// What you see here may be used freely, but it comes with no support or
// guarantee.

#pragma once

typedef double special_func_scalar_type;

// {{{ cephes_polevl

/*
 * DESCRIPTION:
 *
 * Evaluates polynomial of degree N:
 *
 *                     2          N
 * y  =  C  + C x + C x  +...+ C x
 *        0    1     2          N
 *
 * Coefficients are stored in reverse order:
 *
 * coef[0] = C  , ..., coef[N] = C  .
 *            N                   0
 *
 *  The function p1evl() assumes that coef[N] = 1.0 and is
 * omitted from the array.  Its calling arguments are
 * otherwise the same as polevl().
 *
 */

special_func_scalar_type cephes_polevl(special_func_scalar_type x, __constant const special_func_scalar_type *coef, int N)
{
  special_func_scalar_type ans;
  int i;
  __constant const special_func_scalar_type *p;

  p = coef;
  ans = *p++;
  i = N;

  do
    ans = ans * x  +  *p++;
  while( --i );

  return( ans );
}

// }}}

// {{{ cephes_p1evl

special_func_scalar_type cephes_p1evl( special_func_scalar_type x, __constant const special_func_scalar_type *coef, int N )
{
  special_func_scalar_type ans;
  __constant const special_func_scalar_type *p;
  int i;

  p = coef;
  ans = x + *p++;
  i = N-1;

  do
    ans = ans * x  + *p++;
  while( --i );

  return( ans );
}

// }}}

// {{{ boost_evaluate_rational

special_func_scalar_type boost_evaluate_rational_backend(__constant const special_func_scalar_type* num, __constant const special_func_scalar_type* denom, special_func_scalar_type z, int count)
{
   special_func_scalar_type s1, s2;
   if(z <= 1)
   {
      s1 = num[count-1];
      s2 = denom[count-1];
      for(int i = (int)count - 2; i >= 0; --i)
      {
         s1 *= z;
         s2 *= z;
         s1 += num[i];
         s2 += denom[i];
      }
   }
   else
   {
      z = 1 / z;
      s1 = num[0];
      s2 = denom[0];
      for(unsigned i = 1; i < count; ++i)
      {
         s1 *= z;
         s2 *= z;
         s1 += num[i];
         s2 += denom[i];
      }
   }
   return s1 / s2;
}

#define boost_evaluate_rational(num, denom, z) \
  boost_evaluate_rational_backend(num, denom, z, sizeof(num)/sizeof(special_func_scalar_type))

// }}}

// vim: fdm=marker
