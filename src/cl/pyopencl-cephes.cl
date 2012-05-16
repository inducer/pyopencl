//  Ported from Cephes by
//  Andreas Kloeckner (C) 2012
//
// Cephes Math Library Release 2.8:  June, 2000
// Copyright 1984, 1987, 1989, 1992, 2000 by Stephen L. Moshier
// What you see here may be used freely, but it comes with no support or
// guarantee.

#pragma once

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

double cephes_polevl(double x, __constant const double *coef, int N)
{
  double ans;
  int i;
  __constant const double *p;

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

double cephes_p1evl( double x, __constant const double *coef, int N )
{
  double ans;
  __constant const double *p;
  int i;

  p = coef;
  ans = x + *p++;
  i = N-1;

  do
    ans = ans * x  + *p++;
  while( --i );

  return( ans );
}
