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

#include <pyopencl-eval-tbl.cl>
#include <pyopencl-airy.cl>

typedef double bessel_j_scalar_type;
// FIXME: T is really a bad name
typedef bessel_j_scalar_type T;

// {{{ bessel_j0

__constant const bessel_j_scalar_type bessel_j0_P1[] = {
     -4.1298668500990866786e+11,
     2.7282507878605942706e+10,
     -6.2140700423540120665e+08,
     6.6302997904833794242e+06,
     -3.6629814655107086448e+04,
     1.0344222815443188943e+02,
     -1.2117036164593528341e-01
};
__constant const bessel_j_scalar_type bessel_j0_Q1[] = {
     2.3883787996332290397e+12,
     2.6328198300859648632e+10,
     1.3985097372263433271e+08,
     4.5612696224219938200e+05,
     9.3614022392337710626e+02,
     1.0,
     0.0
};
__constant const bessel_j_scalar_type bessel_j0_P2[] = {
     -1.8319397969392084011e+03,
     -1.2254078161378989535e+04,
     -7.2879702464464618998e+03,
     1.0341910641583726701e+04,
     1.1725046279757103576e+04,
     4.4176707025325087628e+03,
     7.4321196680624245801e+02,
     4.8591703355916499363e+01
};
__constant const bessel_j_scalar_type bessel_j0_Q2[] = {
     -3.5783478026152301072e+05,
     2.4599102262586308984e+05,
     -8.4055062591169562211e+04,
     1.8680990008359188352e+04,
     -2.9458766545509337327e+03,
     3.3307310774649071172e+02,
     -2.5258076240801555057e+01,
     1.0
};
__constant const bessel_j_scalar_type bessel_j0_PC[] = {
     2.2779090197304684302e+04,
     4.1345386639580765797e+04,
     2.1170523380864944322e+04,
     3.4806486443249270347e+03,
     1.5376201909008354296e+02,
     8.8961548424210455236e-01
};
__constant const bessel_j_scalar_type bessel_j0_QC[] = {
     2.2779090197304684318e+04,
     4.1370412495510416640e+04,
     2.1215350561880115730e+04,
     3.5028735138235608207e+03,
     1.5711159858080893649e+02,
     1.0
};
__constant const bessel_j_scalar_type bessel_j0_PS[] = {
    -8.9226600200800094098e+01,
    -1.8591953644342993800e+02,
    -1.1183429920482737611e+02,
    -2.2300261666214198472e+01,
    -1.2441026745835638459e+00,
    -8.8033303048680751817e-03
};
__constant const bessel_j_scalar_type bessel_j0_QS[] = {
     5.7105024128512061905e+03,
     1.1951131543434613647e+04,
     7.2642780169211018836e+03,
     1.4887231232283756582e+03,
     9.0593769594993125859e+01,
     1.0
};

bessel_j_scalar_type bessel_j0(bessel_j_scalar_type x)
{
    const bessel_j_scalar_type x1  =  2.4048255576957727686e+00,
          x2  =  5.5200781102863106496e+00,
          x11 =  6.160e+02,
          x12 =  -1.42444230422723137837e-03,
          x21 =  1.4130e+03,
          x22 =  5.46860286310649596604e-04;

    bessel_j_scalar_type value, factor, r, rc, rs;

    if (x < 0)
    {
        x = -x;                         // even function
    }
    if (x == 0)
    {
        return 1;
    }
    if (x <= 4)                       // x in (0, 4]
    {
        bessel_j_scalar_type y = x * x;
        r = boost_evaluate_rational(bessel_j0_P1, bessel_j0_Q1, y);
        factor = (x + x1) * ((x - x11/256) - x12);
        value = factor * r;
    }
    else if (x <= 8.0)                  // x in (4, 8]
    {
        bessel_j_scalar_type y = 1 - (x * x)/64;
        r = boost_evaluate_rational(bessel_j0_P2, bessel_j0_Q2, y);
        factor = (x + x2) * ((x - x21/256) - x22);
        value = factor * r;
    }
    else                                // x in (8, \infty)
    {
        bessel_j_scalar_type y = 8 / x;
        bessel_j_scalar_type y2 = y * y;
        bessel_j_scalar_type z = x - 0.25f * M_PI;
        rc = boost_evaluate_rational(bessel_j0_PC, bessel_j0_QC, y2);
        rs = boost_evaluate_rational(bessel_j0_PS, bessel_j0_QS, y2);
        factor = sqrt(2 / (x * M_PI));
        value = factor * (rc * cos(z) - y * rs * sin(z));
    }

    return value;
}

// }}}

// {{{ bessel_j1

__constant const bessel_j_scalar_type bessel_j1_P1[] = {
     -1.4258509801366645672e+11,
     6.6781041261492395835e+09,
     -1.1548696764841276794e+08,
     9.8062904098958257677e+05,
     -4.4615792982775076130e+03,
     1.0650724020080236441e+01,
     -1.0767857011487300348e-02
};
__constant const bessel_j_scalar_type bessel_j1_Q1[] = {
     4.1868604460820175290e+12,
     4.2091902282580133541e+10,
     2.0228375140097033958e+08,
     5.9117614494174794095e+05,
     1.0742272239517380498e+03,
     1.0,
     0.0
};
__constant const bessel_j_scalar_type bessel_j1_P2[] = {
     -1.7527881995806511112e+16,
     1.6608531731299018674e+15,
     -3.6658018905416665164e+13,
     3.5580665670910619166e+11,
     -1.8113931269860667829e+09,
     5.0793266148011179143e+06,
     -7.5023342220781607561e+03,
     4.6179191852758252278e+00
};
__constant const bessel_j_scalar_type bessel_j1_Q2[] = {
     1.7253905888447681194e+18,
     1.7128800897135812012e+16,
     8.4899346165481429307e+13,
     2.7622777286244082666e+11,
     6.4872502899596389593e+08,
     1.1267125065029138050e+06,
     1.3886978985861357615e+03,
     1.0
};
__constant const bessel_j_scalar_type bessel_j1_PC[] = {
    -4.4357578167941278571e+06,
    -9.9422465050776411957e+06,
    -6.6033732483649391093e+06,
    -1.5235293511811373833e+06,
    -1.0982405543459346727e+05,
    -1.6116166443246101165e+03,
    0.0
};
__constant const bessel_j_scalar_type bessel_j1_QC[] = {
    -4.4357578167941278568e+06,
    -9.9341243899345856590e+06,
    -6.5853394797230870728e+06,
    -1.5118095066341608816e+06,
    -1.0726385991103820119e+05,
    -1.4550094401904961825e+03,
    1.0
};
__constant const bessel_j_scalar_type bessel_j1_PS[] = {
     3.3220913409857223519e+04,
     8.5145160675335701966e+04,
     6.6178836581270835179e+04,
     1.8494262873223866797e+04,
     1.7063754290207680021e+03,
     3.5265133846636032186e+01,
     0.0
};
__constant const bessel_j_scalar_type bessel_j1_QS[] = {
     7.0871281941028743574e+05,
     1.8194580422439972989e+06,
     1.4194606696037208929e+06,
     4.0029443582266975117e+05,
     3.7890229745772202641e+04,
     8.6383677696049909675e+02,
     1.0
};


bessel_j_scalar_type bessel_j1(bessel_j_scalar_type x)
{
    const bessel_j_scalar_type x1  =  3.8317059702075123156e+00,
                   x2  =  7.0155866698156187535e+00,
                   x11 =  9.810e+02,
                   x12 =  -3.2527979248768438556e-04,
                   x21 =  1.7960e+03,
                   x22 =  -3.8330184381246462950e-05;

    bessel_j_scalar_type value, factor, r, rc, rs, w;

    w = fabs(x);
    if (x == 0)
    {
        return 0;
    }
    if (w <= 4)                       // w in (0, 4]
    {
        bessel_j_scalar_type y = x * x;
        r = boost_evaluate_rational(bessel_j1_P1, bessel_j1_Q1, y);
        factor = w * (w + x1) * ((w - x11/256) - x12);
        value = factor * r;
    }
    else if (w <= 8)                  // w in (4, 8]
    {
        bessel_j_scalar_type y = x * x;
        r = boost_evaluate_rational(bessel_j1_P2, bessel_j1_Q2, y);
        factor = w * (w + x2) * ((w - x21/256) - x22);
        value = factor * r;
    }
    else                                // w in (8, \infty)
    {
        bessel_j_scalar_type y = 8 / w;
        bessel_j_scalar_type y2 = y * y;
        bessel_j_scalar_type z = w - 0.75f * M_PI;
        rc = boost_evaluate_rational(bessel_j1_PC, bessel_j1_QC, y2);
        rs = boost_evaluate_rational(bessel_j1_PS, bessel_j1_QS, y2);
        factor = sqrt(2 / (w * M_PI));
        value = factor * (rc * cos(z) - y * rs * sin(z));
    }

    if (x < 0)
    {
        value *= -1;                 // odd function
    }
    return value;
}

// }}}

// {{{ bessel_recur

/* Reduce the order by backward recurrence.
 * AMS55 #9.1.27 and 9.1.73.
 */

#define BESSEL_BIG  1.44115188075855872E+17

double bessel_recur(double *n, double x, double *newn, int cancel )
{
  double pkm2, pkm1, pk, qkm2, qkm1;
  /* double pkp1; */
  double k, ans, qk, xk, yk, r, t, kf;
  const double big = BESSEL_BIG;
  int nflag, ctr;

  /* continued fraction for Jn(x)/Jn-1(x)  */
  if( *n < 0.0 )
    nflag = 1;
  else
    nflag = 0;

fstart:

#if DEBUG
  printf( "recur: n = %.6e, newn = %.6e, cfrac = ", *n, *newn );
#endif

  pkm2 = 0.0;
  qkm2 = 1.0;
  pkm1 = x;
  qkm1 = *n + *n;
  xk = -x * x;
  yk = qkm1;
  ans = 1.0;
  ctr = 0;
  do
  {
    yk += 2.0;
    pk = pkm1 * yk +  pkm2 * xk;
    qk = qkm1 * yk +  qkm2 * xk;
    pkm2 = pkm1;
    pkm1 = pk;
    qkm2 = qkm1;
    qkm1 = qk;
    if( qk != 0 )
      r = pk/qk;
    else
      r = 0.0;
    if( r != 0 )
    {
      t = fabs( (ans - r)/r );
      ans = r;
    }
    else
      t = 1.0;

    if( ++ctr > 1000 )
    {
      //mtherr( "jv", UNDERFLOW );
      pk = nan((uint)24);

      goto done;
    }
    if( t < DBL_EPSILON )
      goto done;

    if( fabs(pk) > big )
    {
      pkm2 /= big;
      pkm1 /= big;
      qkm2 /= big;
      qkm1 /= big;
    }
  }
  while( t > DBL_EPSILON );

done:

#if DEBUG
  printf( "%.6e\n", ans );
#endif

  /* Change n to n-1 if n < 0 and the continued fraction is small
  */
  if( nflag > 0 )
  {
    if( fabs(ans) < 0.125 )
    {
      nflag = -1;
      *n = *n - 1.0;
      goto fstart;
    }
  }


  kf = *newn;

  /* backward recurrence
   *              2k
   *  J   (x)  =  --- J (x)  -  J   (x)
   *   k-1         x   k         k+1
   */

  pk = 1.0;
  pkm1 = 1.0/ans;
  k = *n - 1.0;
  r = 2 * k;
  do
  {
    pkm2 = (pkm1 * r  -  pk * x) / x;
    /*  pkp1 = pk; */
    pk = pkm1;
    pkm1 = pkm2;
    r -= 2.0;
    /*
       t = fabs(pkp1) + fabs(pk);
       if( (k > (kf + 2.5)) && (fabs(pkm1) < 0.25*t) )
       {
       k -= 1.0;
       t = x*x;
       pkm2 = ( (r*(r+2.0)-t)*pk - r*x*pkp1 )/t;
       pkp1 = pk;
       pk = pkm1;
       pkm1 = pkm2;
       r -= 2.0;
       }
       */
    k -= 1.0;
  }
  while( k > (kf + 0.5) );

  /* Take the larger of the last two iterates
   * on the theory that it may have less cancellation error.
   */

  if( cancel )
  {
    if( (kf >= 0.0) && (fabs(pk) > fabs(pkm1)) )
    {
      k += 1.0;
      pkm2 = pk;
    }
  }
  *newn = k;
#if DEBUG
  printf( "newn %.6e rans %.6e\n", k, pkm2 );
#endif
  return( pkm2 );
}

// }}}

// {{{ bessel_jvs

#define BESSEL_MAXGAM 171.624376956302725
#define BESSEL_MAXLOG 7.09782712893383996843E2

/* Ascending power series for Jv(x).
 * AMS55 #9.1.10.
 */

double bessel_jvs(double n, double x)
{
  double t, u, y, z, k;
  int ex;
  int sgngam = 1;

  z = -x * x / 4.0;
  u = 1.0;
  y = u;
  k = 1.0;
  t = 1.0;

  while( t > DBL_EPSILON )
  {
    u *= z / (k * (n+k));
    y += u;
    k += 1.0;
    if( y != 0 )
      t = fabs( u/y );
  }
#if DEBUG
  printf( "power series=%.5e ", y );
#endif
  t = frexp( 0.5*x, &ex );
  ex = ex * n;
  if(  (ex > -1023)
      && (ex < 1023)
      && (n > 0.0)
      && (n < (BESSEL_MAXGAM-1.0)) )
  {
    t = pow( 0.5*x, n ) / tgamma( n + 1.0 );
#if DEBUG
    printf( "pow(.5*x, %.4e)/gamma(n+1)=%.5e\n", n, t );
#endif
    y *= t;
  }
  else
  {
#if DEBUG
    z = n * log(0.5*x);
    k = lgamma( n+1.0 );
    t = z - k;
    printf( "log pow=%.5e, lgam(%.4e)=%.5e\n", z, n+1.0, k );
#else
    t = n * log(0.5*x) - lgamma(n + 1.0);
#endif
    if( y < 0 )
    {
      sgngam = -sgngam;
      y = -y;
    }
    t += log(y);
#if DEBUG
    printf( "log y=%.5e\n", log(y) );
#endif
    if( t < -BESSEL_MAXLOG )
    {
      return( 0.0 );
    }
    if( t > BESSEL_MAXLOG )
    {
      // mtherr( "Jv", OVERFLOW );
      return( DBL_MAX);
    }
    y = sgngam * exp( t );
  }
  return(y);
}

// }}}

// {{{ bessel_jnt

__constant const double bessel_jnt_PF2[] = {
 -9.0000000000000000000e-2,
  8.5714285714285714286e-2
};
__constant const double bessel_jnt_PF3[] = {
  1.3671428571428571429e-1,
 -5.4920634920634920635e-2,
 -4.4444444444444444444e-3
};
__constant const double bessel_jnt_PF4[] = {
  1.3500000000000000000e-3,
 -1.6036054421768707483e-1,
  4.2590187590187590188e-2,
  2.7330447330447330447e-3
};
__constant const double bessel_jnt_PG1[] = {
 -2.4285714285714285714e-1,
  1.4285714285714285714e-2
};
__constant const double bessel_jnt_PG2[] = {
 -9.0000000000000000000e-3,
  1.9396825396825396825e-1,
 -1.1746031746031746032e-2
};
__constant const double bessel_jnt_PG3[] = {
  1.9607142857142857143e-2,
 -1.5983694083694083694e-1,
  6.3838383838383838384e-3
};

double bessel_jnt(double n, double x)
{
  double z, zz, z3;
  double cbn, n23, cbtwo;
  double ai, aip, bi, bip;      /* Airy functions */
  double nk, fk, gk, pp, qq;
  double F[5], G[4];
  int k;

  cbn = cbrt(n);
  z = (x - n)/cbn;
  cbtwo = cbrt( 2.0 );

  /* Airy function */
  zz = -cbtwo * z;
  airy( zz, &ai, &aip, &bi, &bip );

  /* polynomials in expansion */
  zz = z * z;
  z3 = zz * z;
  F[0] = 1.0;
  F[1] = -z/5.0;
  F[2] = cephes_polevl( z3, bessel_jnt_PF2, 1 ) * zz;
  F[3] = cephes_polevl( z3, bessel_jnt_PF3, 2 );
  F[4] = cephes_polevl( z3, bessel_jnt_PF4, 3 ) * z;
  G[0] = 0.3 * zz;
  G[1] = cephes_polevl( z3, bessel_jnt_PG1, 1 );
  G[2] = cephes_polevl( z3, bessel_jnt_PG2, 2 ) * z;
  G[3] = cephes_polevl( z3, bessel_jnt_PG3, 2 ) * zz;
#if DEBUG
  for( k=0; k<=4; k++ )
    printf( "F[%d] = %.5E\n", k, F[k] );
  for( k=0; k<=3; k++ )
    printf( "G[%d] = %.5E\n", k, G[k] );
#endif
  pp = 0.0;
  qq = 0.0;
  nk = 1.0;
  n23 = cbrt( n * n );

  for( k=0; k<=4; k++ )
  {
    fk = F[k]*nk;
    pp += fk;
    if( k != 4 )
    {
      gk = G[k]*nk;
      qq += gk;
    }
#if DEBUG
    printf("fk[%d] %.5E, gk[%d] %.5E\n", k, fk, k, gk );
#endif
    nk /= n23;
  }

  fk = cbtwo * ai * pp/cbn  +  cbrt(4.0) * aip * qq/n;
  return(fk);
}

// }}}

// {{{ bessel_jnx

__constant const double bessel_jnx_lambda[] = {
  1.0,
  1.041666666666666666666667E-1,
  8.355034722222222222222222E-2,
  1.282265745563271604938272E-1,
  2.918490264641404642489712E-1,
  8.816272674437576524187671E-1,
  3.321408281862767544702647E+0,
  1.499576298686255465867237E+1,
  7.892301301158651813848139E+1,
  4.744515388682643231611949E+2,
  3.207490090890661934704328E+3
};
__constant const double bessel_jnx_mu[] = {
  1.0,
 -1.458333333333333333333333E-1,
 -9.874131944444444444444444E-2,
 -1.433120539158950617283951E-1,
 -3.172272026784135480967078E-1,
 -9.424291479571202491373028E-1,
 -3.511203040826354261542798E+0,
 -1.572726362036804512982712E+1,
 -8.228143909718594444224656E+1,
 -4.923553705236705240352022E+2,
 -3.316218568547972508762102E+3
};
__constant const double bessel_jnx_P1[] = {
 -2.083333333333333333333333E-1,
  1.250000000000000000000000E-1
};
__constant const double bessel_jnx_P2[] = {
  3.342013888888888888888889E-1,
 -4.010416666666666666666667E-1,
  7.031250000000000000000000E-2
};
__constant const double bessel_jnx_P3[] = {
 -1.025812596450617283950617E+0,
  1.846462673611111111111111E+0,
 -8.912109375000000000000000E-1,
  7.324218750000000000000000E-2
};
__constant const double bessel_jnx_P4[] = {
  4.669584423426247427983539E+0,
 -1.120700261622299382716049E+1,
  8.789123535156250000000000E+0,
 -2.364086914062500000000000E+0,
  1.121520996093750000000000E-1
};
__constant const double bessel_jnx_P5[] = {
 -2.8212072558200244877E1,
  8.4636217674600734632E1,
 -9.1818241543240017361E1,
  4.2534998745388454861E1,
 -7.3687943594796316964E0,
  2.27108001708984375E-1
};
__constant const double bessel_jnx_P6[] = {
  2.1257013003921712286E2,
 -7.6525246814118164230E2,
  1.0599904525279998779E3,
 -6.9957962737613254123E2,
  2.1819051174421159048E2,
 -2.6491430486951555525E1,
  5.7250142097473144531E-1
};
__constant const double bessel_jnx_P7[] = {
 -1.9194576623184069963E3,
  8.0617221817373093845E3,
 -1.3586550006434137439E4,
  1.1655393336864533248E4,
 -5.3056469786134031084E3,
  1.2009029132163524628E3,
 -1.0809091978839465550E2,
  1.7277275025844573975E0
};

double bessel_jnx(double n, double x)
{
  double zeta, sqz, zz, zp, np;
  double cbn, n23, t, z, sz;
  double pp, qq, z32i, zzi;
  double ak, bk, akl, bkl;
  int sign, doa, dob, nflg, k, s, tk, tkp1, m;
  double u[8];
  double ai, aip, bi, bip;

  /* Test for x very close to n.
   * Use expansion for transition region if so.
   */
  cbn = cbrt(n);
  z = (x - n)/cbn;
  if( fabs(z) <= 0.7 )
    return( bessel_jnt(n,x) );

  z = x/n;
  zz = 1.0 - z*z;
  if( zz == 0.0 )
    return(0.0);

  if( zz > 0.0 )
  {
    sz = sqrt( zz );
    t = 1.5 * (log( (1.0+sz)/z ) - sz );        /* zeta ** 3/2          */
    zeta = cbrt( t * t );
    nflg = 1;
  }
  else
  {
    sz = sqrt(-zz);
    t = 1.5 * (sz - acos(1.0/z));
    zeta = -cbrt( t * t );
    nflg = -1;
  }
  z32i = fabs(1.0/t);
  sqz = cbrt(t);

  /* Airy function */
  n23 = cbrt( n * n );
  t = n23 * zeta;

#if DEBUG
  printf("zeta %.5E, Airy(%.5E)\n", zeta, t );
#endif
  airy( t, &ai, &aip, &bi, &bip );

  /* polynomials in expansion */
  u[0] = 1.0;
  zzi = 1.0/zz;
  u[1] = cephes_polevl( zzi, bessel_jnx_P1, 1 )/sz;
  u[2] = cephes_polevl( zzi, bessel_jnx_P2, 2 )/zz;
  u[3] = cephes_polevl( zzi, bessel_jnx_P3, 3 )/(sz*zz);
  pp = zz*zz;
  u[4] = cephes_polevl( zzi, bessel_jnx_P4, 4 )/pp;
  u[5] = cephes_polevl( zzi, bessel_jnx_P5, 5 )/(pp*sz);
  pp *= zz;
  u[6] = cephes_polevl( zzi, bessel_jnx_P6, 6 )/pp;
  u[7] = cephes_polevl( zzi, bessel_jnx_P7, 7 )/(pp*sz);

#if DEBUG
  for( k=0; k<=7; k++ )
    printf( "u[%d] = %.5E\n", k, u[k] );
#endif

  pp = 0.0;
  qq = 0.0;
  np = 1.0;
  /* flags to stop when terms get larger */
  doa = 1;
  dob = 1;
  akl = DBL_MAX;
  bkl = DBL_MAX;

  for( k=0; k<=3; k++ )
  {
    tk = 2 * k;
    tkp1 = tk + 1;
    zp = 1.0;
    ak = 0.0;
    bk = 0.0;
    for( s=0; s<=tk; s++ )
    {
      if( doa )
      {
        if( (s & 3) > 1 )
          sign = nflg;
        else
          sign = 1;
        ak += sign * bessel_jnx_mu[s] * zp * u[tk-s];
      }

      if( dob )
      {
        m = tkp1 - s;
        if( ((m+1) & 3) > 1 )
          sign = nflg;
        else
          sign = 1;
        bk += sign * bessel_jnx_lambda[s] * zp * u[m];
      }
      zp *= z32i;
    }

    if( doa )
    {
      ak *= np;
      t = fabs(ak);
      if( t < akl )
      {
        akl = t;
        pp += ak;
      }
      else
        doa = 0;
    }

    if( dob )
    {
      bk += bessel_jnx_lambda[tkp1] * zp * u[0];
      bk *= -np/sqz;
      t = fabs(bk);
      if( t < bkl )
      {
        bkl = t;
        qq += bk;
      }
      else
        dob = 0;
    }
#if DEBUG
    printf("a[%d] %.5E, b[%d] %.5E\n", k, ak, k, bk );
#endif
    if( np < DBL_EPSILON )
      break;
    np /= n*n;
  }

  /* normalizing factor ( 4*zeta/(1 - z**2) )**1/4      */
  t = 4.0 * zeta/zz;
  t = sqrt( sqrt(t) );

  t *= ai*pp/cbrt(n)  +  aip*qq/(n23*n);
  return(t);
}

// }}}

// {{{ bessel_hankel

/* Hankel's asymptotic expansion
 * for large x.
 * AMS55 #9.2.5.
 */

double bessel_hankel( double n, double x )
{
  double t, u, z, k, sign, conv;
  double p, q, j, m, pp, qq;
  int flag;

  m = 4.0*n*n;
  j = 1.0;
  z = 8.0 * x;
  k = 1.0;
  p = 1.0;
  u = (m - 1.0)/z;
  q = u;
  sign = 1.0;
  conv = 1.0;
  flag = 0;
  t = 1.0;
  pp = 1.0e38;
  qq = 1.0e38;

  while( t > DBL_EPSILON )
  {
    k += 2.0;
    j += 1.0;
    sign = -sign;
    u *= (m - k * k)/(j * z);
    p += sign * u;
    k += 2.0;
    j += 1.0;
    u *= (m - k * k)/(j * z);
    q += sign * u;
    t = fabs(u/p);
    if( t < conv )
    {
      conv = t;
      qq = q;
      pp = p;
      flag = 1;
    }
    /* stop if the terms start getting larger */
    if( (flag != 0) && (t > conv) )
    {
#if DEBUG
      printf( "Hankel: convergence to %.4E\n", conv );
#endif
      goto hank1;
    }
  }

hank1:
  u = x - (0.5*n + 0.25) * M_PI;
  t = sqrt( 2.0/(M_PI*x) ) * ( pp * cos(u) - qq * sin(u) );
#if DEBUG
  printf( "hank: %.6e\n", t );
#endif
  return( t );
}
// }}}

// {{{ bessel_jv

// SciPy says jn has no advantage over jv, so alias the two.

#define bessel_jn bessel_jv

double bessel_jv(double n, double x)
{
  double k, q, t, y, an;
  int i, sign, nint;

  nint = 0;     /* Flag for integer n */
  sign = 1;     /* Flag for sign inversion */
  an = fabs( n );
  y = floor( an );
  if( y == an )
  {
    nint = 1;
    i = an - 16384.0 * floor( an/16384.0 );
    if( n < 0.0 )
    {
      if( i & 1 )
        sign = -sign;
      n = an;
    }
    if( x < 0.0 )
    {
      if( i & 1 )
        sign = -sign;
      x = -x;
    }
    if( n == 0.0 )
      return( bessel_j0(x) );
    if( n == 1.0 )
      return( sign * bessel_j1(x) );
  }

  if( (x < 0.0) && (y != an) )
  {
    // mtherr( "Jv", DOMAIN );
    // y = 0.0;
    y = nan((uint)22);
    goto done;
  }

  y = fabs(x);

  if( y < DBL_EPSILON )
    goto underf;

  k = 3.6 * sqrt(y);
  t = 3.6 * sqrt(an);
  if( (y < t) && (an > 21.0) )
    return( sign * bessel_jvs(n,x) );
  if( (an < k) && (y > 21.0) )
    return( sign * bessel_hankel(n,x) );

  if( an < 500.0 )
  {
    /* Note: if x is too large, the continued
     * fraction will fail; but then the
     * Hankel expansion can be used.
     */
    if( nint != 0 )
    {
      k = 0.0;
      q = bessel_recur( &n, x, &k, 1 );
      if( k == 0.0 )
      {
        y = bessel_j0(x)/q;
        goto done;
      }
      if( k == 1.0 )
      {
        y = bessel_j1(x)/q;
        goto done;
      }
    }

    if( an > 2.0 * y )
      goto rlarger;

    if( (n >= 0.0) && (n < 20.0)
        && (y > 6.0) && (y < 20.0) )
    {
      /* Recur backwards from a larger value of n
      */
rlarger:
      k = n;

      y = y + an + 1.0;
      if( y < 30.0 )
        y = 30.0;
      y = n + floor(y-n);
      q = bessel_recur( &y, x, &k, 0 );
      y = bessel_jvs(y,x) * q;
      goto done;
    }

    if( k <= 30.0 )
    {
      k = 2.0;
    }
    else if( k < 90.0 )
    {
      k = (3*k)/4;
    }
    if( an > (k + 3.0) )
    {
      if( n < 0.0 )
        k = -k;
      q = n - floor(n);
      k = floor(k) + q;
      if( n > 0.0 )
        q = bessel_recur( &n, x, &k, 1 );
      else
      {
        t = k;
        k = n;
        q = bessel_recur( &t, x, &k, 1 );
        k = t;
      }
      if( q == 0.0 )
      {
underf:
        y = 0.0;
        goto done;
      }
    }
    else
    {
      k = n;
      q = 1.0;
    }

    /* boundary between convergence of
     * power series and Hankel expansion
     */
    y = fabs(k);
    if( y < 26.0 )
      t = (0.0083*y + 0.09)*y + 12.9;
    else
      t = 0.9 * y;

    if( x > t )
      y = bessel_hankel(k,x);
    else
      y = bessel_jvs(k,x);
#if DEBUG
    printf( "y = %.16e, recur q = %.16e\n", y, q );
#endif
    if( n > 0.0 )
      y /= q;
    else
      y *= q;
  }

  else
  {
    /* For large n, use the uniform expansion
     * or the transitional expansion.
     * But if x is of the order of n**2,
     * these may blow up, whereas the
     * Hankel expansion will then work.
     */
    if( n < 0.0 )
    {
      //mtherr( "Jv", TLOSS );
      //y = 0.0;
      y = nan((uint)23);
      goto done;
    }
    t = x/n;
    t /= n;
    if( t > 0.3 )
      y = bessel_hankel(n,x);
    else
      y = bessel_jnx(n,x);
  }

done:   return( sign * y);
}

// }}}

// vim: fdm=marker
