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
#include <pyopencl-bessel-j.cl>

typedef double bessel_y_scalar_type;

// {{{ bessel_y0

__constant const bessel_y_scalar_type bessel_y0_P1[] = {
     1.0723538782003176831e+11,
    -8.3716255451260504098e+09,
     2.0422274357376619816e+08,
    -2.1287548474401797963e+06,
     1.0102532948020907590e+04,
    -1.8402381979244993524e+01,
};
__constant const bessel_y_scalar_type bessel_y0_Q1[] = {
     5.8873865738997033405e+11,
     8.1617187777290363573e+09,
     5.5662956624278251596e+07,
     2.3889393209447253406e+05,
     6.6475986689240190091e+02,
     1.0,
};
__constant const bessel_y_scalar_type bessel_y0_P2[] = {
    -2.2213976967566192242e+13,
    -5.5107435206722644429e+11,
     4.3600098638603061642e+10,
    -6.9590439394619619534e+08,
     4.6905288611678631510e+06,
    -1.4566865832663635920e+04,
     1.7427031242901594547e+01,
};
__constant const bessel_y_scalar_type bessel_y0_Q2[] = {
     4.3386146580707264428e+14,
     5.4266824419412347550e+12,
     3.4015103849971240096e+10,
     1.3960202770986831075e+08,
     4.0669982352539552018e+05,
     8.3030857612070288823e+02,
     1.0,
};
__constant const bessel_y_scalar_type bessel_y0_P3[] = {
    -8.0728726905150210443e+15,
     6.7016641869173237784e+14,
    -1.2829912364088687306e+11,
    -1.9363051266772083678e+11,
     2.1958827170518100757e+09,
    -1.0085539923498211426e+07,
     2.1363534169313901632e+04,
    -1.7439661319197499338e+01,
};
__constant const bessel_y_scalar_type bessel_y0_Q3[] = {
     3.4563724628846457519e+17,
     3.9272425569640309819e+15,
     2.2598377924042897629e+13,
     8.6926121104209825246e+10,
     2.4727219475672302327e+08,
     5.3924739209768057030e+05,
     8.7903362168128450017e+02,
     1.0,
};
__constant const bessel_y_scalar_type bessel_y0_PC[] = {
     2.2779090197304684302e+04,
     4.1345386639580765797e+04,
     2.1170523380864944322e+04,
     3.4806486443249270347e+03,
     1.5376201909008354296e+02,
     8.8961548424210455236e-01,
};
__constant const bessel_y_scalar_type bessel_y0_QC[] = {
     2.2779090197304684318e+04,
     4.1370412495510416640e+04,
     2.1215350561880115730e+04,
     3.5028735138235608207e+03,
     1.5711159858080893649e+02,
     1.0,
};
__constant const bessel_y_scalar_type bessel_y0_PS[] = {
    -8.9226600200800094098e+01,
    -1.8591953644342993800e+02,
    -1.1183429920482737611e+02,
    -2.2300261666214198472e+01,
    -1.2441026745835638459e+00,
    -8.8033303048680751817e-03,
};
__constant const bessel_y_scalar_type bessel_y0_QS[] = {
     5.7105024128512061905e+03,
     1.1951131543434613647e+04,
     7.2642780169211018836e+03,
     1.4887231232283756582e+03,
     9.0593769594993125859e+01,
     1.0,
};

bessel_y_scalar_type bessel_y0(bessel_y_scalar_type x)
{
    const bessel_y_scalar_type
          x1  =  8.9357696627916752158e-01,
          x2  =  3.9576784193148578684e+00,
          x3  =  7.0860510603017726976e+00,
          x11 =  2.280e+02,
          x12 =  2.9519662791675215849e-03,
          x21 =  1.0130e+03,
          x22 =  6.4716931485786837568e-04,
          x31 =  1.8140e+03,
          x32 =  1.1356030177269762362e-04;

    bessel_y_scalar_type value, factor, r, rc, rs;

    if (x < 0)
    {
       //return policies::raise_domain_error<T>(function,
       //    "Got x = %1% but x must be non-negative, complex result not supported.", x, pol);
       return nan((uint)22);
    }
    if (x == 0)
    {
        return -DBL_MAX;
    }
    if (x <= 3)                       // x in (0, 3]
    {
        bessel_y_scalar_type y = x * x;
        bessel_y_scalar_type z = 2 * log(x/x1) * bessel_j0(x) / M_PI;
        r = boost_evaluate_rational(bessel_y0_P1, bessel_y0_Q1, y);
        factor = (x + x1) * ((x - x11/256) - x12);
        value = z + factor * r;
    }
    else if (x <= 5.5f)                  // x in (3, 5.5]
    {
        bessel_y_scalar_type y = x * x;
        bessel_y_scalar_type z = 2 * log(x/x2) * bessel_j0(x) / M_PI;
        r = boost_evaluate_rational(bessel_y0_P2, bessel_y0_Q2, y);
        factor = (x + x2) * ((x - x21/256) - x22);
        value = z + factor * r;
    }
    else if (x <= 8)                  // x in (5.5, 8]
    {
        bessel_y_scalar_type y = x * x;
        bessel_y_scalar_type z = 2 * log(x/x3) * bessel_j0(x) / M_PI;
        r = boost_evaluate_rational(bessel_y0_P3, bessel_y0_Q3, y);
        factor = (x + x3) * ((x - x31/256) - x32);
        value = z + factor * r;
    }
    else                                // x in (8, \infty)
    {
        bessel_y_scalar_type y = 8 / x;
        bessel_y_scalar_type y2 = y * y;
        bessel_y_scalar_type z = x - 0.25f * M_PI;
        rc = boost_evaluate_rational(bessel_y0_PC, bessel_y0_QC, y2);
        rs = boost_evaluate_rational(bessel_y0_PS, bessel_y0_QS, y2);
        factor = sqrt(2 / (x * M_PI));
        value = factor * (rc * sin(z) + y * rs * cos(z));
    }

    return value;
}

// }}}

// {{{ bessel_y1

__constant const bessel_y_scalar_type bessel_y1_P1[] = {
     4.0535726612579544093e+13,
     5.4708611716525426053e+12,
    -3.7595974497819597599e+11,
     7.2144548214502560419e+09,
    -5.9157479997408395984e+07,
     2.2157953222280260820e+05,
    -3.1714424660046133456e+02,
};
__constant const bessel_y_scalar_type bessel_y1_Q1[] = {
     3.0737873921079286084e+14,
     4.1272286200406461981e+12,
     2.7800352738690585613e+10,
     1.2250435122182963220e+08,
     3.8136470753052572164e+05,
     8.2079908168393867438e+02,
     1.0,
};
__constant const bessel_y_scalar_type bessel_y1_P2[] = {
     1.1514276357909013326e+19,
    -5.6808094574724204577e+18,
    -2.3638408497043134724e+16,
     4.0686275289804744814e+15,
    -5.9530713129741981618e+13,
     3.7453673962438488783e+11,
    -1.1957961912070617006e+09,
     1.9153806858264202986e+06,
    -1.2337180442012953128e+03,
};
__constant const bessel_y_scalar_type bessel_y1_Q2[] = {
     5.3321844313316185697e+20,
     5.6968198822857178911e+18,
     3.0837179548112881950e+16,
     1.1187010065856971027e+14,
     3.0221766852960403645e+11,
     6.3550318087088919566e+08,
     1.0453748201934079734e+06,
     1.2855164849321609336e+03,
     1.0,
};
__constant const bessel_y_scalar_type bessel_y1_PC[] = {
    -4.4357578167941278571e+06,
    -9.9422465050776411957e+06,
    -6.6033732483649391093e+06,
    -1.5235293511811373833e+06,
    -1.0982405543459346727e+05,
    -1.6116166443246101165e+03,
     0.0,
};
__constant const bessel_y_scalar_type bessel_y1_QC[] = {
    -4.4357578167941278568e+06,
    -9.9341243899345856590e+06,
    -6.5853394797230870728e+06,
    -1.5118095066341608816e+06,
    -1.0726385991103820119e+05,
    -1.4550094401904961825e+03,
     1.0,
};
__constant const bessel_y_scalar_type bessel_y1_PS[] = {
     3.3220913409857223519e+04,
     8.5145160675335701966e+04,
     6.6178836581270835179e+04,
     1.8494262873223866797e+04,
     1.7063754290207680021e+03,
     3.5265133846636032186e+01,
     0.0,
};
__constant const bessel_y_scalar_type bessel_y1_QS[] = {
     7.0871281941028743574e+05,
     1.8194580422439972989e+06,
     1.4194606696037208929e+06,
     4.0029443582266975117e+05,
     3.7890229745772202641e+04,
     8.6383677696049909675e+02,
     1.0,
};

bessel_y_scalar_type bessel_y1(bessel_y_scalar_type x)
{
    const bessel_y_scalar_type 
      x1  =  2.1971413260310170351e+00,
      x2  =  5.4296810407941351328e+00,
      x11 =  5.620e+02,
      x12 =  1.8288260310170351490e-03,
      x21 =  1.3900e+03,
      x22 = -6.4592058648672279948e-06
    ;
    bessel_y_scalar_type value, factor, r, rc, rs;

    if (x <= 0)
    {
      // domain error
      return nan((uint)22);
    }
    if (x <= 4)                       // x in (0, 4]
    {
        bessel_y_scalar_type y = x * x;
        bessel_y_scalar_type z = 2 * log(x/x1) * bessel_j1(x) / M_PI;
        r = boost_evaluate_rational(bessel_y1_P1, bessel_y1_Q1, y);
        factor = (x + x1) * ((x - x11/256) - x12) / x;
        value = z + factor * r;
    }
    else if (x <= 8)                  // x in (4, 8]
    {
        bessel_y_scalar_type y = x * x;
        bessel_y_scalar_type z = 2 * log(x/x2) * bessel_j1(x) / M_PI;
        r = boost_evaluate_rational(bessel_y1_P2, bessel_y1_Q2, y);
        factor = (x + x2) * ((x - x21/256) - x22) / x;
        value = z + factor * r;
    }
    else                                // x in (8, \infty)
    {
        bessel_y_scalar_type y = 8 / x;
        bessel_y_scalar_type y2 = y * y;
        bessel_y_scalar_type z = x - 0.75f * M_PI;
        rc = boost_evaluate_rational(bessel_y1_PC, bessel_y1_QC, y2);
        rs = boost_evaluate_rational(bessel_y1_PS, bessel_y1_QS, y2);
        factor = sqrt(2 / (x * M_PI));
        value = factor * (rc * sin(z) + y * rs * cos(z));
    }

    return value;
}

// }}}

// {{{ bessel_yn

bessel_y_scalar_type bessel_yn_small_z(int n, bessel_y_scalar_type z, bessel_y_scalar_type* scale)
{
   //
   // See http://functions.wolfram.com/Bessel-TypeFunctions/BesselY/06/01/04/01/02/
   //
   // Note that when called we assume that x < epsilon and n is a positive integer.
   //
   // BOOST_ASSERT(n >= 0);
   // BOOST_ASSERT((z < policies::get_epsilon<T, Policy>()));

   if(n == 0)
   {
      return (2 / M_PI) * (log(z / 2) +  M_E);
   }
   else if(n == 1)
   {
      return (z / M_PI) * log(z / 2) 
         - 2 / (M_PI * z) 
         - (z / (2 * M_PI)) * (1 - 2 * M_E);
   }
   else if(n == 2)
   {
      return (z * z) / (4 * M_PI) * log(z / 2) 
         - (4 / (M_PI * z * z)) 
         - ((z * z) / (8 * M_PI)) * (3./2 - 2 * M_E);
   }
   else
   {
      bessel_y_scalar_type p = pow(z / 2, (bessel_y_scalar_type) n);
      bessel_y_scalar_type result = -((tgamma((bessel_y_scalar_type) n) / M_PI));
      if(p * DBL_MAX < result)
      {
         bessel_y_scalar_type div = DBL_MAX / 8;
         result /= div;
         *scale /= div;
         if(p * DBL_MAX < result)
         {
            return -DBL_MAX;
         }
      }
      return result / p;
   }
}




bessel_y_scalar_type bessel_yn(int n, bessel_y_scalar_type x)
{
    //BOOST_MATH_STD_USING
    bessel_y_scalar_type value, factor, current, prev;

    //using namespace boost::math::tools;

    if ((x == 0) && (n == 0))
    {
       return -DBL_MAX;
    }
    if (x <= 0)
    {
       //return policies::raise_domain_error<T>(function,
            //"Got x = %1%, but x must be > 0, complex result not supported.", x, pol);
       return nan((uint)22);
    }

    //
    // Reflection comes first:
    //
    if (n < 0)
    {
        factor = (n & 0x1) ? -1 : 1;  // Y_{-n}(z) = (-1)^n Y_n(z)
        n = -n;
    }
    else
    {
        factor = 1;
    }

    if(x < DBL_EPSILON)
    {
       bessel_y_scalar_type scale = 1;
       value = bessel_yn_small_z(n, x, &scale);
       if(DBL_MAX * fabs(scale) < fabs(value))
          return copysign((bessel_y_scalar_type) 1, scale) * copysign((bessel_y_scalar_type) 1, value) * DBL_MAX;
       value /= scale;
    }
    else if (n == 0)
    {
        value = bessel_y0(x);
    }
    else if (n == 1)
    {
        value = factor * bessel_y1(x);
    }
    else
    {
       prev = bessel_y0(x);
       current = bessel_y1(x);
       int k = 1;
       // BOOST_ASSERT(k < n);
       do
       {
           bessel_y_scalar_type fact = 2 * k / x;
           if((DBL_MAX - fabs(prev)) / fact < fabs(current))
           {
              prev /= current;
              factor /= current;
              current = 1;
           }
           value = fact * current - prev;
           prev = current;
           current = value;
           ++k;
       }
       while(k < n);
       if(fabs(DBL_MAX * factor) < fabs(value))
          return sign(value) * sign(value) * DBL_MAX;
       value /= factor;
    }
    return value;
}

// }}}

// vim: fdm=marker
