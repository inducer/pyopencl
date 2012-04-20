//  Copyright (c) 2006 Xiaogang Zhang, John Maddock
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See
//  http://www.boost.org/LICENSE_1_0.txt)

#pragma once

typedef double T;

#define MAX_SERIES_ITERATIONS 10000

// {{{ evaluate_rational

T evaluate_rational_backend(__constant const T* num, __constant const T* denom, T z, int count)
{
   T s1, s2;
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

// }}}

#define evaluate_rational(num, denom, z) \
  evaluate_rational_backend(num, denom, z, sizeof(num)/sizeof(T))

// {{{ j0

__constant const T bessel_j0_P1[] = {
     -4.1298668500990866786e+11,
     2.7282507878605942706e+10,
     -6.2140700423540120665e+08,
     6.6302997904833794242e+06,
     -3.6629814655107086448e+04,
     1.0344222815443188943e+02,
     -1.2117036164593528341e-01
};
__constant const T bessel_j0_Q1[] = {
     2.3883787996332290397e+12,
     2.6328198300859648632e+10,
     1.3985097372263433271e+08,
     4.5612696224219938200e+05,
     9.3614022392337710626e+02,
     1.0,
     0.0
};
__constant const T bessel_j0_P2[] = {
     -1.8319397969392084011e+03,
     -1.2254078161378989535e+04,
     -7.2879702464464618998e+03,
     1.0341910641583726701e+04,
     1.1725046279757103576e+04,
     4.4176707025325087628e+03,
     7.4321196680624245801e+02,
     4.8591703355916499363e+01
};
__constant const T bessel_j0_Q2[] = {
     -3.5783478026152301072e+05,
     2.4599102262586308984e+05,
     -8.4055062591169562211e+04,
     1.8680990008359188352e+04,
     -2.9458766545509337327e+03,
     3.3307310774649071172e+02,
     -2.5258076240801555057e+01,
     1.0
};
__constant const T bessel_j0_PC[] = {
     2.2779090197304684302e+04,
     4.1345386639580765797e+04,
     2.1170523380864944322e+04,
     3.4806486443249270347e+03,
     1.5376201909008354296e+02,
     8.8961548424210455236e-01
};
__constant const T bessel_j0_QC[] = {
     2.2779090197304684318e+04,
     4.1370412495510416640e+04,
     2.1215350561880115730e+04,
     3.5028735138235608207e+03,
     1.5711159858080893649e+02,
     1.0
};
__constant const T bessel_j0_PS[] = {
    -8.9226600200800094098e+01,
    -1.8591953644342993800e+02,
    -1.1183429920482737611e+02,
    -2.2300261666214198472e+01,
    -1.2441026745835638459e+00,
    -8.8033303048680751817e-03
};
__constant const T bessel_j0_QS[] = {
     5.7105024128512061905e+03,
     1.1951131543434613647e+04,
     7.2642780169211018836e+03,
     1.4887231232283756582e+03,
     9.0593769594993125859e+01,
     1.0
};

T bessel_j0(T x)
{
    const T x1  =  2.4048255576957727686e+00,
          x2  =  5.5200781102863106496e+00,
          x11 =  6.160e+02,
          x12 =  -1.42444230422723137837e-03,
          x21 =  1.4130e+03,
          x22 =  5.46860286310649596604e-04;

    T value, factor, r, rc, rs;

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
        T y = x * x;
        r = evaluate_rational(bessel_j0_P1, bessel_j0_Q1, y);
        factor = (x + x1) * ((x - x11/256) - x12);
        value = factor * r;
    }
    else if (x <= 8.0)                  // x in (4, 8]
    {
        T y = 1 - (x * x)/64;
        r = evaluate_rational(bessel_j0_P2, bessel_j0_Q2, y);
        factor = (x + x2) * ((x - x21/256) - x22);
        value = factor * r;
    }
    else                                // x in (8, \infty)
    {
        T y = 8 / x;
        T y2 = y * y;
        T z = x - 0.25f * M_PI;
        rc = evaluate_rational(bessel_j0_PC, bessel_j0_QC, y2);
        rs = evaluate_rational(bessel_j0_PS, bessel_j0_QS, y2);
        factor = sqrt(2 / (x * M_PI));
        value = factor * (rc * cos(z) - y * rs * sin(z));
    }

    return value;
}

// }}}

// {{{ j1

__constant const T bessel_j1_P1[] = {
     -1.4258509801366645672e+11,
     6.6781041261492395835e+09,
     -1.1548696764841276794e+08,
     9.8062904098958257677e+05,
     -4.4615792982775076130e+03,
     1.0650724020080236441e+01,
     -1.0767857011487300348e-02
};
__constant const T bessel_j1_Q1[] = {
     4.1868604460820175290e+12,
     4.2091902282580133541e+10,
     2.0228375140097033958e+08,
     5.9117614494174794095e+05,
     1.0742272239517380498e+03,
     1.0,
     0.0
};
__constant const T bessel_j1_P2[] = {
     -1.7527881995806511112e+16,
     1.6608531731299018674e+15,
     -3.6658018905416665164e+13,
     3.5580665670910619166e+11,
     -1.8113931269860667829e+09,
     5.0793266148011179143e+06,
     -7.5023342220781607561e+03,
     4.6179191852758252278e+00
};
__constant const T bessel_j1_Q2[] = {
     1.7253905888447681194e+18,
     1.7128800897135812012e+16,
     8.4899346165481429307e+13,
     2.7622777286244082666e+11,
     6.4872502899596389593e+08,
     1.1267125065029138050e+06,
     1.3886978985861357615e+03,
     1.0
};
__constant const T bessel_j1_PC[] = {
    -4.4357578167941278571e+06,
    -9.9422465050776411957e+06,
    -6.6033732483649391093e+06,
    -1.5235293511811373833e+06,
    -1.0982405543459346727e+05,
    -1.6116166443246101165e+03,
    0.0
};
__constant const T bessel_j1_QC[] = {
    -4.4357578167941278568e+06,
    -9.9341243899345856590e+06,
    -6.5853394797230870728e+06,
    -1.5118095066341608816e+06,
    -1.0726385991103820119e+05,
    -1.4550094401904961825e+03,
    1.0
};
__constant const T bessel_j1_PS[] = {
     3.3220913409857223519e+04,
     8.5145160675335701966e+04,
     6.6178836581270835179e+04,
     1.8494262873223866797e+04,
     1.7063754290207680021e+03,
     3.5265133846636032186e+01,
     0.0
};
__constant const T bessel_j1_QS[] = {
     7.0871281941028743574e+05,
     1.8194580422439972989e+06,
     1.4194606696037208929e+06,
     4.0029443582266975117e+05,
     3.7890229745772202641e+04,
     8.6383677696049909675e+02,
     1.0
};


T bessel_j1(T x)
{
    const T x1  =  3.8317059702075123156e+00,
                   x2  =  7.0155866698156187535e+00,
                   x11 =  9.810e+02,
                   x12 =  -3.2527979248768438556e-04,
                   x21 =  1.7960e+03,
                   x22 =  -3.8330184381246462950e-05;

    T value, factor, r, rc, rs, w;

    w = fabs(x);
    if (x == 0)
    {
        return 0;
    }
    if (w <= 4)                       // w in (0, 4]
    {
        T y = x * x;
        r = evaluate_rational(bessel_j1_P1, bessel_j1_Q1, y);
        factor = w * (w + x1) * ((w - x11/256) - x12);
        value = factor * r;
    }
    else if (w <= 8)                  // w in (4, 8]
    {
        T y = x * x;
        r = evaluate_rational(bessel_j1_P2, bessel_j1_Q2, y);
        factor = w * (w + x2) * ((w - x21/256) - x22);
        value = factor * r;
    }
    else                                // w in (8, \infty)
    {
        T y = 8 / w;
        T y2 = y * y;
        T z = w - 0.75f * M_PI;
        rc = evaluate_rational(bessel_j1_PC, bessel_j1_QC, y2);
        rs = evaluate_rational(bessel_j1_PS, bessel_j1_QS, y2);
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

// {{{ asymptotic_bessel_y_large_x_2

inline T asymptotic_bessel_j_limit(const T v)
{
   // double case:
   T v2 = max((T) 3, v * v);
   return v2 * 33 /*73*/;
}

inline T asymptotic_bessel_amplitude(T v, T x)
{
   // Calculate the amplitude of J(v, x) and Y(v, x) for large
   // x: see A&S 9.2.28.
   T s = 1;
   T mu = 4 * v * v;
   T txq = 2 * x;
   txq *= txq;

   s += (mu - 1) / (2 * txq);
   s += 3 * (mu - 1) * (mu - 9) / (txq * txq * 8);
   s += 15 * (mu - 1) * (mu - 9) * (mu - 25) / (txq * txq * txq * 8 * 6);

   return sqrt(s * 2 / (M_PI * x));
}

T asymptotic_bessel_phase_mx(T v, T x)
{
   //
   // Calculate the phase of J(v, x) and Y(v, x) for large x.
   // See A&S 9.2.29.
   // Note that the result returned is the phase less x.
   //
   T mu = 4 * v * v;
   T denom = 4 * x;
   T denom_mult = denom * denom;

   T s = -M_PI * (v / 2 + 0.25f);
   s += (mu - 1) / (2 * denom);
   denom *= denom_mult;
   s += (mu - 1) * (mu - 25) / (6 * denom);
   denom *= denom_mult;
   s += (mu - 1) * (mu * mu - 114 * mu + 1073) / (5 * denom);
   denom *= denom_mult;
   s += (mu - 1) * (5 * mu * mu * mu - 1535 * mu * mu + 54703 * mu - 375733) / (14 * denom);
   return s;
}


// }}}

// {{{ CF1_jy

// Evaluate continued fraction fv = J_(v+1) / J_v, see
// Abramowitz and Stegun, Handbook of Mathematical Functions, 1972, 9.1.73
int CF1_jy(T v, T x, T* fv, int* sign)
{
    T C, D, f, a, b, delta, tiny, tolerance;
    unsigned long k;
    int s = 1;

    // |x| <= |v|, CF1_jy converges rapidly
    // |x| > |v|, CF1_jy needs O(|x|) iterations to converge

    // modified Lentz's method, see
    // Lentz, Applied Optics, vol 15, 668 (1976)
    tolerance = 2 * DBL_EPSILON;
    tiny = sqrt(DBL_MIN);
    C = f = tiny;                           // b0 = 0, replace with tiny
    D = 0;
    for (k = 1; k < MAX_SERIES_ITERATIONS * 100; k++)
    {
        a = -1;
        b = 2 * (v + k) / x;
        C = b + a / C;
        D = b + a * D;
        if (C == 0) { C = tiny; }
        if (D == 0) { D = tiny; }
        D = 1 / D;
        delta = C * D;
        f *= delta;
        if (D < 0) { s = -s; }
        if (fabs(delta - 1) < tolerance) 
        { break; }
    }

    // policies::check_series_iterations<T>("boost::math::bessel_jy<%1%>(%1%,%1%) in CF1_jy", k / 100, pol);
    *fv = -f;
    *sign = s;                              // sign of denominator

    return 0;
}

// }}}

// {{{ bessel_j_small_z_series

typedef struct 
{
   unsigned N;
   T v;
   T mult;
   T term;
} bessel_j_small_z_series_term;

void bessel_j_small_z_series_term_init(bessel_j_small_z_series_term *self, T v_, T x)
{
  self->N = 0;
  self->v = v_;
  self->mult = x / 2;
  self->mult *= -self->mult;
  self->term = 1;
}

T bessel_j_small_z_series_term_do(bessel_j_small_z_series_term *self)
{
  T r = self->term;
  ++self->N;
  self->term *= self->mult / (self->N * (self->N + self->v));
  return r;
}


//
// Series evaluation for BesselJ(v, z) as z -> 0.
// See http://functions.wolfram.com/Bessel-TypeFunctions/BesselJ/06/01/04/01/01/0003/
// Converges rapidly for all z << v.
//
inline T bessel_j_small_z_series(T v, T x)
{
   T prefix;

   int const max_factorial = 170; // long double value from boost

   if(v < max_factorial)
   {
      prefix = pow(x / 2, v) / tgamma(v+1);
   }
   else
   {
      prefix = v * log(x / 2) - lgamma(v+1);
      prefix = exp(prefix);
   }
   if(0 == prefix)
      return prefix;

   bessel_j_small_z_series_term s;
   bessel_j_small_z_series_term_init(&s, v, x);

   T factor = DBL_EPSILON;
   int counter = MAX_SERIES_ITERATIONS;

   T result = 0;
   T next_term;
   do
   {
      next_term = bessel_j_small_z_series_term_do(&s);
      result += next_term;
   }
   while((fabs(factor * result) < fabs(next_term)) && --counter);

   // policies::check_series_iterations<T>("boost::math::bessel_j_small_z_series<%1%>(%1%,%1%)", max_iter, pol);

   return prefix * result;
}

// }}}

// {{{ jn

inline T asymptotic_bessel_j_large_x_2(T v, T x)
{
   // See A&S 9.2.19.
   // Get the phase and amplitude:
   T ampl = asymptotic_bessel_amplitude(v, x);
   T phase = asymptotic_bessel_phase_mx(v, x);
   //
   // Calculate the sine of the phase, using:
   // cos(x+p) = cos(x)cos(p) - sin(x)sin(p)
   //
   T sin_phase = cos(phase) * cos(x) - sin(phase) * sin(x);
   return sin_phase * ampl;
}

T bessel_jn(int n, T x)
{
    T value = 0, factor, current, prev, next;

    //
    // Reflection has to come first:
    //
    if (n < 0)
    {
        factor = (n & 0x1) ? -1 : 1;  // J_{-n}(z) = (-1)^n J_n(z)
        n = -n;
    }
    else
    {
        factor = 1;
    }
    //
    // Special cases:
    //
    if (n == 0)
    {
        return factor * bessel_j0(x);
    }
    if (n == 1)
    {
        return factor * bessel_j1(x);
    }

    if (x == 0)                             // n >= 2
    {
        return 0;
    }

    if(fabs(x) > asymptotic_bessel_j_limit(n))
      return factor * asymptotic_bessel_j_large_x_2(n, x);

    // BOOST_ASSERT(n > 1);
    T scale = 1;
    if (n < fabs(x))                         // forward recurrence
    {
        prev = bessel_j0(x);
        current = bessel_j1(x);
        for (int k = 1; k < n; k++)
        {
            T fact = 2 * k / x;
            if((DBL_MAX - fabs(prev)) / fabs(fact) < fabs(current))
            {
               scale /= current;
               prev /= current;
               current = 1;
            }
            value = fact * current - prev;
            prev = current;
            current = value;
        }
    }
    else if(x < 1)
    {
       return factor * bessel_j_small_z_series((T) n, x);
    }
    else                                    // backward recurrence
    {
        T fn; int s;                        // fn = J_(n+1) / J_n
        // |x| <= n, fast convergence for continued fraction CF1
        CF1_jy((T) n, x, &fn, &s);
        prev = fn;
        current = 1;
        for (int k = n; k > 0; k--)
        {
            T fact = 2 * k / x;
            if((DBL_MAX - fabs(prev)) / fact < fabs(current))
            {
               prev /= current;
               scale /= current;
               current = 1;
            }
            next = fact * current - prev;
            prev = current;
            current = next;
        }
        value = bessel_j0(x) / current;       // normalization
        scale = 1 / scale;
    }
    value *= factor;

    if(DBL_MAX * scale < fabs(value))
       return nan(22ul);

    return value / scale;
}

// }}}

// vim: fdm=marker
