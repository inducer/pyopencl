/*
Evaluate Hankel function of first kind of order 0 and 1 for argument z
anywhere in the complex plane.

Copyright (C) Vladimir Rokhlin
Copyright (C) 2010-2012 Leslie Greengard and Zydrunas Gimbutas
Copyright (C) 2015 Andreas Kloeckner

Auto-translated from
https://github.com/zgimbutas/fmmlib2d/blob/master/src/hank103.f
using
https://github.com/inducer/pyopencl/tree/master/contrib/fortran-to-opencl

Originally licensed under GPL, permission to license under MIT granted via email
by Vladimir Rokhlin on May 25, 2015 and by Zydrunas Gimbutas on May 17, 2015.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

void hank103(cdouble_t z, cdouble_t *h0, cdouble_t *h1, int ifexpon);
void hank103u(cdouble_t z, int *ier, cdouble_t *h0, cdouble_t *h1, int ifexpon);
void hank103p(__constant cdouble_t *p, int m, cdouble_t z, cdouble_t *f);
void hank103a(cdouble_t z, cdouble_t *h0, cdouble_t *h1, int ifexpon);
void hank103l(cdouble_t z, cdouble_t *h0, cdouble_t *h1, int ifexpon);
void hank103r(cdouble_t z, int *ier, cdouble_t *h0, cdouble_t *h1, int ifexpon);

/*
 * this subroutine evaluates the hankel functions H_0^1, H_1^1
 * for an arbitrary user-specified complex number z. The user
 * also has the option of evaluating the functions h0, h1
 * scaled by the (complex) coefficient e^{-i \cdot z}. This
 * subroutine is a modification of the subroutine hank102
 * (see), different from the latter by having the parameter
 * ifexpon. Please note that the subroutine hank102 is in
 * turn a slightly accelerated version of the old hank101
 * (see). The principal claim to fame of all three is that
 * they are valid on the whole  complex plane, and are
 * reasonably accurate (14-digit relative accuracy) and
 * reasonably fast. Also, please note that all three have not
 * been carefully tested in the third quadrant (both x and y
 * negative); some sort of numerical trouble is possible
 * (though has not been observed) for LARGE z in the third
 * quadrant.
 *
 * ifexpon = 1 will cause the subroutine to evaluate the Hankel functions
 *     honestly
 * ifexpon = 0 will cause the subroutine to scale the Hankel functions
 *     by e^{-i \cdot z}.
 */

void hankel_01_complex(cdouble_t z, cdouble_t *h0, cdouble_t *h1, int ifexpon)
{
  cdouble_t cclog;
  cdouble_t cd;
  cdouble_t fj0;
  cdouble_t fj1;
  cdouble_t h0r;
  cdouble_t h0u;
  cdouble_t h1r;
  cdouble_t h1u;
  double half_;
  int ier;
  cdouble_t ima = cdouble_new(0.0e0, 1.0e0);
  double pi = 0.31415926535897932e+01;
  cdouble_t ser2;
  cdouble_t ser3;
  double subt;
  cdouble_t y0;
  cdouble_t y1;
  cdouble_t z2;
  cdouble_t zr;
  cdouble_t zu;
  if (cdouble_imag(z) < 0)
    goto label_1400;

  hank103u(z, & ier, & (* h0), & (* h1), ifexpon);
  return;
  label_1400:
  ;

  if (cdouble_real(z) < 0)
    goto label_2000;

  hank103r(z, & ier, & (* h0), & (* h1), ifexpon);
  return;
  label_2000:
  ;

  zu = cdouble_conj(z);
  zr = cdouble_rmul(- 1, zu);
  hank103u(zu, & ier, & h0u, & h1u, ifexpon);
  hank103r(zr, & ier, & h0r, & h1r, ifexpon);
  if (ifexpon == 1)
    goto label_3000;

  subt = fabs(cdouble_imag(zu));
  cd = cdouble_exp(cdouble_add(cdouble_fromreal((- 1) * subt), cdouble_mul(ima, zu)));
  h0u = cdouble_mul(h0u, cd);
  h1u = cdouble_mul(h1u, cd);
  cd = cdouble_exp(cdouble_add(cdouble_fromreal((- 1) * subt), cdouble_mul(ima, zr)));
  h0r = cdouble_mul(h0r, cd);
  h1r = cdouble_mul(h1r, cd);
  label_3000:
  ;

  half_ = 1;
  half_ = half_ / 2;
  y0 = cdouble_divide(cdouble_rmul(half_, cdouble_add(h0u, h0r)), ima);
  fj0 = cdouble_rmul((- 1) * half_, cdouble_add(h0u, cdouble_rmul(- 1, h0r)));
  y1 = cdouble_divide(cdouble_rmul((- 1) * half_, cdouble_add(h1u, cdouble_rmul(- 1, h1r))), ima);
  fj1 = cdouble_rmul(half_, cdouble_add(h1u, h1r));
  z2 = cdouble_rmul(- 1, cdouble_conj(z));
  cclog = cdouble_log(z2);
  ser2 = cdouble_add(y0, cdouble_rmul(- 1, cdouble_mul(cdouble_divider(cdouble_rmul(2, fj0), pi), cclog)));
  ser3 = cdouble_add(y1, cdouble_rmul(- 1, cdouble_mul(cdouble_divider(cdouble_rmul(2, fj1), pi), cclog)));
  fj0 = cdouble_conj(fj0);
  fj1 = cdouble_rmul(- 1, cdouble_conj(fj1));
  ser2 = cdouble_conj(ser2);
  ser3 = cdouble_rmul(- 1, cdouble_conj(ser3));
  cclog = cdouble_log(z);
  y0 = cdouble_add(ser2, cdouble_mul(cdouble_divider(cdouble_rmul(2, fj0), pi), cclog));
  y1 = cdouble_add(ser3, cdouble_mul(cdouble_divider(cdouble_rmul(2, fj1), pi), cclog));
  * h0 = cdouble_add(fj0, cdouble_mul(ima, y0));
  * h1 = cdouble_add(fj1, cdouble_mul(ima, y1));
  if (ifexpon == 1)
    return;

  cd = cdouble_exp(cdouble_add(cdouble_fromreal(subt), cdouble_rmul(- 1, cdouble_mul(ima, z))));
  * h0 = cdouble_mul(* h0, cd);
  * h1 = cdouble_mul(* h1, cd);
}

__constant double hank103u_c0p1[] = {(- 1) * 0.6619836118357782e-12, (- 1) * 0.6619836118612709e-12, (- 1) * 0.7307514264754200e-21, 0.3928160926261892e-10, 0.5712712520172854e-09, (- 1) * 0.5712712519967086e-09, (- 1) * 0.1083820384008718e-07, (- 1) * 0.1894529309455499e-18, 0.7528123700585197e-07, 0.7528123700841491e-07, 0.1356544045548053e-16, (- 1) * 0.8147940452202855e-06, (- 1) * 0.3568198575016769e-05, 0.3568198574899888e-05, 0.2592083111345422e-04, 0.4209074870019400e-15, (- 1) * 0.7935843289157352e-04, (- 1) * 0.7935843289415642e-04, (- 1) * 0.6848330800445365e-14, 0.4136028298630129e-03, 0.9210433149997867e-03, (- 1) * 0.9210433149680665e-03, (- 1) * 0.3495306809056563e-02, (- 1) * 0.6469844672213905e-13, 0.5573890502766937e-02, 0.5573890503000873e-02, 0.3767341857978150e-12, (- 1) * 0.1439178509436339e-01, (- 1) * 0.1342403524448708e-01, 0.1342403524340215e-01, 0.8733016209933828e-02, 0.1400653553627576e-11, 0.2987361261932706e-01, 0.2987361261607835e-01, (- 1) * 0.3388096836339433e-11, (- 1) * 0.1690673895793793e+00, 0.2838366762606121e+00, (- 1) * 0.2838366762542546e+00, 0.7045107746587499e+00, (- 1) * 0.5363893133864181e-11, (- 1) * 0.7788044738211666e+00, (- 1) * 0.7788044738130360e+00, 0.5524779104964783e-11, 0.1146003459721775e+01, 0.6930697486173089e+00, (- 1) * 0.6930697486240221e+00, (- 1) * 0.7218270272305891e+00, 0.3633022466839301e-11, 0.3280924142354455e+00, 0.3280924142319602e+00, (- 1) * 0.1472323059106612e-11, (- 1) * 0.2608421334424268e+00, (- 1) * 0.9031397649230536e-01, 0.9031397649339185e-01, 0.5401342784296321e-01, (- 1) * 0.3464095071668884e-12, (- 1) * 0.1377057052946721e-01, (- 1) * 0.1377057052927901e-01, 0.4273263742980154e-13, 0.5877224130705015e-02, 0.1022508471962664e-02, (- 1) * 0.1022508471978459e-02, (- 1) * 0.2789107903871137e-03, 0.2283984571396129e-14, 0.2799719727019427e-04, 0.2799719726970900e-04, (- 1) * 0.3371218242141487e-16, (- 1) * 0.3682310515545645e-05, (- 1) * 0.1191412910090512e-06, 0.1191412910113518e-06};
__constant double hank103u_c0p2[] = {0.5641895835516786e+00, (- 1) * 0.5641895835516010e+00, (- 1) * 0.3902447089770041e-09, (- 1) * 0.3334441074447365e-11, (- 1) * 0.7052368835911731e-01, (- 1) * 0.7052368821797083e-01, 0.1957299315085370e-08, (- 1) * 0.3126801711815631e-06, (- 1) * 0.3967331737107949e-01, 0.3967327747706934e-01, 0.6902866639752817e-04, 0.3178420816292497e-06, 0.4080457166061280e-01, 0.4080045784614144e-01, (- 1) * 0.2218731025620065e-04, 0.6518438331871517e-02, 0.9798339748600499e-01, (- 1) * 0.9778028374972253e-01, (- 1) * 0.3151825524811773e+00, (- 1) * 0.7995603166188139e-03, 0.1111323666639636e+01, 0.1116791178994330e+01, 0.1635711249533488e-01, (- 1) * 0.8527067497983841e+01, (- 1) * 0.2595553689471247e+02, 0.2586942834408207e+02, 0.1345583522428299e+03, 0.2002017907999571e+00, (- 1) * 0.3086364384881525e+03, (- 1) * 0.3094609382885628e+03, (- 1) * 0.1505974589617013e+01, 0.1250150715797207e+04, 0.2205210257679573e+04, (- 1) * 0.2200328091885836e+04, (- 1) * 0.6724941072552172e+04, (- 1) * 0.7018887749450317e+01, 0.8873498980910335e+04, 0.8891369384353965e+04, 0.2008805099643591e+02, (- 1) * 0.2030681426035686e+05, (- 1) * 0.2010017782384992e+05, 0.2006046282661137e+05, 0.3427941581102808e+05, 0.3432892927181724e+02, (- 1) * 0.2511417407338804e+05, (- 1) * 0.2516567363193558e+05, (- 1) * 0.3318253740485142e+02, 0.3143940826027085e+05, 0.1658466564673543e+05, (- 1) * 0.1654843151976437e+05, (- 1) * 0.1446345041326510e+05, (- 1) * 0.1645433213663233e+02, 0.5094709396573681e+04, 0.5106816671258367e+04, 0.3470692471612145e+01, (- 1) * 0.2797902324245621e+04, (- 1) * 0.5615581955514127e+03, 0.5601021281020627e+03, 0.1463856702925587e+03, 0.1990076422327786e+00, (- 1) * 0.9334741618922085e+01, (- 1) * 0.9361368967669095e+01};
__constant double hank103u_c1p1[] = {0.4428361927253983e-12, (- 1) * 0.4428361927153559e-12, (- 1) * 0.2575693161635231e-10, (- 1) * 0.2878656317479645e-21, 0.3658696304107867e-09, 0.3658696304188925e-09, 0.7463138750413651e-19, (- 1) * 0.6748894854135266e-08, (- 1) * 0.4530098210372099e-07, 0.4530098210271137e-07, 0.4698787882823243e-06, 0.5343848349451927e-17, (- 1) * 0.1948662942158171e-05, (- 1) * 0.1948662942204214e-05, (- 1) * 0.1658085463182409e-15, 0.1316906100496570e-04, 0.3645368564036497e-04, (- 1) * 0.3645368563934748e-04, (- 1) * 0.1633458547818390e-03, (- 1) * 0.2697770638600506e-14, 0.2816784976551660e-03, 0.2816784976676616e-03, 0.2548673351180060e-13, (- 1) * 0.6106478245116582e-03, 0.2054057459296899e-03, (- 1) * 0.2054057460218446e-03, (- 1) * 0.6254962367291260e-02, 0.1484073406594994e-12, 0.1952900562500057e-01, 0.1952900562457318e-01, (- 1) * 0.5517611343746895e-12, (- 1) * 0.8528074392467523e-01, (- 1) * 0.1495138141086974e+00, 0.1495138141099772e+00, 0.4394907314508377e+00, (- 1) * 0.1334677126491326e-11, (- 1) * 0.1113740586940341e+01, (- 1) * 0.1113740586937837e+01, 0.2113005088866033e-11, 0.1170212831401968e+01, 0.1262152242318805e+01, (- 1) * 0.1262152242322008e+01, (- 1) * 0.1557810619605511e+01, 0.2176383208521897e-11, 0.8560741701626648e+00, 0.8560741701600203e+00, (- 1) * 0.1431161194996653e-11, (- 1) * 0.8386735092525187e+00, (- 1) * 0.3651819176599290e+00, 0.3651819176613019e+00, 0.2811692367666517e+00, (- 1) * 0.5799941348040361e-12, (- 1) * 0.9494630182937280e-01, (- 1) * 0.9494630182894480e-01, 0.1364615527772751e-12, 0.5564896498129176e-01, 0.1395239688792536e-01, (- 1) * 0.1395239688799950e-01, (- 1) * 0.5871314703753967e-02, 0.1683372473682212e-13, 0.1009157100083457e-02, 0.1009157100077235e-02, (- 1) * 0.8997331160162008e-15, (- 1) * 0.2723724213360371e-03, (- 1) * 0.2708696587599713e-04, 0.2708696587618830e-04, 0.3533092798326666e-05, (- 1) * 0.1328028586935163e-16, (- 1) * 0.1134616446885126e-06, (- 1) * 0.1134616446876064e-06};
__constant double hank103u_c1p2[] = {(- 1) * 0.5641895835446003e+00, (- 1) * 0.5641895835437973e+00, 0.3473016376419171e-10, (- 1) * 0.3710264617214559e-09, 0.2115710836381847e+00, (- 1) * 0.2115710851180242e+00, 0.3132928887334847e-06, 0.2064187785625558e-07, (- 1) * 0.6611954881267806e-01, (- 1) * 0.6611997176900310e-01, (- 1) * 0.3386004893181560e-05, 0.7146557892862998e-04, (- 1) * 0.5728505088320786e-01, 0.5732906930408979e-01, (- 1) * 0.6884187195973806e-02, (- 1) * 0.2383737409286457e-03, 0.1170452203794729e+00, 0.1192356405185651e+00, 0.8652871239920498e-02, (- 1) * 0.3366165876561572e+00, (- 1) * 0.1203989383538728e+01, 0.1144625888281483e+01, 0.9153684260534125e+01, 0.1781426600949249e+00, (- 1) * 0.2740411284066946e+02, (- 1) * 0.2834461441294877e+02, (- 1) * 0.2192611071606340e+01, 0.1445470231392735e+03, 0.3361116314072906e+03, (- 1) * 0.3270584743216529e+03, (- 1) * 0.1339254798224146e+04, (- 1) * 0.1657618537130453e+02, 0.2327097844591252e+04, 0.2380960024514808e+04, 0.7760611776965994e+02, (- 1) * 0.7162513471480693e+04, (- 1) * 0.9520608696419367e+04, 0.9322604506839242e+04, 0.2144033447577134e+05, 0.2230232555182369e+03, (- 1) * 0.2087584364240919e+05, (- 1) * 0.2131762020653283e+05, (- 1) * 0.3825699231499171e+03, 0.3582976792594737e+05, 0.2642632405857713e+05, (- 1) * 0.2585137938787267e+05, (- 1) * 0.3251446505037506e+05, (- 1) * 0.3710875194432116e+03, 0.1683805377643986e+05, 0.1724393921722052e+05, 0.1846128226280221e+03, (- 1) * 0.1479735877145448e+05, (- 1) * 0.5258288893282565e+04, 0.5122237462705988e+04, 0.2831540486197358e+04, 0.3905972651440027e+02, (- 1) * 0.5562781548969544e+03, (- 1) * 0.5726891190727206e+03, (- 1) * 0.2246192560136119e+01, 0.1465347141877978e+03, 0.9456733342595993e+01, (- 1) * 0.9155767836700837e+01};
void hank103u(cdouble_t z, int *ier, cdouble_t *h0, cdouble_t *h1, int ifexpon)
{




  cdouble_t ccex;
  cdouble_t cd;
  double com;
  double d;
  double done;
  cdouble_t ima = cdouble_new(0.0e0, 1.0e0);
  int m;
  double thresh1;
  double thresh2;
  double thresh3;
  cdouble_t zzz9;
  * ier = 0;
  com = cdouble_real(z);
  if (cdouble_imag(z) >= 0)
    goto label_1200;

  * ier = 4;
  return;
  label_1200:
  ;

  done = 1;
  thresh1 = 1;
  thresh2 = 3.7 * 3.7;
  thresh3 = 400;
  d = cdouble_real(cdouble_mul(z, cdouble_conj(z)));
  if ((d < thresh1) || (d > thresh3))
    goto label_3000;

  if (d > thresh2)
    goto label_2000;

  cd = cdouble_rdivide(done, cdouble_sqrt(z));
  ccex = cd;
  if (ifexpon == 1)
    ccex = cdouble_mul(ccex, cdouble_exp(cdouble_mul(ima, z)));

  zzz9 = cdouble_powr(z, 9);
  m = 35;
  hank103p((__constant cdouble_t *) (& (* hank103u_c0p1)), m, cd, & (* h0));
  * h0 = cdouble_mul(cdouble_mul(* h0, ccex), zzz9);
  hank103p((__constant cdouble_t *) (& (* hank103u_c1p1)), m, cd, & (* h1));
  * h1 = cdouble_mul(cdouble_mul(* h1, ccex), zzz9);
  return;
  label_2000:
  ;

  cd = cdouble_rdivide(done, cdouble_sqrt(z));
  ccex = cd;
  if (ifexpon == 1)
    ccex = cdouble_mul(ccex, cdouble_exp(cdouble_mul(ima, z)));

  m = 31;
  hank103p((__constant cdouble_t *) (& (* hank103u_c0p2)), m, cd, & (* h0));
  * h0 = cdouble_mul(* h0, ccex);
  m = 31;
  hank103p((__constant cdouble_t *) (& (* hank103u_c1p2)), m, cd, & (* h1));
  * h1 = cdouble_mul(* h1, ccex);
  return;
  label_3000:
  ;

  if (d > 50.e0)
    goto label_4000;

  hank103l(z, & (* h0), & (* h1), ifexpon);
  return;
  label_4000:
  ;

  hank103a(z, & (* h0), & (* h1), ifexpon);
}

void hank103p(__constant cdouble_t *p, int m, cdouble_t z, cdouble_t *f)
{
  int i;

  * f = p[m - 1];
  for (i = m + (- 1); i >= 1; i += - 1)
  {
    * f = cdouble_add(cdouble_mul(* f, z), p[i - 1]);
    label_1200:
    ;

  }

}

__constant double hank103a_p[] = {0.1000000000000000e+01, (- 1) * 0.7031250000000000e-01, 0.1121520996093750e+00, (- 1) * 0.5725014209747314e+00, 0.6074042001273483e+01, (- 1) * 0.1100171402692467e+03, 0.3038090510922384e+04, (- 1) * 0.1188384262567833e+06, 0.6252951493434797e+07, (- 1) * 0.4259392165047669e+09, 0.3646840080706556e+11, (- 1) * 0.3833534661393944e+13, 0.4854014686852901e+15, (- 1) * 0.7286857349377657e+17, 0.1279721941975975e+20, (- 1) * 0.2599382102726235e+22, 0.6046711487532401e+24, (- 1) * 0.1597065525294211e+27};
__constant double hank103a_p1[] = {0.1000000000000000e+01, 0.1171875000000000e+00, (- 1) * 0.1441955566406250e+00, 0.6765925884246826e+00, (- 1) * 0.6883914268109947e+01, 0.1215978918765359e+03, (- 1) * 0.3302272294480852e+04, 0.1276412726461746e+06, (- 1) * 0.6656367718817687e+07, 0.4502786003050393e+09, (- 1) * 0.3833857520742789e+11, 0.4011838599133198e+13, (- 1) * 0.5060568503314726e+15, 0.7572616461117957e+17, (- 1) * 0.1326257285320556e+20, 0.2687496750276277e+22, (- 1) * 0.6238670582374700e+24, 0.1644739123064188e+27};
__constant double hank103a_q[] = {(- 1) * 0.1250000000000000e+00, 0.7324218750000000e-01, (- 1) * 0.2271080017089844e+00, 0.1727727502584457e+01, (- 1) * 0.2438052969955606e+02, 0.5513358961220206e+03, (- 1) * 0.1825775547429317e+05, 0.8328593040162893e+06, (- 1) * 0.5006958953198893e+08, 0.3836255180230434e+10, (- 1) * 0.3649010818849834e+12, 0.4218971570284096e+14, (- 1) * 0.5827244631566907e+16, 0.9476288099260110e+18, (- 1) * 0.1792162323051699e+21, 0.3900121292034000e+23, (- 1) * 0.9677028801069847e+25, 0.2715581773544907e+28};
__constant double hank103a_q1[] = {0.3750000000000000e+00, (- 1) * 0.1025390625000000e+00, 0.2775764465332031e+00, (- 1) * 0.1993531733751297e+01, 0.2724882731126854e+02, (- 1) * 0.6038440767050702e+03, 0.1971837591223663e+05, (- 1) * 0.8902978767070679e+06, 0.5310411010968522e+08, (- 1) * 0.4043620325107754e+10, 0.3827011346598606e+12, (- 1) * 0.4406481417852279e+14, 0.6065091351222699e+16, (- 1) * 0.9833883876590680e+18, 0.1855045211579829e+21, (- 1) * 0.4027994121281017e+23, 0.9974783533410457e+25, (- 1) * 0.2794294288720121e+28};
void hank103a(cdouble_t z, cdouble_t *h0, cdouble_t *h1, int ifexpon)
{
  cdouble_t cccexp;
  cdouble_t cdd;
  cdouble_t cdumb = cdouble_new(0.70710678118654757e+00, (- 1) * 0.70710678118654746e+00);
  double done = 1.0e0;
  int i;
  cdouble_t ima = cdouble_new(0.0e0, 1.0e0);
  int m;


  double pi = 0.31415926535897932e+01;
  cdouble_t pp;
  cdouble_t pp1;


  cdouble_t qq;
  cdouble_t qq1;
  cdouble_t zinv;
  cdouble_t zinv22;
  m = 10;
  zinv = cdouble_rdivide(done, z);
  pp = cdouble_fromreal(hank103a_p[m - 1]);
  pp1 = cdouble_fromreal(hank103a_p1[m - 1]);
  zinv22 = cdouble_mul(zinv, zinv);
  qq = cdouble_fromreal(hank103a_q[m - 1]);
  qq1 = cdouble_fromreal(hank103a_q1[m - 1]);
  for (i = m + (- 1); i >= 1; i += - 1)
  {
    pp = cdouble_add(cdouble_fromreal(hank103a_p[i - 1]), cdouble_mul(pp, zinv22));
    pp1 = cdouble_add(cdouble_fromreal(hank103a_p1[i - 1]), cdouble_mul(pp1, zinv22));
    qq = cdouble_add(cdouble_fromreal(hank103a_q[i - 1]), cdouble_mul(qq, zinv22));
    qq1 = cdouble_add(cdouble_fromreal(hank103a_q1[i - 1]), cdouble_mul(qq1, zinv22));
    label_1600:
    ;

  }

  qq = cdouble_mul(qq, zinv);
  qq1 = cdouble_mul(qq1, zinv);
  cccexp = cdouble_fromreal(1);
  if (ifexpon == 1)
    cccexp = cdouble_exp(cdouble_mul(ima, z));

  cdd = cdouble_sqrt(cdouble_rmul(2 / pi, zinv));
  * h0 = cdouble_add(pp, cdouble_mul(ima, qq));
  * h0 = cdouble_mul(cdouble_mul(cdouble_mul(cdd, cdumb), cccexp), * h0);
  * h1 = cdouble_add(pp1, cdouble_mul(ima, qq1));
  * h1 = cdouble_rmul(- 1, cdouble_mul(cdouble_mul(cdouble_mul(cdouble_mul(cdd, cccexp), cdumb), * h1), ima));
}

__constant double hank103l_cj0[] = {0.1000000000000000e+01, (- 1) * 0.2500000000000000e+00, 0.1562500000000000e-01, (- 1) * 0.4340277777777778e-03, 0.6781684027777778e-05, (- 1) * 0.6781684027777778e-07, 0.4709502797067901e-09, (- 1) * 0.2402807549524439e-11, 0.9385966990329841e-14, (- 1) * 0.2896903392077112e-16, 0.7242258480192779e-19, (- 1) * 0.1496334396734045e-21, 0.2597802772107717e-24, (- 1) * 0.3842903509035085e-27, 0.4901662639075363e-30, (- 1) * 0.5446291821194848e-33};
__constant double hank103l_cj1[] = {(- 1) * 0.5000000000000000e+00, 0.6250000000000000e-01, (- 1) * 0.2604166666666667e-02, 0.5425347222222222e-04, (- 1) * 0.6781684027777778e-06, 0.5651403356481481e-08, (- 1) * 0.3363930569334215e-10, 0.1501754718452775e-12, (- 1) * 0.5214426105738801e-15, 0.1448451696038556e-17, (- 1) * 0.3291935672814899e-20, 0.6234726653058522e-23, (- 1) * 0.9991549123491221e-26, 0.1372465538941102e-28, (- 1) * 0.1633887546358454e-31, 0.1701966194123390e-34};
__constant double hank103l_ser2[] = {0.2500000000000000e+00, (- 1) * 0.2343750000000000e-01, 0.7957175925925926e-03, (- 1) * 0.1412850839120370e-04, 0.1548484519675926e-06, (- 1) * 0.1153828185281636e-08, 0.6230136717695511e-11, (- 1) * 0.2550971742728932e-13, 0.8195247730999099e-16, (- 1) * 0.2121234517551702e-18, 0.4518746345057852e-21, (- 1) * 0.8061529302289970e-24, 0.1222094716680443e-26, (- 1) * 0.1593806157473552e-29, 0.1807204342667468e-32, (- 1) * 0.1798089518115172e-35};
__constant double hank103l_ser2der[] = {0.5000000000000000e+00, (- 1) * 0.9375000000000000e-01, 0.4774305555555556e-02, (- 1) * 0.1130280671296296e-03, 0.1548484519675926e-05, (- 1) * 0.1384593822337963e-07, 0.8722191404773715e-10, (- 1) * 0.4081554788366291e-12, 0.1475144591579838e-14, (- 1) * 0.4242469035103405e-17, 0.9941241959127275e-20, (- 1) * 0.1934767032549593e-22, 0.3177446263369152e-25, (- 1) * 0.4462657240925946e-28, 0.5421613028002404e-31, (- 1) * 0.5753886457968550e-34};
void hank103l(cdouble_t z, cdouble_t *h0, cdouble_t *h1, int ifexpon)
{
  cdouble_t cd;
  cdouble_t cdddlog;


  cdouble_t fj0;
  cdouble_t fj1;
  double gamma = 0.5772156649015328606e+00;
  int i;
  cdouble_t ima = cdouble_new(0.0e0, 1.0e0);
  int m;
  double pi = 0.31415926535897932e+01;


  double two = 2.0e0;
  cdouble_t y0;
  cdouble_t y1;
  cdouble_t z2;
  m = 16;
  fj0 = cdouble_fromreal(0);
  fj1 = cdouble_fromreal(0);
  y0 = cdouble_fromreal(0);
  y1 = cdouble_fromreal(0);
  z2 = cdouble_mul(z, z);
  cd = cdouble_fromreal(1);
  for (i = 1; i <= m; i += 1)
  {
    fj0 = cdouble_add(fj0, cdouble_rmul(hank103l_cj0[i - 1], cd));
    fj1 = cdouble_add(fj1, cdouble_rmul(hank103l_cj1[i - 1], cd));
    y1 = cdouble_add(y1, cdouble_rmul(hank103l_ser2der[i - 1], cd));
    cd = cdouble_mul(cd, z2);
    y0 = cdouble_add(y0, cdouble_rmul(hank103l_ser2[i - 1], cd));
    label_1800:
    ;

  }

  fj1 = cdouble_rmul(- 1, cdouble_mul(fj1, z));
  cdddlog = cdouble_add(cdouble_fromreal(gamma), cdouble_log(cdouble_divider(z, two)));
  y0 = cdouble_add(cdouble_mul(cdddlog, fj0), y0);
  y0 = cdouble_rmul(two / pi, y0);
  y1 = cdouble_mul(y1, z);
  y1 = cdouble_add(cdouble_add(cdouble_rmul(- 1, cdouble_mul(cdddlog, fj1)), cdouble_divide(fj0, z)), y1);
  y1 = cdouble_divider(cdouble_rmul((- 1) * two, y1), pi);
  * h0 = cdouble_add(fj0, cdouble_mul(ima, y0));
  * h1 = cdouble_add(fj1, cdouble_mul(ima, y1));
  if (ifexpon == 1)
    return;

  cd = cdouble_exp(cdouble_rmul(- 1, cdouble_mul(ima, z)));
  * h0 = cdouble_mul(* h0, cd);
  * h1 = cdouble_mul(* h1, cd);
}

__constant double hank103r_c0p1[] = {(- 1) * 0.4268441995428495e-23, 0.4374027848105921e-23, 0.9876152216238049e-23, (- 1) * 0.1065264808278614e-20, 0.6240598085551175e-19, 0.6658529985490110e-19, (- 1) * 0.5107210870050163e-17, (- 1) * 0.2931746613593983e-18, 0.1611018217758854e-15, (- 1) * 0.1359809022054077e-15, (- 1) * 0.7718746693707326e-15, 0.6759496139812828e-14, (- 1) * 0.1067620915195442e-12, (- 1) * 0.1434699000145826e-12, 0.3868453040754264e-11, 0.7061853392585180e-12, (- 1) * 0.6220133527871203e-10, 0.3957226744337817e-10, 0.3080863675628417e-09, (- 1) * 0.1154618431281900e-08, 0.7793319486868695e-08, 0.1502570745460228e-07, (- 1) * 0.1978090852638430e-06, (- 1) * 0.7396691873499030e-07, 0.2175857247417038e-05, (- 1) * 0.8473534855334919e-06, (- 1) * 0.1053381327609720e-04, 0.2042555121261223e-04, (- 1) * 0.4812568848956982e-04, (- 1) * 0.1961519090873697e-03, 0.1291714391689374e-02, 0.9234422384950050e-03, (- 1) * 0.1113890671502769e-01, 0.9053687375483149e-03, 0.5030666896877862e-01, (- 1) * 0.4923119348218356e-01, 0.5202355973926321e+00, (- 1) * 0.1705244841954454e+00, (- 1) * 0.1134990486611273e+01, (- 1) * 0.1747542851820576e+01, 0.8308174484970718e+01, 0.2952358687641577e+01, (- 1) * 0.3286074510100263e+02, 0.1126542966971545e+02, 0.6576015458463394e+02, (- 1) * 0.1006116996293757e+03, 0.3216834899377392e+02, 0.3614005342307463e+03, (- 1) * 0.6653878500833375e+03, (- 1) * 0.6883582242804924e+03, 0.2193362007156572e+04, 0.2423724600546293e+03, (- 1) * 0.3665925878308203e+04, 0.2474933189642588e+04, 0.1987663383445796e+04, (- 1) * 0.7382586600895061e+04, 0.4991253411017503e+04, 0.1008505017740918e+05, (- 1) * 0.1285284928905621e+05, (- 1) * 0.5153674821668470e+04, 0.1301656757246985e+05, (- 1) * 0.4821250366504323e+04, (- 1) * 0.4982112643422311e+04, 0.9694070195648748e+04, (- 1) * 0.1685723189234701e+04, (- 1) * 0.6065143678129265e+04, 0.2029510635584355e+04, 0.1244402339119502e+04, (- 1) * 0.4336682903961364e+03, 0.8923209875101459e+02};
__constant double hank103r_c0p2[] = {0.5641895835569398e+00, (- 1) * 0.5641895835321127e+00, (- 1) * 0.7052370223565544e-01, (- 1) * 0.7052369923405479e-01, (- 1) * 0.3966909368581382e-01, 0.3966934297088857e-01, 0.4130698137268744e-01, 0.4136196771522681e-01, 0.6240742346896508e-01, (- 1) * 0.6553556513852438e-01, (- 1) * 0.3258849904760676e-01, (- 1) * 0.7998036854222177e-01, (- 1) * 0.3988006311955270e+01, 0.1327373751674479e+01, 0.6121789346915312e+02, (- 1) * 0.9251865216627577e+02, 0.4247064992018806e+03, 0.2692553333489150e+04, (- 1) * 0.4374691601489926e+05, (- 1) * 0.3625248208112831e+05, 0.1010975818048476e+07, (- 1) * 0.2859360062580096e+05, (- 1) * 0.1138970241206912e+08, 0.1051097979526042e+08, 0.2284038899211195e+08, (- 1) * 0.2038012515235694e+09, 0.1325194353842857e+10, 0.1937443530361381e+10, (- 1) * 0.2245999018652171e+11, (- 1) * 0.5998903865344352e+10, 0.1793237054876609e+12, (- 1) * 0.8625159882306147e+11, (- 1) * 0.5887763042735203e+12, 0.1345331284205280e+13, (- 1) * 0.2743432269370813e+13, (- 1) * 0.8894942160272255e+13, 0.4276463113794564e+14, 0.2665019886647781e+14, (- 1) * 0.2280727423955498e+15, 0.3686908790553973e+14, 0.5639846318168615e+15, (- 1) * 0.6841529051615703e+15, 0.9901426799966038e+14, 0.2798406605978152e+16, (- 1) * 0.4910062244008171e+16, (- 1) * 0.5126937967581805e+16, 0.1387292951936756e+17, 0.1043295727224325e+16, (- 1) * 0.1565204120687265e+17, 0.1215262806973577e+17, 0.3133802397107054e+16, (- 1) * 0.1801394550807078e+17, 0.4427598668012807e+16, 0.6923499968336864e+16};
__constant double hank103r_c1p1[] = {(- 1) * 0.4019450270734195e-23, (- 1) * 0.4819240943285824e-23, 0.1087220822839791e-20, 0.1219058342725899e-21, (- 1) * 0.7458149572694168e-19, 0.5677825613414602e-19, 0.8351815799518541e-18, (- 1) * 0.5188585543982425e-17, 0.1221075065755962e-15, 0.1789261470637227e-15, (- 1) * 0.6829972121890858e-14, (- 1) * 0.1497462301804588e-14, 0.1579028042950957e-12, (- 1) * 0.9414960303758800e-13, (- 1) * 0.1127570848999746e-11, 0.3883137940932639e-11, (- 1) * 0.3397569083776586e-10, (- 1) * 0.6779059427459179e-10, 0.1149529442506273e-08, 0.4363087909873751e-09, (- 1) * 0.1620182360840298e-07, 0.6404695607668289e-08, 0.9651461037419628e-07, (- 1) * 0.1948572160668177e-06, 0.6397881896749446e-06, 0.2318661930507743e-05, (- 1) * 0.1983192412396578e-04, (- 1) * 0.1294811208715315e-04, 0.2062663873080766e-03, (- 1) * 0.2867633324735777e-04, (- 1) * 0.1084309075952914e-02, 0.1227880935969686e-02, 0.2538406015667726e-03, (- 1) * 0.1153316815955356e-01, 0.4520140008266983e-01, 0.5693944718258218e-01, (- 1) * 0.9640790976658534e+00, (- 1) * 0.6517135574036008e+00, 0.2051491829570049e+01, (- 1) * 0.1124151010077572e+01, (- 1) * 0.3977380460328048e+01, 0.8200665483661009e+01, (- 1) * 0.7950131652215817e+01, (- 1) * 0.3503037697046647e+02, 0.9607320812492044e+02, 0.7894079689858070e+02, (- 1) * 0.3749002890488298e+03, (- 1) * 0.8153831134140778e+01, 0.7824282518763973e+03, (- 1) * 0.6035276543352174e+03, (- 1) * 0.5004685759675768e+03, 0.2219009060854551e+04, (- 1) * 0.2111301101664672e+04, (- 1) * 0.4035632271617418e+04, 0.7319737262526823e+04, 0.2878734389521922e+04, (- 1) * 0.1087404934318719e+05, 0.3945740567322783e+04, 0.6727823761148537e+04, (- 1) * 0.1253555346597302e+05, 0.3440468371829973e+04, 0.1383240926370073e+05, (- 1) * 0.9324927373036743e+04, (- 1) * 0.6181580304530313e+04, 0.6376198146666679e+04, (- 1) * 0.1033615527971958e+04, (- 1) * 0.1497604891055181e+04, 0.1929025541588262e+04, (- 1) * 0.4219760183545219e+02, (- 1) * 0.4521162915353207e+03};
__constant double hank103r_c1p2[] = {(- 1) * 0.5641895835431980e+00, (- 1) * 0.5641895835508094e+00, 0.2115710934750869e+00, (- 1) * 0.2115710923186134e+00, (- 1) * 0.6611607335011594e-01, (- 1) * 0.6611615414079688e-01, (- 1) * 0.5783289433408652e-01, 0.5785737744023628e-01, 0.8018419623822896e-01, 0.8189816020440689e-01, 0.1821045296781145e+00, (- 1) * 0.2179738973008740e+00, 0.5544705668143094e+00, 0.2224466316444440e+01, (- 1) * 0.8563271248520645e+02, (- 1) * 0.4394325758429441e+02, 0.2720627547071340e+04, (- 1) * 0.6705390850875292e+03, (- 1) * 0.3936221960600770e+05, 0.5791730432605451e+05, (- 1) * 0.1976787738827811e+06, (- 1) * 0.1502498631245144e+07, 0.2155317823990686e+08, 0.1870953796705298e+08, (- 1) * 0.4703995711098311e+09, 0.3716595906453190e+07, 0.5080557859012385e+10, (- 1) * 0.4534199223888966e+10, (- 1) * 0.1064438211647413e+11, 0.8612243893745942e+11, (- 1) * 0.5466017687785078e+12, (- 1) * 0.8070950386640701e+12, 0.9337074941225827e+13, 0.2458379240643264e+13, (- 1) * 0.7548692171244579e+14, 0.3751093169954336e+14, 0.2460677431350039e+15, (- 1) * 0.5991919372881911e+15, 0.1425679408434606e+16, 0.4132221939781502e+16, (- 1) * 0.2247506469468969e+17, (- 1) * 0.1269771078165026e+17, 0.1297336292749026e+18, (- 1) * 0.2802626909791308e+17, (- 1) * 0.3467137222813017e+18, 0.4773955215582192e+18, (- 1) * 0.2347165776580206e+18, (- 1) * 0.2233638097535785e+19, 0.5382350866778548e+19, 0.4820328886922998e+19, (- 1) * 0.1928978948099345e+20, 0.1575498747750907e+18, 0.3049162180215152e+20, (- 1) * 0.2837046201123502e+20, (- 1) * 0.5429391644354291e+19, 0.6974653380104308e+20, (- 1) * 0.5322120857794536e+20, (- 1) * 0.6739879079691706e+20, 0.6780343087166473e+20, 0.1053455984204666e+20, (- 1) * 0.2218784058435737e+20, 0.1505391868530062e+20};
void hank103r(cdouble_t z, int *ier, cdouble_t *h0, cdouble_t *h1, int ifexpon)
{




  cdouble_t cccexp;
  cdouble_t cd;
  cdouble_t cdd;
  double d;
  double done;
  cdouble_t ima = cdouble_new(0.0e0, 1.0e0);
  int m;
  double thresh1;
  double thresh2;
  double thresh3;
  cdouble_t zz18;
  * ier = 0;
  if ((cdouble_real(z) >= 0) && (cdouble_imag(z) <= 0))
    goto label_1400;

  * ier = 4;
  return;
  label_1400:
  ;

  done = 1;
  thresh1 = 16;
  thresh2 = 64;
  thresh3 = 400;
  d = cdouble_real(cdouble_mul(z, cdouble_conj(z)));
  if ((d < thresh1) || (d > thresh3))
    goto label_3000;

  if (d > thresh2)
    goto label_2000;

  cccexp = cdouble_fromreal(1);
  if (ifexpon == 1)
    cccexp = cdouble_exp(cdouble_mul(ima, z));

  cdd = cdouble_rdivide(done, cdouble_sqrt(z));
  cd = cdouble_rdivide(done, z);
  zz18 = cdouble_powr(z, 18);
  m = 35;
  hank103p((__constant cdouble_t *) (& (* hank103r_c0p1)), m, cd, & (* h0));
  * h0 = cdouble_mul(cdouble_mul(cdouble_mul(* h0, cdd), cccexp), zz18);
  hank103p((__constant cdouble_t *) (& (* hank103r_c1p1)), m, cd, & (* h1));
  * h1 = cdouble_mul(cdouble_mul(cdouble_mul(* h1, cdd), cccexp), zz18);
  return;
  label_2000:
  ;

  cd = cdouble_rdivide(done, z);
  cdd = cdouble_sqrt(cd);
  cccexp = cdouble_fromreal(1);
  if (ifexpon == 1)
    cccexp = cdouble_exp(cdouble_mul(ima, z));

  m = 27;
  hank103p((__constant cdouble_t *) (& (* hank103r_c0p2)), m, cd, & (* h0));
  * h0 = cdouble_mul(cdouble_mul(* h0, cccexp), cdd);
  m = 31;
  hank103p((__constant cdouble_t *) (& (* hank103r_c1p2)), m, cd, & (* h1));
  * h1 = cdouble_mul(cdouble_mul(* h1, cccexp), cdd);
  return;
  label_3000:
  ;

  if (d > 50.e0)
    goto label_4000;

  hank103l(z, & (* h0), & (* h1), ifexpon);
  return;
  label_4000:
  ;

  hank103a(z, & (* h0), & (* h1), ifexpon);
}

