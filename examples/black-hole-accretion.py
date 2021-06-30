#!/usr/bin/env python3
#
# TrouNoir model using PyOpenCL or PyCUDA
#
# CC BY-NC-SA 2019 : <emmanuel.quemener@ens-lyon.fr>
#
# Part of matrix programs from: https://forge.cbp.ens-lyon.fr/svn/bench4gpu/
#
# Thanks to Andreas Klockner for PyOpenCL and PyCUDA:
# http://mathema.tician.de/software/pyopencl
#
# Original code programmed in Fortran 77 in mars 1994
# for Practical Work of Numerical Simulation
# DEA (old Master2) in astrophysics and spatial techniques in Meudon
# by Herve Aussel & Emmanuel Quemener
#
# Conversion in C done by Emmanuel Quemener in august 1997
# GPUfication in OpenCL under Python in july 2019
# GPUfication in CUDA under Python in august 2019
#
# Thanks to :
#
# - Herve Aussel for his part of code of black body spectrum
# - Didier Pelat for his help to perform this work
# - Jean-Pierre Luminet for his article published in 1979
# - Numerical Recipes for Runge Kutta recipes
# - Luc Blanchet for his disponibility about my questions in General Relativity
# - Pierre Lena for his passion about science and vulgarisation

# If crash on OpenCL Intel implementation, add following options and force
# export PYOPENCL_COMPILER_OUTPUT=1
# export CL_CONFIG_USE_VECTORIZER=True
# export CL_CONFIG_CPU_VECTORIZER_MODE=16

import pyopencl as cl
import numpy
import time
import sys
import getopt
from socket import gethostname


def DictionariesAPI():
    PhysicsList = {"Einstein": 0, "Newton": 1}
    return PhysicsList


#
# Blank space below to simplify debugging on OpenCL code
#


BlobOpenCL = """

#define PI (float)3.14159265359e0f
#define nbr 256

#define EINSTEIN 0
#define NEWTON 1

#ifdef SETTRACKPOINTS
#define TRACKPOINTS SETTRACKPOINTS
#else
#define TRACKPOINTS 2048
#endif

float atanp(float x,float y)
{
  float angle;

  angle=atan2(y,x);

  if (angle<0.e0f)
    {
      angle+=(float)2.e0f*PI;
    }

  return angle;
}

float f(float v)
{
  return v;
}

#if PHYSICS == NEWTON
float g(float u,float m,float b)
{
  return (-u);
}
#else
float g(float u,float m,float b)
{
  return (3.e0f*m/b*pow(u,2)-u);
}
#endif

void calcul(float *us,float *vs,float up,float vp,
            float h,float m,float b)
{
  float c0,c1,c2,c3,d0,d1,d2,d3;

  c0=h*f(vp);
  c1=h*f(vp+c0/2.e0f);
  c2=h*f(vp+c1/2.e0f);
  c3=h*f(vp+c2);
  d0=h*g(up,m,b);
  d1=h*g(up+d0/2.e0f,m,b);
  d2=h*g(up+d1/2.e0f,m,b);
  d3=h*g(up+d2,m,b);

  *us=up+(c0+2.e0f*c1+2.e0f*c2+c3)/6.e0f;
  *vs=vp+(d0+2.e0f*d1+2.e0f*d2+d3)/6.e0f;
}

void rungekutta(float *ps,float *us,float *vs,
                float pp,float up,float vp,
                float h,float m,float b)
{
  calcul(us,vs,up,vp,h,m,b);
  *ps=pp+h;
}

float decalage_spectral(float r,float b,float phi,
                        float tho,float m)
{
  return (sqrt(1-3*m/r)/(1+sqrt(m/pow(r,3))*b*sin(tho)*sin(phi)));
}

float spectre(float rf,int q,float b,float db,
              float h,float r,float m,float bss)
{
  float flx;

//  flx=exp(q*log(r/m))*pow(rf,4)*b*db*h;
  flx=exp(q*log(r/m)+4.e0f*log(rf))*b*db*h;
  return(flx);
}

float spectre_cn(float rf32,float b32,float db32,
                 float h32,float r32,float m32,float bss32)
{

#define MYFLOAT float

  MYFLOAT rf=(MYFLOAT)(rf32);
  MYFLOAT b=(MYFLOAT)(b32);
  MYFLOAT db=(MYFLOAT)(db32);
  MYFLOAT h=(MYFLOAT)(h32);
  MYFLOAT r=(MYFLOAT)(r32);
  MYFLOAT m=(MYFLOAT)(m32);
  MYFLOAT bss=(MYFLOAT)(bss32);

  MYFLOAT flx;
  MYFLOAT nu_rec,nu_em,qu,temp_em,flux_int;
  int fi,posfreq;

#define planck 6.62e-34f
#define k 1.38e-23f
#define c2 9.e16f
#define temp 3.e7f
#define m_point 1.e0f

#define lplanck (log(6.62e0f)-34.e0f*log(10.e0f))
#define lk (log(1.38e0f)-23.e0f*log(10.e0f))
#define lc2 (log(9.e0f)+16.e0f*log(10.e0f))

  MYFLOAT v=1.e0f-3.e0f/r;

  qu=1.e0f/sqrt((1.e0f-3.e0f/r)*r)*(sqrt(r)-sqrt(6.e0f)+sqrt(3.e0f)/2.e0f*log((sqrt(r)+sqrt(3.e0f))/(sqrt(r)-sqrt(3.e0f))* 0.17157287525380988e0f ));  // # noqa: E501

  temp_em=temp*sqrt(m)*exp(0.25e0f*log(m_point)-0.75e0f*log(r)-0.125e0f*log(v)+0.25e0f*log(fabs(qu)));

  flux_int=0.e0f;
  flx=0.e0f;

  for (fi=0;fi<nbr;fi++)
    {
      nu_em=bss*(MYFLOAT)fi/(MYFLOAT)nbr;
      nu_rec=nu_em*rf;
      posfreq=(int)(nu_rec*(MYFLOAT)nbr/bss);
      if ((posfreq>0)&&(posfreq<nbr))
        {
          // Initial version
          // flux_int=2.*planck/c2*pow(nu_em,3)/(exp(planck*nu_em/(k*temp_em))-1.);
          // Version with log used
          //flux_int=2.*exp(lplanck-lc2+3.*log(nu_em))/(exp(exp(lplanck-lk+log(nu_em/temp_em)))-1.);
          // flux_int*=pow(rf,3)*b*db*h;
          //flux_int*=exp(3.e0f*log(rf))*b*db*h;
          flux_int=2.e0f*exp(lplanck-lc2+3.e0f*log(nu_em))/(exp(exp(lplanck-lk+log(nu_em/temp_em)))-1.e0f)*exp(3.e0f*log(rf))*b*db*h;

          flx+=flux_int;
        }
    }

  return((float)(flx));
}

void impact(float phi,float r,float b,float tho,float m,
            float *zp,float *fp,
            int q,float db,
            float h,int raie)
{
  float flx,rf,bss;

  rf=decalage_spectral(r,b,phi,tho,m);

  if (raie==0)
    {
      bss=1.e19f;
      flx=spectre_cn(rf,b,db,h,r,m,bss);
    }
  else
    {
      bss=2.e0f;
      flx=spectre(rf,q,b,db,h,r,m,bss);
    }

  *zp=1.e0f/rf;
  *fp=flx;

}

__kernel void EachPixel(__global float *zImage,__global float *fImage,
                        float Mass,float InternalRadius,
                        float ExternalRadius,float Angle,
                        int Line)
{
   uint xi=(uint)get_global_id(0);
   uint yi=(uint)get_global_id(1);
   uint sizex=(uint)get_global_size(0);
   uint sizey=(uint)get_global_size(1);

   // Perform trajectory for each pixel, exit on hit

  float m,rs,ri,re,tho;
  int q,raie;

  m=Mass;
  rs=2.e0f*m;
  ri=InternalRadius;
  re=ExternalRadius;
  tho=Angle;
  q=-2;
  raie=Line;

  float bmx,db,b,h;
  float rp0,rps;
  float phi,phd;
  uint nh=0;
  float zp=0.e0f,fp=0.e0f;

  // Autosize for image
  bmx=1.25e0f*re;

  h=4.e0f*PI/(float)TRACKPOINTS;

  // set origin as center of image
  float x=(float)xi-(float)(sizex/2)+(float)5.e-1f;
  float y=(float)yi-(float)(sizey/2)+(float)5.e-1f;
  // angle extracted from cylindric symmetry
  phi=atanp(x,y);
  phd=atanp(cos(phi)*sin(tho),cos(tho));


  float up,vp,pp,us,vs,ps;

  // impact parameter
  b=sqrt(x*x+y*y)*(float)2.e0f/(float)sizex*bmx;
  // step of impact parameter;
  db=bmx/(float)(sizex);

  up=0.e0f;
  vp=1.e0f;
  pp=0.e0f;

  rungekutta(&ps,&us,&vs,pp,up,vp,h,m,b);

  rps=fabs(b/us);
  rp0=rps;

  int ExitOnImpact=0;

  do
  {
     nh++;
     pp=ps;
     up=us;
     vp=vs;
     rungekutta(&ps,&us,&vs,pp,up,vp,h,m,b);
     rps=fabs(b/us);
     ExitOnImpact = ((fmod(pp,PI)<fmod(phd,PI))&&(fmod(ps,PI)>fmod(phd,PI)))&&(rps>=ri)&&(rps<=re)?1:0;

  } while ((rps>=rs)&&(rps<=rp0)&&(ExitOnImpact==0)&&(nh<TRACKPOINTS));


  if (ExitOnImpact==1) {
     impact(phi,rps,b,tho,m,&zp,&fp,q,db,h,raie);
  }
  else
  {
     zp=0.e0f;
     fp=0.e0f;
  }

  barrier(CLK_GLOBAL_MEM_FENCE);

  zImage[yi+sizex*xi]=(float)zp;
  fImage[yi+sizex*xi]=(float)fp;
}

__kernel void Pixel(__global float *zImage,__global float *fImage,
                    __global float *Trajectories,__global int *IdLast,
                    uint ImpactParameter,
                    float Mass,float InternalRadius,
                    float ExternalRadius,float Angle,
                    int Line)
{
   uint xi=(uint)get_global_id(0);
   uint yi=(uint)get_global_id(1);
   uint sizex=(uint)get_global_size(0);
   uint sizey=(uint)get_global_size(1);

   // Perform trajectory for each pixel

  float m,ri,re,tho;
  int q,raie;

  m=Mass;
  ri=InternalRadius;
  re=ExternalRadius;
  tho=Angle;
  q=-2;
  raie=Line;

  float bmx,db,b,h;
  float phi,phd,php,nr,r;
  float zp=0.e0f,fp=0.e0f;

  // Autosize for image, 25% greater than external radius
  bmx=1.25e0f*re;

  // Angular step of integration
  h=4.e0f*PI/(float)TRACKPOINTS;

  // Step of Impact Parameter
  db=bmx/(2.e0f*(float)ImpactParameter);

  // set origin as center of image
  float x=(float)xi-(float)(sizex/2)+(float)5.e-1f;
  float y=(float)yi-(float)(sizey/2)+(float)5.e-1f;

  // angle extracted from cylindric symmetry
  phi=atanp(x,y);
  phd=atanp(cos(phi)*sin(tho),cos(tho));

  // Real Impact Parameter
  b=sqrt(x*x+y*y)*bmx/(float)ImpactParameter;

  // Integer Impact Parameter
  uint bi=(uint)sqrt(x*x+y*y);

  int HalfLap=0,ExitOnImpact=0,ni;

  if (bi<ImpactParameter)
  {
    do
    {
      php=phd+(float)HalfLap*PI;
      nr=php/h;
      ni=(int)nr;

      if (ni<IdLast[bi])
      {
        r=(Trajectories[bi*TRACKPOINTS+ni+1]-Trajectories[bi*TRACKPOINTS+ni])*(nr-ni*1.e0f)+Trajectories[bi*TRACKPOINTS+ni];
      }
      else
      {
        r=Trajectories[bi*TRACKPOINTS+ni];
      }

      if ((r<=re)&&(r>=ri))
      {
        ExitOnImpact=1;
        impact(phi,r,b,tho,m,&zp,&fp,q,db,h,raie);
      }

      HalfLap++;
    } while ((HalfLap<=2)&&(ExitOnImpact==0));

  }

  barrier(CLK_GLOBAL_MEM_FENCE);

  zImage[yi+sizex*xi]=zp;
  fImage[yi+sizex*xi]=fp;
}

__kernel void Circle(__global float *Trajectories,__global int *IdLast,
                     __global float *zImage,__global float *fImage,
                     float Mass,float InternalRadius,
                     float ExternalRadius,float Angle,
                     int Line)
{
   // Integer Impact Parameter ID
   int bi=get_global_id(0);
   // Integer points on circle
   int i=get_global_id(1);
   // Integer Impact Parameter Size (half of image)
   int bmaxi=get_global_size(0);
   // Integer Points on circle
   int imx=get_global_size(1);

   // Perform trajectory for each pixel

  float m,ri,re,tho;
  int q,raie;

  m=Mass;
  ri=InternalRadius;
  re=ExternalRadius;
  tho=Angle;
  raie=Line;

  float bmx,db,b,h;
  float phi,phd;
  float zp=0.e0f,fp=0.e0f;

  // Autosize for image
  bmx=1.25e0f*re;

  // Angular step of integration
  h=4.e0f*PI/(float)TRACKPOINTS;

  // impact parameter
  b=(float)bi/(float)bmaxi*bmx;
  db=bmx/(2.e0f*(float)bmaxi);

  phi=2.e0f*PI/(float)imx*(float)i;
  phd=atanp(cos(phi)*sin(tho),cos(tho));
  int yi=(int)((float)bi*sin(phi))+bmaxi;
  int xi=(int)((float)bi*cos(phi))+bmaxi;

  int HalfLap=0,ExitOnImpact=0,ni;
  float php,nr,r;

  do
  {
     php=phd+(float)HalfLap*PI;
     nr=php/h;
     ni=(int)nr;

     if (ni<IdLast[bi])
     {
        r=(Trajectories[bi*TRACKPOINTS+ni+1]-Trajectories[bi*TRACKPOINTS+ni])*(nr-ni*1.e0f)+Trajectories[bi*TRACKPOINTS+ni];
     }
     else
     {
        r=Trajectories[bi*TRACKPOINTS+ni];
     }

     if ((r<=re)&&(r>=ri))
     {
        ExitOnImpact=1;
        impact(phi,r,b,tho,m,&zp,&fp,q,db,h,raie);
     }

     HalfLap++;
  } while ((HalfLap<=2)&&(ExitOnImpact==0));

  zImage[yi+2*bmaxi*xi]=zp;
  fImage[yi+2*bmaxi*xi]=fp;

  barrier(CLK_GLOBAL_MEM_FENCE);

}

__kernel void Trajectory(__global float *Trajectories,__global int *IdLast,
                         float Mass,float InternalRadius,
                         float ExternalRadius,float Angle,
                         int Line)
{
  // Integer Impact Parameter ID
  int bi=get_global_id(0);
  // Integer Impact Parameter Size (half of image)
  int bmaxi=get_global_size(0);

  // Perform trajectory for each pixel

  float m,rs,re;

  m=Mass;
  rs=2.e0f*m;
  re=ExternalRadius;

  float bmx,b,h;
  int nh;

  // Autosize for image
  bmx=1.25e0f*re;

  // Angular step of integration
  h=4.e0f*PI/(float)TRACKPOINTS;

  // impact parameter
  b=(float)bi/(float)bmaxi*bmx;

  float up,vp,pp,us,vs,ps;

  up=0.e0f;
  vp=1.e0f;

  pp=0.e0f;
  nh=0;

  rungekutta(&ps,&us,&vs,pp,up,vp,h,m,b);

  // b versus us
  float bvus=fabs(b/us);
  float bvus0=bvus;
  Trajectories[bi*TRACKPOINTS+nh]=bvus;

  do
  {
     nh++;
     pp=ps;
     up=us;
     vp=vs;
     rungekutta(&ps,&us,&vs,pp,up,vp,h,m,b);
     bvus=fabs(b/us);
     Trajectories[bi*TRACKPOINTS+nh]=bvus;

  } while ((bvus>=rs)&&(bvus<=bvus0));

  IdLast[bi]=nh;

  barrier(CLK_GLOBAL_MEM_FENCE);

}

__kernel void EachCircle(__global float *zImage,__global float *fImage,
                         float Mass,float InternalRadius,
                         float ExternalRadius,float Angle,
                         int Line)
{
   // Integer Impact Parameter ID
   uint bi=(uint)get_global_id(0);
   // Integer Impact Parameter Size (half of image)
   uint bmaxi=(uint)get_global_size(0);

  private float Trajectory[TRACKPOINTS];

  float m,rs,ri,re,tho;
  int raie,q;

  m=Mass;
  rs=2.e0f*m;
  ri=InternalRadius;
  re=ExternalRadius;
  tho=Angle;
  q=-2;
  raie=Line;

  float bmx,db,b,h;
  uint nh;


  // Autosize for image
  bmx=1.25e0f*re;

  // Angular step of integration
  h=4.e0f*PI/(float)TRACKPOINTS;

  // impact parameter
  b=(float)bi/(float)bmaxi*bmx;
  db=bmx/(2.e0f*(float)bmaxi);

  float up,vp,pp,us,vs,ps;

  up=0.e0f;
  vp=1.e0f;

  pp=0.e0f;
  nh=0;

  rungekutta(&ps,&us,&vs,pp,up,vp,h,m,b);

  // b versus us
  float bvus=fabs(b/us);
  float bvus0=bvus;
  Trajectory[nh]=bvus;

  do
  {
     nh++;
     pp=ps;
     up=us;
     vp=vs;
     rungekutta(&ps,&us,&vs,pp,up,vp,h,m,b);
     bvus=(float)fabs(b/us);
     Trajectory[nh]=bvus;

  } while ((bvus>=rs)&&(bvus<=bvus0));


  for (uint i=(uint)nh+1;i<TRACKPOINTS;i++) {
     Trajectory[i]=0.e0f;
  }


  uint imx=(uint)(16*bi);

  for (uint i=0;i<imx;i++)
  {
     float zp=0.e0f,fp=0.e0f;
     float phi=2.e0f*PI/(float)imx*(float)i;
     float phd=atanp(cos(phi)*sin(tho),cos(tho));
     uint yi=(uint)((float)bi*sin(phi)+bmaxi);
     uint xi=(uint)((float)bi*cos(phi)+bmaxi);

     uint HalfLap=0,ExitOnImpact=0,ni;
     float php,nr,r;

     do
     {
        php=phd+(float)HalfLap*PI;
        nr=php/h;
        ni=(int)nr;

        if (ni<nh)
        {
           r=(Trajectory[ni+1]-Trajectory[ni])*(nr-ni*1.e0f)+Trajectory[ni];
        }
        else
        {
           r=Trajectory[ni];
        }

        if ((r<=re)&&(r>=ri))
        {
           ExitOnImpact=1;
           impact(phi,r,b,tho,m,&zp,&fp,q,db,h,raie);
        }

        HalfLap++;

     } while ((HalfLap<=2)&&(ExitOnImpact==0));

     zImage[yi+2*bmaxi*xi]=zp;
     fImage[yi+2*bmaxi*xi]=fp;

  }

  barrier(CLK_GLOBAL_MEM_FENCE);

}

__kernel void Original(__global float *zImage,__global float *fImage,
                       uint Size,float Mass,float InternalRadius,
                       float ExternalRadius,float Angle,
                       int Line)
{
   // Integer Impact Parameter Size (half of image)
   uint bmaxi=(uint)Size;

   float Trajectory[TRACKPOINTS];

   // Perform trajectory for each pixel

   float m,rs,ri,re,tho;
   int raie,q;

   m=Mass;
   rs=2.e0f*m;
   ri=InternalRadius;
   re=ExternalRadius;
   tho=Angle;
   q=-2;
   raie=Line;

   float bmx,db,b,h;
   uint nh;

   // Autosize for image
   bmx=1.25e0f*re;

   // Angular step of integration
   h=4.e0f*PI/(float)TRACKPOINTS;

   // Integer Impact Parameter ID
   for (int bi=0;bi<bmaxi;bi++)
   {
      // impact parameter
      b=(float)bi/(float)bmaxi*bmx;
      db=bmx/(2.e0f*(float)bmaxi);

      float up,vp,pp,us,vs,ps;

      up=0.e0f;
      vp=1.e0f;

      pp=0.e0f;
      nh=0;

      rungekutta(&ps,&us,&vs,pp,up,vp,h,m,b);

      // b versus us
      float bvus=fabs(b/us);
      float bvus0=bvus;
      Trajectory[nh]=bvus;

      do
      {
         nh++;
         pp=ps;
         up=us;
         vp=vs;
         rungekutta(&ps,&us,&vs,pp,up,vp,h,m,b);
         bvus=fabs(b/us);
         Trajectory[nh]=bvus;

      } while ((bvus>=rs)&&(bvus<=bvus0));

      for (uint i=(uint)nh+1;i<TRACKPOINTS;i++) {
         Trajectory[i]=0.e0f;
      }

      int imx=(int)(16*bi);

      for (int i=0;i<imx;i++)
      {
         float zp=0.e0f,fp=0.e0f;
         float phi=2.e0f*PI/(float)imx*(float)i;
         float phd=atanp(cos(phi)*sin(tho),cos(tho));
         uint yi=(uint)((float)bi*sin(phi)+bmaxi);
         uint xi=(uint)((float)bi*cos(phi)+bmaxi);

         uint HalfLap=0,ExitOnImpact=0,ni;
         float php,nr,r;

         do
         {
            php=phd+(float)HalfLap*PI;
            nr=php/h;
            ni=(int)nr;

            if (ni<nh)
            {
               r=(Trajectory[ni+1]-Trajectory[ni])*(nr-ni*1.e0f)+Trajectory[ni];
            }
            else
            {
               r=Trajectory[ni];
            }

            if ((r<=re)&&(r>=ri))
            {
               ExitOnImpact=1;
               impact(phi,r,b,tho,m,&zp,&fp,q,db,h,raie);
            }

            HalfLap++;

         } while ((HalfLap<=2)&&(ExitOnImpact==0));

         zImage[yi+2*bmaxi*xi]=zp;
         fImage[yi+2*bmaxi*xi]=fp;

      }

   }

   barrier(CLK_GLOBAL_MEM_FENCE);

}
"""


def KernelCodeCuda():
    BlobCUDA = """

#define PI (float)3.14159265359
#define nbr 256

#define EINSTEIN 0
#define NEWTON 1

#ifdef SETTRACKPOINTS
#define TRACKPOINTS SETTRACKPOINTS
#else
#define TRACKPOINTS
#endif
__device__ float nothing(float x)
{
  return(x);
}

__device__ float atanp(float x,float y)
{
  float angle;

  angle=atan2(y,x);

  if (angle<0.e0f)
    {
      angle+=(float)2.e0f*PI;
    }

  return(angle);
}

__device__ float f(float v)
{
  return(v);
}

#if PHYSICS == NEWTON
__device__ float g(float u,float m,float b)
{
  return (-u);
}
#else
__device__ float g(float u,float m,float b)
{
  return (3.e0f*m/b*pow(u,2)-u);
}
#endif

__device__ void calcul(float *us,float *vs,float up,float vp,
                       float h,float m,float b)
{
  float c0,c1,c2,c3,d0,d1,d2,d3;

  c0=h*f(vp);
  c1=h*f(vp+c0/2.);
  c2=h*f(vp+c1/2.);
  c3=h*f(vp+c2);
  d0=h*g(up,m,b);
  d1=h*g(up+d0/2.,m,b);
  d2=h*g(up+d1/2.,m,b);
  d3=h*g(up+d2,m,b);

  *us=up+(c0+2.*c1+2.*c2+c3)/6.;
  *vs=vp+(d0+2.*d1+2.*d2+d3)/6.;
}

__device__ void rungekutta(float *ps,float *us,float *vs,
                           float pp,float up,float vp,
                           float h,float m,float b)
{
  calcul(us,vs,up,vp,h,m,b);
  *ps=pp+h;
}

__device__ float decalage_spectral(float r,float b,float phi,
                                   float tho,float m)
{
  return (sqrt(1-3*m/r)/(1+sqrt(m/pow(r,3))*b*sin(tho)*sin(phi)));
}

__device__ float spectre(float rf,int q,float b,float db,
                         float h,float r,float m,float bss)
{
  float flx;

//  flx=exp(q*log(r/m))*pow(rf,4)*b*db*h;
  flx=exp(q*log(r/m)+4.*log(rf))*b*db*h;
  return(flx);
}

__device__ float spectre_cn(float rf32,float b32,float db32,
                            float h32,float r32,float m32,float bss32)
{

#define MYFLOAT float

  MYFLOAT rf=(MYFLOAT)(rf32);
  MYFLOAT b=(MYFLOAT)(b32);
  MYFLOAT db=(MYFLOAT)(db32);
  MYFLOAT h=(MYFLOAT)(h32);
  MYFLOAT r=(MYFLOAT)(r32);
  MYFLOAT m=(MYFLOAT)(m32);
  MYFLOAT bss=(MYFLOAT)(bss32);

  MYFLOAT flx;
  MYFLOAT nu_rec,nu_em,qu,temp_em,flux_int;
  int fi,posfreq;

#define planck 6.62e-34
#define k 1.38e-23
#define c2 9.e16
#define temp 3.e7
#define m_point 1.

#define lplanck (log(6.62)-34.*log(10.))
#define lk (log(1.38)-23.*log(10.))
#define lc2 (log(9.)+16.*log(10.))

  MYFLOAT v=1.-3./r;

  qu=1./sqrt((1.-3./r)*r)*(sqrt(r)-sqrt(6.)+sqrt(3.)/2.*log((sqrt(r)+sqrt(3.))/(sqrt(r)-sqrt(3.))* 0.17157287525380988 ));  // # noqa: #051

  temp_em=temp*sqrt(m)*exp(0.25*log(m_point)-0.75*log(r)-0.125*log(v)+0.25*log(fabs(qu)));

  flux_int=0.;
  flx=0.;

  for (fi=0;fi<nbr;fi++)
    {
      nu_em=bss*(MYFLOAT)fi/(MYFLOAT)nbr;
      nu_rec=nu_em*rf;
      posfreq=(int)(nu_rec*(MYFLOAT)nbr/bss);
      if ((posfreq>0)&&(posfreq<nbr))
        {
          // Initial version
          // flux_int=2.*planck/c2*pow(nu_em,3)/(exp(planck*nu_em/(k*temp_em))-1.);
          // Version with log used
          //flux_int=2.*exp(lplanck-lc2+3.*log(nu_em))/(exp(exp(lplanck-lk+log(nu_em/temp_em)))-1.);
          // flux_int*=pow(rf,3)*b*db*h;
          //flux_int*=exp(3.*log(rf))*b*db*h;
          flux_int=2.*exp(lplanck-lc2+3.*log(nu_em))/(exp(exp(lplanck-lk+log(nu_em/temp_em)))-1.)*exp(3.*log(rf))*b*db*h;

          flx+=flux_int;
        }
    }

  return((float)(flx));
}

__device__ void impact(float phi,float r,float b,float tho,float m,
                       float *zp,float *fp,
                       int q,float db,
                       float h,int raie)
{
  float flx,rf,bss;

  rf=decalage_spectral(r,b,phi,tho,m);

  if (raie==0)
    {
      bss=1.e19;
      flx=spectre_cn(rf,b,db,h,r,m,bss);
    }
  else
    {
      bss=2.;
      flx=spectre(rf,q,b,db,h,r,m,bss);
    }

  *zp=1./rf;
  *fp=flx;

}

__global__ void EachPixel(float *zImage,float *fImage,
                          float Mass,float InternalRadius,
                          float ExternalRadius,float Angle,
                          int Line)
{
   uint xi=(uint)(blockIdx.x*blockDim.x+threadIdx.x);
   uint yi=(uint)(blockIdx.y*blockDim.y+threadIdx.y);
   uint sizex=(uint)gridDim.x*blockDim.x;
   uint sizey=(uint)gridDim.y*blockDim.y;


   // Perform trajectory for each pixel, exit on hit

  float m,rs,ri,re,tho;
  int q,raie;

  m=Mass;
  rs=2.*m;
  ri=InternalRadius;
  re=ExternalRadius;
  tho=Angle;
  q=-2;
  raie=Line;

  float bmx,db,b,h;
  float rp0,rpp,rps;
  float phi,phd;
  int nh;
  float zp,fp;

  // Autosize for image
  bmx=1.25*re;
  b=0.;

  h=4.e0f*PI/(float)TRACKPOINTS;

  // set origin as center of image
  float x=(float)xi-(float)(sizex/2)+(float)5e-1f;
  float y=(float)yi-(float)(sizey/2)+(float)5e-1f;
  // angle extracted from cylindric symmetry
  phi=atanp(x,y);
  phd=atanp(cos(phi)*sin(tho),cos(tho));

  float up,vp,pp,us,vs,ps;

  // impact parameter
  b=sqrt(x*x+y*y)*(float)2.e0f/(float)sizex*bmx;
  // step of impact parameter;
//  db=bmx/(float)(sizex/2);
  db=bmx/(float)(sizex);

  up=0.;
  vp=1.;
  pp=0.;
  nh=0;

  rungekutta(&ps,&us,&vs,pp,up,vp,h,m,b);

  rps=fabs(b/us);
  rp0=rps;

  int ExitOnImpact=0;

  do
  {
     nh++;
     pp=ps;
     up=us;
     vp=vs;
     rungekutta(&ps,&us,&vs,pp,up,vp,h,m,b);
     rpp=rps;
     rps=fabs(b/us);
     ExitOnImpact = ((fmod(pp,PI)<fmod(phd,PI))&&(fmod(ps,PI)>fmod(phd,PI)))&&(rps>ri)&&(rps<re)?1:0;

  } while ((rps>=rs)&&(rps<=rp0)&&(ExitOnImpact==0));

  if (ExitOnImpact==1) {
     impact(phi,rpp,b,tho,m,&zp,&fp,q,db,h,raie);
  }
  else
  {
     zp=0.e0f;
     fp=0.e0f;
  }

  __syncthreads();

  zImage[yi+sizex*xi]=(float)zp;
  fImage[yi+sizex*xi]=(float)fp;
}

__global__ void Pixel(float *zImage,float *fImage,
                      float *Trajectories,int *IdLast,
                      uint ImpactParameter,
                      float Mass,float InternalRadius,
                      float ExternalRadius,float Angle,
                      int Line)
{
   uint xi=(uint)(blockIdx.x*blockDim.x+threadIdx.x);
   uint yi=(uint)(blockIdx.y*blockDim.y+threadIdx.y);
   uint sizex=(uint)gridDim.x*blockDim.x;
   uint sizey=(uint)gridDim.y*blockDim.y;

  // Perform trajectory for each pixel

  float m,ri,re,tho;
  int q,raie;

  m=Mass;
  ri=InternalRadius;
  re=ExternalRadius;
  tho=Angle;
  q=-2;
  raie=Line;

  float bmx,db,b,h;
  float phi,phd,php,nr,r;
  float zp=0,fp=0;
  // Autosize for image, 25% greater than external radius
  bmx=1.25e0f*re;

  // Angular step of integration
  h=4.e0f*PI/(float)TRACKPOINTS;

  // Step of Impact Parameter
  db=bmx/(2.e0f*(float)ImpactParameter);

  // set origin as center of image
  float x=(float)xi-(float)(sizex/2)+(float)5e-1f;
  float y=(float)yi-(float)(sizey/2)+(float)5e-1f;
  // angle extracted from cylindric symmetry
  phi=atanp(x,y);
  phd=atanp(cos(phi)*sin(tho),cos(tho));

  // Real Impact Parameter
  b=sqrt(x*x+y*y)*bmx/(float)ImpactParameter;

  // Integer Impact Parameter
  uint bi=(uint)sqrt(x*x+y*y);

  int HalfLap=0,ExitOnImpact=0,ni;

  if (bi<ImpactParameter)
  {
    do
    {
      php=phd+(float)HalfLap*PI;
      nr=php/h;
      ni=(int)nr;

      if (ni<IdLast[bi])
      {
        r=(Trajectories[bi*TRACKPOINTS+ni+1]-Trajectories[bi*TRACKPOINTS+ni])*(nr-ni*1.e0f)+Trajectories[bi*TRACKPOINTS+ni];
      }
      else
      {
        r=Trajectories[bi*TRACKPOINTS+ni];
      }

      if ((r<=re)&&(r>=ri))
      {
        ExitOnImpact=1;
        impact(phi,r,b,tho,m,&zp,&fp,q,db,h,raie);
      }

      HalfLap++;
    } while ((HalfLap<=2)&&(ExitOnImpact==0));

  }

  zImage[yi+sizex*xi]=zp;
  fImage[yi+sizex*xi]=fp;
}

__global__ void Circle(float *Trajectories,int *IdLast,
                       float *zImage,float *fImage,
                       float Mass,float InternalRadius,
                       float ExternalRadius,float Angle,
                       int Line)
{
   // Integer Impact Parameter ID
   int bi=blockIdx.x*blockDim.x+threadIdx.x;
   // Integer points on circle
   int i=blockIdx.y*blockDim.y+threadIdx.y;
   // Integer Impact Parameter Size (half of image)
   int bmaxi=gridDim.x*blockDim.x;
   // Integer Points on circle
   int imx=gridDim.y*blockDim.y;

   // Perform trajectory for each pixel

  float m,ri,re,tho;
  int q,raie;

  m=Mass;
  ri=InternalRadius;
  re=ExternalRadius;
  tho=Angle;
  raie=Line;

  float bmx,db,b,h;
  float phi,phd;
  float zp=0,fp=0;

  // Autosize for image
  bmx=1.25e0f*re;

  // Angular step of integration
  h=4.e0f*PI/(float)TRACKPOINTS;

  // impact parameter
  b=(float)bi/(float)bmaxi*bmx;
  db=bmx/(2.e0f*(float)bmaxi);

  phi=2.e0f*PI/(float)imx*(float)i;
  phd=atanp(cos(phi)*sin(tho),cos(tho));
  int yi=(int)((float)bi*sin(phi))+bmaxi;
  int xi=(int)((float)bi*cos(phi))+bmaxi;

  int HalfLap=0,ExitOnImpact=0,ni;
  float php,nr,r;

  do
  {
     php=phd+(float)HalfLap*PI;
     nr=php/h;
     ni=(int)nr;

     if (ni<IdLast[bi])
     {
        r=(Trajectories[bi*TRACKPOINTS+ni+1]-Trajectories[bi*TRACKPOINTS+ni])*(nr-ni*1.e0f)+Trajectories[bi*TRACKPOINTS+ni];
     }
     else
     {
        r=Trajectories[bi*TRACKPOINTS+ni];
     }

     if ((r<=re)&&(r>=ri))
     {
        ExitOnImpact=1;
        impact(phi,r,b,tho,m,&zp,&fp,q,db,h,raie);
     }

     HalfLap++;
  } while ((HalfLap<=2)&&(ExitOnImpact==0));

  zImage[yi+2*bmaxi*xi]=zp;
  fImage[yi+2*bmaxi*xi]=fp;

}

__global__ void Trajectory(float *Trajectories,int *IdLast,
                           float Mass,float InternalRadius,
                           float ExternalRadius,float Angle,
                           int Line)
{
  // Integer Impact Parameter ID
  int bi=blockIdx.x*blockDim.x+threadIdx.x;
  // Integer Impact Parameter Size (half of image)
  int bmaxi=gridDim.x*blockDim.x;

  // Perform trajectory for each pixel

  float m,rs,re;

  m=Mass;
  rs=2.e0f*m;
  re=ExternalRadius;

  float bmx,b,h;
  int nh;

  // Autosize for image
  bmx=1.25e0f*re;

  // Angular step of integration
  h=4.e0f*PI/(float)TRACKPOINTS;

  // impact parameter
  b=(float)bi/(float)bmaxi*bmx;

  float up,vp,pp,us,vs,ps;

  up=0.e0f;
  vp=1.e0f;
  pp=0.e0f;
  nh=0;

  rungekutta(&ps,&us,&vs,pp,up,vp,h,m,b);

  // b versus us
  float bvus=fabs(b/us);
  float bvus0=bvus;
  Trajectories[bi*TRACKPOINTS+nh]=bvus;

  do
  {
     nh++;
     pp=ps;
     up=us;
     vp=vs;
     rungekutta(&ps,&us,&vs,pp,up,vp,h,m,b);
     bvus=fabs(b/us);
     Trajectories[bi*TRACKPOINTS+nh]=bvus;

  } while ((bvus>=rs)&&(bvus<=bvus0));

  IdLast[bi]=nh;

}

__global__ void EachCircle(float *zImage,float *fImage,
                           float Mass,float InternalRadius,
                           float ExternalRadius,float Angle,
                           int Line)
{
  // Integer Impact Parameter ID
  int bi=blockIdx.x*blockDim.x+threadIdx.x;

  // Integer Impact Parameter Size (half of image)
  int bmaxi=gridDim.x*blockDim.x;

  float Trajectory[2048];

  // Perform trajectory for each pixel

  float m,rs,ri,re,tho;
  int raie,q;

  m=Mass;
  rs=2.*m;
  ri=InternalRadius;
  re=ExternalRadius;
  tho=Angle;
  q=-2;
  raie=Line;

  float bmx,db,b,h;
  int nh;

  // Autosize for image
  bmx=1.25e0f*re;

  // Angular step of integration
  h=4.e0f*PI/(float)TRACKPOINTS;

  // impact parameter
  b=(float)bi/(float)bmaxi*bmx;
  db=bmx/(2.e0f*(float)bmaxi);

  float up,vp,pp,us,vs,ps;

  up=0.;
  vp=1.;
  pp=0.;
  nh=0;

  rungekutta(&ps,&us,&vs,pp,up,vp,h,m,b);

  // b versus us
  float bvus=fabs(b/us);
  float bvus0=bvus;
  Trajectory[nh]=bvus;

  do
  {
     nh++;
     pp=ps;
     up=us;
     vp=vs;
     rungekutta(&ps,&us,&vs,pp,up,vp,h,m,b);
     bvus=fabs(b/us);
     Trajectory[nh]=bvus;

  } while ((bvus>=rs)&&(bvus<=bvus0));

  int imx=(int)(16*bi);

  for (int i=0;i<imx;i++)
  {
     float zp=0,fp=0;
     float phi=2.*PI/(float)imx*(float)i;
     float phd=atanp(cos(phi)*sin(tho),cos(tho));
     uint yi=(uint)((float)bi*sin(phi)+bmaxi);
     uint xi=(uint)((float)bi*cos(phi)+bmaxi);

     int HalfLap=0,ExitOnImpact=0,ni;
     float php,nr,r;

     do
     {
        php=phd+(float)HalfLap*PI;
        nr=php/h;
        ni=(int)nr;

        if (ni<nh)
        {
           r=(Trajectory[ni+1]-Trajectory[ni])*(nr-ni*1.)+Trajectory[ni];
        }
        else
        {
           r=Trajectory[ni];
        }

        if ((r<=re)&&(r>=ri))
        {
           ExitOnImpact=1;
           impact(phi,r,b,tho,m,&zp,&fp,q,db,h,raie);
        }

        HalfLap++;

     } while ((HalfLap<=2)&&(ExitOnImpact==0));

   __syncthreads();

   zImage[yi+2*bmaxi*xi]=zp;
   fImage[yi+2*bmaxi*xi]=fp;

  }

}

__global__ void Original(float *zImage,float *fImage,
                         uint Size,float Mass,float InternalRadius,
                         float ExternalRadius,float Angle,
                         int Line)
{
   // Integer Impact Parameter Size (half of image)
   uint bmaxi=(uint)Size;

   float Trajectory[TRACKPOINTS];

   // Perform trajectory for each pixel

   float m,rs,ri,re,tho;
   int raie,q;

   m=Mass;
   rs=2.e0f*m;
   ri=InternalRadius;
   re=ExternalRadius;
   tho=Angle;
   q=-2;
   raie=Line;

   float bmx,db,b,h;
   int nh;

   // Autosize for image
   bmx=1.25e0f*re;

   // Angular step of integration
   h=4.e0f*PI/(float)TRACKPOINTS;

   // Integer Impact Parameter ID
   for (int bi=0;bi<bmaxi;bi++)
   {
      // impact parameter
      b=(float)bi/(float)bmaxi*bmx;
      db=bmx/(2.e0f*(float)bmaxi);

      float up,vp,pp,us,vs,ps;

      up=0.;
      vp=1.;
      pp=0.;
      nh=0;

      rungekutta(&ps,&us,&vs,pp,up,vp,h,m,b);

      // b versus us
      float bvus=fabs(b/us);
      float bvus0=bvus;
      Trajectory[nh]=bvus;

      do
      {
         nh++;
         pp=ps;
         up=us;
         vp=vs;
         rungekutta(&ps,&us,&vs,pp,up,vp,h,m,b);
         bvus=fabs(b/us);
         Trajectory[nh]=bvus;

      } while ((bvus>=rs)&&(bvus<=bvus0));

      for (uint i=(uint)nh+1;i<TRACKPOINTS;i++) {
         Trajectory[i]=0.e0f;
      }

      int imx=(int)(16*bi);

      for (int i=0;i<imx;i++)
      {
         float zp=0,fp=0;
         float phi=2.e0f*PI/(float)imx*(float)i;
         float phd=atanp(cos(phi)*sin(tho),cos(tho));
         uint yi=(uint)((float)bi*sin(phi)+bmaxi);
         uint xi=(uint)((float)bi*cos(phi)+bmaxi);

         int HalfLap=0,ExitOnImpact=0,ni;
         float php,nr,r;

         do
         {
            php=phd+(float)HalfLap*PI;
            nr=php/h;
            ni=(int)nr;

            if (ni<nh)
            {
               r=(Trajectory[ni+1]-Trajectory[ni])*(nr-ni*1.)+Trajectory[ni];
            }
            else
            {
               r=Trajectory[ni];
            }

            if ((r<=re)&&(r>=ri))
            {
               ExitOnImpact=1;
               impact(phi,r,b,tho,m,&zp,&fp,q,db,h,raie);
            }

            HalfLap++;

         } while ((HalfLap<=2)&&(ExitOnImpact==0));

         zImage[yi+2*bmaxi*xi]=zp;
         fImage[yi+2*bmaxi*xi]=fp;

      }

   }

}
"""
    return BlobCUDA


# def ImageOutput(sigma,prefix,Colors):
#     import matplotlib.pyplot as plt
#     start_time=time.time()
#     if Colors == 'Red2Yellow':
#         plt.imsave("%s.png" % prefix, sigma, cmap='afmhot')
#     else:
#         plt.imsave("%s.png" % prefix, sigma, cmap='Greys_r')
#     save_time = time.time()-start_time
#     print("Save image as %s.png file" % prefix)
#     print("Save Time : %f" % save_time)


def ImageOutput(sigma, prefix, Colors):
    from PIL import Image

    Max = sigma.max()
    Min = sigma.min()
    # Normalize value as 8bits Integer
    SigmaInt = (255 * (sigma - Min) / (Max - Min)).astype("uint8")
    image = Image.fromarray(SigmaInt)
    image.save("%s.jpg" % prefix)


def BlackHoleCL(zImage, fImage, InputCL):
    Device = InputCL["Device"]
    Mass = InputCL["Mass"]
    InternalRadius = InputCL["InternalRadius"]
    ExternalRadius = InputCL["ExternalRadius"]
    Angle = InputCL["Angle"]
    Method = InputCL["Method"]
    TrackPoints = InputCL["TrackPoints"]
    Physics = InputCL["Physics"]
    NoImage = InputCL["NoImage"]
    TrackSave = InputCL["TrackSave"]

    PhysicsList = DictionariesAPI()

    if InputCL["BlackBody"]:
        # Spectrum is Black Body one
        Line = 0
    else:
        # Spectrum is Monochromatic Line one
        Line = 1

    Trajectories = numpy.zeros(
        (int(InputCL["Size"] / 2), InputCL["TrackPoints"]), dtype=numpy.float32
    )
    IdLast = numpy.zeros(int(InputCL["Size"] / 2), dtype=numpy.int32)

    # Je detecte un peripherique GPU dans la liste des peripheriques
    Id = 0
    HasXPU = False
    for platform in cl.get_platforms():
        for device in platform.get_devices():
            if Id == Device:
                PF4XPU = platform.name
                XPU = device
                print("CPU/GPU selected: ", device.name.lstrip())
                HasXPU = True
            Id += 1

    if not HasXPU:
        print("No XPU #%i found in all of %i devices, sorry..." % (Device, Id - 1))
        sys.exit()

    ctx = cl.Context([XPU])
    queue = cl.CommandQueue(
        ctx, properties=cl.command_queue_properties.PROFILING_ENABLE
    )

    BuildOptions = "-DPHYSICS=%i -DSETTRACKPOINTS=%i " % (
        PhysicsList[Physics],
        InputCL["TrackPoints"],
    )

    print("My Platform is ", PF4XPU)

    if (
        "Intel" in PF4XPU
        or "Experimental" in PF4XPU
        or "Clover" in PF4XPU
        or "Portable" in PF4XPU
    ):
        print("No extra options for Intel and Clover!")
    else:
        BuildOptions = BuildOptions + " -cl-mad-enable"

    BlackHoleCL = cl.Program(ctx, BlobOpenCL).build(options=BuildOptions)

    # Je recupere les flag possibles pour les buffers
    mf = cl.mem_flags

    if Method == "TrajectoPixel" or Method == "TrajectoCircle":
        TrajectoriesCL = cl.Buffer(
            ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=Trajectories
        )
        IdLastCL = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=IdLast)

    zImageCL = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=zImage)
    fImageCL = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=fImage)

    start_time = time.time()

    if Method == "EachPixel":
        CLLaunch = BlackHoleCL.EachPixel(
            queue,
            (zImage.shape[0], zImage.shape[1]),
            None,
            zImageCL,
            fImageCL,
            numpy.float32(Mass),
            numpy.float32(InternalRadius),
            numpy.float32(ExternalRadius),
            numpy.float32(Angle),
            numpy.int32(Line),
        )
        CLLaunch.wait()
    elif Method == "Original":
        CLLaunch = BlackHoleCL.Original(
            queue,
            (1,),
            None,
            zImageCL,
            fImageCL,
            numpy.uint32(zImage.shape[0] / 2),
            numpy.float32(Mass),
            numpy.float32(InternalRadius),
            numpy.float32(ExternalRadius),
            numpy.float32(Angle),
            numpy.int32(Line),
        )
        CLLaunch.wait()
    elif Method == "EachCircle":
        CLLaunch = BlackHoleCL.EachCircle(
            queue,
            (int(zImage.shape[0] / 2),),
            None,
            zImageCL,
            fImageCL,
            numpy.float32(Mass),
            numpy.float32(InternalRadius),
            numpy.float32(ExternalRadius),
            numpy.float32(Angle),
            numpy.int32(Line),
        )
        CLLaunch.wait()
    elif Method == "TrajectoCircle":
        CLLaunch = BlackHoleCL.Trajectory(
            queue,
            (Trajectories.shape[0],),
            None,
            TrajectoriesCL,
            IdLastCL,
            numpy.float32(Mass),
            numpy.float32(InternalRadius),
            numpy.float32(ExternalRadius),
            numpy.float32(Angle),
            numpy.int32(Line),
        )

        CLLaunch = BlackHoleCL.Circle(
            queue,
            (Trajectories.shape[0], int(zImage.shape[0] * 4)),
            None,
            TrajectoriesCL,
            IdLastCL,
            zImageCL,
            fImageCL,
            numpy.float32(Mass),
            numpy.float32(InternalRadius),
            numpy.float32(ExternalRadius),
            numpy.float32(Angle),
            numpy.int32(Line),
        )
        CLLaunch.wait()
    else:
        CLLaunch = BlackHoleCL.Trajectory(
            queue,
            (Trajectories.shape[0],),
            None,
            TrajectoriesCL,
            IdLastCL,
            numpy.float32(Mass),
            numpy.float32(InternalRadius),
            numpy.float32(ExternalRadius),
            numpy.float32(Angle),
            numpy.int32(Line),
        )

        CLLaunch = BlackHoleCL.Pixel(
            queue,
            (zImage.shape[0], zImage.shape[1]),
            None,
            zImageCL,
            fImageCL,
            TrajectoriesCL,
            IdLastCL,
            numpy.uint32(Trajectories.shape[0]),
            numpy.float32(Mass),
            numpy.float32(InternalRadius),
            numpy.float32(ExternalRadius),
            numpy.float32(Angle),
            numpy.int32(Line),
        )
        CLLaunch.wait()

    compute = time.time() - start_time

    cl.enqueue_copy(queue, zImage, zImageCL).wait()
    cl.enqueue_copy(queue, fImage, fImageCL).wait()
    if Method == "TrajectoPixel" or Method == "TrajectoCircle":
        cl.enqueue_copy(queue, Trajectories, TrajectoriesCL).wait()
        cl.enqueue_copy(queue, IdLast, IdLastCL).wait()
    elapsed = time.time() - start_time
    print("\nCompute Time : %f" % compute)
    print("Elapsed Time : %f\n" % elapsed)

    zMaxPosition = numpy.where(zImage[:, :] == zImage.max())
    fMaxPosition = numpy.where(fImage[:, :] == fImage.max())
    print(
        "Z max @(%f,%f) : %f"
        % (
            (
                1.0 * zMaxPosition[1][0] / zImage.shape[1] - 0.5,
                1.0 * zMaxPosition[0][0] / zImage.shape[0] - 0.5,
                zImage.max(),
            )
        )
    )
    print(
        "Flux max @(%f,%f) : %f"
        % (
            (
                1.0 * fMaxPosition[1][0] / fImage.shape[1] - 0.5,
                1.0 * fMaxPosition[0][0] / fImage.shape[0] - 0.5,
                fImage.max(),
            )
        )
    )
    zImageCL.release()
    fImageCL.release()

    if Method == "TrajectoPixel" or Method == "TrajectoCircle":
        if not NoImage:
            AngleStep = 4 * numpy.pi / TrackPoints
            Angles = numpy.arange(0.0, 4 * numpy.pi, AngleStep)
            Angles.shape = (1, TrackPoints)

            if TrackSave:
                # numpy.savetxt("TrouNoirTrajectories_%s.csv" % ImageInfo,
                #               numpy.transpose(numpy.concatenate((Angles,Trajectories),axis=0)),
                #               delimiter=' ', fmt='%.2e')
                numpy.savetxt(
                    "TrouNoirTrajectories.csv",
                    numpy.transpose(numpy.concatenate((Angles, Trajectories),
                        axis=0)),
                    delimiter=" ",
                    fmt="%.2e",
                )

        TrajectoriesCL.release()
        IdLastCL.release()

    return elapsed


def BlackHoleCUDA(zImage, fImage, InputCL):
    Device = InputCL["Device"]
    Mass = InputCL["Mass"]
    InternalRadius = InputCL["InternalRadius"]
    ExternalRadius = InputCL["ExternalRadius"]
    Angle = InputCL["Angle"]
    Method = InputCL["Method"]
    TrackPoints = InputCL["TrackPoints"]
    Physics = InputCL["Physics"]
    Threads = InputCL["Threads"]

    PhysicsList = DictionariesAPI()

    if InputCL["BlackBody"]:
        # Spectrum is Black Body one
        Line = 0
    else:
        # Spectrum is Monochromatic Line one
        Line = 1

    Trajectories = numpy.zeros(
        (int(InputCL["Size"] / 2), InputCL["TrackPoints"]), dtype=numpy.float32
    )
    IdLast = numpy.zeros(int(InputCL["Size"] / 2), dtype=numpy.int32)

    try:
        # For PyCUDA import
        import pycuda.driver as cuda
        from pycuda.compiler import SourceModule

        cuda.init()
        for Id in range(cuda.Device.count()):
            if Id == Device:
                XPU = cuda.Device(Id)
                print("GPU selected %s" % XPU.name())
        print

    except ImportError:
        print("Platform does not seem to support CUDA")

    Context = XPU.make_context()

    try:
        mod = SourceModule(
            KernelCodeCuda(),
            options=[
                "--compiler-options",
                "-DPHYSICS=%i -DSETTRACKPOINTS=%i"
                % (PhysicsList[Physics], TrackPoints),
            ],
        )
        print("Compilation seems to be OK")
    except Exception:
        print("Compilation seems to break")

    EachPixelCU = mod.get_function("EachPixel")
    OriginalCU = mod.get_function("Original")
    EachCircleCU = mod.get_function("EachCircle")
    TrajectoryCU = mod.get_function("Trajectory")
    PixelCU = mod.get_function("Pixel")
    CircleCU = mod.get_function("Circle")

    TrajectoriesCU = cuda.mem_alloc(Trajectories.size * Trajectories.dtype.itemsize)
    cuda.memcpy_htod(TrajectoriesCU, Trajectories)
    zImageCU = cuda.mem_alloc(zImage.size * zImage.dtype.itemsize)
    cuda.memcpy_htod(zImageCU, zImage)
    fImageCU = cuda.mem_alloc(fImage.size * fImage.dtype.itemsize)
    cuda.memcpy_htod(zImageCU, fImage)
    IdLastCU = cuda.mem_alloc(IdLast.size * IdLast.dtype.itemsize)
    cuda.memcpy_htod(IdLastCU, IdLast)

    start_time = time.time()

    if Method == "EachPixel":
        EachPixelCU(
            zImageCU,
            fImageCU,
            numpy.float32(Mass),
            numpy.float32(InternalRadius),
            numpy.float32(ExternalRadius),
            numpy.float32(Angle),
            numpy.int32(Line),
            grid=(int(zImage.shape[0] / Threads), int(zImage.shape[1] / Threads)),
            block=(Threads, Threads, 1),
        )
    elif Method == "EachCircle":
        EachCircleCU(
            zImageCU,
            fImageCU,
            numpy.float32(Mass),
            numpy.float32(InternalRadius),
            numpy.float32(ExternalRadius),
            numpy.float32(Angle),
            numpy.int32(Line),
            grid=(int(zImage.shape[0] / Threads / 2), 1),
            block=(Threads, 1, 1),
        )
    elif Method == "Original":
        OriginalCU(
            zImageCU,
            fImageCU,
            numpy.uint32(zImage.shape[0] / 2),
            numpy.float32(Mass),
            numpy.float32(InternalRadius),
            numpy.float32(ExternalRadius),
            numpy.float32(Angle),
            numpy.int32(Line),
            grid=(1, 1),
            block=(1, 1, 1),
        )
    elif Method == "TrajectoCircle":
        TrajectoryCU(
            TrajectoriesCU,
            IdLastCU,
            numpy.float32(Mass),
            numpy.float32(InternalRadius),
            numpy.float32(ExternalRadius),
            numpy.float32(Angle),
            numpy.int32(Line),
            grid=(int(Trajectories.shape[0] / Threads), 1),
            block=(Threads, 1, 1),
        )

        CircleCU(
            TrajectoriesCU,
            IdLastCU,
            zImageCU,
            fImageCU,
            numpy.float32(Mass),
            numpy.float32(InternalRadius),
            numpy.float32(ExternalRadius),
            numpy.float32(Angle),
            numpy.int32(Line),
            grid=(
                int(Trajectories.shape[0] / Threads),
                int(zImage.shape[0] * 4 / Threads),
            ),
            block=(Threads, Threads, 1),
        )
    else:
        # Default method: TrajectoPixel
        TrajectoryCU(
            TrajectoriesCU,
            IdLastCU,
            numpy.float32(Mass),
            numpy.float32(InternalRadius),
            numpy.float32(ExternalRadius),
            numpy.float32(Angle),
            numpy.int32(Line),
            grid=(int(Trajectories.shape[0] / Threads), 1),
            block=(Threads, 1, 1),
        )

        PixelCU(
            zImageCU,
            fImageCU,
            TrajectoriesCU,
            IdLastCU,
            numpy.uint32(Trajectories.shape[0]),
            numpy.float32(Mass),
            numpy.float32(InternalRadius),
            numpy.float32(ExternalRadius),
            numpy.float32(Angle),
            numpy.int32(Line),
            grid=(int(zImage.shape[0] / Threads), int(zImage.shape[1] / Threads), 1),
            block=(Threads, Threads, 1),
        )

    Context.synchronize()

    compute = time.time() - start_time

    cuda.memcpy_dtoh(zImage, zImageCU)
    cuda.memcpy_dtoh(fImage, fImageCU)
    if Method == "TrajectoPixel" or Method == "TrajectoCircle":
        cuda.memcpy_dtoh(Trajectories, TrajectoriesCU)
    elapsed = time.time() - start_time
    print("\nCompute Time : %f" % compute)
    print("Elapsed Time : %f\n" % elapsed)

    zMaxPosition = numpy.where(zImage[:, :] == zImage.max())
    fMaxPosition = numpy.where(fImage[:, :] == fImage.max())
    print(
        "Z max @(%f,%f) : %f"
        % (
            (
                1.0 * zMaxPosition[1][0] / zImage.shape[1] - 0.5,
                1.0 * zMaxPosition[0][0] / zImage.shape[0] - 0.5,
                zImage.max(),
            )
        )
    )
    print(
        "Flux max @(%f,%f) : %f"
        % (
            (
                1.0 * fMaxPosition[1][0] / fImage.shape[1] - 0.5,
                1.0 * fMaxPosition[0][0] / fImage.shape[0] - 0.5,
                fImage.max(),
            )
        )
    )

    Context.pop()

    Context.detach()

    if Method == "TrajectoPixel" or Method == "TrajectoCircle":
        if not NoImage:
            AngleStep = 4 * numpy.pi / TrackPoints
            Angles = numpy.arange(0.0, 4 * numpy.pi, AngleStep)
            Angles.shape = (1, TrackPoints)

            # numpy.savetxt("TrouNoirTrajectories_%s.csv" % ImageInfo,
            #               numpy.transpose(numpy.concatenate((Angles,Trajectories),axis=0)),
            #               delimiter=' ', fmt='%.2e')
            numpy.savetxt(
                "TrouNoirTrajectories.csv",
                numpy.transpose(numpy.concatenate((Angles, Trajectories), axis=0)),
                delimiter=" ",
                fmt="%.2e",
            )

    return elapsed


if __name__ == "__main__":

    # Default device: first one!
    Device = 0
    # Default implementation: OpenCL, most versatile!
    GpuStyle = "OpenCL"
    Mass = 1.0
    # Internal Radius 3 times de Schwarzschild Radius
    InternalRadius = 6.0 * Mass
    #
    ExternalRadius = 12.0
    #
    # Angle with normal to disc 10 degrees
    Angle = numpy.pi / 180.0 * (90.0 - 10.0)
    # Radiation of disc : BlackBody or Monochromatic
    BlackBody = False
    # Size of image
    Size = 1024
    # Variable Type
    VariableType = "FP32"
    # ?
    q = -2
    # Method of resolution
    Method = "TrajectoPixel"
    # Colors for output image
    Colors = "Greyscale"
    # Physics
    Physics = "Einstein"
    # No output as image
    NoImage = False
    # Threads in CUDA
    Threads = 32
    # Trackpoints of trajectories
    TrackPoints = 2048
    # Tracksave of trajectories
    TrackSave = False

    HowToUse = "%s -h [Help] -b [BlackBodyEmission] -j [TrackSave] -n [NoImage] -p <Einstein/Newton> -s <SizeInPixels> -m <Mass> -i <DiscInternalRadius> -x <DiscExternalRadius> -a <AngleAboveDisc> -d <DeviceId> -c <Greyscale/Red2Yellow> -g <CUDA/OpenCL> -o <EachPixel/TrajectoCircle/TrajectoPixel/EachCircle/Original> -t <ThreadsInCuda> -v <FP32/FP64> -k <TrackPoints>"  # noqa: E501

    try:
        opts, args = getopt.getopt(
            sys.argv[1:],
            "hbnjs:m:i:x:a:d:g:v:o:t:c:p:k:",
            [
                "tracksave",
                "blackbody",
                "noimage",
                "camera",
                "size=",
                "mass=",
                "internal=",
                "external=",
                "angle=",
                "device=",
                "gpustyle=",
                "variabletype=",
                "method=",
                "threads=",
                "colors=",
                "physics=",
                "trackpoints=",
            ],
        )
    except getopt.GetoptError:
        print(HowToUse % sys.argv[0])
        sys.exit(2)

    # List of Devices
    Devices = []
    Alu = {}

    for opt, arg in opts:
        if opt == "-h":
            print(HowToUse % sys.argv[0])

            print("\nInformations about devices detected under OpenCL API:")
            # For PyOpenCL import
            try:
                Id = 0
                for platform in cl.get_platforms():
                    for device in platform.get_devices():
                        # deviceType=cl.device_type.to_string(device.type)
                        deviceType = "xPU"
                        print(
                            "Device #%i from %s of type %s : %s"
                            % (
                                Id,
                                platform.vendor.lstrip(),
                                deviceType,
                                device.name.lstrip(),
                            )
                        )
                        Id = Id + 1

            except Exception:
                print("Your platform does not seem to support OpenCL")

            print("\nInformations about devices detected under CUDA API:")
            # For PyCUDA import
            try:
                import pycuda.driver as cuda

                cuda.init()
                for Id in range(cuda.Device.count()):
                    device = cuda.Device(Id)
                    print("Device #%i of type GPU : %s" % (Id, device.name()))
                print
            except Exception:
                print("Your platform does not seem to support CUDA")

            sys.exit()

        elif opt in ("-d", "--device"):
            #            Devices.append(int(arg))
            Device = int(arg)
        elif opt in ("-g", "--gpustyle"):
            GpuStyle = arg
        elif opt in ("-v", "--variabletype"):
            VariableType = arg
        elif opt in ("-s", "--size"):
            Size = int(arg)
        elif opt in ("-k", "--trackpoints"):
            TrackPoints = int(arg)
        elif opt in ("-m", "--mass"):
            Mass = float(arg)
        elif opt in ("-i", "--internal"):
            InternalRadius = float(arg)
        elif opt in ("-e", "--external"):
            ExternalRadius = float(arg)
        elif opt in ("-a", "--angle"):
            Angle = numpy.pi / 180.0 * (90.0 - float(arg))
        elif opt in ("-b", "--blackbody"):
            BlackBody = True
        elif opt in ("-j", "--tracksave"):
            TrackSave = True
        elif opt in ("-n", "--noimage"):
            NoImage = True
        elif opt in ("-o", "--method"):
            Method = arg
        elif opt in ("-t", "--threads"):
            Threads = int(arg)
        elif opt in ("-c", "--colors"):
            Colors = arg
        elif opt in ("-p", "--physics"):
            Physics = arg

    print("Device Identification selected : %s" % Device)
    print("GpuStyle used : %s" % GpuStyle)
    print("VariableType : %s" % VariableType)
    print("Size : %i" % Size)
    print("Mass : %f" % Mass)
    print("Internal Radius : %f" % InternalRadius)
    print("External Radius : %f" % ExternalRadius)
    print("Angle with normal of (in radians) : %f" % Angle)
    print("Black Body Disc Emission (monochromatic instead) : %s" % BlackBody)
    print("Method of resolution : %s" % Method)
    print("Colors for output images : %s" % Colors)
    print("Physics used for Trajectories : %s" % Physics)
    print("Trackpoints of Trajectories : %i" % TrackPoints)
    print("Tracksave of Trajectories : %i" % TrackSave)

    if GpuStyle == "CUDA":
        print("\nSelection of CUDA device")
        try:
            # For PyCUDA import
            import pycuda.driver as cuda

            cuda.init()
            for Id in range(cuda.Device.count()):
                device = cuda.Device(Id)
                print("Device #%i of type GPU : %s" % (Id, device.name()))
                if Id in Devices:
                    Alu[Id] = "GPU"

        except ImportError:
            print("Platform does not seem to support CUDA")

    if GpuStyle == "OpenCL":
        print("\nSelection of OpenCL device")
        try:
            # For PyOpenCL import
            import pyopencl as cl

            Id = 0
            for platform in cl.get_platforms():
                for device in platform.get_devices():
                    # deviceType=cl.device_type.to_string(device.type)
                    deviceType = "xPU"
                    print(
                        "Device #%i from %s of type %s : %s"
                        % (
                            Id,
                            platform.vendor.lstrip().rstrip(),
                            deviceType,
                            device.name.lstrip().rstrip(),
                        )
                    )

                    if Id in Devices:
                        # Set the Alu as detected Device Type
                        Alu[Id] = deviceType
                    Id = Id + 1
        except ImportError:
            print("Platform does not seem to support OpenCL")

    zImage = numpy.zeros((Size, Size), dtype=numpy.float32)
    fImage = numpy.zeros((Size, Size), dtype=numpy.float32)

    InputCL = {}
    InputCL["Device"] = Device
    InputCL["GpuStyle"] = GpuStyle
    InputCL["VariableType"] = VariableType
    InputCL["Size"] = Size
    InputCL["Mass"] = Mass
    InputCL["InternalRadius"] = InternalRadius
    InputCL["ExternalRadius"] = ExternalRadius
    InputCL["Angle"] = Angle
    InputCL["BlackBody"] = BlackBody
    InputCL["Method"] = Method
    InputCL["TrackPoints"] = TrackPoints
    InputCL["Physics"] = Physics
    InputCL["Threads"] = Threads
    InputCL["NoImage"] = NoImage
    InputCL["TrackSave"] = TrackSave

    if GpuStyle == "OpenCL":
        duration = BlackHoleCL(zImage, fImage, InputCL)
    else:
        duration = BlackHoleCUDA(zImage, fImage, InputCL)

    Hostname = gethostname()
    Date = time.strftime("%Y%m%d_%H%M%S")
    ImageInfo = "%s_Device%i_%s_%s" % (Method, Device, Hostname, Date)

    if not NoImage:
        ImageOutput(zImage, "TrouNoirZ_%s" % ImageInfo, Colors)
        ImageOutput(fImage, "TrouNoirF_%s" % ImageInfo, Colors)
