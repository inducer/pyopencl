#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NBody Demonstrator implemented in OpenCL, rendering OpenGL

By default, rendering in OpenGL is disabled. Add -g option to activate.

Part of matrix programs from: https://forge.cbp.ens-lyon.fr/svn/bench4gpu/

CC BY-NC-SA 2011 : Emmanuel QUEMENER <emmanuel.quemener@gmail.com>
Cecill v2 : Emmanuel QUEMENER <emmanuel.quemener@gmail.com>

Thanks to Andreas Klockner for PyOpenCL:
http://mathema.tician.de/software/pyopencl

"""
import getopt
import sys
import time
import numpy as np
import pyopencl as cl
import pyopencl.array
from numpy.random import randint as nprnd


def DictionariesAPI():
    Marsaglia = {"CONG": 0, "SHR3": 1, "MWC": 2, "KISS": 3}
    Computing = {"FP32": 0, "FP64": 1}
    Interaction = {"Force": 0, "Potential": 1}
    Artevasion = {"None": 0, "NegExp": 1, "CorRad": 2}
    return (Marsaglia, Computing, Interaction, Artevasion)


BlobOpenCL = """
#define TFP32 0
#define TFP64 1

#define TFORCE 0
#define TPOTENTIAL 1

#define NONE 0
#define NEGEXP 1
#define CORRAD 2

#if TYPE == TFP32
#define MYFLOAT4 float4
#define MYFLOAT8 float8
#define MYFLOAT float
#define DISTANCE fast_distance
#else
#define MYFLOAT4 double4
#define MYFLOAT8 double8
#define MYFLOAT double
#define DISTANCE distance
#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif
#endif

#define znew  ((zmwc=36969*(zmwc&65535)+(zmwc>>16))<<16)
#define wnew  ((wmwc=18000*(wmwc&65535)+(wmwc>>16))&65535)
#define MWC   (znew+wnew)
#define SHR3  (jsr=(jsr=(jsr=jsr^(jsr<<17))^(jsr>>13))^(jsr<<5))
#define CONG  (jcong=69069*jcong+1234567)
#define KISS  ((MWC^CONG)+SHR3)

#define MWCfp (MYFLOAT)(MWC * 2.3283064365386963e-10f)
#define KISSfp (MYFLOAT)(KISS * 2.3283064365386963e-10f)
#define SHR3fp (MYFLOAT)(SHR3 * 2.3283064365386963e-10f)
#define CONGfp (MYFLOAT)(CONG * 2.3283064365386963e-10f)

#define PI (MYFLOAT)3.141592653589793238e0f

#define SMALL_NUM (MYFLOAT)1.e-9f

#define CoreRadius (MYFLOAT)(1.e0f)

// Create my own Distance implementation: distance buggy on Oland AMD chipset

MYFLOAT MyDistance(MYFLOAT4 n,MYFLOAT4 m)
{
    private MYFLOAT x2,y2,z2;
    x2=n.s0-m.s0;
    x2*=x2;
    y2=n.s1-m.s1;
    y2*=y2;
    z2=n.s2-m.s2;
    z2*=z2;
    return(sqrt(x2+y2+z2));
}

// Potential between 2 m,n bodies
MYFLOAT PairPotential(MYFLOAT4 m,MYFLOAT4 n)
#if ARTEVASION == NEGEXP
// Add exp(-r) to numerator to avoid divergence for low distances
{
    MYFLOAT r=DISTANCE(n,m);
    return((-1.e0f+exp(-r))/r);
}
#elif ARTEVASION == CORRAD
// Add Core Radius to avoid divergence for low distances
{
    MYFLOAT r=DISTANCE(n,m);
    return(-1.e0f/sqrt(r*r+CoreRadius*CoreRadius));
}
#else
// Classical potential in 1/r
{
//    return((MYFLOAT)(-1.e0f)/(MyDistance(m,n)));
    return((MYFLOAT)(-1.e0f)/(DISTANCE(n,m)));
}
#endif

// Interaction based of Force as gradient of Potential
MYFLOAT4 Interaction(MYFLOAT4 m,MYFLOAT4 n)
#if INTERACTION == TFORCE
#if ARTEVASION == NEGEXP
// Force gradient of potential, set as (1-exp(-r))/r
{
    private MYFLOAT r=MyDistance(n,m);
    private MYFLOAT num=1.e0f+exp(-r)*(r-1.e0f);
    return((n-m)*num/(MYFLOAT)(r*r*r));
}
#elif ARTEVASION == CORRAD
// Force gradient of potential, (Core Radius) set as 1/sqrt(r**2+CoreRadius**2)
{
    private MYFLOAT r=MyDistance(n,m);
    private MYFLOAT den=sqrt(r*r+CoreRadius*CoreRadius);
    return((n-m)/(MYFLOAT)(den*den*den));
}
#else
// Simplest implementation of force (equals to acceleration)
// seems to bo bad (numerous artevasions)
// MYFLOAT4 InteractionForce(MYFLOAT4 m,MYFLOAT4 n)
{
    private MYFLOAT r=MyDistance(n,m);
    return((n-m)/(MYFLOAT)(r*r*r));
}
#endif
#else
// Force definited as gradient of potential
// Estimate potential and proximate potential to estimate force
{
    // 1/1024 seems to be a good factor: larger one provides bad results
    private MYFLOAT epsilon=(MYFLOAT)(1.e0f/1024);
    private MYFLOAT4 er=normalize(n-m);
    private MYFLOAT4 dr=er*(MYFLOAT)epsilon;

    return(er/epsilon*(PairPotential(m,n)-PairPotential(m+dr,n)));
}
#endif

MYFLOAT AtomicPotential(__global MYFLOAT4* clDataX,int gid)
{
    private MYFLOAT potential=(MYFLOAT)0.e0f;
    private MYFLOAT4 x=clDataX[gid];

    for (int i=0;i<get_global_size(0);i++)
    {
        if (gid != i)
        potential+=PairPotential(x,clDataX[i]);
    }

    barrier(CLK_GLOBAL_MEM_FENCE);
    return(potential);
}

MYFLOAT AtomicPotentialCoM(__global MYFLOAT4* clDataX,__global MYFLOAT4* clCoM,int gid) // # noqa: E501
{
    return(PairPotential(clDataX[gid],clCoM[0]));
}

// Elements from : http://doswa.com/2009/01/02/fourth-order-runge-kutta-numerical-integration.html

MYFLOAT8 AtomicRungeKutta(__global MYFLOAT4* clDataInX,__global MYFLOAT4* clDataInV,int gid,MYFLOAT dt)
{
    private MYFLOAT4 a0,v0,x0,a1,v1,x1,a2,v2,x2,a3,v3,x3,a4,v4,x4,xf,vf;
    MYFLOAT4 DT=dt*(MYFLOAT4)(1.e0f,1.e0f,1.e0f,1.e0f);

    a0=(MYFLOAT4)(0.e0f,0.e0f,0.e0f,0.e0f);
    v0=(MYFLOAT4)clDataInV[gid];
    x0=(MYFLOAT4)clDataInX[gid];
    int N = get_global_size(0);

    for (private int i=0;i<N;i++)
    {
        if (gid != i)
        a0+=Interaction(x0,clDataInX[i]);
    }

    a1=(MYFLOAT4)(0.e0f,0.e0f,0.e0f,0.e0f);
    v1=a0*dt+v0;
    x1=v0*dt+x0;
    for (private int j=0;j<N;j++)
    {
        if (gid != j)
        a1+=Interaction(x1,clDataInX[j]);
    }

    a2=(MYFLOAT4)(0.e0f,0.e0f,0.e0f,0.e0f);
    v2=a1*(MYFLOAT)(dt/2.e0f)+v0;
    x2=v1*(MYFLOAT)(dt/2.e0f)+x0;
    for (private int k=0;k<N;k++)
    {
        if (gid != k)
        a2+=Interaction(x2,clDataInX[k]);
    }

    a3=(MYFLOAT4)(0.e0f,0.e0f,0.e0f,0.e0f);
    v3=a2*(MYFLOAT)(dt/2.e0f)+v0;
    x3=v2*(MYFLOAT)(dt/2.e0f)+x0;
    for (private int l=0;l<N;l++)
    {
        if (gid != l)
        a3+=Interaction(x3,clDataInX[l]);
    }

    a4=(MYFLOAT4)(0.e0f,0.e0f,0.e0f,0.e0f);
    v4=a3*dt+v0;
    x4=v3*dt+x0;
    for (private int m=0;m<N;m++)
    {
        if (gid != m)
        a4+=Interaction(x4,clDataInX[m]);
    }

    xf=x0+dt*(v1+(MYFLOAT)2.e0f*(v2+v3)+v4)/(MYFLOAT)6.e0f;
    vf=v0+dt*(a1+(MYFLOAT)2.e0f*(a2+a3)+a4)/(MYFLOAT)6.e0f;

    return((MYFLOAT8)(xf.s0,xf.s1,xf.s2,0.e0f,vf.s0,vf.s1,vf.s2,0.e0f));
}

MYFLOAT8 AtomicHeun(__global MYFLOAT4* clDataInX,__global MYFLOAT4* clDataInV,int gid,MYFLOAT dt)
{
    private MYFLOAT4 x0,v0,a0,x1,v1,a1,xf,vf;
    MYFLOAT4 Dt=dt*(MYFLOAT4)(1.e0f,1.e0f,1.e0f,1.e0f);

    x0=(MYFLOAT4)clDataInX[gid];
    v0=(MYFLOAT4)clDataInV[gid];
    a0=(MYFLOAT4)(0.e0f,0.e0f,0.e0f,0.e0f);

    for (private int i=0;i<get_global_size(0);i++)
    {
        if (gid != i)
        a0+=Interaction(x0,clDataInX[i]);
    }

    a1=(MYFLOAT4)(0.e0f,0.e0f,0.e0f,0.e0f);
    //v1=v0+dt*a0;
    //x1=x0+dt*v0;
    v1=dt*a0+v0;
    x1=dt*v0+x0;

    for (private int j=0;j<get_global_size(0);j++)
    {
        if (gid != j)
        a1+=Interaction(x1,clDataInX[j]);
    }

    vf=v0+dt*(a0+a1)/(MYFLOAT)2.e0f;
    xf=x0+dt*(v0+v1)/(MYFLOAT)2.e0f;

    return((MYFLOAT8)(xf.s0,xf.s1,xf.s2,0.e0f,vf.s0,vf.s1,vf.s2,0.e0f));
}

MYFLOAT8 AtomicImplicitEuler(__global MYFLOAT4* clDataInX,__global MYFLOAT4* clDataInV,int gid,MYFLOAT dt)
{
    MYFLOAT4 x0,v0,a,xf,vf;

    x0=(MYFLOAT4)clDataInX[gid];
    v0=(MYFLOAT4)clDataInV[gid];
    a=(MYFLOAT4)(0.e0f,0.e0f,0.e0f,0.e0f);

    for (private int i=0;i<get_global_size(0);i++)
    {
        if (gid != i)
          a+=Interaction(x0,clDataInX[i]);
    }

    vf=v0+dt*a;
    xf=x0+dt*vf;

    return((MYFLOAT8)(xf.s0,xf.s1,xf.s2,0.e0f,vf.s0,vf.s1,vf.s2,0.e0f));
}

MYFLOAT8 AtomicExplicitEuler(__global MYFLOAT4* clDataInX,__global MYFLOAT4* clDataInV,int gid,MYFLOAT dt)
{
    MYFLOAT4 x0,v0,a,xf,vf;

    x0=(MYFLOAT4)clDataInX[gid];
    v0=(MYFLOAT4)clDataInV[gid];
    a=(MYFLOAT4)(0.e0f,0.e0f,0.e0f,0.e0f);

    for (private int i=0;i<get_global_size(0);i++)
    {
        if (gid != i)
        a+=Interaction(x0,clDataInX[i]);
    }

    vf=v0+dt*a;
    xf=x0+dt*v0;

    return((MYFLOAT8)(xf.s0,xf.s1,xf.s2,0.e0f,vf.s0,vf.s1,vf.s2,0.e0f));
}

__kernel void InBallSplutterPoints(__global MYFLOAT4* clDataX,
                                   MYFLOAT diameter,uint seed_z,uint seed_w)
{
    private int gid=get_global_id(0);
    private uint zmwc=seed_z+gid;
    private uint wmwc=seed_w+(gid+1)%2;
    private MYFLOAT Heat;

    for (int i=0;i<gid;i++)
    {
        Heat=MWCfp;
    }

// More accurate distribution based on spherical coordonates
// Disactivated because of AMD Oland GPU crash on launch
//     private MYFLOAT Radius,Theta,Phi,PosX,PosY,PosZ,SinTheta;
//     Radius=MWCfp*diameter/2.e0f;
//     Theta=(MYFLOAT)acos((float)(-2.e0f*MWCfp+1.0e0f));
//     Phi=(MYFLOAT)(2.e0f*PI*MWCfp);
//     SinTheta=sin((float)Theta);
//     PosX=cos((float)Phi)*Radius*SinTheta;
//     PosY=sin((float)Phi)*Radius*SinTheta;
//     PosZ=cos((float)Theta)*Radius;
//     clDataX[gid]=(MYFLOAT4)(PosX,PosY,PosZ,0.e0f);

    private MYFLOAT Radius=diameter/2.e0f;
    private MYFLOAT Length=diameter;
    private MYFLOAT4 Position;
    while (Length>Radius) {
       Position=(MYFLOAT4)((MWCfp-0.5e0f)*diameter,(MWCfp-0.5e0f)*diameter,(MWCfp-0.5e0f)*diameter,0.e0f);
       Length=(MYFLOAT)length((MYFLOAT4)Position);
    }

    clDataX[gid]=Position;

    barrier(CLK_GLOBAL_MEM_FENCE);
}

__kernel void InBoxSplutterPoints(__global MYFLOAT4* clDataX, MYFLOAT box,
                             uint seed_z,uint seed_w)
{
    int gid=get_global_id(0);
    uint zmwc=seed_z+gid;
    uint wmwc=seed_w-gid;
    private MYFLOAT Heat;

    for (int i=0;i<gid;i++)
    {
        Heat=MWCfp;
    }

    clDataX[gid]=(MYFLOAT4)((MWCfp-0.5e0f)*box,(MWCfp-0.5e0f)*box,(MWCfp-0.5e0f)*box,0.e0f);

    barrier(CLK_GLOBAL_MEM_FENCE);
}

__kernel void SplutterStress(__global MYFLOAT4* clDataX,__global MYFLOAT4* clDataV,__global MYFLOAT4* clCoM, MYFLOAT velocity,uint seed_z,uint seed_w)
{
    int gid = get_global_id(0);
    MYFLOAT N = (MYFLOAT)get_global_size(0);
    uint zmwc=seed_z+(uint)gid;
    uint wmwc=seed_w-(uint)gid;
    MYFLOAT4 CrossVector,SpeedVector,FromCoM;
    MYFLOAT Heat,ThetaA,PhiA,ThetaB,PhiB,Length,tA,tB,Polar;

    for (int i=0;i<gid;i++)
    {
        Heat=MWCfp;
    }

    // cast to float for sin,cos are NEEDED by Mesa FP64 implementation!
    // Implemention on AMD Oland are probably broken in float

    FromCoM=(MYFLOAT4)(clDataX[gid]-clCoM[0]);
    Length=length(FromCoM);
    //Theta=acos(FromCoM.z/Length);
    //Phi=atan(FromCoM.y/FromCoM.x);
    // First tangential vector to sphere of length radius
    ThetaA=acos(FromCoM.x/Length)+5.e-1f*PI;
    PhiA=atan(FromCoM.y/FromCoM.z);
    // Second tangential vector to sphere of length radius
    ThetaB=acos((float)(FromCoM.x/Length));
    PhiB=atan((float)(FromCoM.y/FromCoM.z))+5.e-1f*PI;
    // (x,y) random coordonates to plane tangential to sphere
    Polar=MWCfp*2.e0f*PI;
    tA=cos((float)Polar);
    tB=sin((float)Polar);

    // Exception for 2 particules to ovoid shifting
    if (get_global_size(0)==2) {
       CrossVector=(MYFLOAT4)(1.e0f,1.e0f,1.e0f,0.e0f);
    } else {
       CrossVector.s0=tA*cos((float)ThetaA)+tB*cos((float)ThetaB);
       CrossVector.s1=tA*sin((float)ThetaA)*sin((float)PhiA)+tB*sin((float)ThetaB)*sin((float)PhiB);
       CrossVector.s2=tA*sin((float)ThetaA)*cos((float)PhiA)+tB*sin((float)ThetaB)*cos((float)PhiB);
       CrossVector.s3=0.e0f;
    }

    if (velocity<SMALL_NUM) {
       SpeedVector=(MYFLOAT4)normalize(cross(FromCoM,CrossVector))*sqrt((-AtomicPotential(clDataX,gid)/(MYFLOAT)2.e0f));
    }
    else
    {

       SpeedVector=(MYFLOAT4)((MWCfp-5e-1f)*velocity,(MWCfp-5e-1f)*velocity,
                              (MWCfp-5e-1f)*velocity,0.e0f);
    }
    clDataV[gid]=SpeedVector;
    barrier(CLK_GLOBAL_MEM_FENCE);
}

__kernel void RungeKutta(__global MYFLOAT4* clDataX,__global MYFLOAT4* clDataV,MYFLOAT h)
{
    private int gid = get_global_id(0);
    private MYFLOAT8 clDataGid;

    clDataGid=AtomicRungeKutta(clDataX,clDataV,gid,h);
    barrier(CLK_GLOBAL_MEM_FENCE);
    clDataX[gid]=clDataGid.s0123;
    clDataV[gid]=clDataGid.s4567;
}

__kernel void Heun(__global MYFLOAT4* clDataX,__global MYFLOAT4* clDataV,MYFLOAT h)
{
    private int gid = get_global_id(0);
    private MYFLOAT8 clDataGid;

    clDataGid=AtomicHeun(clDataX,clDataV,gid,h);
    barrier(CLK_GLOBAL_MEM_FENCE);
    clDataX[gid]=clDataGid.s0123;
    clDataV[gid]=clDataGid.s4567;
}

__kernel void ImplicitEuler(__global MYFLOAT4* clDataX,__global MYFLOAT4* clDataV,MYFLOAT h)
{
    private int gid = get_global_id(0);
    private MYFLOAT8 clDataGid;

    clDataGid=AtomicImplicitEuler(clDataX,clDataV,gid,h);
    barrier(CLK_GLOBAL_MEM_FENCE);
    clDataX[gid]=clDataGid.s0123;
    clDataV[gid]=clDataGid.s4567;
}

__kernel void ExplicitEuler(__global MYFLOAT4* clDataX,__global MYFLOAT4* clDataV,MYFLOAT h)
{
    private int gid = get_global_id(0);
    private MYFLOAT8 clDataGid;

    clDataGid=AtomicExplicitEuler(clDataX,clDataV,gid,h);
    barrier(CLK_GLOBAL_MEM_FENCE);
    clDataX[gid]=clDataGid.s0123;
    clDataV[gid]=clDataGid.s4567;
}

__kernel void CoMPotential(__global MYFLOAT4* clDataX,__global MYFLOAT4* clCoM,__global MYFLOAT* clPotential)
{
    int gid = get_global_id(0);

    clPotential[gid]=PairPotential(clDataX[gid],clCoM[0]);
}

__kernel void Potential(__global MYFLOAT4* clDataX,__global MYFLOAT* clPotential)
{
    int gid = get_global_id(0);

    MYFLOAT potential=(MYFLOAT)0.e0f;
    MYFLOAT4 x=clDataX[gid];

    for (int i=0;i<get_global_size(0);i++)
    {
        if (gid != i)
        potential+=PairPotential(x,clDataX[i]);
    }

    barrier(CLK_GLOBAL_MEM_FENCE);
    clPotential[gid]=potential*(MYFLOAT)5.e-1f;
}

__kernel void CenterOfMass(__global MYFLOAT4* clDataX,__global MYFLOAT4* clCoM,int Size)
{
    MYFLOAT4 CoM=clDataX[0];

    for (int i=1;i<Size;i++)
    {
        CoM+=clDataX[i];
    }

    barrier(CLK_GLOBAL_MEM_FENCE);
    clCoM[0]=(MYFLOAT4)(CoM.s0,CoM.s1,CoM.s2,0.e0f)/(MYFLOAT)Size;
}

__kernel void Kinetic(__global MYFLOAT4* clDataV,__global MYFLOAT* clKinetic)
{
    int gid = get_global_id(0);

    barrier(CLK_GLOBAL_MEM_FENCE);
    MYFLOAT d=(MYFLOAT)length(clDataV[gid]);
    clKinetic[gid]=(MYFLOAT)5.e-1f*(MYFLOAT)(d*d);
}

"""


def MainOpenCL(clDataX, clDataV, Step, Method):
    time_start = time.time()
    if Method == "RungeKutta":
        CLLaunch = MyRoutines.RungeKutta(
            queue, (Number, 1), None, clDataX, clDataV, Step
        )
    elif Method == "ExplicitEuler":
        CLLaunch = MyRoutines.ExplicitEuler(
            queue, (Number, 1), None, clDataX, clDataV, Step
        )
    elif Method == "Heun":
        CLLaunch = MyRoutines.Heun(queue, (Number, 1), None, clDataX, clDataV, Step)
    else:
        CLLaunch = MyRoutines.ImplicitEuler(
            queue, (Number, 1), None, clDataX, clDataV, Step
        )
    CLLaunch.wait()
    Elapsed = time.time() - time_start
    return Elapsed


def display(*args):
    global MyDataX, MyDataV, clDataX, clDataV, Step, Method, Number, Iterations, \
            Durations, Verbose, SpeedRendering

    gl.glClearColor(0.0, 0.0, 0.0, 0.0)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
    gl.glColor3f(1.0, 1.0, 1.0)

    MainOpenCL(clDataX, clDataV, Step, Method)
    if SpeedRendering:
        cl.enqueue_copy(queue, MyDataV, clDataV)
        MyDataV.reshape(Number, 4)[:, 3] = 1
        gl.glVertexPointerf(MyDataV.reshape(Number, 4))
    else:
        cl.enqueue_copy(queue, MyDataX, clDataX)
        MyDataX.reshape(Number, 4)[:, 3] = 1
        gl.glVertexPointerf(MyDataX.reshape(Number, 4))

    if Verbose:
        print("Positions for #%s iteration: %s" % (Iterations, MyDataX))
    else:
        sys.stdout.write(".")
        sys.stdout.flush()
    Durations = np.append(Durations, MainOpenCL(clDataX, clDataV, Step, Method))
    gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
    gl.glDrawArrays(gl.GL_POINTS, 0, Number)
    gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
    gl.glFlush()
    Iterations += 1
    glut.glutSwapBuffers()


def halt():
    pass


def keyboard(k, x, y):
    global ViewRZ, SpeedRendering
    LC_Z = glut.as_8_bit("z")
    UC_Z = glut.as_8_bit("Z")
    Plus = glut.as_8_bit("+")
    Minus = glut.as_8_bit("-")
    Switch = glut.as_8_bit("s")

    Zoom = 1
    if k == LC_Z:
        ViewRZ += 1.0
    elif k == UC_Z:
        ViewRZ -= 1.0
    elif k == Plus:
        Zoom *= 2.0
    elif k == Minus:
        Zoom /= 2.0
    elif k == Switch:
        if SpeedRendering:
            SpeedRendering = False
        else:
            SpeedRendering = True
    elif ord(k) == 27:  # Escape
        glut.glutLeaveMainLoop()
        return False
    else:
        return
    gl.glRotatef(ViewRZ, 0.0, 0.0, 1.0)
    gl.glScalef(Zoom, Zoom, Zoom)
    glut.glutPostRedisplay()


def special(k, x, y):
    global ViewRX, ViewRY

    Step = 1.0
    if k == glut.GLUT_KEY_UP:
        ViewRX += Step
    elif k == glut.GLUT_KEY_DOWN:
        ViewRX -= Step
    elif k == glut.GLUT_KEY_LEFT:
        ViewRY += Step
    elif k == glut.GLUT_KEY_RIGHT:
        ViewRY -= Step
    else:
        return
    gl.glRotatef(ViewRX, 1.0, 0.0, 0.0)
    gl.glRotatef(ViewRY, 0.0, 1.0, 0.0)
    glut.glutPostRedisplay()


def setup_viewport():
    global SizeOfBox
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    gl.glOrtho(-SizeOfBox, SizeOfBox, -SizeOfBox, SizeOfBox, -SizeOfBox, SizeOfBox)
    glut.glutPostRedisplay()


def reshape(w, h):
    gl.glViewport(0, 0, w, h)
    setup_viewport()


if __name__ == "__main__":

    global Number, Step, clDataX, clDataV, MyDataX, MyDataV, Method, SizeOfBox, \
            Iterations, Verbose, Durations

    # ValueType
    ValueType = "FP32"

    class MyFloat(np.float32):
        pass

    #    clType8=cl_array.vec.float8
    # Set defaults values
    np.set_printoptions(precision=2)
    # Id of Device : 1 is for first find !
    Device = 0
    # Number of bodies is integer
    Number = 2
    # Number of iterations (for standalone execution)
    Iterations = 10
    # Size of shape
    SizeOfShape = MyFloat(1.0)
    # Initial velocity of particules
    Velocity = MyFloat(1.0)
    # Step
    Step = MyFloat(1.0 / 32)
    # Method of integration
    Method = "ImplicitEuler"
    # InitialRandom
    InitialRandom = False
    # RNG Marsaglia Method
    RNG = "MWC"
    # Viriel Distribution of stress
    VirielStress = True
    # Verbose
    Verbose = False
    # OpenGL real time rendering
    OpenGL = False
    # Speed rendering
    SpeedRendering = False
    # Counter ArtEvasions Measures (artefact evasion)
    CoArEv = "None"
    # Shape to distribute
    Shape = "Ball"
    # Type of Interaction
    InterType = "Force"

    HowToUse = "%s -h [Help] -r [InitialRandom] -g [OpenGL] -e [VirielStress] -o [Verbose] -p [Potential] -x <None|NegExp|CorRad> -d <DeviceId> -n <NumberOfParticules> -i <Iterations> -z <SizeOfBoxOrBall> -v <Velocity> -s <Step> -b <Ball|Box> -m <ImplicitEuler|RungeKutta|ExplicitEuler|Heun> -t <FP32|FP64>"  # noqa: E501

    try:
        opts, args = getopt.getopt(
            sys.argv[1:],
            "rpgehod:n:i:z:v:s:m:t:b:x:",
            [
                "random",
                "potential",
                "coarev",
                "opengl",
                "viriel",
                "verbose",
                "device=",
                "number=",
                "iterations=",
                "size=",
                "velocity=",
                "step=",
                "method=",
                "valuetype=",
                "shape=",
            ],
        )
    except getopt.GetoptError:
        print(HowToUse % sys.argv[0])
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h":
            print(HowToUse % sys.argv[0])

            print("\nInformations about devices detected under OpenCL:")
            try:
                Id = 0
                for platform in cl.get_platforms():
                    for device in platform.get_devices():
                        # Failed now because of POCL implementation
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
                sys.exit()
            except ImportError:
                print("Your platform does not seem to support OpenCL")
                sys.exit()

        elif opt in ("-t", "--valuetype"):
            if arg == "FP64":

                class MyFloat(np.float64):
                    pass

            else:

                class MyFloat(np.float32):
                    pass

            ValueType = arg
        elif opt in ("-d", "--device"):
            Device = int(arg)
        elif opt in ("-m", "--method"):
            Method = arg
        elif opt in ("-b", "--shape"):
            Shape = arg
            if Shape != "Ball" or Shape != "Box":
                print("Wrong argument: set to Ball")
        elif opt in ("-n", "--number"):
            Number = int(arg)
        elif opt in ("-i", "--iterations"):
            Iterations = int(arg)
        elif opt in ("-z", "--size"):
            SizeOfShape = MyFloat(arg)
        elif opt in ("-v", "--velocity"):
            Velocity = MyFloat(arg)
            VirielStress = False
        elif opt in ("-s", "--step"):
            Step = MyFloat(arg)
        elif opt in ("-r", "--random"):
            InitialRandom = True
        elif opt in ("-c", "--check"):
            CheckEnergies = True
        elif opt in ("-e", "--viriel"):
            VirielStress = True
        elif opt in ("-g", "--opengl"):
            OpenGL = True
        elif opt in ("-p", "--potential"):
            InterType = "Potential"
        elif opt in ("-x", "--coarev"):
            CoArEv = arg
        elif opt in ("-o", "--verbose"):
            Verbose = True

    SizeOfShape = np.sqrt(MyFloat(SizeOfShape * Number))
    Velocity = MyFloat(Velocity)
    Step = MyFloat(Step)

    print("Device choosed : %s" % Device)
    print("Number of particules : %s" % Number)
    print("Size of Shape : %s" % SizeOfShape)
    print("Initial velocity : %s" % Velocity)
    print("Step of iteration : %s" % Step)
    print("Number of iterations : %s" % Iterations)
    print("Method of resolution : %s" % Method)
    print("Initial Random for RNG Seed : %s" % InitialRandom)
    print("ValueType is : %s" % ValueType)
    print("Viriel distribution of stress : %s" % VirielStress)
    print("OpenGL real time rendering : %s" % OpenGL)
    print("Speed rendering : %s" % SpeedRendering)
    print("Interaction type : %s" % InterType)
    print("Counter Artevasion type : %s" % CoArEv)

    # Create Numpy array of CL vector with 8 FP32
    MyCoM = np.zeros(4, dtype=MyFloat)
    MyDataX = np.zeros(Number * 4, dtype=MyFloat)
    MyDataV = np.zeros(Number * 4, dtype=MyFloat)
    MyPotential = np.zeros(Number, dtype=MyFloat)
    MyKinetic = np.zeros(Number, dtype=MyFloat)

    Marsaglia, Computing, Interaction, Artevasion = DictionariesAPI()

    # Scan the OpenCL arrays
    Id = 0
    HasXPU = False
    for platform in cl.get_platforms():
        for device in platform.get_devices():
            if Id == Device:
                PlatForm = platform
                XPU = device
                print("CPU/GPU selected: ", device.name.lstrip())
                print("Platform selected: ", platform.name)
                HasXPU = True
            Id += 1

    if not HasXPU:
        print("No XPU #%i found in all of %i devices, sorry..." % (Device, Id - 1))
        sys.exit()

    # Create Context
    try:
        ctx = cl.Context([XPU])
        queue = cl.CommandQueue(
            ctx, properties=cl.command_queue_properties.PROFILING_ENABLE
        )
    except Exception:
        print("Crash during context creation")

    # Build all routines used for the computing

    # BuildOptions="-cl-mad-enable -cl-kernel-arg-info -cl-fast-relaxed-math -cl-std=CL1.2 -DTRNG=%i -DTYPE=%i" % (Marsaglia[RNG],Computing[ValueType])  # noqa: E501
    BuildOptions = "-cl-mad-enable -cl-fast-relaxed-math -DTRNG=%i -DTYPE=%i -DINTERACTION=%i -DARTEVASION=%i" % (  # noqa: E501
        Marsaglia[RNG],
        Computing[ValueType],
        Interaction[InterType],
        Artevasion[CoArEv],
    )

    if (
        "Intel" in PlatForm.name
        or "Experimental" in PlatForm.name
        or "Clover" in PlatForm.name
        or "Portable" in PlatForm.name
    ):
        MyRoutines = cl.Program(ctx, BlobOpenCL).build(options=BuildOptions)
    else:
        MyRoutines = cl.Program(ctx, BlobOpenCL).build(
            options=BuildOptions + " -cl-strict-aliasing"
        )

    mf = cl.mem_flags
    # Read/Write approach for buffering
    clDataX = cl.Buffer(ctx, mf.READ_WRITE, MyDataX.nbytes)
    clDataV = cl.Buffer(ctx, mf.READ_WRITE, MyDataV.nbytes)
    clPotential = cl.Buffer(ctx, mf.READ_WRITE, MyPotential.nbytes)
    clKinetic = cl.Buffer(ctx, mf.READ_WRITE, MyKinetic.nbytes)
    clCoM = cl.Buffer(ctx, mf.READ_WRITE, MyCoM.nbytes)

    # Write/HostPointer approach for buffering
    # clDataX = cl.Buffer(ctx, mf.WRITE_ONLY|mf.COPY_HOST_PTR,hostbuf=MyDataX)
    # clDataV = cl.Buffer(ctx, mf.WRITE_ONLY|mf.COPY_HOST_PTR,hostbuf=MyDataV)
    # clPotential = cl.Buffer(ctx, mf.WRITE_ONLY|mf.COPY_HOST_PTR,hostbuf=MyPotential)  # noqa: E501
    # clKinetic = cl.Buffer(ctx, mf.WRITE_ONLY|mf.COPY_HOST_PTR,hostbuf=MyKinetic)
    # clCoM = cl.Buffer(ctx, mf.WRITE_ONLY|mf.COPY_HOST_PTR,hostbuf=MyCoM)

    print("All particles superimposed.")

    # Set particles to RNG points
    if InitialRandom:
        seed_w = np.uint32(nprnd(2 ** 32))
        seed_z = np.uint32(nprnd(2 ** 32))
    else:
        seed_w = np.uint32(19710211)
        seed_z = np.uint32(20081010)

    if Shape == "Ball":
        MyRoutines.InBallSplutterPoints(
            queue, (Number, 1), None, clDataX, SizeOfShape, seed_w, seed_z
        )
    else:
        MyRoutines.InBoxSplutterPoints(
            queue, (Number, 1), None, clDataX, SizeOfShape, seed_w, seed_z
        )

    print("All particules distributed")

    CLLaunch = MyRoutines.CenterOfMass(
        queue, (1, 1), None, clDataX, clCoM, np.int32(Number)
    )
    CLLaunch.wait()
    cl.enqueue_copy(queue, MyCoM, clCoM)
    print("Center Of Mass estimated: (%s,%s,%s)" % (MyCoM[0], MyCoM[1], MyCoM[2]))

    if VirielStress:
        CLLaunch = MyRoutines.SplutterStress(
            queue,
            (Number, 1),
            None,
            clDataX,
            clDataV,
            clCoM,
            MyFloat(0.0),
            np.uint32(110271),
            np.uint32(250173),
        )
    else:
        CLLaunch = MyRoutines.SplutterStress(
            queue,
            (Number, 1),
            None,
            clDataX,
            clDataV,
            clCoM,
            Velocity,
            np.uint32(110271),
            np.uint32(250173),
        )
    CLLaunch.wait()

    print("All particules stressed")

    CLLaunch = MyRoutines.Potential(queue, (Number, 1), None, clDataX, clPotential)
    CLLaunch = MyRoutines.Kinetic(queue, (Number, 1), None, clDataV, clKinetic)
    CLLaunch.wait()
    cl.enqueue_copy(queue, MyPotential, clPotential)
    cl.enqueue_copy(queue, MyKinetic, clKinetic)
    print(
        "Energy estimated: Viriel=%s Potential=%s Kinetic=%s\n"
        % (
            np.sum(MyPotential) + 2 * np.sum(MyKinetic),
            np.sum(MyPotential),
            np.sum(MyKinetic),
        )
    )

    if SpeedRendering:
        SizeOfBox = max(2 * MyKinetic)
    else:
        SizeOfBox = SizeOfShape

    if OpenGL:
        print("\tTiny documentation to interact OpenGL rendering:\n")
        print("\t<Left|Right> Rotate around X axis")
        print("\t  <Up|Down>  Rotate around Y axis")
        print("\t   <z|Z>     Rotate around Z axis")
        print("\t   <-|+>     Unzoom/Zoom")
        print("\t    <s>      Toggle to display Positions or Velocities")
        print("\t   <Esc>     Quit\n")

    wall_time_start = time.time()

    Durations = np.array([], dtype=MyFloat)
    print("Starting!")
    if OpenGL:
        import OpenGL.GL as gl
        import OpenGL.GLUT as glut

        global ViewRX, ViewRY, ViewRZ
        Iterations = 0
        ViewRX, ViewRY, ViewRZ = 0.0, 0.0, 0.0
        # Launch OpenGL Loop
        glut.glutInit(sys.argv)
        glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGB)
        glut.glutSetOption(glut.GLUT_ACTION_ON_WINDOW_CLOSE,
                glut.GLUT_ACTION_CONTINUE_EXECUTION)
        glut.glutInitWindowSize(512, 512)
        glut.glutCreateWindow(b"NBodyGL")
        setup_viewport()
        glut.glutReshapeFunc(reshape)
        glut.glutDisplayFunc(display)
        glut.glutIdleFunc(display)
        #   glutMouseFunc(mouse)
        glut.glutSpecialFunc(special)
        glut.glutKeyboardFunc(keyboard)
        glut.glutMainLoop()
    else:
        for iteration in range(Iterations):
            Elapsed = MainOpenCL(clDataX, clDataV, Step, Method)
            if Verbose:
                # print("Duration of #%s iteration: %s" % (iteration,Elapsed))
                cl.enqueue_copy(queue, MyDataX, clDataX)
                print("Positions for #%s iteration: %s" % (iteration, MyDataX))
            else:
                sys.stdout.write(".")
                sys.stdout.flush()
            Durations = np.append(Durations, Elapsed)

    print("\nEnding!")

    MyRoutines.CenterOfMass(queue, (1, 1), None, clDataX, clCoM, np.int32(Number))
    CLLaunch = MyRoutines.Potential(queue, (Number, 1), None, clDataX, clPotential)
    CLLaunch = MyRoutines.Kinetic(queue, (Number, 1), None, clDataV, clKinetic)
    CLLaunch.wait()
    cl.enqueue_copy(queue, MyCoM, clCoM)
    cl.enqueue_copy(queue, MyPotential, clPotential)
    cl.enqueue_copy(queue, MyKinetic, clKinetic)
    print("\nCenter Of Mass estimated: (%s,%s,%s)" % (MyCoM[0], MyCoM[1], MyCoM[2]))
    print(
        "Energy estimated: Viriel=%s Potential=%s Kinetic=%s\n"
        % (
            np.sum(MyPotential) + 2.0 * np.sum(MyKinetic),
            np.sum(MyPotential),
            np.sum(MyKinetic),
        )
    )

    print(
        "Duration stats on device %s with %s iterations :\n\tMean:\t%s\n\tMedian:\t%s\n\tStddev:\t%s\n\tMin:\t%s\n\tMax:\t%s\n\n\tVariability:\t%s\n"  # noqa: E501
        % (
            Device,
            Iterations,
            np.mean(Durations),
            np.median(Durations),
            np.std(Durations),
            np.min(Durations),
            np.max(Durations),
            np.std(Durations) / np.median(Durations),
        )
    )

    # FPS: 1/Elapsed
    FPS = np.ones(len(Durations))
    FPS /= Durations

    print(
        "FPS stats on device %s with %s iterations :\n\tMean:\t%s\n\tMedian:\t%s\n\tStddev:\t%s\n\tMin:\t%s\n\tMax:\t%s\n"  # noqa: E501
        % (
            Device,
            Iterations,
            np.mean(FPS),
            np.median(FPS),
            np.std(FPS),
            np.min(FPS),
            np.max(FPS),
        )
    )

    # Contraction of Square*Size*Hertz: Size*Size/Elapsed
    Squertz = np.ones(len(Durations))
    Squertz *= Number * Number
    Squertz /= Durations

    print(
        "Squertz in log10 & complete stats on device %s with %s iterations :\n\tMean:\t%s\t%s\n\tMedian:\t%s\t%s\n\tStddev:\t%s\t%s\n\tMin:\t%s\t%s\n\tMax:\t%s\t%s\n"  # noqa: E501
        % (
            Device,
            Iterations,
            np.log10(np.mean(Squertz)),
            np.mean(Squertz),
            np.log10(np.median(Squertz)),
            np.median(Squertz),
            np.log10(np.std(Squertz)),
            np.std(Squertz),
            np.log10(np.min(Squertz)),
            np.min(Squertz),
            np.log10(np.max(Squertz)),
            np.max(Squertz),
        )
    )

    clDataX.release()
    clDataV.release()
    clKinetic.release()
    clPotential.release()
