#!/usr/bin/env python3

#
# Pi-by-MonteCarlo using PyCUDA/PyOpenCL
#
# performs an estimation of Pi using Monte Carlo method
# a large amount of iterations is divided and distributed to compute units
# a lot of options are provided to perform scalabilty tests
#
# use -h for complete set of options
#
# CC BY-NC-SA 2011 : Emmanuel QUEMENER <emmanuel.quemener@gmail.com>
# Cecill v2 : Emmanuel QUEMENER <emmanuel.quemener@gmail.com>
#

# Thanks to Andreas Klockner for PyCUDA:
# http://mathema.tician.de/software/pycuda
# Thanks to Andreas Klockner for PyOpenCL:
# http://mathema.tician.de/software/pyopencl
#

# 2013-01-01 : problems with launch timeout
# http://stackoverflow.com/questions/497685/how-do-you-get-around-the-maximum-cuda-run-time
# Option "Interactive" "0" in /etc/X11/xorg.conf

# Common tools
import numpy
import sys
import getopt
import time
import itertools
from socket import gethostname


class PenStacle:
    """Pentacle of Statistics from data"""

    Avg = 0
    Med = 0
    Std = 0
    Min = 0
    Max = 0

    def __init__(self, Data):
        self.Avg = numpy.average(Data)
        self.Med = numpy.median(Data)
        self.Std = numpy.std(Data)
        self.Max = numpy.max(Data)
        self.Min = numpy.min(Data)

    def display(self):
        print("%s %s %s %s %s" % (self.Avg, self.Med, self.Std, self.Min, self.Max))


class Experience:
    """Metrology for experiences"""

    DeviceStyle = ""
    DeviceId = 0
    AvgD = 0
    MedD = 0
    StdD = 0
    MinD = 0
    MaxD = 0
    AvgR = 0
    MedR = 0
    StdR = 0
    MinR = 0
    MaxR = 0

    def __init__(self, DeviceStyle, DeviceId, Iterations):
        self.DeviceStyle = DeviceStyle
        self.DeviceId = DeviceId
        self.Iterations

    def Metrology(self, Data):
        Duration = PenStacle(Data)
        Rate = PenStacle(Iterations / Data)
        print("Duration %s" % Duration)
        print("Rate %s" % Rate)


def DictionariesAPI():
    Marsaglia = {"CONG": 0, "SHR3": 1, "MWC": 2, "KISS": 3}
    Computing = {"INT32": 0, "INT64": 1, "FP32": 2, "FP64": 3}
    Test = {True: 1, False: 0}
    return (Marsaglia, Computing, Test)


# find prime factors of a number
# Get for WWW :
# http://pythonism.wordpress.com/2008/05/17/looking-at-factorisation-in-python/
def PrimeFactors(x):

    factorlist = numpy.array([]).astype("uint32")
    loop = 2
    while loop <= x:
        if x % loop == 0:
            x /= loop
            factorlist = numpy.append(factorlist, [loop])
        else:
            loop += 1
    return factorlist


# Try to find the best thread number in Hybrid approach (Blocks&Threads)
# output is thread number
def BestThreadsNumber(jobs):
    factors = PrimeFactors(jobs)
    matrix = numpy.append([factors], [factors[::-1]], axis=0)
    threads = 1
    for factor in matrix.transpose().ravel():
        threads = threads * factor
        if threads * threads > jobs or threads > 512:
            break
    return int(threads)


# Predicted Amdahl Law (Reduced with s=1-p)
def AmdahlR(N, T1, p):
    return T1 * (1 - p + p / N)


# Predicted Amdahl Law
def Amdahl(N, T1, s, p):
    return T1 * (s + p / N)


# Predicted Mylq Law with first order
def Mylq(N, T1, s, c, p):
    return T1 * (s + p / N) + c * N


# Predicted Mylq Law with second order
def Mylq2(N, T1, s, c1, c2, p):
    return T1 * (s + p / N) + c1 * N + c2 * N * N


def KernelCodeCuda():
    KERNEL_CODE_CUDA = """
#define TCONG 0
#define TSHR3 1
#define TMWC 2
#define TKISS 3

#define TINT32 0
#define TINT64 1
#define TFP32 2
#define TFP64 3

#define IFTHEN 1

// Marsaglia RNG very simple implementation

#define znew  ((z=36969*(z&65535)+(z>>16))<<16)
#define wnew  ((w=18000*(w&65535)+(w>>16))&65535)
#define MWC   (znew+wnew)
#define SHR3  (jsr=(jsr=(jsr=jsr^(jsr<<17))^(jsr>>13))^(jsr<<5))
#define CONG  (jcong=69069*jcong+1234567)
#define KISS  ((MWC^CONG)+SHR3)

#define MWCfp MWC * 2.328306435454494e-10f
#define KISSfp KISS * 2.328306435454494e-10f
#define SHR3fp SHR3 * 2.328306435454494e-10f
#define CONGfp CONG * 2.328306435454494e-10f

__device__ ulong MainLoop(ulong iterations,uint seed_w,uint seed_z,size_t work)
{

#if TRNG == TCONG
   uint jcong=seed_z+work;
#elif TRNG == TSHR3
   uint jsr=seed_w+work;
#elif TRNG == TMWC
   uint z=seed_z+work;
   uint w=seed_w+work;
#elif TRNG == TKISS
   uint jcong=seed_z+work;
   uint jsr=seed_w+work;
   uint z=seed_z-work;
   uint w=seed_w-work;
#endif

   ulong total=0;

   for (ulong i=0;i<iterations;i++) {

#if TYPE == TINT32
    #define THEONE 1073741824
    #if TRNG == TCONG
        uint x=CONG>>17 ;
        uint y=CONG>>17 ;
    #elif TRNG == TSHR3
        uint x=SHR3>>17 ;
        uint y=SHR3>>17 ;
    #elif TRNG == TMWC
        uint x=MWC>>17 ;
        uint y=MWC>>17 ;
    #elif TRNG == TKISS
        uint x=KISS>>17 ;
        uint y=KISS>>17 ;
    #endif
#elif TYPE == TINT64
    #define THEONE 4611686018427387904
    #if TRNG == TCONG
        ulong x=(ulong)(CONG>>1) ;
        ulong y=(ulong)(CONG>>1) ;
    #elif TRNG == TSHR3
        ulong x=(ulong)(SHR3>>1) ;
        ulong y=(ulong)(SHR3>>1) ;
    #elif TRNG == TMWC
        ulong x=(ulong)(MWC>>1) ;
        ulong y=(ulong)(MWC>>1) ;
    #elif TRNG == TKISS
        ulong x=(ulong)(KISS>>1) ;
        ulong y=(ulong)(KISS>>1) ;
    #endif
#elif TYPE == TFP32
    #define THEONE 1.0f
    #if TRNG == TCONG
        float x=CONGfp ;
        float y=CONGfp ;
    #elif TRNG == TSHR3
        float x=SHR3fp ;
        float y=SHR3fp ;
    #elif TRNG == TMWC
        float x=MWCfp ;
        float y=MWCfp ;
    #elif TRNG == TKISS
      float x=KISSfp ;
      float y=KISSfp ;
    #endif
#elif TYPE == TFP64
    #define THEONE 1.0f
    #if TRNG == TCONG
        double x=(double)CONGfp ;
        double y=(double)CONGfp ;
    #elif TRNG == TSHR3
        double x=(double)SHR3fp ;
        double y=(double)SHR3fp ;
    #elif TRNG == TMWC
        double x=(double)MWCfp ;
        double y=(double)MWCfp ;
    #elif TRNG == TKISS
        double x=(double)KISSfp ;
        double y=(double)KISSfp ;
    #endif
#endif

#if TEST == IFTHEN
      if ((x*x+y*y) <=THEONE) {
         total+=1;
      }
#else
      ulong inside=((x*x+y*y) <= THEONE) ? 1:0;
      total+=inside;
#endif
   }

   return(total);
}

__global__ void MainLoopBlocks(ulong *s,ulong iterations,uint seed_w,uint seed_z)
{
   ulong total=MainLoop(iterations,seed_z,seed_w,blockIdx.x);
   s[blockIdx.x]=total;
   __syncthreads();

}

__global__ void MainLoopThreads(ulong *s,ulong iterations,uint seed_w,uint seed_z)
{
   ulong total=MainLoop(iterations,seed_z,seed_w,threadIdx.x);
   s[threadIdx.x]=total;
   __syncthreads();

}

__global__ void MainLoopHybrid(ulong *s,ulong iterations,uint seed_w,uint seed_z)
{
   ulong total=MainLoop(iterations,seed_z,seed_w,blockDim.x*blockIdx.x+threadIdx.x);
   s[blockDim.x*blockIdx.x+threadIdx.x]=total;
   __syncthreads();
}

"""
    return KERNEL_CODE_CUDA


def KernelCodeOpenCL():
    KERNEL_CODE_OPENCL = """
#define TCONG 0
#define TSHR3 1
#define TMWC 2
#define TKISS 3

#define TINT32 0
#define TINT64 1
#define TFP32 2
#define TFP64 3

#define IFTHEN 1

// Marsaglia RNG very simple implementation
#define znew  ((z=36969*(z&65535)+(z>>16))<<16)
#define wnew  ((w=18000*(w&65535)+(w>>16))&65535)

#define MWC   (znew+wnew)
#define SHR3  (jsr=(jsr=(jsr=jsr^(jsr<<17))^(jsr>>13))^(jsr<<5))
#define CONG  (jcong=69069*jcong+1234567)
#define KISS  ((MWC^CONG)+SHR3)

#define MWCfp MWC * 2.328306435454494e-10f
#define KISSfp KISS * 2.328306435454494e-10f
#define CONGfp CONG * 2.328306435454494e-10f
#define SHR3fp SHR3 * 2.328306435454494e-10f

ulong MainLoop(ulong iterations,uint seed_z,uint seed_w,size_t work)
{

#if TRNG == TCONG
   uint jcong=seed_z+work;
#elif TRNG == TSHR3
   uint jsr=seed_w+work;
#elif TRNG == TMWC
   uint z=seed_z+work;
   uint w=seed_w+work;
#elif TRNG == TKISS
   uint jcong=seed_z+work;
   uint jsr=seed_w+work;
   uint z=seed_z-work;
   uint w=seed_w-work;
#endif

   ulong total=0;

   for (ulong i=0;i<iterations;i++) {

#if TYPE == TINT32
    #define THEONE 1073741824
    #if TRNG == TCONG
        uint x=CONG>>17 ;
        uint y=CONG>>17 ;
    #elif TRNG == TSHR3
        uint x=SHR3>>17 ;
        uint y=SHR3>>17 ;
    #elif TRNG == TMWC
        uint x=MWC>>17 ;
        uint y=MWC>>17 ;
    #elif TRNG == TKISS
        uint x=KISS>>17 ;
        uint y=KISS>>17 ;
    #endif
#elif TYPE == TINT64
    #define THEONE 4611686018427387904
    #if TRNG == TCONG
        ulong x=(ulong)(CONG>>1) ;
        ulong y=(ulong)(CONG>>1) ;
    #elif TRNG == TSHR3
        ulong x=(ulong)(SHR3>>1) ;
        ulong y=(ulong)(SHR3>>1) ;
    #elif TRNG == TMWC
        ulong x=(ulong)(MWC>>1) ;
        ulong y=(ulong)(MWC>>1) ;
    #elif TRNG == TKISS
        ulong x=(ulong)(KISS>>1) ;
        ulong y=(ulong)(KISS>>1) ;
    #endif
#elif TYPE == TFP32
    #define THEONE 1.0f
    #if TRNG == TCONG
        float x=CONGfp ;
        float y=CONGfp ;
    #elif TRNG == TSHR3
        float x=SHR3fp ;
        float y=SHR3fp ;
    #elif TRNG == TMWC
        float x=MWCfp ;
        float y=MWCfp ;
    #elif TRNG == TKISS
      float x=KISSfp ;
      float y=KISSfp ;
    #endif
#elif TYPE == TFP64
#pragma OPENCL EXTENSION cl_khr_fp64: enable
    #define THEONE 1.0f
    #if TRNG == TCONG
        double x=(double)CONGfp ;
        double y=(double)CONGfp ;
    #elif TRNG == TSHR3
        double x=(double)SHR3fp ;
        double y=(double)SHR3fp ;
    #elif TRNG == TMWC
        double x=(double)MWCfp ;
        double y=(double)MWCfp ;
    #elif TRNG == TKISS
        double x=(double)KISSfp ;
        double y=(double)KISSfp ;
    #endif
#endif

#if TEST == IFTHEN
      if ((x*x+y*y) <= THEONE) {
         total+=1;
      }
#else
      ulong inside=((x*x+y*y) <= THEONE) ? 1:0;
      total+=inside;
#endif
   }

   return(total);
}

__kernel void MainLoopGlobal(
    __global ulong *s,ulong iterations,uint seed_w,uint seed_z)
{
   ulong total=MainLoop(iterations,seed_z,seed_w,get_global_id(0));
   barrier(CLK_GLOBAL_MEM_FENCE);
   s[get_global_id(0)]=total;
}

__kernel void MainLoopLocal(
    __global ulong *s,ulong iterations,uint seed_w,uint seed_z)
{
   ulong total=MainLoop(iterations,seed_z,seed_w,get_local_id(0));
   barrier(CLK_LOCAL_MEM_FENCE);
   s[get_local_id(0)]=total;
}

__kernel void MainLoopHybrid(
    __global ulong *s,ulong iterations,uint seed_w,uint seed_z)
{
   ulong total=MainLoop(iterations,seed_z,seed_w,get_global_id(0));
   barrier(CLK_GLOBAL_MEM_FENCE || CLK_LOCAL_MEM_FENCE);
   s[get_global_id(0)]=total;
}

"""
    return KERNEL_CODE_OPENCL


def MetropolisCuda(InputCU):

    print("Inside ", InputCU)

    iterations = InputCU["Iterations"]
    steps = InputCU["Steps"]
    blocks = InputCU["Blocks"]
    threads = InputCU["Threads"]
    Device = InputCU["Device"]
    RNG = InputCU["RNG"]
    ValueType = InputCU["ValueType"]
    Seeds = InputCU["Seeds"]

    Marsaglia, Computing, Test = DictionariesAPI()

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

    circle = numpy.zeros(blocks * threads).astype(numpy.uint64)
    circleCU = cuda.InOut(circle)
    # circleCU = cuda.mem_alloc(circle.size*circle.dtype.itemize)
    # cuda.memcpy_htod(circleCU, circle)

    Context = XPU.make_context()

    try:
        mod = SourceModule(
            KernelCodeCuda(),
            options=[
                "--compiler-options",
                "-DTRNG=%i -DTYPE=%s" % (Marsaglia[RNG], Computing[ValueType]),
            ],
        )
        # mod = SourceModule(KernelCodeCuda(),nvcc='nvcc',keep=True)
        # Needed to set the compiler via ccbin for CUDA9 implementation
        # mod = SourceModule(KernelCodeCuda(),options=['-ccbin','clang-3.9','--compiler-options','-DTRNG=%i' % Marsaglia[RNG],'-DTYPE=%s' % Computing[ValueType],'-DTEST=%s' % Test[TestType]],keep=True)  # noqa: E501
    except Exception:
        print("Compilation seems to break")

    MetropolisBlocksCU = mod.get_function("MainLoopBlocks")  # noqa: F841
    MetropolisThreadsCU = mod.get_function("MainLoopThreads")  # noqa: F841
    MetropolisHybridCU = mod.get_function("MainLoopHybrid")

    MyDuration = numpy.zeros(steps)

    jobs = blocks * threads

    iterationsCU = numpy.uint64(iterations / jobs)
    if iterations % jobs != 0:
        iterationsCU += numpy.uint64(1)

    for i in range(steps):
        start_time = time.time()

        try:
            MetropolisHybridCU(
                circleCU,
                numpy.uint64(iterationsCU),
                numpy.uint32(Seeds[0]),
                numpy.uint32(Seeds[1]),
                grid=(blocks, 1),
                block=(threads, 1, 1),
            )
        except Exception:
            print("Crash during CUDA call")

        elapsed = time.time() - start_time
        print(
            "(Blocks/Threads)=(%i,%i) method done in %.2f s..."
            % (blocks, threads, elapsed)
        )

        MyDuration[i] = elapsed

    OutputCU = {
        "Inside": sum(circle),
        "NewIterations": numpy.uint64(iterationsCU * jobs),
        "Duration": MyDuration,
    }
    print(OutputCU)
    Context.pop()

    Context.detach()

    return OutputCU


def MetropolisOpenCL(InputCL):

    import pyopencl as cl

    iterations = InputCL["Iterations"]
    steps = InputCL["Steps"]
    blocks = InputCL["Blocks"]
    threads = InputCL["Threads"]
    Device = InputCL["Device"]
    RNG = InputCL["RNG"]
    ValueType = InputCL["ValueType"]
    TestType = InputCL["IfThen"]
    Seeds = InputCL["Seeds"]

    Marsaglia, Computing, Test = DictionariesAPI()

    # Initialisation des variables en les CASTant correctement
    Id = 0
    HasXPU = False
    for platform in cl.get_platforms():
        for device in platform.get_devices():
            if Id == Device:
                XPU = device
                print("CPU/GPU selected: ", device.name.lstrip())
                HasXPU = True
            Id += 1
            # print(Id)

    if not HasXPU:
        print("No XPU #%i found in all of %i devices, sorry..." % (Device, Id - 1))
        sys.exit()

    # Je cree le contexte et la queue pour son execution
    try:
        ctx = cl.Context(devices=[XPU])
        queue = cl.CommandQueue(
            ctx, properties=cl.command_queue_properties.PROFILING_ENABLE
        )
    except Exception:
        print("Crash during context creation")

    # Je recupere les flag possibles pour les buffers
    mf = cl.mem_flags

    circle = numpy.zeros(blocks * threads).astype(numpy.uint64)
    circleCL = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=circle)

    MetropolisCL = cl.Program(ctx, KernelCodeOpenCL()).build(
        options="-cl-mad-enable -cl-fast-relaxed-math -DTRNG=%i -DTYPE=%s -DTEST=%s"
        % (Marsaglia[RNG], Computing[ValueType], Test[TestType])
    )

    MyDuration = numpy.zeros(steps)

    jobs = blocks * threads

    iterationsCL = numpy.uint64(iterations / jobs)
    if iterations % jobs != 0:
        iterationsCL += 1

    for i in range(steps):
        start_time = time.time()
        if threads == 1:
            CLLaunch = MetropolisCL.MainLoopGlobal(
                queue,
                (blocks,),
                None,
                circleCL,
                numpy.uint64(iterationsCL),
                numpy.uint32(Seeds[0]),
                numpy.uint32(Seeds[1]),
            )
        else:
            CLLaunch = MetropolisCL.MainLoopHybrid(
                queue,
                (jobs,),
                (threads,),
                circleCL,
                numpy.uint64(iterationsCL),
                numpy.uint32(Seeds[0]),
                numpy.uint32(Seeds[1]),
            )

        CLLaunch.wait()
        cl.enqueue_copy(queue, circle, circleCL).wait()

        elapsed = time.time() - start_time
        print(
            "(Blocks/Threads)=(%i,%i) method done in %.2f s..."
            % (blocks, threads, elapsed)
        )

        # Elapsed method based on CLLaunch doesn't work for Beignet OpenCL
        # elapsed = 1e-9*(CLLaunch.profile.end - CLLaunch.profile.start)

        # print circle,numpy.mean(circle),numpy.median(circle),numpy.std(circle)
        MyDuration[i] = elapsed
        # AllPi=4./numpy.float32(iterationsCL)*circle.astype(numpy.float32)
        # MyPi[i]=numpy.median(AllPi)
        # print MyPi[i],numpy.std(AllPi),MyDuration[i]

    circleCL.release()

    OutputCL = {
        "Inside": sum(circle),
        "NewIterations": numpy.uint64(iterationsCL * jobs),
        "Duration": MyDuration,
    }
    # print(OutputCL)
    return OutputCL


def FitAndPrint(N, D, Curves):

    from scipy.optimize import curve_fit
    import matplotlib.pyplot as plt

    try:
        coeffs_Amdahl, matcov_Amdahl = curve_fit(Amdahl, N, D)

        D_Amdahl = Amdahl(N, coeffs_Amdahl[0], coeffs_Amdahl[1], coeffs_Amdahl[2])
        coeffs_Amdahl[1] = coeffs_Amdahl[1] * coeffs_Amdahl[0] / D[0]
        coeffs_Amdahl[2] = coeffs_Amdahl[2] * coeffs_Amdahl[0] / D[0]
        coeffs_Amdahl[0] = D[0]
        print(
            "Amdahl Normalized: T=%.2f(%.6f+%.6f/N)"
            % (coeffs_Amdahl[0], coeffs_Amdahl[1], coeffs_Amdahl[2])
        )
    except Exception:
        print("Impossible to fit for Amdahl law : only %i elements" % len(D))

    try:
        coeffs_AmdahlR, matcov_AmdahlR = curve_fit(AmdahlR, N, D)

        # D_AmdahlR = AmdahlR(N, coeffs_AmdahlR[0], coeffs_AmdahlR[1])
        coeffs_AmdahlR[1] = coeffs_AmdahlR[1] * coeffs_AmdahlR[0] / D[0]
        coeffs_AmdahlR[0] = D[0]
        print(
            "Amdahl Reduced Normalized: T=%.2f(%.6f+%.6f/N)"
            % (coeffs_AmdahlR[0], 1 - coeffs_AmdahlR[1], coeffs_AmdahlR[1])
        )

    except Exception:
        print("Impossible to fit for Reduced Amdahl law : only %i elements" % len(D))

    try:
        coeffs_Mylq, matcov_Mylq = curve_fit(Mylq, N, D)

        coeffs_Mylq[1] = coeffs_Mylq[1] * coeffs_Mylq[0] / D[0]
        # coeffs_Mylq[2]=coeffs_Mylq[2]*coeffs_Mylq[0]/D[0]
        coeffs_Mylq[3] = coeffs_Mylq[3] * coeffs_Mylq[0] / D[0]
        coeffs_Mylq[0] = D[0]
        print(
            "Mylq Normalized : T=%.2f(%.6f+%.6f/N)+%.6f*N"
            % (coeffs_Mylq[0], coeffs_Mylq[1], coeffs_Mylq[3], coeffs_Mylq[2])
        )
        D_Mylq = Mylq(N, coeffs_Mylq[0], coeffs_Mylq[1], coeffs_Mylq[2],
                coeffs_Mylq[3])
    except Exception:
        print("Impossible to fit for Mylq law : only %i elements" % len(D))

    try:
        coeffs_Mylq2, matcov_Mylq2 = curve_fit(Mylq2, N, D)

        coeffs_Mylq2[1] = coeffs_Mylq2[1] * coeffs_Mylq2[0] / D[0]
        # coeffs_Mylq2[2]=coeffs_Mylq2[2]*coeffs_Mylq2[0]/D[0]
        # coeffs_Mylq2[3]=coeffs_Mylq2[3]*coeffs_Mylq2[0]/D[0]
        coeffs_Mylq2[4] = coeffs_Mylq2[4] * coeffs_Mylq2[0] / D[0]
        coeffs_Mylq2[0] = D[0]
        print(
            "Mylq 2nd order Normalized: T=%.2f(%.6f+%.6f/N)+%.6f*N+%.6f*N^2"
            % (
                coeffs_Mylq2[0],
                coeffs_Mylq2[1],
                coeffs_Mylq2[4],
                coeffs_Mylq2[2],
                coeffs_Mylq2[3],
            )
        )

    except Exception:
        print("Impossible to fit for 2nd order Mylq law : only %i elements" % len(D))

    if Curves:
        plt.xlabel("Number of Threads/work Items")
        plt.ylabel("Total Elapsed Time")

        (Experience,) = plt.plot(N, D, "ro")
    try:
        (pAmdahl,) = plt.plot(N, D_Amdahl, label="Loi de Amdahl")
        (pMylq,) = plt.plot(N, D_Mylq, label="Loi de Mylq")
    except Exception:
        print("Fit curves seem not to be available")

    plt.legend()
    plt.show()


if __name__ == "__main__":

    # Set defaults values

    # GPU style can be Cuda (Nvidia implementation) or OpenCL
    GpuStyle = "OpenCL"
    # Iterations is integer
    Iterations = 1000000000
    # BlocksBlocks in first number of Blocks to explore
    BlocksBegin = 1024
    # BlocksEnd is last number of Blocks to explore
    BlocksEnd = 1024
    # BlocksStep is the step of Blocks to explore
    BlocksStep = 1
    # ThreadsBlocks in first number of Blocks to explore
    ThreadsBegin = 1
    # ThreadsEnd is last number of Blocks to explore
    ThreadsEnd = 1
    # ThreadsStep is the step of Blocks to explore
    ThreadsStep = 1
    # Redo is the times to redo the test to improve metrology
    Redo = 1
    # OutMetrology is method for duration estimation : False is GPU inside
    OutMetrology = False
    Metrology = "InMetro"
    # Curves is True to print the curves
    Curves = False
    # Fit is True to print the curves
    Fit = False
    # Inside based on If
    IfThen = False
    # Marsaglia RNG
    RNG = "MWC"
    # Value type : INT32, INT64, FP32, FP64
    ValueType = "FP32"
    # Seeds for RNG
    Seeds = 110271, 101008

    HowToUse = "%s -o (Out of Core Metrology) -c (Print Curves) -k (Case On IfThen) -d <DeviceId> -g <CUDA/OpenCL> -i <Iterations> -b <BlocksBegin> -e <BlocksEnd> -s <BlocksStep> -f <ThreadsFirst> -l <ThreadsLast> -t <ThreadssTep> -r <RedoToImproveStats> -m <SHR3/CONG/MWC/KISS> -v <INT32/INT64/FP32/FP64>"  # noqa: E501

    try:
        opts, args = getopt.getopt(
            sys.argv[1:],
            "hockg:i:b:e:s:f:l:t:r:d:m:v:",
            [
                "gpustyle=",
                "iterations=",
                "blocksBegin=",
                "blocksEnd=",
                "blocksStep=",
                "threadsFirst=",
                "threadsLast=",
                "threadssTep=",
                "redo=",
                "device=",
                "marsaglia=",
                "valuetype=",
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

        elif opt == "-o":
            OutMetrology = True
            Metrology = "OutMetro"
        elif opt == "-c":
            Curves = True
        elif opt == "-k":
            IfThen = True
        elif opt in ("-d", "--device"):
            Devices.append(int(arg))
        elif opt in ("-g", "--gpustyle"):
            GpuStyle = arg
        elif opt in ("-m", "--marsaglia"):
            RNG = arg
        elif opt in ("-v", "--valuetype"):
            ValueType = arg
        elif opt in ("-i", "--iterations"):
            Iterations = numpy.uint64(arg)
        elif opt in ("-b", "--blocksbegin"):
            BlocksBegin = int(arg)
            BlocksEnd = BlocksBegin
        elif opt in ("-e", "--blocksend"):
            BlocksEnd = int(arg)
        elif opt in ("-s", "--blocksstep"):
            BlocksStep = int(arg)
        elif opt in ("-f", "--threadsfirst"):
            ThreadsBegin = int(arg)
            ThreadsEnd = ThreadsBegin
        elif opt in ("-l", "--threadslast"):
            ThreadsEnd = int(arg)
        elif opt in ("-t", "--threadsstep"):
            ThreadsStep = int(arg)
        elif opt in ("-r", "--redo"):
            Redo = int(arg)

    # If no device has been specified, take the first one!
    if len(Devices) == 0:
        Devices.append(0)

    print("Devices Identification : %s" % Devices)
    print("GpuStyle used : %s" % GpuStyle)
    print("Iterations : %s" % Iterations)
    print("Number of Blocks on begin : %s" % BlocksBegin)
    print("Number of Blocks on end : %s" % BlocksEnd)
    print("Step on Blocks : %s" % BlocksStep)
    print("Number of Threads on begin : %s" % ThreadsBegin)
    print("Number of Threads on end : %s" % ThreadsEnd)
    print("Step on Threads : %s" % ThreadsStep)
    print("Number of redo : %s" % Redo)
    print("Metrology done out of XPU : %r" % OutMetrology)
    print("Type of Marsaglia RNG used : %s" % RNG)
    print("Type of variable : %s" % ValueType)

    if GpuStyle == "CUDA":
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

    # print(Devices,Alu)

    BlocksList = range(BlocksBegin, BlocksEnd + BlocksStep, BlocksStep)
    ThreadsList = range(ThreadsBegin, ThreadsEnd + ThreadsStep, ThreadsStep)

    ExploredJobs = numpy.array([]).astype(numpy.uint32)
    ExploredBlocks = numpy.array([]).astype(numpy.uint32)
    ExploredThreads = numpy.array([]).astype(numpy.uint32)
    avgD = numpy.array([]).astype(numpy.float32)
    medD = numpy.array([]).astype(numpy.float32)
    stdD = numpy.array([]).astype(numpy.float32)
    minD = numpy.array([]).astype(numpy.float32)
    maxD = numpy.array([]).astype(numpy.float32)
    avgR = numpy.array([]).astype(numpy.float32)
    medR = numpy.array([]).astype(numpy.float32)
    stdR = numpy.array([]).astype(numpy.float32)
    minR = numpy.array([]).astype(numpy.float32)
    maxR = numpy.array([]).astype(numpy.float32)

    for Blocks, Threads in itertools.product(BlocksList, ThreadsList):

        # print Blocks,Threads
        circle = numpy.zeros(Blocks * Threads).astype(numpy.uint64)
        ExploredJobs = numpy.append(ExploredJobs, Blocks * Threads)
        ExploredBlocks = numpy.append(ExploredBlocks, Blocks)
        ExploredThreads = numpy.append(ExploredThreads, Threads)

        if OutMetrology:
            DurationItem = numpy.array([]).astype(numpy.float32)
            Duration = numpy.array([]).astype(numpy.float32)
            Rate = numpy.array([]).astype(numpy.float32)
            for i in range(Redo):
                start = time.time()
                if GpuStyle == "CUDA":
                    try:
                        InputCU = {}
                        InputCU["Iterations"] = Iterations
                        InputCU["Steps"] = 1
                        InputCU["Blocks"] = Blocks
                        InputCU["Threads"] = Threads
                        InputCU["Device"] = Devices[0]
                        InputCU["RNG"] = RNG
                        InputCU["Seeds"] = Seeds
                        InputCU["ValueType"] = ValueType
                        InputCU["IfThen"] = IfThen
                        OutputCU = MetropolisCuda(InputCU)
                        Inside = OutputCU["Circle"]
                        NewIterations = OutputCU["NewIterations"]
                        Duration = OutputCU["Duration"]
                    except Exception:
                        print(
                            "Problem with (%i,%i) // computations on Cuda"
                            % (Blocks, Threads)
                        )
                elif GpuStyle == "OpenCL":
                    try:
                        InputCL = {}
                        InputCL["Iterations"] = Iterations
                        InputCL["Steps"] = 1
                        InputCL["Blocks"] = Blocks
                        InputCL["Threads"] = Threads
                        InputCL["Device"] = Devices[0]
                        InputCL["RNG"] = RNG
                        InputCL["Seeds"] = Seeds
                        InputCL["ValueType"] = ValueType
                        InputCL["IfThen"] = IfThen
                        OutputCL = MetropolisOpenCL(InputCL)
                        Inside = OutputCL["Circle"]
                        NewIterations = OutputCL["NewIterations"]
                        Duration = OutputCL["Duration"]
                    except Exception:
                        print(
                            "Problem with (%i,%i) // computations on OpenCL"
                            % (Blocks, Threads)
                        )
                Duration = numpy.append(Duration, time.time() - start)
                Rate = numpy.append(Rate, NewIterations / Duration[-1])
        else:
            if GpuStyle == "CUDA":
                try:
                    InputCU = {}
                    InputCU["Iterations"] = Iterations
                    InputCU["Steps"] = Redo
                    InputCU["Blocks"] = Blocks
                    InputCU["Threads"] = Threads
                    InputCU["Device"] = Devices[0]
                    InputCU["RNG"] = RNG
                    InputCU["Seeds"] = Seeds
                    InputCU["ValueType"] = ValueType
                    InputCU["IfThen"] = IfThen
                    OutputCU = MetropolisCuda(InputCU)
                    Inside = OutputCU["Inside"]
                    NewIterations = OutputCU["NewIterations"]
                    Duration = OutputCU["Duration"]
                    pycuda.context.pop()  # noqa: F821
                except Exception:
                    print(
                        "Problem with (%i,%i) // computations on Cuda"
                        % (Blocks, Threads)
                    )
            elif GpuStyle == "OpenCL":
                try:
                    InputCL = {}
                    InputCL["Iterations"] = Iterations
                    InputCL["Steps"] = Redo
                    InputCL["Blocks"] = Blocks
                    InputCL["Threads"] = Threads
                    InputCL["Device"] = Devices[0]
                    InputCL["RNG"] = RNG
                    InputCL["Seeds"] = Seeds
                    InputCL["ValueType"] = ValueType
                    InputCL["IfThen"] = IfThen
                    OutputCL = MetropolisOpenCL(InputCL)
                    Inside = OutputCL["Inside"]
                    NewIterations = OutputCL["NewIterations"]
                    Duration = OutputCL["Duration"]
                except Exception:
                    print(
                        "Problem with (%i,%i) // computations on OpenCL"
                        % (Blocks, Threads)
                    )
            Rate = NewIterations / Duration[-1]
            print(
                "Itops %i\nLogItops %.2f "
                % (int(Rate), numpy.log(Rate) / numpy.log(10))
            )
            print("Pi estimation %.8f" % (4.0 / NewIterations * Inside))

        avgD = numpy.append(avgD, numpy.average(Duration))
        medD = numpy.append(medD, numpy.median(Duration))
        stdD = numpy.append(stdD, numpy.std(Duration))
        minD = numpy.append(minD, numpy.min(Duration))
        maxD = numpy.append(maxD, numpy.max(Duration))
        avgR = numpy.append(avgR, numpy.average(Rate))
        medR = numpy.append(medR, numpy.median(Rate))
        stdR = numpy.append(stdR, numpy.std(Rate))
        minR = numpy.append(minR, numpy.min(Rate))
        maxR = numpy.append(maxR, numpy.max(Rate))

        print(
            "%.2f %.2f %.2f %.2f %.2f %i %i %i %i %i"
            % (
                avgD[-1],
                medD[-1],
                stdD[-1],
                minD[-1],
                maxD[-1],
                avgR[-1],
                medR[-1],
                stdR[-1],
                minR[-1],
                maxR[-1],
            )
        )

        numpy.savez(
            "Pi_%s_%s_%s_%s_%s_%s_%s_%s_%.8i_Device%i_%s_%s"
            % (
                ValueType,
                RNG,
                Alu[Devices[0]],
                GpuStyle,
                BlocksBegin,
                BlocksEnd,
                ThreadsBegin,
                ThreadsEnd,
                Iterations,
                Devices[0],
                Metrology,
                gethostname(),
            ),
            (
                ExploredBlocks,
                ExploredThreads,
                avgD,
                medD,
                stdD,
                minD,
                maxD,
                avgR,
                medR,
                stdR,
                minR,
                maxR,
            ),
        )
        ToSave = [
            ExploredBlocks,
            ExploredThreads,
            avgD,
            medD,
            stdD,
            minD,
            maxD,
            avgR,
            medR,
            stdR,
            minR,
            maxR,
        ]
        numpy.savetxt(
            "Pi_%s_%s_%s_%s_%s_%s_%s_%i_%.8i_Device%i_%s_%s"
            % (
                ValueType,
                RNG,
                Alu[Devices[0]],
                GpuStyle,
                BlocksBegin,
                BlocksEnd,
                ThreadsBegin,
                ThreadsEnd,
                Iterations,
                Devices[0],
                Metrology,
                gethostname(),
            ),
            numpy.transpose(ToSave),
            fmt="%i %i %e %e %e %e %e %i %i %i %i %i",
        )

    if Fit:
        FitAndPrint(ExploredJobs, median, Curves)  # noqa: F821, E501 # FIXME: undefined var 'median'
