all:
	g++ -c -Wall -DHAVE_GL=1 -DPYOPENCL_PRETEND_CL_VERSION=4112 -fpic wrap_cl.cpp wrap_constants.cpp bitlog.cpp
	#g++ -c -Wall -DPYOPENCL_PRETEND_CL_VERSION=4112 -fpic wrap_cl.cpp wrap_constants.cpp bitlog.cpp
	g++ -shared -o libwrapcl.so wrap_cl.o wrap_constants.o bitlog.o
	cp libwrapcl.so ../../pyopencl/
	cp wrap_cl_core.h ../../pyopencl/wrap_cl_core.h
