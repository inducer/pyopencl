all:
	g++ -c -Wall -DPYOPENCL_PRETEND_CL_VERSION=4112 -Werror -fpic wrap_cl.c
	g++ -shared -o libwrapcl.so wrap_cl.o
	cp libwrapcl.so ../../pyopencl/
	cp wrap_cl_core.h ../../pyopencl/wrap_cl_core.h
