#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

set -e -x

mkdir -p ~/deps
cd ~/deps

# v2.3.5 is the first release that honors OCL_ICD_FILENAMES (issue 797)
git clone --branch v2.3.5 https://github.com/OCL-dev/ocl-icd
cd ocl-icd
curl -L -O https://raw.githubusercontent.com/conda-forge/ocl-icd-feedstock/e2c03e3ddb1ff86630ccf80dc7b87a81640025ea/recipe/install-headers.patch
# Context after the version-script rule moved in v2.3.5. Install the same
# Khronos headers, plus cl_function_types.h (included by cl_icd.h) and cl_layer.h.
git apply install-headers.patch || git apply <<'ENDPATCH'
diff --git a/Makefile.am b/Makefile.am
--- a/Makefile.am
+++ b/Makefile.am
@@ -55,6 +55,48 @@ libOpenCL_la_LDFLAGS += -Wl,--version-script,ocl_icd_loader_gen.map
 libOpenCL_la_DEPENDS = ocl_icd_loader_gen.map
 endif
 
+clincludedir = $(includedir)/CL
+clinclude_HEADERS = \
+    khronos-headers/CL/cl_d3d10.h \
+    khronos-headers/CL/cl_d3d11.h \
+    khronos-headers/CL/cl_dx9_media_sharing.h \
+    khronos-headers/CL/cl_dx9_media_sharing_intel.h \
+    khronos-headers/CL/cl_egl.h \
+    khronos-headers/CL/cl_ext.h \
+    khronos-headers/CL/cl_ext_intel.h \
+    khronos-headers/CL/cl_function_types.h \
+    khronos-headers/CL/cl_gl_ext.h \
+    khronos-headers/CL/cl_gl.h \
+    khronos-headers/CL/cl.h \
+    khronos-headers/CL/cl_half.h \
+    khronos-headers/CL/cl_icd.h \
+    khronos-headers/CL/cl_layer.h \
+    khronos-headers/CL/cl_platform.h \
+    khronos-headers/CL/cl_va_api_media_sharing_intel.h \
+    khronos-headers/CL/cl_version.h \
+    khronos-headers/CL/opencl.h
+
+openclincludedir = $(includedir)/OpenCL
+openclinclude_HEADERS = \
+    khronos-headers/OpenCL/cl_d3d10.h \
+    khronos-headers/OpenCL/cl_d3d11.h \
+    khronos-headers/OpenCL/cl_dx9_media_sharing.h \
+    khronos-headers/OpenCL/cl_dx9_media_sharing_intel.h \
+    khronos-headers/OpenCL/cl_egl.h \
+    khronos-headers/OpenCL/cl_ext.h \
+    khronos-headers/OpenCL/cl_ext_intel.h \
+    khronos-headers/OpenCL/cl_function_types.h \
+    khronos-headers/OpenCL/cl_gl_ext.h \
+    khronos-headers/OpenCL/cl_gl.h \
+    khronos-headers/OpenCL/cl.h \
+    khronos-headers/OpenCL/cl_half.h \
+    khronos-headers/OpenCL/cl_icd.h \
+    khronos-headers/OpenCL/cl_layer.h \
+    khronos-headers/OpenCL/cl_platform.h \
+    khronos-headers/OpenCL/cl_va_api_media_sharing_intel.h \
+    khronos-headers/OpenCL/cl_version.h \
+    khronos-headers/OpenCL/opencl.h
+
 ####################################
 # A utility to list loaded layers
 bin_PROGRAMS = cllayerinfo
ENDPATCH
curl -L -O https://github.com/isuruf/ocl-icd/commit/307f2267100a2d1383f0c4a77344b127c0857588.patch
# _open_drivers already takes num_icds. Keep the PYOPENCL_HOME lookup from
# that commit, and count those ICDs after OCL_ICD_FILENAMES.
git apply 307f2267100a2d1383f0c4a77344b127c0857588.patch || git apply <<'ENDPATCH'
diff --git a/ocl_icd_loader.c b/ocl_icd_loader.c
--- a/ocl_icd_loader.c
+++ b/ocl_icd_loader.c
@@ -893,6 +893,19 @@ static void __initClIcd( void ) {
   cl_uint num_icds = 0;
   int is_dir = 0;
   DIR *dir = NULL;
+  DIR *dir2 = NULL;
+  char pyopencl_path[4096];
+  const char* pyopencl_home = getenv("PYOPENCL_HOME");
+  pyopencl_path[0] = '\0';
+  if (pyopencl_home != NULL && pyopencl_home[0] != '\0') {
+    size_t home_len = strnlen(pyopencl_home, sizeof(pyopencl_path));
+    if (home_len > sizeof(pyopencl_path) - sizeof("/.libs")) {
+      debug(D_WARN, "PYOPENCL_HOME is too long; ignoring it");
+    } else {
+      memcpy(pyopencl_path, pyopencl_home, home_len);
+      memcpy(pyopencl_path + home_len, "/.libs", sizeof("/.libs"));
+    }
+  }
   const char* dir_path=getenv("OCL_ICD_VENDORS");
   const char* vendor_path=getenv("OPENCL_VENDOR_PATH");
   char* icd_filenames=getenv_dup("OCL_ICD_FILENAMES");
@@ -950,10 +963,16 @@ static void __initClIcd( void ) {
         debug(D_DUMP, "%s is not a directory, trying to use it as a ICD libname",
 	  dir_path);
       }
-      goto abort;
+    } else {
+      num_icds += _find_num_icds(dir);
+    }
+    if (pyopencl_path[0] != '\0') {
+      dir2 = opendir(pyopencl_path);
+      debug(D_LOG, "%s\n", pyopencl_path);
+      if (dir2 != NULL) {
+        num_icds += _find_num_icds(dir2);
+      }
     }
-
-    num_icds += _find_num_icds(dir);
     if(num_icds == 0) {
       goto abort;
     }
@@ -995,7 +1014,12 @@ static void __initClIcd( void ) {
       num_icds = _load_icd(num_icds, dir_path);
     }
   } else {
-    num_icds = _open_drivers(num_icds, dir, dir_path);
+    if (dir != NULL) {
+      num_icds = _open_drivers(num_icds, dir, dir_path);
+    }
+    if (dir2 != NULL) {
+      num_icds = _open_drivers(num_icds, dir2, pyopencl_path);
+    }
   }
   if(num_icds == 0) {
     goto abort;
@@ -1014,6 +1038,12 @@ static void __initClIcd( void ) {
   if (dir != NULL){
     closedir(dir);
   }
+  if (dir2 != NULL){
+    if (closedir(dir2) != 0) {
+      debug(D_WARN, "Cannot close ICD directory '%s': %s",
+            pyopencl_path, strerror(errno));
+    }
+  }
   __initLayers();
   return;
  abort:
@@ -1025,6 +1055,12 @@ static void __initClIcd( void ) {
   if (dir != NULL){
     closedir(dir);
   }
+  if (dir2 != NULL){
+    if (closedir(dir2) != 0) {
+      debug(D_WARN, "Cannot close ICD directory '%s': %s",
+            pyopencl_path, strerror(errno));
+    }
+  }
   if (icd_filenames){
     free(icd_filenames);
   }
ENDPATCH
git grep -q OCL_ICD_FILENAMES
autoreconf -i
chmod +x configure
./configure --prefix=/usr
make -j4
make install

# Bundle license files
echo "PyOpenCL wheel includes ocl-icd which is licensed as below" >> ${SCRIPT_DIR}/../LICENSE
cat ~/deps/ocl-icd/COPYING >> ${SCRIPT_DIR}/../LICENSE
