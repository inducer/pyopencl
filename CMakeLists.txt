cmake_minimum_required(VERSION 2.6)
project (pyopencl)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/submodules/pocl/cmake")

set(KERNELLIB_HOST_CPU_VARIANTS "distro")
 
add_subdirectory(submodules/llvm)
add_subdirectory(submodules/pocl)
