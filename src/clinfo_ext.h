/* Include OpenCL header, and define OpenCL extensions, since what is and is not
 * available in the official headers is very system-dependent */

#ifndef _EXT_H
#define _EXT_H

#if (defined(__APPLE__) && !defined(PYOPENCL_APPLE_USE_CL_H))
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

/* These two defines were introduced in the 1.2 headers
 * on 2012-11-30, so earlier versions don't have them
 * (e.g. Debian wheezy)
 */

#ifndef CL_DEVICE_IMAGE_PITCH_ALIGNMENT
#define CL_DEVICE_IMAGE_PITCH_ALIGNMENT                 0x104A
#define CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT          0x104B
#endif

/*
 * Extensions
 */

/* cl_khr_icd */
#define CL_PLATFORM_ICD_SUFFIX_KHR			0x0920
#define CL_PLATFORM_NOT_FOUND_KHR			-1001


/* cl_khr_fp64 */
#define CL_DEVICE_DOUBLE_FP_CONFIG			0x1032

/* cl_khr_fp16 */
#define CL_DEVICE_HALF_FP_CONFIG			0x1033

/* cl_khr_terminate_context */
#define CL_DEVICE_TERMINATE_CAPABILITY_KHR		0x200F

/* cl_nv_device_attribute_query */
#define CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV		0x4000
#define CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV		0x4001
#define CL_DEVICE_REGISTERS_PER_BLOCK_NV		0x4002
#define CL_DEVICE_WARP_SIZE_NV				0x4003
#define CL_DEVICE_GPU_OVERLAP_NV			0x4004
#define CL_DEVICE_KERNEL_EXEC_TIMEOUT_NV		0x4005
#define CL_DEVICE_INTEGRATED_MEMORY_NV			0x4006
#define CL_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT_NV	0x4007
#define CL_DEVICE_PCI_BUS_ID_NV				0x4008
#define CL_DEVICE_PCI_SLOT_ID_NV			0x4009

/* cl_ext_atomic_counters_{32,64} */
#define CL_DEVICE_MAX_ATOMIC_COUNTERS_EXT		0x4032

/* cl_amd_device_attribute_query */
#define CL_DEVICE_PROFILING_TIMER_OFFSET_AMD		0x4036
#define CL_DEVICE_TOPOLOGY_AMD				0x4037
#define CL_DEVICE_BOARD_NAME_AMD			0x4038
#define CL_DEVICE_GLOBAL_FREE_MEMORY_AMD		0x4039
#define CL_DEVICE_SIMD_PER_COMPUTE_UNIT_AMD		0x4040
#define CL_DEVICE_SIMD_WIDTH_AMD			0x4041
#define CL_DEVICE_SIMD_INSTRUCTION_WIDTH_AMD		0x4042
#define CL_DEVICE_WAVEFRONT_WIDTH_AMD			0x4043
#define CL_DEVICE_GLOBAL_MEM_CHANNELS_AMD		0x4044
#define CL_DEVICE_GLOBAL_MEM_CHANNEL_BANKS_AMD		0x4045
#define CL_DEVICE_GLOBAL_MEM_CHANNEL_BANK_WIDTH_AMD	0x4046
#define CL_DEVICE_LOCAL_MEM_SIZE_PER_COMPUTE_UNIT_AMD	0x4047
#define CL_DEVICE_LOCAL_MEM_BANKS_AMD			0x4048
#define CL_DEVICE_THREAD_TRACE_SUPPORTED_AMD		0x4049
#define CL_DEVICE_GFXIP_MAJOR_AMD			0x404A
#define CL_DEVICE_GFXIP_MINOR_AMD			0x404B
#define CL_DEVICE_AVAILABLE_ASYNC_QUEUES_AMD		0x404C
#define CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_AMD         0x4030
#define CL_DEVICE_MAX_WORK_GROUP_SIZE_AMD               0x4031
#define CL_DEVICE_PREFERRED_CONSTANT_BUFFER_SIZE_AMD    0x4033
#define CL_DEVICE_PCIE_ID_AMD                           0x4034

#ifndef CL_DEVICE_TOPOLOGY_TYPE_PCIE_AMD
#define CL_DEVICE_TOPOLOGY_TYPE_PCIE_AMD		1

typedef union
{
	struct { cl_uint type; cl_uint data[5]; } raw;
	struct { cl_uint type; cl_char unused[17]; cl_char bus; cl_char device; cl_char function; } pcie;
} cl_device_topology_amd;
#endif

/* cl_amd_offline_devices */
#define CL_CONTEXT_OFFLINE_DEVICES_AMD			0x403F

/* cl_ext_device_fission */
#define cl_ext_device_fission				1

typedef cl_ulong  cl_device_partition_property_ext;

#define CL_DEVICE_PARTITION_EQUALLY_EXT			0x4050
#define CL_DEVICE_PARTITION_BY_COUNTS_EXT		0x4051
#define CL_DEVICE_PARTITION_BY_NAMES_EXT		0x4052
#define CL_DEVICE_PARTITION_BY_NAMES_INTEL		0x4052 /* cl_intel_device_partition_by_names */
#define CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN_EXT	0x4053

#define CL_DEVICE_PARENT_DEVICE_EXT			0x4054
#define CL_DEVICE_PARTITION_TYPES_EXT			0x4055
#define CL_DEVICE_AFFINITY_DOMAINS_EXT			0x4056
#define CL_DEVICE_REFERENCE_COUNT_EXT			0x4057
#define CL_DEVICE_PARTITION_STYLE_EXT			0x4058

#define CL_AFFINITY_DOMAIN_L1_CACHE_EXT			0x1
#define CL_AFFINITY_DOMAIN_L2_CACHE_EXT			0x2
#define CL_AFFINITY_DOMAIN_L3_CACHE_EXT			0x3
#define CL_AFFINITY_DOMAIN_L4_CACHE_EXT			0x4
#define CL_AFFINITY_DOMAIN_NUMA_EXT			0x10
#define CL_AFFINITY_DOMAIN_NEXT_FISSIONABLE_EXT		0x100

/* cl_intel_advanced_motion_estimation */
#define CL_DEVICE_ME_VERSION_INTEL			0x407E

/* cl_qcom_ext_host_ptr */
#define CL_DEVICE_EXT_MEM_PADDING_IN_BYTES_QCOM		0x40A0
#define CL_DEVICE_PAGE_SIZE_QCOM			0x40A1

/* cl_khr_spir */
#define CL_DEVICE_SPIR_VERSIONS				0x40E0

/* cl_altera_device_temperature */
#define CL_DEVICE_CORE_TEMPERATURE_ALTERA		0x40F3

/* cl_intel_simultaneous_sharing */
#define CL_DEVICE_SIMULTANEOUS_INTEROPS_INTEL		0x4104
#define CL_DEVICE_NUM_SIMULTANEOUS_INTEROPS_INTEL	0x4105

#endif
