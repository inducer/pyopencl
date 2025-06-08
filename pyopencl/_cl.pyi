__copyright__ = "Copyright (C) 2025 University of Illinois Board of Trustees"

__license__ = """
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
"""

from collections.abc import Callable, Sequence
from enum import IntEnum, auto
from typing import TYPE_CHECKING, Generic, Literal, overload

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

import pyopencl._monkeypatch
from pyopencl.typing import DTypeT, HasBufferInterface, KernelArg, SVMInnerT, WaitList

if TYPE_CHECKING:
    from pyopencl import Program

class _ErrorRecord:
    def __init__(self, msg: str, code: int, routine: str) -> None: ...
    def routine(self) -> str: ...
    def code(self) -> int: ...
    def what(self) -> str: ...
    def is_out_of_memory(self) -> bool: ...

class Error(Exception):
    code: int
    routine: str
    what: _ErrorRecord

    __str__ = pyopencl._monkeypatch.error_str

class MemoryError(Error):
    pass

class LogicError(Error):
    pass

class RuntimeError(Error):
    pass

class status_code(IntEnum):  # noqa: N801
    SUCCESS = auto()
    DEVICE_NOT_FOUND = auto()
    DEVICE_NOT_AVAILABLE = auto()
    COMPILER_NOT_AVAILABLE = auto()
    MEM_OBJECT_ALLOCATION_FAILURE = auto()
    OUT_OF_RESOURCES = auto()
    OUT_OF_HOST_MEMORY = auto()
    PROFILING_INFO_NOT_AVAILABLE = auto()
    MEM_COPY_OVERLAP = auto()
    IMAGE_FORMAT_MISMATCH = auto()
    IMAGE_FORMAT_NOT_SUPPORTED = auto()
    BUILD_PROGRAM_FAILURE = auto()
    MAP_FAILURE = auto()
    INVALID_VALUE = auto()
    INVALID_DEVICE_TYPE = auto()
    INVALID_PLATFORM = auto()
    INVALID_DEVICE = auto()
    INVALID_CONTEXT = auto()
    INVALID_QUEUE_PROPERTIES = auto()
    INVALID_COMMAND_QUEUE = auto()
    INVALID_HOST_PTR = auto()
    INVALID_MEM_OBJECT = auto()
    INVALID_IMAGE_FORMAT_DESCRIPTOR = auto()
    INVALID_IMAGE_SIZE = auto()
    INVALID_SAMPLER = auto()
    INVALID_BINARY = auto()
    INVALID_BUILD_OPTIONS = auto()
    INVALID_PROGRAM = auto()
    INVALID_PROGRAM_EXECUTABLE = auto()
    INVALID_KERNEL_NAME = auto()
    INVALID_KERNEL_DEFINITION = auto()
    INVALID_KERNEL = auto()
    INVALID_ARG_INDEX = auto()
    INVALID_ARG_VALUE = auto()
    INVALID_ARG_SIZE = auto()
    INVALID_KERNEL_ARGS = auto()
    INVALID_WORK_DIMENSION = auto()
    INVALID_WORK_GROUP_SIZE = auto()
    INVALID_WORK_ITEM_SIZE = auto()
    INVALID_GLOBAL_OFFSET = auto()
    INVALID_EVENT_WAIT_LIST = auto()
    INVALID_EVENT = auto()
    INVALID_OPERATION = auto()
    INVALID_GL_OBJECT = auto()
    INVALID_BUFFER_SIZE = auto()
    INVALID_MIP_LEVEL = auto()
    PLATFORM_NOT_FOUND_KHR = auto()
    MISALIGNED_SUB_BUFFER_OFFSET = auto()
    EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST = auto()
    INVALID_GLOBAL_WORK_SIZE = auto()
    COMPILE_PROGRAM_FAILURE = auto()
    LINKER_NOT_AVAILABLE = auto()
    LINK_PROGRAM_FAILURE = auto()
    DEVICE_PARTITION_FAILED = auto()
    KERNEL_ARG_INFO_NOT_AVAILABLE = auto()
    INVALID_IMAGE_DESCRIPTOR = auto()
    INVALID_COMPILER_OPTIONS = auto()
    INVALID_LINKER_OPTIONS = auto()
    INVALID_DEVICE_PARTITION_COUNT = auto()
    INVALID_PIPE_SIZE = auto()
    INVALID_DEVICE_QUEUE = auto()
    INVALID_SPEC_ID = auto()
    MAX_SIZE_RESTRICTION_EXCEEDED = auto()
    to_string = classmethod(pyopencl._monkeypatch.to_string)

class platform_info(IntEnum):  # noqa: N801
    PROFILE = auto()
    VERSION = auto()
    NAME = auto()
    VENDOR = auto()
    EXTENSIONS = auto()
    HOST_TIMER_RESOLUTION = auto()
    NUMERIC_VERSION = auto()
    EXTENSIONS_WITH_VERSION = auto()
    to_string = classmethod(pyopencl._monkeypatch.to_string)

class device_type(IntEnum):  # noqa: N801
    DEFAULT = auto()
    CPU = auto()
    GPU = auto()
    ACCELERATOR = auto()
    CUSTOM = auto()
    ALL = auto()
    to_string = classmethod(pyopencl._monkeypatch.to_string)

class device_info(IntEnum):  # noqa: N801
    TYPE = auto()
    VENDOR_ID = auto()
    MAX_COMPUTE_UNITS = auto()
    MAX_WORK_ITEM_DIMENSIONS = auto()
    MAX_WORK_GROUP_SIZE = auto()
    MAX_WORK_ITEM_SIZES = auto()
    PREFERRED_VECTOR_WIDTH_CHAR = auto()
    PREFERRED_VECTOR_WIDTH_SHORT = auto()
    PREFERRED_VECTOR_WIDTH_INT = auto()
    PREFERRED_VECTOR_WIDTH_LONG = auto()
    PREFERRED_VECTOR_WIDTH_FLOAT = auto()
    PREFERRED_VECTOR_WIDTH_DOUBLE = auto()
    MAX_CLOCK_FREQUENCY = auto()
    ADDRESS_BITS = auto()
    MAX_READ_IMAGE_ARGS = auto()
    MAX_WRITE_IMAGE_ARGS = auto()
    MAX_MEM_ALLOC_SIZE = auto()
    IMAGE2D_MAX_WIDTH = auto()
    IMAGE2D_MAX_HEIGHT = auto()
    IMAGE3D_MAX_WIDTH = auto()
    IMAGE3D_MAX_HEIGHT = auto()
    IMAGE3D_MAX_DEPTH = auto()
    IMAGE_SUPPORT = auto()
    MAX_PARAMETER_SIZE = auto()
    MAX_SAMPLERS = auto()
    MEM_BASE_ADDR_ALIGN = auto()
    MIN_DATA_TYPE_ALIGN_SIZE = auto()
    SINGLE_FP_CONFIG = auto()
    DOUBLE_FP_CONFIG = auto()
    HALF_FP_CONFIG = auto()
    GLOBAL_MEM_CACHE_TYPE = auto()
    GLOBAL_MEM_CACHELINE_SIZE = auto()
    GLOBAL_MEM_CACHE_SIZE = auto()
    GLOBAL_MEM_SIZE = auto()
    MAX_CONSTANT_BUFFER_SIZE = auto()
    MAX_CONSTANT_ARGS = auto()
    LOCAL_MEM_TYPE = auto()
    LOCAL_MEM_SIZE = auto()
    ERROR_CORRECTION_SUPPORT = auto()
    PROFILING_TIMER_RESOLUTION = auto()
    ENDIAN_LITTLE = auto()
    AVAILABLE = auto()
    COMPILER_AVAILABLE = auto()
    EXECUTION_CAPABILITIES = auto()
    QUEUE_PROPERTIES = auto()
    QUEUE_ON_HOST_PROPERTIES = auto()
    NAME = auto()
    VENDOR = auto()
    DRIVER_VERSION = auto()
    VERSION = auto()
    PROFILE = auto()
    EXTENSIONS = auto()
    PLATFORM = auto()
    PREFERRED_VECTOR_WIDTH_HALF = auto()
    HOST_UNIFIED_MEMORY = auto()
    NATIVE_VECTOR_WIDTH_CHAR = auto()
    NATIVE_VECTOR_WIDTH_SHORT = auto()
    NATIVE_VECTOR_WIDTH_INT = auto()
    NATIVE_VECTOR_WIDTH_LONG = auto()
    NATIVE_VECTOR_WIDTH_FLOAT = auto()
    NATIVE_VECTOR_WIDTH_DOUBLE = auto()
    NATIVE_VECTOR_WIDTH_HALF = auto()
    OPENCL_C_VERSION = auto()
    COMPUTE_CAPABILITY_MAJOR_NV = auto()
    COMPUTE_CAPABILITY_MINOR_NV = auto()
    REGISTERS_PER_BLOCK_NV = auto()
    WARP_SIZE_NV = auto()
    GPU_OVERLAP_NV = auto()
    KERNEL_EXEC_TIMEOUT_NV = auto()
    INTEGRATED_MEMORY_NV = auto()
    ATTRIBUTE_ASYNC_ENGINE_COUNT_NV = auto()
    PCI_BUS_ID_NV = auto()
    PCI_SLOT_ID_NV = auto()
    PCI_DOMAIN_ID_NV = auto()
    PROFILING_TIMER_OFFSET_AMD = auto()
    TOPOLOGY_AMD = auto()
    BOARD_NAME_AMD = auto()
    GLOBAL_FREE_MEMORY_AMD = auto()
    SIMD_PER_COMPUTE_UNIT_AMD = auto()
    SIMD_WIDTH_AMD = auto()
    SIMD_INSTRUCTION_WIDTH_AMD = auto()
    WAVEFRONT_WIDTH_AMD = auto()
    GLOBAL_MEM_CHANNELS_AMD = auto()
    GLOBAL_MEM_CHANNEL_BANKS_AMD = auto()
    GLOBAL_MEM_CHANNEL_BANK_WIDTH_AMD = auto()
    LOCAL_MEM_SIZE_PER_COMPUTE_UNIT_AMD = auto()
    LOCAL_MEM_BANKS_AMD = auto()
    THREAD_TRACE_SUPPORTED_AMD = auto()
    GFXIP_MAJOR_AMD = auto()
    GFXIP_MINOR_AMD = auto()
    AVAILABLE_ASYNC_QUEUES_AMD = auto()
    PREFERRED_WORK_GROUP_SIZE_AMD = auto()
    MAX_WORK_GROUP_SIZE_AMD = auto()
    PREFERRED_CONSTANT_BUFFER_SIZE_AMD = auto()
    PCIE_ID_AMD = auto()
    MAX_ATOMIC_COUNTERS_EXT = auto()
    LINKER_AVAILABLE = auto()
    BUILT_IN_KERNELS = auto()
    IMAGE_MAX_BUFFER_SIZE = auto()
    IMAGE_MAX_ARRAY_SIZE = auto()
    PARENT_DEVICE = auto()
    PARTITION_MAX_SUB_DEVICES = auto()
    PARTITION_PROPERTIES = auto()
    PARTITION_AFFINITY_DOMAIN = auto()
    PARTITION_TYPE = auto()
    REFERENCE_COUNT = auto()
    PREFERRED_INTEROP_USER_SYNC = auto()
    PRINTF_BUFFER_SIZE = auto()
    IMAGE_PITCH_ALIGNMENT = auto()
    IMAGE_BASE_ADDRESS_ALIGNMENT = auto()
    MAX_READ_WRITE_IMAGE_ARGS = auto()
    MAX_GLOBAL_VARIABLE_SIZE = auto()
    QUEUE_ON_DEVICE_PROPERTIES = auto()
    QUEUE_ON_DEVICE_PREFERRED_SIZE = auto()
    QUEUE_ON_DEVICE_MAX_SIZE = auto()
    MAX_ON_DEVICE_QUEUES = auto()
    MAX_ON_DEVICE_EVENTS = auto()
    SVM_CAPABILITIES = auto()
    GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE = auto()
    MAX_PIPE_ARGS = auto()
    PIPE_MAX_ACTIVE_RESERVATIONS = auto()
    PIPE_MAX_PACKET_SIZE = auto()
    PREFERRED_PLATFORM_ATOMIC_ALIGNMENT = auto()
    PREFERRED_GLOBAL_ATOMIC_ALIGNMENT = auto()
    PREFERRED_LOCAL_ATOMIC_ALIGNMENT = auto()
    IL_VERSION = auto()
    MAX_NUM_SUB_GROUPS = auto()
    SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS = auto()
    NUMERIC_VERSION = auto()
    EXTENSIONS_WITH_VERSION = auto()
    ILS_WITH_VERSION = auto()
    BUILT_IN_KERNELS_WITH_VERSION = auto()
    ATOMIC_MEMORY_CAPABILITIES = auto()
    ATOMIC_FENCE_CAPABILITIES = auto()
    NON_UNIFORM_WORK_GROUP_SUPPORT = auto()
    OPENCL_C_ALL_VERSIONS = auto()
    PREFERRED_WORK_GROUP_SIZE_MULTIPLE = auto()
    WORK_GROUP_COLLECTIVE_FUNCTIONS_SUPPORT = auto()
    GENERIC_ADDRESS_SPACE_SUPPORT = auto()
    OPENCL_C_FEATURES = auto()
    DEVICE_ENQUEUE_CAPABILITIES = auto()
    PIPE_SUPPORT = auto()
    ME_VERSION_INTEL = auto()
    EXT_MEM_PADDING_IN_BYTES_QCOM = auto()
    PAGE_SIZE_QCOM = auto()
    SPIR_VERSIONS = auto()
    SIMULTANEOUS_INTEROPS_INTEL = auto()
    NUM_SIMULTANEOUS_INTEROPS_INTEL = auto()
    to_string = classmethod(pyopencl._monkeypatch.to_string)

class device_topology_type_amd(IntEnum):  # noqa: N801
    PCIE = auto()
    to_string = classmethod(pyopencl._monkeypatch.to_string)

class device_fp_config(IntEnum):  # noqa: N801
    DENORM = auto()
    INF_NAN = auto()
    ROUND_TO_NEAREST = auto()
    ROUND_TO_ZERO = auto()
    ROUND_TO_INF = auto()
    FMA = auto()
    SOFT_FLOAT = auto()
    CORRECTLY_ROUNDED_DIVIDE_SQRT = auto()
    to_string = classmethod(pyopencl._monkeypatch.to_string)

class device_mem_cache_type(IntEnum):  # noqa: N801
    NONE = auto()
    READ_ONLY_CACHE = auto()
    READ_WRITE_CACHE = auto()
    to_string = classmethod(pyopencl._monkeypatch.to_string)

class device_local_mem_type(IntEnum):  # noqa: N801
    LOCAL = auto()
    GLOBAL = auto()
    to_string = classmethod(pyopencl._monkeypatch.to_string)

class device_exec_capabilities(IntEnum):  # noqa: N801
    KERNEL = auto()
    NATIVE_KERNEL = auto()
    to_string = classmethod(pyopencl._monkeypatch.to_string)

class device_svm_capabilities(IntEnum):  # noqa: N801
    COARSE_GRAIN_BUFFER = auto()
    FINE_GRAIN_BUFFER = auto()
    FINE_GRAIN_SYSTEM = auto()
    ATOMICS = auto()
    to_string = classmethod(pyopencl._monkeypatch.to_string)

class command_queue_properties(IntEnum):  # noqa: N801
    _zero = 0
    OUT_OF_ORDER_EXEC_MODE_ENABLE = auto()
    PROFILING_ENABLE = auto()
    ON_DEVICE = auto()
    ON_DEVICE_DEFAULT = auto()
    to_string = classmethod(pyopencl._monkeypatch.to_string)

class context_info(IntEnum):  # noqa: N801
    REFERENCE_COUNT = auto()
    DEVICES = auto()
    PROPERTIES = auto()
    NUM_DEVICES = auto()
    INTEROP_USER_SYNC = auto()
    to_string = classmethod(pyopencl._monkeypatch.to_string)

class gl_context_info(IntEnum):  # noqa: N801
    to_string = classmethod(pyopencl._monkeypatch.to_string)

class context_properties(IntEnum):  # noqa: N801
    PLATFORM = auto()
    OFFLINE_DEVICES_AMD = auto()
    to_string = classmethod(pyopencl._monkeypatch.to_string)

class command_queue_info(IntEnum):  # noqa: N801
    CONTEXT = auto()
    DEVICE = auto()
    REFERENCE_COUNT = auto()
    PROPERTIES = auto()
    PROPERTIES_ARRAY = auto()
    to_string = classmethod(pyopencl._monkeypatch.to_string)

class queue_properties(IntEnum):  # noqa: N801
    PROPERTIES = auto()
    SIZE = auto()
    DEVICE_DEFAULT = auto()
    to_string = classmethod(pyopencl._monkeypatch.to_string)

class mem_flags(IntEnum):  # noqa: N801
    READ_WRITE = auto()
    WRITE_ONLY = auto()
    READ_ONLY = auto()
    USE_HOST_PTR = auto()
    ALLOC_HOST_PTR = auto()
    COPY_HOST_PTR = auto()
    USE_PERSISTENT_MEM_AMD = auto()
    HOST_WRITE_ONLY = auto()
    HOST_READ_ONLY = auto()
    HOST_NO_ACCESS = auto()
    KERNEL_READ_AND_WRITE = auto()
    to_string = classmethod(pyopencl._monkeypatch.to_string)

class svm_mem_flags(IntEnum):  # noqa: N801
    READ_WRITE = auto()
    WRITE_ONLY = auto()
    READ_ONLY = auto()
    SVM_FINE_GRAIN_BUFFER = auto()
    SVM_ATOMICS = auto()
    to_string = classmethod(pyopencl._monkeypatch.to_string)

class channel_order(IntEnum):  # noqa: N801
    R = auto()
    A = auto()
    RG = auto()
    RA = auto()
    RGB = auto()
    RGBA = auto()
    BGRA = auto()
    INTENSITY = auto()
    LUMINANCE = auto()
    Rx = auto()
    RGx = auto()
    RGBx = auto()
    sRGB = auto()  # noqa: N815
    sRGBx = auto()  # noqa: N815
    sRGBA = auto()  # noqa: N815
    sBGRA = auto()  # noqa: N815
    ABGR = auto()
    to_string = classmethod(pyopencl._monkeypatch.to_string)

class channel_type(IntEnum):  # noqa: N801
    SNORM_INT8 = auto()
    SNORM_INT16 = auto()
    UNORM_INT8 = auto()
    UNORM_INT16 = auto()
    UNORM_SHORT_565 = auto()
    UNORM_SHORT_555 = auto()
    UNORM_INT_101010 = auto()
    SIGNED_INT8 = auto()
    SIGNED_INT16 = auto()
    SIGNED_INT32 = auto()
    UNSIGNED_INT8 = auto()
    UNSIGNED_INT16 = auto()
    UNSIGNED_INT32 = auto()
    HALF_FLOAT = auto()
    FLOAT = auto()
    UNORM_INT_101010_2 = auto()
    to_string = classmethod(pyopencl._monkeypatch.to_string)

class mem_object_type(IntEnum):  # noqa: N801
    BUFFER = auto()
    IMAGE2D = auto()
    IMAGE3D = auto()
    IMAGE2D_ARRAY = auto()
    IMAGE1D = auto()
    IMAGE1D_ARRAY = auto()
    IMAGE1D_BUFFER = auto()
    PIPE = auto()
    to_string = classmethod(pyopencl._monkeypatch.to_string)

class mem_info(IntEnum):  # noqa: N801
    TYPE = auto()
    FLAGS = auto()
    SIZE = auto()
    HOST_PTR = auto()
    MAP_COUNT = auto()
    REFERENCE_COUNT = auto()
    CONTEXT = auto()
    ASSOCIATED_MEMOBJECT = auto()
    OFFSET = auto()
    USES_SVM_POINTER = auto()
    PROPERTIES = auto()
    to_string = classmethod(pyopencl._monkeypatch.to_string)

class image_info(IntEnum):  # noqa: N801
    FORMAT = auto()
    ELEMENT_SIZE = auto()
    ROW_PITCH = auto()
    SLICE_PITCH = auto()
    WIDTH = auto()
    HEIGHT = auto()
    DEPTH = auto()
    ARRAY_SIZE = auto()
    BUFFER = auto()
    NUM_MIP_LEVELS = auto()
    NUM_SAMPLES = auto()
    to_string = classmethod(pyopencl._monkeypatch.to_string)

class pipe_info(IntEnum):  # noqa: N801
    PACKET_SIZE = auto()
    MAX_PACKETS = auto()
    PROPERTIES = auto()
    to_string = classmethod(pyopencl._monkeypatch.to_string)

class pipe_properties(IntEnum):  # noqa: N801
    PACKET_SIZE = auto()
    MAX_PACKETS = auto()
    to_string = classmethod(pyopencl._monkeypatch.to_string)

class addressing_mode(IntEnum):  # noqa: N801
    NONE = auto()
    CLAMP_TO_EDGE = auto()
    CLAMP = auto()
    REPEAT = auto()
    MIRRORED_REPEAT = auto()
    to_string = classmethod(pyopencl._monkeypatch.to_string)

class filter_mode(IntEnum):  # noqa: N801
    NEAREST = auto()
    LINEAR = auto()
    to_string = classmethod(pyopencl._monkeypatch.to_string)

class sampler_info(IntEnum):  # noqa: N801
    REFERENCE_COUNT = auto()
    CONTEXT = auto()
    NORMALIZED_COORDS = auto()
    ADDRESSING_MODE = auto()
    FILTER_MODE = auto()
    MIP_FILTER_MODE = auto()
    LOD_MIN = auto()
    LOD_MAX = auto()
    PROPERTIES = auto()
    MIP_FILTER_MODE_KHR = auto()
    LOD_MIN_KHR = auto()
    LOD_MAX_KHR = auto()
    to_string = classmethod(pyopencl._monkeypatch.to_string)

class sampler_properties(IntEnum):  # noqa: N801
    NORMALIZED_COORDS = auto()
    ADDRESSING_MODE = auto()
    FILTER_MODE = auto()
    to_string = classmethod(pyopencl._monkeypatch.to_string)

class map_flags(IntEnum):  # noqa: N801
    READ = auto()
    WRITE = auto()
    WRITE_INVALIDATE_REGION = auto()
    to_string = classmethod(pyopencl._monkeypatch.to_string)

class program_info(IntEnum):  # noqa: N801
    REFERENCE_COUNT = auto()
    CONTEXT = auto()
    NUM_DEVICES = auto()
    DEVICES = auto()
    SOURCE = auto()
    BINARY_SIZES = auto()
    BINARIES = auto()
    NUM_KERNELS = auto()
    KERNEL_NAMES = auto()
    IL = auto()
    SCOPE_GLOBAL_CTORS_PRESENT = auto()
    SCOPE_GLOBAL_DTORS_PRESENT = auto()
    to_string = classmethod(pyopencl._monkeypatch.to_string)

class program_build_info(IntEnum):  # noqa: N801
    STATUS = auto()
    OPTIONS = auto()
    LOG = auto()
    BINARY_TYPE = auto()
    GLOBAL_VARIABLE_TOTAL_SIZE = auto()
    to_string = classmethod(pyopencl._monkeypatch.to_string)

class program_binary_type(IntEnum):  # noqa: N801
    NONE = auto()
    COMPILED_OBJECT = auto()
    LIBRARY = auto()
    EXECUTABLE = auto()
    to_string = classmethod(pyopencl._monkeypatch.to_string)

class kernel_info(IntEnum):  # noqa: N801
    FUNCTION_NAME = auto()
    NUM_ARGS = auto()
    REFERENCE_COUNT = auto()
    CONTEXT = auto()
    PROGRAM = auto()
    ATTRIBUTES = auto()
    to_string = classmethod(pyopencl._monkeypatch.to_string)

class kernel_arg_info(IntEnum):  # noqa: N801
    ADDRESS_QUALIFIER = auto()
    ACCESS_QUALIFIER = auto()
    TYPE_NAME = auto()
    TYPE_QUALIFIER = auto()
    NAME = auto()
    to_string = classmethod(pyopencl._monkeypatch.to_string)

class kernel_arg_address_qualifier(IntEnum):  # noqa: N801
    GLOBAL = auto()
    LOCAL = auto()
    CONSTANT = auto()
    PRIVATE = auto()
    to_string = classmethod(pyopencl._monkeypatch.to_string)

class kernel_arg_access_qualifier(IntEnum):  # noqa: N801
    READ_ONLY = auto()
    WRITE_ONLY = auto()
    READ_WRITE = auto()
    NONE = auto()
    to_string = classmethod(pyopencl._monkeypatch.to_string)

class kernel_arg_type_qualifier(IntEnum):  # noqa: N801
    NONE = auto()
    CONST = auto()
    RESTRICT = auto()
    VOLATILE = auto()
    PIPE = auto()
    to_string = classmethod(pyopencl._monkeypatch.to_string)

class kernel_work_group_info(IntEnum):  # noqa: N801
    WORK_GROUP_SIZE = auto()
    COMPILE_WORK_GROUP_SIZE = auto()
    LOCAL_MEM_SIZE = auto()
    PREFERRED_WORK_GROUP_SIZE_MULTIPLE = auto()
    PRIVATE_MEM_SIZE = auto()
    GLOBAL_WORK_SIZE = auto()
    to_string = classmethod(pyopencl._monkeypatch.to_string)

class kernel_sub_group_info(IntEnum):  # noqa: N801
    MAX_SUB_GROUP_SIZE_FOR_NDRANGE = auto()
    SUB_GROUP_COUNT_FOR_NDRANGE = auto()
    LOCAL_SIZE_FOR_SUB_GROUP_COUNT = auto()
    MAX_NUM_SUB_GROUPS = auto()
    COMPILE_NUM_SUB_GROUPS = auto()
    to_string = classmethod(pyopencl._monkeypatch.to_string)

class event_info(IntEnum):  # noqa: N801
    COMMAND_QUEUE = auto()
    COMMAND_TYPE = auto()
    REFERENCE_COUNT = auto()
    COMMAND_EXECUTION_STATUS = auto()
    CONTEXT = auto()
    to_string = classmethod(pyopencl._monkeypatch.to_string)

class command_type(IntEnum):  # noqa: N801
    NDRANGE_KERNEL = auto()
    TASK = auto()
    NATIVE_KERNEL = auto()
    READ_BUFFER = auto()
    WRITE_BUFFER = auto()
    COPY_BUFFER = auto()
    READ_IMAGE = auto()
    WRITE_IMAGE = auto()
    COPY_IMAGE = auto()
    COPY_IMAGE_TO_BUFFER = auto()
    COPY_BUFFER_TO_IMAGE = auto()
    MAP_BUFFER = auto()
    MAP_IMAGE = auto()
    UNMAP_MEM_OBJECT = auto()
    MARKER = auto()
    ACQUIRE_GL_OBJECTS = auto()
    RELEASE_GL_OBJECTS = auto()
    READ_BUFFER_RECT = auto()
    WRITE_BUFFER_RECT = auto()
    COPY_BUFFER_RECT = auto()
    USER = auto()
    BARRIER = auto()
    MIGRATE_MEM_OBJECTS = auto()
    FILL_BUFFER = auto()
    FILL_IMAGE = auto()
    SVM_FREE = auto()
    SVM_MEMCPY = auto()
    SVM_MEMFILL = auto()
    SVM_MAP = auto()
    SVM_UNMAP = auto()
    SVM_MIGRATE_MEM = auto()
    to_string = classmethod(pyopencl._monkeypatch.to_string)

class command_execution_status(IntEnum):  # noqa: N801
    COMPLETE = auto()
    RUNNING = auto()
    SUBMITTED = auto()
    QUEUED = auto()
    to_string = classmethod(pyopencl._monkeypatch.to_string)

class profiling_info(IntEnum):  # noqa: N801
    QUEUED = auto()
    SUBMIT = auto()
    START = auto()
    END = auto()
    COMPLETE = auto()
    to_string = classmethod(pyopencl._monkeypatch.to_string)

class mem_migration_flags(IntEnum):  # noqa: N801
    HOST = auto()
    CONTENT_UNDEFINED = auto()
    to_string = classmethod(pyopencl._monkeypatch.to_string)

class device_partition_property(IntEnum):  # noqa: N801
    EQUALLY = auto()
    BY_COUNTS = auto()
    BY_COUNTS_LIST_END = auto()
    BY_AFFINITY_DOMAIN = auto()
    to_string = classmethod(pyopencl._monkeypatch.to_string)

class device_affinity_domain(IntEnum):  # noqa: N801
    NUMA = auto()
    L4_CACHE = auto()
    L3_CACHE = auto()
    L2_CACHE = auto()
    L1_CACHE = auto()
    NEXT_PARTITIONABLE = auto()
    to_string = classmethod(pyopencl._monkeypatch.to_string)

class device_atomic_capabilities(IntEnum):  # noqa: N801
    ORDER_RELAXED = auto()
    ORDER_ACQ_REL = auto()
    ORDER_SEQ_CST = auto()
    SCOPE_WORK_ITEM = auto()
    SCOPE_WORK_GROUP = auto()
    SCOPE_DEVICE = auto()
    SCOPE_ALL_DEVICES = auto()
    to_string = classmethod(pyopencl._monkeypatch.to_string)

class device_device_enqueue_capabilities(IntEnum):  # noqa: N801
    SUPPORTED = auto()
    REPLACEABLE_DEFAULT = auto()
    to_string = classmethod(pyopencl._monkeypatch.to_string)

class version_bits(IntEnum):  # noqa: N801
    MAJOR_BITS = auto()
    MINOR_BITS = auto()
    PATCH_BITS = auto()
    MAJOR_MASK = auto()
    MINOR_MASK = auto()
    PATCH_MASK = auto()
    to_string = classmethod(pyopencl._monkeypatch.to_string)

class khronos_vendor_id(IntEnum):  # noqa: N801
    CODEPLAY = auto()

    to_string = classmethod(pyopencl._monkeypatch.to_string)

class gl_object_type(IntEnum):  # noqa: N801
    BUFFER = auto()
    TEXTURE2D = auto()
    TEXTURE3D = auto()
    RENDERBUFFER = auto()

    to_string = pyopencl._monkeypatch.to_string

class gl_texture_info(IntEnum):  # noqa: N801
    TEXTURE_TARGET = auto()
    MIPMAP_LEVEL = auto()

    to_string = pyopencl._monkeypatch.to_string

class NameVersion:
    def __init__(self, version: int = 0, name: str = "") -> None: ...

    @property
    def version(self) -> int: ...

    @version.setter
    def version(self, arg: int, /) -> None: ...

    @property
    def name(self) -> str: ...

    @name.setter
    def name(self, arg: str, /) -> None: ...

class DeviceTopologyAmd:
    def __init__(self, bus: int = 0, device: int = 0, function: int = 0) -> None: ...

    @property
    def type(self) -> int: ...

    @type.setter
    def type(self, arg: int, /) -> None: ...

    @property
    def bus(self) -> int: ...

    @bus.setter
    def bus(self, arg: int, /) -> None: ...

    @property
    def device(self) -> int: ...

    @device.setter
    def device(self, arg: int, /) -> None: ...

    @property
    def function(self) -> int: ...

    @function.setter
    def function(self, arg: int, /) -> None: ...

def get_cl_header_version() -> tuple[int, int]: ...

def get_platforms() -> list[Platform]: ...

class Platform:
    def get_info(self, arg: platform_info, /) -> object: ...

    def get_devices(self, device_type: int = 4294967295) -> list[Device]: ...

    @override
    def __hash__(self) -> int: ...

    @overload
    def __eq__(self, arg: Platform, /) -> bool: ...

    @overload
    def __eq__(self, obj: object | None) -> bool: ...

    @staticmethod
    def from_int_ptr(int_ptr_value: int, retain: bool = True) -> Platform: ...

    @property
    def int_ptr(self) -> int: ...

    __repr__ = pyopencl._monkeypatch.platform_repr

    _get_cl_version = pyopencl._monkeypatch.generic_get_cl_version

    profile: str
    version: str
    name: str
    vendor: str
    extensions: str
    host_timer_resolution: int
    numeric_version: int
    extensions_with_version: Sequence[NameVersion]

class Device:
    def get_info(self, arg: device_info, /) -> object: ...

    @overload
    def __eq__(self, arg: Device, /) -> bool: ...

    @overload
    def __eq__(self, obj: object | None) -> bool: ...

    @override
    def __hash__(self) -> int: ...

    def create_sub_devices(self,
            arg: Sequence[device_partition_property],
        /) -> list[Device]: ...

    @staticmethod
    def from_int_ptr(int_ptr_value: int, retain: bool = True) -> Device: ...

    @property
    def int_ptr(self) -> int: ...

    def device_and_host_timer(self) -> tuple[int, int]: ...

    def host_timer(self) -> int: ...

    _get_cl_version = pyopencl._monkeypatch.generic_get_cl_version

    __repr__ = pyopencl._monkeypatch.device_repr

    type: device_type
    vendor_id: int
    max_compute_units: int
    max_work_item_dimensions: int
    max_work_group_size: int
    max_work_item_sizes: Sequence[int]
    preferred_vector_width_char: int
    preferred_vector_width_short: int
    preferred_vector_width_int: int
    preferred_vector_width_long: int
    preferred_vector_width_float: int
    preferred_vector_width_double: int
    max_clock_frequency: int
    address_bits: int
    max_read_image_args: int
    max_write_image_args: int
    max_mem_alloc_size: int
    image2d_max_width: int
    image2d_max_height: int
    image3d_max_width: int
    image3d_max_height: int
    image3d_max_depth: int
    image_support: bool
    max_parameter_size: int
    max_samplers: int
    mem_base_addr_align: int
    min_data_type_align_size: int
    single_fp_config: int
    double_fp_config: int
    half_fp_config: int
    global_mem_cache_type: device_mem_cache_type
    global_mem_cacheline_size: int
    global_mem_cache_size: int
    global_mem_size: int
    max_constant_buffer_size: int
    max_constant_args: int
    local_mem_type: device_local_mem_type
    local_mem_size: int
    error_correction_support: bool
    profiling_timer_resolution: int
    endian_little: bool
    available: bool
    compiler_available: bool
    execution_capabilities: int
    queue_properties: int
    queue_on_host_properties: int
    name: str
    vendor: str
    driver_version: str
    version: str
    profile: str
    extensions: str
    platform: Platform
    preferred_vector_width_half: int
    host_unified_memory: bool
    native_vector_width_char: int
    native_vector_width_short: int
    native_vector_width_int: int
    native_vector_width_long: int
    native_vector_width_float: int
    native_vector_width_double: int
    native_vector_width_half: int
    opencl_c_version: str
    compute_capability_major_nv: int
    compute_capability_minor_nv: int
    registers_per_block_nv: int
    warp_size_nv: int
    gpu_overlap_nv: bool
    kernel_exec_timeout_nv: bool
    integrated_memory_nv: bool
    attribute_async_engine_count_nv: int
    pci_bus_id_nv: int
    pci_slot_id_nv: int
    pci_domain_id_nv: int
    profiling_timer_offset_amd: int
    topology_amd: DeviceTopologyAmd
    board_name_amd: str
    global_free_memory_amd: int
    simd_per_compute_unit_amd: int
    simd_width_amd: int
    simd_instruction_width_amd: int
    wavefront_width_amd: int
    global_mem_channels_amd: int
    global_mem_channel_banks_amd: int
    global_mem_channel_bank_width_amd: int
    local_mem_size_per_compute_unit_amd: int
    local_mem_banks_amd: int
    thread_trace_supported_amd: int
    gfxip_major_amd: int
    gfxip_minor_amd: int
    available_async_queues_amd: int
    # preferred_work_group_size_amd
    # max_work_group_size_amd
    # preferred_constant_buffer_size_amd
    # pcie_id_amd
    max_atomic_counters_ext: int
    linker_available: bool
    built_in_kernels: str
    image_max_buffer_size: int
    image_max_array_size: int
    parent_device: Device
    partition_max_sub_devices: int
    partition_properties: int
    partition_affinity_domain: device_affinity_domain
    partition_type: device_partition_property
    reference_count: int
    preferred_interop_user_sync: bool
    printf_buffer_size: int
    # image_pitch_alignment
    # image_base_address_alignment
    max_read_write_image_args: int
    max_global_variable_size: int
    queue_on_device_properties: command_queue_properties
    queue_on_device_preferred_size: int
    queue_on_device_max_size: int
    max_on_device_queues: int
    max_on_device_events: int
    svm_capabilities: device_svm_capabilities
    global_variable_preferred_total_size: int
    max_pipe_args: int
    pipe_max_active_reservations: int
    pipe_max_packet_size: int
    preferred_platform_atomic_alignment: int
    preferred_global_atomic_alignment: int
    preferred_local_atomic_alignment: int
    il_version: int
    max_num_sub_groups: int
    sub_group_independent_forward_progress: bool
    numeric_version: int
    extensions_with_version: Sequence[NameVersion]
    ils_with_version: Sequence[NameVersion]
    built_in_kernels_with_version: Sequence[NameVersion]
    atomic_memory_capabilities: device_atomic_capabilities
    atomic_fence_capabilities: device_atomic_capabilities
    non_uniform_work_group_support: bool
    opencl_c_all_versions: Sequence[NameVersion]
    preferred_work_group_size_multiple: int
    work_group_collective_functions_support: bool
    generic_address_space_support: bool
    opencl_c_features: Sequence[NameVersion]
    device_enqueue_capabilities: device_device_enqueue_capabilities
    pipe_support: bool
    me_version_intel: int
    ext_mem_padding_in_bytes_qcom: int
    page_size_qcom: int
    spir_versions: str
    simultaneous_interops_intel: Sequence[Device]
    num_simultaneous_interops_intel: int

class Context:
    def __init__(self,
            devices: Sequence[Device] | None = None,
            properties: Sequence[context_properties] | None = None,
            dev_type: device_type | None = None
        ) -> None: ...

    def get_info(self, arg: context_info, /) -> object: ...

    @overload
    def __eq__(self, arg: Context, /) -> bool: ...

    @overload
    def __eq__(self, obj: object | None) -> bool: ...

    @override
    def __hash__(self) -> int: ...

    @staticmethod
    def from_int_ptr(int_ptr_value: int, retain: bool = True) -> Context: ...

    @property
    def int_ptr(self) -> int: ...

    def set_default_device_command_queue(self,
            device: Device,
            queue: CommandQueue,
        /) -> None: ...

    _get_cl_version = pyopencl._monkeypatch.context_get_cl_version

    __repr__ = pyopencl._monkeypatch.context_repr

    reference_count: int
    devices: Sequence[Device]
    properties: Sequence[int]
    num_devices: int
    # interop_user_sync:

class CommandQueue:
    def __init__(self,
             context: Context,
             device: Device | None = None,
             properties: command_queue_properties = command_queue_properties._zero
         ) -> None: ...

    def _finalize(self) -> None: ...

    def get_info(self, arg: command_queue_info, /) -> object: ...

    def flush(self) -> None: ...

    def finish(self) -> None: ...

    @overload
    def __eq__(self, arg: CommandQueue, /) -> bool: ...

    @overload
    def __eq__(self, obj: object | None) -> bool: ...

    @override
    def __hash__(self) -> int: ...

    @staticmethod
    def from_int_ptr(int_ptr_value: int, retain: bool = True) -> CommandQueue: ...

    @property
    def int_ptr(self) -> int: ...

    _get_cl_version = pyopencl._monkeypatch.command_queue_get_cl_version

    __enter__ = pyopencl._monkeypatch.command_queue_enter

    __exit__ = pyopencl._monkeypatch.command_queue_exit

    context: Context
    device: Device

    reference_count: int
    properties: command_queue_properties
    properties_array: Sequence[command_queue_properties]

class Event:
    def get_info(self, arg: event_info, /) -> object: ...

    def get_profiling_info(self, arg: int, /) -> object: ...

    def wait(self) -> None: ...

    @overload
    def __eq__(self, arg: Event, /) -> bool: ...

    @overload
    def __eq__(self, obj: object | None) -> bool: ...

    @override
    def __hash__(self) -> int: ...

    @staticmethod
    def from_int_ptr(int_ptr_value: int, retain: bool = True) -> Event: ...

    @property
    def int_ptr(self) -> int: ...

    def set_callback(self,
             arg0: int,
             arg1: Callable[[command_execution_status], None],
         /) -> None: ...

    command_queue: CommandQueue
    command_type: command_type
    reference_count: int
    command_execution_status: command_execution_status
    profile: pyopencl._monkeypatch.ProfilingInfoGetter

class NannyEvent(Event):
    def get_ward(self) -> object: ...

def wait_for_events(arg: Sequence[Event], /) -> None: ...

def _enqueue_marker_with_wait_list(
    queue: CommandQueue,
    wait_for: WaitList = None) -> Event: ...

def _enqueue_marker(queue: CommandQueue) -> Event: ...

def _enqueue_wait_for_events(
    queue: CommandQueue,
    wait_for: WaitList = None) -> None: ...

def _enqueue_barrier_with_wait_list(
    queue: CommandQueue,
    wait_for: WaitList = None) -> Event: ...

def _enqueue_barrier(queue: CommandQueue) -> None: ...

class UserEvent(Event):
    def __init__(self, context: Context) -> None: ...

    def set_status(self, arg: int, /) -> None: ...

class MemoryObjectHolder:
    def get_info(self, arg: mem_info, /) -> object: ...

    def get_host_array(self,
            shape: tuple[int, ...],
            dtype: DTypeT,
            order: Literal["C"] | Literal["F"] = "C"
        ) -> np.ndarray[tuple[int, ...], DTypeT]: ...

    @overload
    def __eq__(self, arg: MemoryObjectHolder, /) -> bool: ...

    @overload
    def __eq__(self, obj: object | None) -> bool: ...

    @override
    def __hash__(self) -> int: ...

    @property
    def int_ptr(self) -> int: ...

    type: int
    flags: int
    size: int
    # FIXME
    # host_ptr
    map_count: int
    reference_count: int
    context: Context
    # associated_memobject
    offset: int
    uses_svm_pointer: bool
    properties: int

class MemoryObject(MemoryObjectHolder):
    def release(self) -> None: ...

    @property
    def hostbuf(self) -> object: ...

    @staticmethod
    def from_int_ptr(int_ptr_value: int, retain: bool = True) -> object: ...

def enqueue_migrate_mem_objects(
        queue: CommandQueue,
        mem_objects: Sequence[MemoryObjectHolder],
        flags: int = 0,
        wait_for: WaitList = None
    ) -> Event: ...

class Buffer(MemoryObject):
    def __init__(
             self,
             context: Context,
             flags: int,
             size: int = 0,
             hostbuf: HasBufferInterface | None = None
         ) -> None: ...

    def get_sub_region(self, origin: int, size: int, flags: int = 0) -> Buffer: ...

    def __getitem__(self, arg: slice, /) -> Buffer: ...

def _enqueue_read_buffer(
        queue: CommandQueue,
        mem: MemoryObjectHolder,
        hostbuf: HasBufferInterface,
        src_offset: int = 0,
        wait_for: WaitList = None,
        is_blocking: bool = True
    ) -> Event: ...

def _enqueue_write_buffer(
        queue: CommandQueue,
        mem: MemoryObjectHolder,
        hostbuf: HasBufferInterface,
        dst_offset: int = 0,
        wait_for: WaitList = None,
        is_blocking: bool = True
    ) -> Event: ...

def _enqueue_copy_buffer(
        queue: CommandQueue,
        src: MemoryObjectHolder,
        dst: MemoryObjectHolder,
        byte_count: int = -1,
        src_offset: int = 0,
        dst_offset: int = 0,
        wait_for: WaitList = None
    ) -> Event: ...

def _enqueue_read_buffer_rect(
        queue: CommandQueue,
        mem: MemoryObjectHolder,
        hostbuf: HasBufferInterface,
        buffer_origin: tuple[int, ...],
        host_origin: tuple[int, ...],
        region: tuple[int, ...],
        buffer_pitches: tuple[int, ...] | None = None,
        host_pitches: tuple[int, ...] | None = None,
        wait_for: WaitList = None,
        is_blocking: bool = True
    ) -> Event: ...

def _enqueue_write_buffer_rect(
        queue: CommandQueue,
        mem: MemoryObjectHolder,
        hostbuf: HasBufferInterface,
        buffer_origin: tuple[int, ...],
        host_origin: tuple[int, ...],
        region: object,
        buffer_pitches: tuple[int, ...] | None = None,
        host_pitches: tuple[int, ...] | None = None,
        wait_for: WaitList = None,
        is_blocking: bool = True
    ) -> Event: ...

def _enqueue_copy_buffer_rect(
        queue: CommandQueue,
        src: MemoryObjectHolder,
        dst: MemoryObjectHolder,
        src_origin: tuple[int, ...],
        dst_origin: tuple[int, ...],
        region: object,
        src_pitches: tuple[int, ...] | None = None,
        dst_pitches: tuple[int, ...] | None = None,
        wait_for: WaitList = None
    ) -> Event: ...

def _enqueue_fill_buffer(
        queue: CommandQueue,
        mem: MemoryObjectHolder,
        pattern: object,
        offset: int,
        size: int,
        wait_for: WaitList = None
    ) -> Event: ...

def enqueue_copy_buffer_p2p_amd(
        platform: Platform,
        queue: CommandQueue,
        src: MemoryObjectHolder,
        dst: MemoryObjectHolder,
        byte_count: int | None = None,
        wait_for: WaitList = None
    ) -> Event: ...

class ImageDescriptor:
    def __init__(self) -> None: ...

    @property
    def image_type(self) -> mem_object_type: ...

    @image_type.setter
    def image_type(self, arg: mem_object_type, /) -> None: ...

    @property
    def shape(self) -> tuple[int, int, int]: ...

    @shape.setter
    def shape(self, arg: tuple[int, int, int], /) -> None: ...

    @property
    def array_size(self) -> int: ...

    @array_size.setter
    def array_size(self, arg: int, /) -> None: ...

    @property
    def pitches(self) -> object: ...

    @pitches.setter
    def pitches(self, arg: tuple[int, int], /) -> None: ...

    @property
    def num_mip_levels(self) -> int: ...

    @num_mip_levels.setter
    def num_mip_levels(self, arg: int, /) -> None: ...

    @property
    def num_samples(self) -> int: ...

    @num_samples.setter
    def num_samples(self, arg: int, /) -> None: ...

    @property
    def buffer(self) -> object: ...

    @buffer.setter
    def buffer(self, buffer: MemoryObject | None) -> None: ...

class Image(MemoryObject):
    # monkeypatched, but apparently pyright doesn't like monkeypatched __init__
    def __init__(self,
                context: Context,
                flags: mem_flags,
                format: ImageFormat,
                shape: tuple[int, ...] | None = None,
                pitches: tuple[int, ...] | None = None,

                hostbuf: HasBufferInterface | None = None,
                is_array: bool = False,
                buffer: Buffer | None = None,
                *,
                desc: ImageDescriptor | None = None,
                _through_create_image: bool = False,
            ) -> None: ...

    @overload
    @staticmethod
    def _custom_init(
            h: Image,
            context: Context,
            flags: int,
            format: ImageFormat,
            shape: tuple[int, ...] | None = None,
            pitches: tuple[int, ...] | None = None,
            hostbuf: object | None = None
        ) -> None: ...

    @overload
    @staticmethod
    def _custom_init(
            h: Image,
            context: Context,
            flags: int,
            format: ImageFormat,
            desc: ImageDescriptor,
            hostbuf: object | None = None
        ) -> None: ...

    def get_image_info(self, arg: image_info, /) -> object: ...

    format: ImageFormat
    element_size: int
    row_pitch: int
    slice_pitch: int
    width: int
    height: int
    depth: int
    array_size: int
    buffer: Buffer
    num_mip_levels: int
    num_samples: int

class ImageFormat:
    def __init__(self, arg0: int, arg1: int, /) -> None: ...

    @property
    def channel_order(self) -> int: ...

    @channel_order.setter
    def channel_order(self, arg: int, /) -> None: ...

    @property
    def channel_data_type(self) -> int: ...

    @channel_data_type.setter
    def channel_data_type(self, arg: int, /) -> None: ...

    @property
    def channel_count(self) -> int: ...

    @property
    def dtype_size(self) -> int: ...

    @property
    def itemsize(self) -> int: ...

    __repr__ = pyopencl._monkeypatch.image_format_repr

    __eq__ = pyopencl._monkeypatch.image_format_eq

    __ne__ = pyopencl._monkeypatch.image_format_ne

    __hash__ = pyopencl._monkeypatch.image_format_hash

def get_supported_image_formats(
        context: Context,
        arg1: mem_flags,
        arg2: mem_object_type,
    /) -> Sequence[ImageFormat]: ...

def _enqueue_read_image(
        queue: CommandQueue,
        mem: Image,
        origin: tuple[int, ...],
        region: tuple[int, ...],
        hostbuf: HasBufferInterface,
        row_pitch: int = 0,
        slice_pitch: int = 0,
        wait_for: WaitList = None,
        is_blocking: bool = True
    ) -> Event: ...

def _enqueue_write_image(
        queue: CommandQueue,
        mem: Image,
        origin: tuple[int, ...],
        region: tuple[int, ...],
        hostbuf: HasBufferInterface,
        row_pitch: int = 0,
        slice_pitch: int = 0,
        wait_for: WaitList = None,
        is_blocking: bool = True
    ) -> Event: ...

def _enqueue_copy_image(
        queue: CommandQueue,
        src: MemoryObjectHolder,
        dest: MemoryObjectHolder,
        src_origin: tuple[int, ...],
        dest_origin: tuple[int, ...],
        region: tuple[int, ...],
        wait_for: WaitList = None
    ) -> Event: ...

def _enqueue_copy_image_to_buffer(
        queue: CommandQueue,
        src: MemoryObjectHolder,
        dest: MemoryObjectHolder,
        origin: tuple[int, ...],
        region: tuple[int, ...],
        offset: int,
        wait_for: WaitList = None
    ) -> Event: ...

def _enqueue_copy_buffer_to_image(
        queue: CommandQueue,
        src: MemoryObjectHolder,
        dest: MemoryObjectHolder,
        offset: int,
        origin: tuple[int, ...],
        region: tuple[int, ...],
        wait_for: WaitList = None
    ) -> Event: ...

def enqueue_fill_image(
        queue: CommandQueue,
        mem: MemoryObjectHolder,
        color: HasBufferInterface,
        origin: tuple[int, ...],
        region: tuple[int, ...],
        wait_for: WaitList = None
    ) -> Event: ...

class Pipe(MemoryObject):
    def __init__(self,
            context: Context,
            flags: int,
            packet_size: int,
            max_packets: int,
            properties: Sequence[pipe_properties] = ()
        ) -> None: ...

    def get_pipe_info(self, arg: pipe_info, /) -> object: ...

    packet_size: int
    max_packets: int
    # FIXME
    # properties:

class MemoryMap:
    def release(self,
            queue: CommandQueue | None = None,
            wait_for: WaitList = None
        ) -> Event: ...

    __enter__ = pyopencl._monkeypatch.memory_map_enter

    __exit__ = pyopencl._monkeypatch.memory_map_exit

def enqueue_map_buffer(
        queue: CommandQueue,
        buf: MemoryObjectHolder,
        flags: map_flags,
        offset: int,
        shape: tuple[int, ...],
        dtype: DTypeT,
        order: Literal["C"] | Literal["F"] = "C",
        strides: tuple[int, ...] | None = None,
        wait_for: WaitList = None,
        is_blocking: bool = True
    ) -> np.ndarray[tuple[int, ...], DTypeT]: ...

def enqueue_map_image(
        queue: CommandQueue,
        img: MemoryObjectHolder,
        flags: int,
        origin: tuple[int, ...],
        region: tuple[int, ...],
        shape: tuple[int, ...],
        dtype: DTypeT,
        order: Literal["C"] | Literal["F"] = "C",
        strides: tuple[int, ...] | None = None,
        wait_for: WaitList = None,
        is_blocking: bool = True
    ) -> np.ndarray[tuple[int, ...], DTypeT]: ...

class SVMPointer:
    @property
    def svm_ptr(self) -> int: ...

    @property
    def size(self) -> int | None: ...

    @property
    def buf(self) -> NDArray[np.uint8]: ...

    map = pyopencl._monkeypatch.svmptr_map

    map_ro = pyopencl._monkeypatch.svmptr_map_ro

    map_rw = pyopencl._monkeypatch.svmptr_map_rw

    _enqueue_unmap = pyopencl._monkeypatch.svmptr__enqueue_unmap

    as_buffer = pyopencl._monkeypatch.svmptr_as_buffer

class SVM(SVMPointer, Generic[SVMInnerT]):
    def __init__(self, arg: SVMInnerT, /) -> None: ...

    @property
    def mem(self) -> SVMInnerT: ...

    map = pyopencl._monkeypatch.svm_map

    map_ro = pyopencl._monkeypatch.svm_map_ro

    map_rw = pyopencl._monkeypatch.svm_map_rw

    _enqueue_unmap = pyopencl._monkeypatch.svm__enqueue_unmap

class SVMAllocation(SVMPointer):
    def __init__(self,
             context: Context,
             size: int,
             alignment: int,
             flags: int,
             queue: CommandQueue | None = None
         ) -> None: ...

    def release(self) -> None: ...

    def enqueue_release(self,
            queue: CommandQueue | None = None,
            wait_for: WaitList = None
        ) -> Event: ...

    @overload
    def __eq__(self, arg: SVMAllocation, /) -> bool: ...

    @overload
    def __eq__(self, obj: object | None) -> bool: ...

    @override
    def __hash__(self) -> int: ...

    def bind_to_queue(self, queue: CommandQueue) -> None: ...

    def unbind_from_queue(self) -> None: ...

    @property
    def _queue(self) -> object: ...

def _enqueue_svm_memcpy(
        queue: CommandQueue,
        is_blocking: int,
        dst: SVMPointer,
        src: SVMPointer,
        wait_for: WaitList = None,
        byte_count: object | None = None
    ) -> Event: ...

def _enqueue_svm_memfill(
        queue: CommandQueue,
        dst: SVMPointer,
        pattern: object,
        byte_count: object | None = None,
        wait_for: WaitList = None
    ) -> Event: ...

def _enqueue_svm_map(
        queue: CommandQueue,
        is_blocking: int,
        flags: int,
        svm: SVMPointer,
        wait_for: WaitList = None,
        size: object | None = None
    ) -> Event: ...

def _enqueue_svm_unmap(
        queue: CommandQueue,
        svm: SVMPointer,
        wait_for: WaitList = None
    ) -> Event: ...

def _enqueue_svm_migrate_mem(
        queue: CommandQueue,
        svms: Sequence[SVMPointer],
        flags: int | None = None,
        wait_for: WaitList = None
    ) -> Event: ...

class Sampler:
    @overload
    def __init__(self, arg0: Context, arg1: Sequence[int], /) -> None: ...

    @overload
    def __init__(self, arg0: Context, arg1: bool, arg2: int, arg3: int, /) -> None: ...

    def get_info(self, arg: sampler_info, /) -> object: ...

    @overload
    def __eq__(self, arg: Sampler, /) -> bool: ...

    @overload
    def __eq__(self, obj: object | None) -> bool: ...

    @override
    def __hash__(self) -> int: ...

    @staticmethod
    def from_int_ptr(int_ptr_value: int, retain: bool = True) -> Sampler: ...

    @property
    def int_ptr(self) -> int: ...

    reference_count: int
    context: Context
    normalized_coords: bool
    addressing_mode: addressing_mode
    filter_mode: filter_mode
    mip_filter_mode: filter_mode
    lod_min: float
    lod_max: float
    properties: Sequence[sampler_properties]
    mip_filter_mode_khr: filter_mode
    lod_min_khr: float
    lod_max_khr: float

class program_kind(IntEnum):  # noqa: N801
    UNKNOWN = auto()
    SOURCE = auto()
    BINARY = auto()
    IL = auto()

    to_string = classmethod(pyopencl._monkeypatch.to_string)

def unload_platform_compiler(arg: Platform, /) -> None: ...

class Kernel:
    def __init__(self, arg0: _Program | Program, arg1: str, /) -> None: ...

    get_info = pyopencl._monkeypatch.kernel_get_info

    get_work_group_info = pyopencl._monkeypatch.kernel_get_work_group_info

    def clone(self) -> Kernel: ...

    def set_arg(self, arg0: int, arg1: KernelArg, /) -> None: ...

    def get_arg_info(self, index: int, param_name: kernel_arg_info, /) -> object: ...

    @overload
    def __eq__(self, arg: Kernel, /) -> bool: ...

    @overload
    def __eq__(self, obj: object | None) -> bool: ...

    @override
    def __hash__(self) -> int: ...

    @staticmethod
    def from_int_ptr(int_ptr_value: int, retain: bool = True) -> Kernel: ...

    @property
    def int_ptr(self) -> int: ...

    def get_sub_group_info(self,
            device: Device,
            param: int,
            input_value: Sequence[int] | int | None = None
        ) -> object: ...

    def __call__(self,
            queue: CommandQueue,
            global_work_size: tuple[int, ...],
            local_work_size: tuple[int, ...] | None,
            *args: KernelArg,
            wait_for: WaitList = None,
            g_times_l: bool = False,
            allow_empty_ndrange: bool = False,
            global_offset: tuple[int, ...] | None = None,
         ) -> Event: ...

    def set_args(self, *args: KernelArg) -> None: ...

    set_scalar_arg_dtypes = pyopencl._monkeypatch.kernel_set_arg_types

    set_arg_types = pyopencl._monkeypatch.kernel_set_arg_types

    capture_call = pyopencl._monkeypatch.kernel_capture_call

    function_name: str
    num_args: int
    reference_count: int
    context: Context
    program: _Program
    attributes: str

class LocalMemory:
    def __init__(self, size: int) -> None: ...

    @property
    def size(self) -> int: ...

def enqueue_nd_range_kernel(
        queue: CommandQueue,
        kernel: Kernel,
        global_work_size: tuple[int, ...],
        local_work_size: tuple[int, ...] | None,
        global_offset: tuple[int, ...] | None = None,
        wait_for: WaitList = None,
        *,
        g_times_l: bool = False,
        allow_empty_ndrange: bool = False
    ) -> Event: ...

def have_gl() -> bool: ...

class GLBuffer(MemoryObject):
    def __init__(self, context: Context, flags: int, bufobj: int) -> None: ...

    def get_gl_object_info(self) -> tuple[gl_object_type, int]: ...

class GLRenderBuffer(MemoryObject):
    def __init__(self, context: Context, flags: int, bufobj: int) -> None: ...

    def get_gl_object_info(self) -> tuple[gl_object_type, int]: ...

class GLTexture(Image):
    def __init__(
             self,
             context: Context,
             flags: int,
             texture_target: int,
             miplevel: int,
             texture: int,
             dims: int
         ) -> None: ...

    def get_gl_object_info(self) -> tuple[gl_object_type, int]: ...

    def get_gl_texture_info(self, arg: gl_texture_info, /) -> int: ...

def enqueue_acquire_gl_objects(
    queue: CommandQueue,
    mem_objects: object,
    wait_for: WaitList = None) -> Event: ...

def enqueue_release_gl_objects(
    queue: CommandQueue,
    mem_objects: object,
    wait_for: WaitList = None) -> Event: ...

def get_gl_context_info_khr(
    properties: object,
    param_name: int,
    platform: object | None = None) -> object: ...

def bitlog2(arg: int, /) -> int: ...

class AllocatorBase:
    def __call__(self, size: int) -> Buffer: ...

class DeferredAllocator(AllocatorBase):
    @overload
    def __init__(self, arg: Context, /) -> None: ...

    @overload
    def __init__(self, queue: Context, mem_flags: int) -> None: ...

class ImmediateAllocator(AllocatorBase):
    @overload
    def __init__(self, arg: CommandQueue, /) -> None: ...

    @overload
    def __init__(self, queue: CommandQueue, mem_flags: int) -> None: ...

class PooledBuffer(MemoryObjectHolder):
    def release(self) -> None: ...

    def bind_to_queue(self, arg: CommandQueue, /) -> None: ...

    def unbind_from_queue(self) -> None: ...

class MemoryPool:
    def __init__(self,
             allocator: AllocatorBase,
             leading_bits_in_bin_id: int = 4
         ) -> None: ...

    def allocate(self, size: int) -> PooledBuffer: ...

    def __call__(self, size: int) -> PooledBuffer: ...

    @property
    def held_blocks(self) -> int: ...

    @property
    def active_blocks(self) -> int: ...

    @property
    def managed_bytes(self) -> int: ...

    @property
    def active_bytes(self) -> int: ...

    def bin_number(self, arg: int, /) -> int: ...

    def alloc_size(self, arg: int, /) -> int: ...

    def free_held(self) -> None: ...

    def stop_holding(self) -> None: ...

class SVMAllocator:
    def __init__(self,
            context: Context,
            alignment: int = 0,
            flags: int = 1,
            queue: CommandQueue | None = None
         ) -> None: ...

    def __call__(self, size: int) -> SVMAllocation: ...

class PooledSVM(SVMPointer):
    def release(self) -> None: ...

    def enqueue_release(self) -> None: ...

    @override
    def __eq__(self, arg: PooledSVM, /) -> bool: ...

    @override
    def __hash__(self) -> int: ...

    def bind_to_queue(self, arg: CommandQueue, /) -> None: ...

    def unbind_from_queue(self) -> None: ...

class SVMPool:
    def __init__(self,
            allocator: SVMAllocator,
            leading_bits_in_bin_id: int = 4
         ) -> None: ...

    def __call__(self, size: int) -> PooledSVM: ...

    @property
    def held_blocks(self) -> int: ...

    @property
    def active_blocks(self) -> int: ...

    @property
    def managed_bytes(self) -> int: ...

    @property
    def active_bytes(self) -> int: ...

    def bin_number(self, arg: int, /) -> int: ...

    def alloc_size(self, arg: int, /) -> int: ...

    def free_held(self) -> None: ...

    def stop_holding(self) -> None: ...

class _Program:
    @overload
    def __init__(self, arg1: Context, arg2: str | bytes) -> None: ...

    @overload
    def __init__(
            self,
            arg1: Context,
            arg2: Sequence[Device],
            arg3: Sequence[bytes]
        ) -> None: ...

    @staticmethod
    def create_with_built_in_kernels(
        context: Context,
        devices: Sequence[Device],
        kernel_names: str) -> _Program: ...

    def kind(self) -> program_kind: ...

    def get_info(self, arg: program_info, /) -> object: ...

    def get_build_info(self, arg0: Device, arg1: int, /) -> object: ...

    def _build(self,
            options: bytes = b"",
            devices: Sequence[Device] | None = None
        ) -> None: ...

    def compile(self,
            options: bytes = b"",
            devices: Sequence[Device] | None = None,
            headers: object = []
        ) -> None: ...

    @staticmethod
    def link(
            context: Context,
            programs: object,
            options: bytes = b"",
            devices: Sequence[Device] | None = None
        ) -> _Program: ...

    def set_specialization_constant(self, spec_id: int, buffer: object) -> None: ...

    @overload
    def __eq__(self, arg: _Program, /) -> bool: ...

    @overload
    def __eq__(self, obj: object | None) -> bool: ...

    @override
    def __hash__(self) -> int: ...

    def all_kernels(self) -> Sequence[Kernel]: ...

    @staticmethod
    def from_int_ptr(int_ptr_value: int, retain: bool = True) -> _Program: ...

    @property
    def int_ptr(self) -> int: ...

    _get_build_logs = pyopencl._monkeypatch.program_get_build_logs

    build = pyopencl._monkeypatch.program_build

def _create_program_with_il(arg0: Context, arg1: bytes, /) -> _Program: ...

class _TestMemoryPool:
    def __init__(self, leading_bits_in_bin_id: int = 4) -> None: ...

    def allocate(self, arg: int, /) -> object: ...

    @property
    def held_blocks(self) -> int: ...

    @property
    def active_blocks(self) -> int: ...

    @property
    def managed_bytes(self) -> int: ...

    @property
    def active_bytes(self) -> int: ...

    def bin_number(self, arg: int, /) -> int: ...

    def alloc_size(self, arg: int, /) -> int: ...

    def free_held(self) -> None: ...

    def stop_holding(self) -> None: ...

    def _set_trace(self, arg: bool, /) -> None: ...

def get_apple_cgl_share_group() -> context_properties: ...
