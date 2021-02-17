.. |comparable| replace:: Instances of this class are hashable, and two
    instances of this class may be compared using *"=="* and *"!="*.
    (Hashability was added in version 2011.2.) Two objects are considered
    the same if the underlying OpenCL object is the same, as established
    by C pointer equality.

.. |buf-iface| replace:: must implement the Python buffer interface.
    (e.g. by being an :class:`numpy.ndarray`)
.. |explain-waitfor| replace:: *wait_for*
    may either be *None* or a list of :class:`pyopencl.Event` instances for
    whose completion this command waits before starting exeuction.
.. |std-enqueue-blurb| replace:: Returns a new :class:`pyopencl.Event`. |explain-waitfor|

.. |copy-depr| replace:: **Note:** This function is deprecated as of PyOpenCL 2011.1.
        Use :func:`~pyopencl.enqueue_copy` instead.

.. |glsize| replace:: *global_size* and *local_size* are tuples of identical length, with
        between one and three entries. *global_size* specifies the overall size
        of the computational grid: one work item will be launched for every
        integer point in the grid. *local_size* specifies the workgroup size,
        which must evenly divide the *global_size* in a dimension-by-dimension
        manner.  *None* may be passed for local_size, in which case the
        implementation will use an implementation-defined workgroup size.
        If *g_times_l* is *True*, the global size will be multiplied by the
        local size. (which makes the behavior more like Nvidia CUDA) In this case,
        *global_size* and *local_size* also do not have to have the same number
        of entries.

.. |empty-nd-range| replace:: *allow_empty_ndrange* is a :class:`bool` indicating
        how an empty NDRange is to be treated, where "empty" means that one or more
        entries of *global_size* or *local_size* are zero. OpenCL itself does not
        allow enqueueing kernels over empty NDRanges. Setting this flag to *True*
        enqueues a marker with a wait list (``clEnqueueMarkerWithWaitList``)
        to obtain the synchronization effects that would have resulted from
        the kernel enqueue.
        Setting *allow_empty_ndrange* to *True* requires OpenCL 1.2 or newer.
