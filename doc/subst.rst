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
        Use :func:`enqueue_copy` instead.
