.. _io_support:

*************
 I/O Support 
*************

.. sectionauthor:: Bryce Adelstein-Lelbach 

HDF5 Libraries 
--------------

The ShenEOS example requires the |hdf5|_ for I/O. To run the ShenEOS
example, you need the HDF5 C++ library compiled with threadsafe support. This
is not the default in most packaged versions of HDF5. It is also suggested 
that you build HDF5 with compression support (which will require |zlib|_). You
will probably need to compile HDF5 from source. The following options should be
passed to the HDF5 configure script:::

    $ ./configure --enable-threadsafe           \
                  --enable-unsupported          \
                  --enable-cxx                  \
                  --with-pthread=/usr           \
                  --disable-hl                  \
                  --disable-static              \
                  --with-zlib=/path/to/zlib
 
