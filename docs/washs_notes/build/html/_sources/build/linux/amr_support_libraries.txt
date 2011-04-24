.. _linux_amr_support_libraries:

***********************
 AMR Support Libraries 
***********************

.. sectionauthor:: Bryce Lelbach (wash) <blelbach@cct.lsu.edu>

Some of the HPX |amr|_ applications written by |matt|_ can optionally use
third-party libraries for multiple-precision mathematics (|mpfr|_, |gmp|_)
and I/O (|rnpl|_). While these libraries are not required, the AMR code may
perform worse without the math libraries, and you will not get SDF output
without the RNPL library.

