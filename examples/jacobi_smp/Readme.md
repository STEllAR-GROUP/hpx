<!-- Copyright (c) 2013 Thomas Heller                                             -->
<!--                                                                              -->
<!-- Distributed under the Boost Software License, Version 1.0. (See accompanying -->
<!-- file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)        -->

This is a version of a Jacobi supposed to run on shared memory machines.
It based on the dataflow ideas as presented in this paper:
http://dl.acm.org/citation.cfm?id=2467126

The example consists of 2 parts, each of the part provides a jacobi smoother
implemented in HPX and OpenMP, while the OpenMP variant includes one with
static and one with dynamic scheduling policies.

The first variant smoothes a regular two-dimensional grid with a simple
5 point stencil. The parameters are the number of grid points in one dimension
and for the HPX version, a block-size parameter which determines the
granularity of the work done. The relevant executables are:
  * jacobi_hpx
  * jacobi_omp_static
  * jacobi_omp_dynamic

The second variant performs a dynamic stencil based on the neighborhood
given by a sparse matrix. The matrix input format is "Matrix Market".
An example matrix can be obtained here:
http://www.cise.ufl.edu/research/sparse/matrices/Janna/Serena.html
Other matrices from that portal work as well.
The relevant executables are:
  * jacobi_nonuniform_hpx
  * jacobi_nonuniform_omp_static
  * jacobi_nonuniform_omp_dynamic

