.. Copyright (c) 2012 Matt Anderson

   Distributed under the Boost Software License, Version 1.0. (See accompanying
   file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

*************
 HPX ShenEOS
*************

The Shen equation of state (EOS) tables of nuclear matter at finite temperature
and density with various electron fractions within the relativistic mean field
(RMF) theory are a set of three dimensional data arrays enabling high precision
interpolation of 19 relevant parameters required for neutron star simulations.
While these tables are currently relatively small in size (about 300 MB), it is
expected that over the next year a new set of tables ensuring higher resolution
will be published. The size of the new tables is expected to be in the range of
several GB. This will prevent loading the whole data set into main memory on
each locality. In conventional, MPI based applications the full tables would
have to be either loaded into each MPI process or a distributed partitioning
scheme would have to be implemented. Both options are either not viable or
difficult to implement.

We created an HPX component encapsulating the non-overlapping partitioning and
distribution of the Shen EOS tables to all available localities, thus reducing
the required memory footprint per locality. A special client side object
ensures the transparent dispatching of interpolation requests to the
appropriate partition corresponding to the locality holding the required part
of the tables. The client side object exposes a simple API for easy
programmability.

This code is based on the work of Christian Ott and O'Connor, authors of the
original |sheneos_tables|_. You can get the original code here:::

    svn checkout --username anon --password anon svn://stellarcollapse.org/projects/EOSdriver

Options
-------

--file : std::string : sheneos_220r_180t_50y_extT_analmu_20100322_SVNr28.h5
    The |hdf5|_ data file containing the |sheneos_tables|_.

--num-ye-points -Y : std::size_t : 20
    The number of points to interpolate on the ye axis.

--num-temp-points -T : std::size_t: 20
    The number of points to interpolate on the temp axis.

--num-rho-points -R : std::size_t : 20
    The number of points to interpolate on the rho axis.

--num-partitions : std::size_t : 32
    The number of partitions to create.

--num-workers : std::size_t : 1
    The number of worker/measurement threads to create per locality.

--seed : std::size_t : 0
    The seed for the pseudo random number generator (if 0, a seed is choosen
    based on the current system time).

Input File Format
-----------------

The input file must be a set of |sheneos_tables|_ in the |hdf5|_ (.h5) format.

Example Run
-----------

::

    sheneos_test --file my_shen_table.h5 --num-partitions 2

.. |sheneos_tables| replace:: ShenEOS tables
.. _sheneos_tables: http://stellarcollapse.org/equationofstate

.. |hdf5| replace:: HDF5
.. _hdf5: http://www.hdfgroup.org/HDF5


