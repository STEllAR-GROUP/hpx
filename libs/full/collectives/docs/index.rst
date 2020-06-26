..
    Copyright (c) 2019 The STE||AR-Group

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _modules_collectives:

===========
collectives
===========

The collectives module exposes a set of distributed collective operations. Those
can be used to exchange data between participating sites in a coordinated way.
At this point the module exposes the following collective primitives:

* :cpp:class:`hpx::collectives::all_reduce`: performs a reduction on data from
  each participating site to each participating site.
* :cpp:class:`hpx::collectives::all_to_all`: each participating site provides its
  element of the data to collect while all participating sites receive the data
  from every other site.
* :cpp:class:`hpx::lcos::barrier`: distributed barrier.
* :cpp:func:`hpx::lcos::broadcast`: performs a given action on all given global
  identifiers.
* :cpp:func:`hpx::lcos::fold`: performs a fold with a given action on all given
  global identifiers.
* :cpp:func:`hpx::lcos::gather`: gathers values from all participating sites.
* :cpp:class:`hpx::lcos::latch`: distributed latch.
* :cpp:func:`hpx::lcos::reduce`: performs a reduction on data from each
  participating site to a root site.
* :cpp:class:`hpx::lcos::spmd_block`: performs the same operation on a local
  image while providing handles to the other images.

See the :ref:`API reference <modules_collectives_api>` of the module for more
details.
