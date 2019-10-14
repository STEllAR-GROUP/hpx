..
    Copyright (c) 2019 The STE||AR-Group

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _libs_collectives:

===========
collectives
===========

The collectives module exposes a set of distributed collective operations. Those
can be used to exchange data between participating sites in a coordinated way.
At this point the module exposes the following collective primitives:

* ``all_to_all``: each participating site provides its element of the data to
  collect while all participating sites receive the data from every other site.

