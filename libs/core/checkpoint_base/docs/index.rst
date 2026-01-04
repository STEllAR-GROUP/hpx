..
    Copyright (c) 2020 The STE||AR-Group

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _modules_checkpoint_base:

===============
checkpoint_base
===============

The checkpoint_base module contains lower level facilities that wrap simple
check-pointing capabilities. This module does not implement special handling
for futures or components, but simply serializes all arguments to or from
a given container.

This module exposes the ``hpx::util::save_checkpoint_data``,
``hpx::util::restore_checkpoint_data``, and ``hpx::util::prepare_checkpoint_data``
APIs. These functions encapsulate the basic serialization functionalities
necessary to save/restore a variadic list of arguments to/from a given data
container.

See the :ref:`API reference <modules_checkpoint_base_api>` of this module for more
details.

