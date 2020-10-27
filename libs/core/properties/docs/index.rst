..
    Copyright (c) 2020 The STE||AR-Group

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _modules_properties:

==========
properties
==========

This module implements the ``require_concept``, ``require``, ``prefer``, and
``query`` customization points for setting and getting properties as defined in
in |p1393|_. The implementation in this module deviates from the proposal in the
use of ``tag_invoke`` for the customization points. All functionality is
experimental and can be accessed in the ``hpx::experimental`` namespace.

See the :ref:`API reference <modules_properties_api>` of this module for more
details.

