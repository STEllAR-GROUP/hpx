..
    Copyright (c) 2020 The STE||AR-Group

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _modules_properties:

==========
properties
==========

This module implements the ``prefer`` customization point for properties in
terms of |p2220|_. This differs from |p1393|_ in that it relies fully on
``tag_invoke`` overloads and fewer base customization points. Actual properties
are defined in modules. All functionality is experimental and can be accessed
through the ``hpx::experimental`` namespace.

See the :ref:`API reference <modules_properties_api>` of this module for more
details.

