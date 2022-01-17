..
    Copyright (c) 2019 The STE||AR-Group

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _modules_overview:

========
Overview
========

|hpx| is organized into different sub-libraries and those in turn into modules.
The libraries and modules are independent, with clear dependencies and no
cycles. As an end-user, the use of these libraries is completely transparent. If
you use e.g. ``add_hpx_executable`` to create a target in your project you will
automatically get all modules as dependencies. See below for a list of the
available libraries and modules. Currently these are nothing more than an
internal grouping and do not affect usage. They cannot be consumed individually
at the moment.

.. toctree::
   :maxdepth: 2

   /libs/core/modules.rst
   /libs/parallelism/modules.rst
   /libs/full/modules.rst
