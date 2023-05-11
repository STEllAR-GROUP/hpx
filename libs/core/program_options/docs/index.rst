..
    Copyright (c) 2019 The STE||AR-Group

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _modules_program_options:

===============
program_options
===============

The module program_options is a direct fork of the |boost_program_options|_ library
(`Boost V1.70.0 <https://www.boost.org/doc/libs/1_70_0/doc/html/program_options.html>`__).
In order to be included as an |hpx| module, the Boost.Program_options library has
been moved to the ``namespace hpx::program_options``. We have also replaced all
Boost facilities the library depends on with either the equivalent facilities
from the standard library or from |hpx|. As a result, the |hpx| program_options module
is fully interface compatible with Boost.Program_options (sans the ``hpx``
namespace and the ``#include <hpx/modules/program_options.hpp>`` changes that need to be
applied to all code relying on this library).

All credit goes to Vladimir Prus, the author of the excellent Boost.Program_options
library. All bugs have been introduced by us.

See the :ref:`API reference <modules_program_options_api>` of the module for more
details.
