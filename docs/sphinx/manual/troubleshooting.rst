..
    Copyright (C) 2019 Mikael Simberg

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _troubleshooting:

===============
Troubleshooting
===============

This section contains commonly encountered problems when compiling or using HPX.

.. _troubleshooting_program_options:

``Undefined reference to boost::program_options``
=================================================

Boost.ProgramOptions is not ABI compatible between all C++ versions and
compilers. Because of this you may see linker errors similar to this:

.. code-block:: text

   ...: undefined reference to `boost::program_options::operator<<(std::ostream&, boost::program_options::options_description const&)'

if you are not linking to a compatible version of Boost.ProgramOptions. We
recommend that you use ``hpx::program_options``, which is part of |hpx|, as a
replacement for ``boost::program_options`` (see :ref:`modules_program_options`).
Until you have migrated to use ``hpx::program_options`` we recommend that you
always build |boost|_ libraries and |hpx| with the same compiler and C++
standard.

.. _troubleshooting_iostreams:

``Undefined reference to hpx::cout``
====================================

You may see an linker error message that looks a bit like this:

.. code-block:: text

   hello_world.cpp:(.text+0x5aa): undefined reference to `hpx::cout'
   hello_world.cpp:(.text+0x5c3): undefined reference to `hpx::iostreams::flush'

This usually happens if you are trying to use |hpx| iostreams functionality such
as ``hpx::cout`` but are not linking against it. The iostreams functionality is
not part of the core |hpx| library, and must be linked to explicitly. Typically
this can be solved by adding ``COMPONENT_DEPENDENCIES iostreams`` to a call to
``add_hpx_library/add_hpx_executable/hpx_setup_target`` if using |cmake|. See
:ref:`creating_hpx_projects` for more details.
