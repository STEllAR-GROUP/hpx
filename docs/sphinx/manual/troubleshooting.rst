..
    Copyright (c) 2022 Dimitra Karatza
    Copyright (C) 2019 Mikael Simberg

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _troubleshooting:

===============
Troubleshooting
===============

Common issues
=============

This section contains commonly encountered problems when compiling or using HPX.

See also the closed issues on `GitHub <hpx_github_closed_issues_>`_ to find out
how other people resolved a similar problem. If nothing of that works, you can
also open a new issue on `GitHub <hpx_github_issues_>`_ or contact us using
one the options found in `Support for deploying and using HPX <hpx_github_support_>`_.

.. _troubleshooting_iostreams:

``HPX::iostreams_component" target not found``
----------------------------------------------

You may see a |cmake|_ error message that looks a bit like this:

.. code-block:: text

   error: `HPX::iostreams_component`` target not found

Simply ensure that |hpx| is installed with ``HPX_WITH_DISTRIBUTED_RUNTIME=ON``
to prevent encountering such error(s). This is required if you want to use
``hpx::cout``.

``Undefined reference to hpx::cout``
------------------------------------

You may see a linker error message that looks a bit like this:

.. code-block:: text

   hello_world.cpp:(.text+0x5aa): undefined reference to `hpx::cout'

This usually happens if you are trying to use |hpx| iostreams functionality such
as ``hpx::cout`` but are not linking against it. The iostreams functionality is
not part of the core |hpx| library, and must be linked to explicitly. Typically
this can be solved by adding ``COMPONENT_DEPENDENCIES iostreams`` to a call to
``add_hpx_library/add_hpx_executable/hpx_setup_target`` if using |cmake|_. See
:ref:`creating_hpx_projects` for more details.

``Build fails with ASIO error``
-------------------------------

You may see an error message that looks a bit like this:

.. code-block:: text

   Cannot open include file asio/io_context.hpp

This can be resolved by using ``-DHPX_WITH_FETCH_ASIO=ON`` to the cmake command line.

See also the corresponding closed :hpx-issue:`5404` for more information.

``Build fails with TCMalloc error``
-----------------------------------

You may see an error message that looks a bit like this:

.. code-block:: text

   Could NOT find TCMalloc (missing: Tcmalloc_LIBRARY Tcmalloc_INCLUDE_DIR)
   ERROR: HPX_WITH_MALLOC was set to tcmalloc, but tcmalloc could not be
   found.  Valid options for HPX_WITH_MALLOC are: system, tcmalloc, jemalloc,
   mimalloc, tbbmalloc, and custom

This can be resolved either by defining ``HPX_WITH_MALLOC=system`` or by installing TCMalloc.
This error occurs when users don't specify an option for ``HPX_WITH_MALLOC``; in that case,
|hpx| will be looking ``tcmalloc``, which is the default value.

Useful suggestions
==================

Reducing compilation time
-------------------------

If you want to significantly reduce compilation time, you can just use the local part of |hpx|
for parallelism by disabling the distributed functionality. Moreover, you can avoid compiling
examples. These can be done with the following flags:

.. code-block:: text

   -DHPX_WITH_NETWORKING=OFF
   -DHPX_WITH_DISTRIBUTED_RUNTIME=OFF
   -DHPX_WITH_EXAMPLES=OFF
   -DHPX_WITH_TESTS=OFF

Linking |hpx| to your application
---------------------------------

If you want to avoid installing and linking |hpx|, you can just build |hpx| and then use the
following flag on your |hpx| application CMake configuration:

.. code-block:: text

   -DHPX_DIR=<build_dir>/lib/cmake/HPX

.. note::
   For this to work you need not to specify ``-DCMAKE_INSTALL_PREFIX`` when building |hpx|.


|hpx|-application build type conformance
----------------------------------------

Your application's build type should align with the HPX build type. For example, if you specified
``-DCMAKE_BUILD_TYPE=Debug`` during the |hpx| compilation, then your application needs to be compiled
with the same flag. We recommend keeping a separate build folder for different build types and just
point accordingly to the type you want by using ``-DHPX_DIR=<build_dir>/lib/cmake/HPX``.
