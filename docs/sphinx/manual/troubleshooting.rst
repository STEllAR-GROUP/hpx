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

See also the closed issues on `GitHub <hpx_github_closed_issues_>`_ to find out 
how other people resolved a similar problem. If nothing of that works, you can 
also open a new issue on `GitHub <hpx_github_issues_>`_ or contact us using
one the options found in `Support for deploying and using HPX <hpx_github_support_>`_.

.. _troubleshooting_iostreams:

``Undefined reference to hpx::cout``
====================================

You may see a linker error message that looks a bit like this:

.. code-block:: text

   hello_world.cpp:(.text+0x5aa): undefined reference to `hpx::cout'

This usually happens if you are trying to use |hpx| iostreams functionality such
as ``hpx::cout`` but are not linking against it. The iostreams functionality is
not part of the core |hpx| library, and must be linked to explicitly. Typically
this can be solved by adding ``COMPONENT_DEPENDENCIES iostreams`` to a call to
``add_hpx_library/add_hpx_executable/hpx_setup_target`` if using |cmake|. See
:ref:`creating_hpx_projects` for more details.

``Fail compiling for examples with hpx::future and co_await``
=============================================================

You may see an error message that looks a bit like this:

.. code-block:: text

   error: coroutines require a traits template; cannot find 'std::coroutine_traits'

This can be resolved by using ``-DHPX_WITH_CXX_STANDARD=20`` to the cmake command line.
Note that a compiler that supports C++20 is needed.

See also the corresponding closed :hpx-issue:`5784`.

``Build fails with ASIO error``
===============================

You may see an error message that looks a bit like this:

.. code-block:: text

   Cannot open include file asio/io_context.hpp

This can be resolved by using ``-DHPX_WITH_FETCH_ASIO=ON`` to the cmake command line.

See also the corresponding closed :hpx-issue:`5404` for more information.


