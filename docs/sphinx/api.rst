..
    Copyright (C) 2018 Mikael Simberg

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

=============
API reference
=============

|hpx| follows a versioning scheme with three numbers: ``major.minor.patch``. We
guarantee no breaking changes in the API for patch releases. Minor releases may
remove or break existing APIs, but only after a deprecation period of at least
two minor releases. In rare cases do we outright remove old and unused
functionality without a deprecation period.

We do not provide any ABI compatibility guarantees between any versions, debug
and release builds, and builds with different C++ standards.

..

   We follow |semver|_ for our API (starting from |hpx| 2.0.0). This means that
   patch releases never change the public API, neither with additions nor
   removals. Minor releases may add new functionality to the public API. Major
   releases may both remove and add functionality to the public API.

   We define the public API as any functionality in the ``hpx`` namespace,
   excluding any ``detail`` or ``experimental`` namespace within the ``hpx``
   namespace. We reserve the right to change any functionality in the ``detail``
   and ``experimental`` namespaces even in patch releases. However, any
   functionality in ``experimental`` is intended for eventual inclusion in the
   public API, and we avoid excessively breaking APIs in the ``experimental``
   namespace. In addition to the above, any macros starting with ``HPX_`` are
   part of the public API.

   We do not provide any ABI compatibility guarantees between any versions,
   debug and release builds, and builds with different C++ standards.

   Our build system provides compatibility guarantees only for |cmake| support.
   Any other build system support may change even in patch releases. The public
   API in terms of our build system are the ``HPX::`` targets provided by
   ``find_package(HPX)``.

Main |hpx| library reference
============================

.. doxygenindex::

Modules reference
=================

.. toctree::
   :maxdepth: 1

{}
