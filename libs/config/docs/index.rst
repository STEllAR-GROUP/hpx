..
    Copyright (c) 2019 The STE||AR-Group

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _libs_config:

======
Config
======

The config component contains macros that identify features of a compiler
as well as platform independent macros to control inlinining, export sets and more.

----------
Attributes
----------

 - :c:macro:`HPX_NOINLINE`
 - :c:macro:`HPX_NORETURN`
 - :c:func:`HPX_DEPRECATED`
 - :c:func:`HPX_FALLTHROUGH`
 - :c:macro:`HPX_FORCEINLINE`
 - :c:macro:`HPX_EXPORT`

------------
Branch Hints
------------

 - :c:func:`HPX_LIKELY`
 - :c:func:`HPX_UNLIKELY`

--------------
Compiler Fence
--------------

 - :c:macro:`HPX_COMPILER_FENCE`
 - :c:macro:`HPX_SMT_PAUSE`

--------------
Native TLS
--------------

 - :c:macro:`HPX_NATIVE_TLS`

-----------------
Compiler specific
-----------------

 - :c:macro:`HPX_GCC_VERSION`
 - :c:macro:`HPX_CLANG_VERSION`
 - :c:macro:`HPX_INTEL_VERSION`
 - :c:macro:`HPX_WINDOWS`
 - :c:macro:`HPX_MSVC`
 - :c:macro:`HPX_MINGW`
 - :c:macro:`HPX_NATIVE_MIC`

---------
Constexpr
---------
 - :c:macro:`HPX_CONSTEXPR`
 - :c:macro:`HPX_CONSTEXPR_OR_CONST`
 - :c:macro:`HPX_CXX14_CONSTEXPR`
 - :c:macro:`HPX_STATIC_CONSTEXPR`

---------
Debugging
---------

 - :c:macro:`HPX_DEBUG`
 - :c:macro:`HPX_BUILD_TYPE`

---------------
Emulate Deleted
---------------

 - :c:macro:`HPX_NON_COPYABLE`

--------------
Lambda Capture
--------------
 - :c:func:`HPX_CAPTURE_FORWARD`
 - :c:func:`HPX_CAPTURE_MOVE`

-------
Version
-------
 - :c:macro:`HPX_VERSION_FULL`
 - :c:macro:`HPX_VERSION_MAJOR`
 - :c:macro:`HPX_VERSION_MINOR`
 - :c:macro:`HPX_VERSION_SUBMINOR`
 - :c:macro:`HPX_VERSION_DATE`
