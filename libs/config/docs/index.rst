..
    Copyright (c) 2019 The STE||AR-Group

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _libs_config:

===========
config
===========

The config component is made of various headers. The contents are described below

-----------------------------
`<hpx/config/attributes.hpp>`
-----------------------------

.. c:macro:: HPX_NOINLINE
This macro can be used as an attribute to a function to not inline it

.. c:macro:: HPX_NORETURN
This macro can be used as an attribute to a function to mark it that it
does not return.

.. c:function:: HPX_DEPRECATED(x)
This can be used to mark an entity as deprecated. The argument `x` specifies a custom
message that is included in the compiler warning.
For more details see
`https://en.cppreference.com/w/cpp/language/attributes/deprecated>`_.

.. c:macro:: HPX_FALLTHROUGH
Indicates that the fall through from the previous case label is intentional and
should not be diagnosed by a compiler that warns on fallthrough.
For more details see
`https://en.cppreference.com/w/cpp/language/attributes/fallthrough>`_.

-------------------------------
`<hpx/config/branch_hints.hpp>`
-------------------------------

.. c:function:: HPX_LIKELY(expr)
Hint at the compiler that `expr` is likely to be true

.. c:function:: HPX_UNLIKELY(expr)
Hint at the compiler that `expr` is likely to be false


---------------------------------
`<hpx/config/compiler_fence.hpp>`
---------------------------------

.. c:macro:: HPX_COMPILER_FENCE
Generates assembly that serves as a fence to the compiler/CPU to disable optimization.
Usually implemented in the form of a memory barrier

.. c:macro:: HPX_SMT_PAUSE
Generates assembly the executes a "pause" instruction. Useful in spinning loops.

--------------------------------------
`<hpx/config/compiler_native_tls.hpp>`
--------------------------------------

.. c:macro:: HPX_NATIVE_TLS
This macro is replaced with the compiler specific keyword/attribute to mark a
variabel as thread local.
For more details see
`https://en.cppreference.com/w/cpp/keyword/thread_local`_.

--------------------------------------
`<hpx/config/compiler_specific.hpp>`
--------------------------------------

.. c:macro:: HPX_GCC_VERSION
Returns the GCC version HPX is compiled with. Only set if compiled with GCC.

.. c:macro:: HPX_CLANG_VERSION
Returns the Clang version HPX is compiled with. Only set if compiled with Clang.

.. c:macro:: HPX_INTEL_VERSION
Returns the Intel Compiler version HPX is compiled with. Only set if compiled
with the Intel Compiler.

.. c:macro:: HPX_WINDOWS
This macro is set, if the compilation is for Windows

.. c:macro:: HPX_MSVC
This macro is set, if the compilation is with MSVC

.. c:macro:: HPX_MINGW
This macro is set, if the compilation is with Mingw

.. c:macro:: HPX_NATIVE_MIC
This macro is set, if the compilation is for Intel Knights Landing

--------------------------------------
`<hpx/config/constexpr.hpp>`
--------------------------------------

.. c:macro:: HPX_CONSTEXPR
This macro evaluates to `constexpr` if the compiler supports it

.. c:macro:: HPX_CONSTEXPR_OR_CONST
This macro evaluates to `constexpr` if the compiler supports it, `const` otherwise

.. c:macro:: HPX_CXX14_CONSTEXPR
This macro evaluates to `constexpr` if the compiler supports C++14 constexpr

.. c:macro:: HPX_STATIC_CONSTEXPR
This macro evaluates to `static :c:macro:HPX_CONSTEXPR_OR_CONST`

--------------------------------------
`<hpx/config/debug.hpp>`
--------------------------------------

.. c:macro:: HPX_DEBUG
This macro is defined if HPX is compiled in debug mode

.. c:macro:: HPX_BUILD_TYPE
Evaluates to `debug` if compiled in debug mode, `release` otherwise.

--------------------------------------
`<hpx/config/emulate_deleted.hpp>`
--------------------------------------

.. c:macro:: HPX_NON_COPYABLE
This macro should be used to mark a class as non-copyable and non-movable

--------------------------------------
`<hpx/config/emulate_deleted.hpp>`
--------------------------------------

.. c:macro:: HPX_EXPORT
This macro should be used to mark a class or function to be exported from HPX or
imported if it is consumed.

--------------------------------------
`<hpx/config/forceinline.hpp>`
--------------------------------------

.. c:macro:: HPX_FORCEINLINE
Marks a function to be forced inline

--------------------------------------
`<hpx/config/lambda_capture.hpp>`
--------------------------------------

.. c:function:: HPX_CAPTURE_FORWARD(var)
Evaluates to `var = std::forward<decltype(var)>(var)` if the compiler supports C++14
Lambdas. Defaults to `var`.

.. c:function:: HPX_CAPTURE_MOVE(var)
Evaluates to `var = std::move(var)` if the compiler supports C++14
Lambdas. Defaults to `var`.

--------------------------------------
`<hpx/config/version.hpp>`
--------------------------------------

.. c:macro:: HPX_VERSION_FULL
Evaluates to the HPX version:
`HPX_VERSION_FULL & 0xFF0000 == :c:macro:HPX_VERSION_MAJOR`
`HPX_VERSION_FULL & 0x00FF00 == :c:macro:HPX_VERSION_MINOR`
`HPX_VERSION_FULL & 0x0000FF == :c:macro:HPX_VERSION_SUBMINOR`

.. c:macro:: HPX_VERSION_MAJOR
Evaluates to the major version of HPX

.. c:macro:: HPX_VERSION_MINOR
Evaluates to the minow version of HPX

.. c:macro:: HPX_VERSION_SUBMINOR
Evaluates to the subminor version of HPX

.. c:macro:: HPX_VERSION_DATE
Evaluates to the release date of this HPX version
