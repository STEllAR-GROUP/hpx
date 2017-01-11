//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_GET_FUNCTION_ADDRESS_FEB_19_2016_0801PM)
#define HPX_TRAITS_GET_FUNCTION_ADDRESS_FEB_19_2016_0801PM

#include <hpx/config.hpp>

#include <cstddef>
#include <memory>

namespace hpx { namespace traits
{
    // By default we return the address of the object which is used to invoke
    // the trait.
    template <typename F, typename Enable = void>
    struct get_function_address
    {
        static std::size_t call(F const& f) HPX_NOEXCEPT
        {
            return reinterpret_cast<std::size_t>(std::addressof(f));
        }
    };

    // For global (and static) functions we return the function address itself
    template <typename R, typename ... Ts>
    struct get_function_address<R (*)(Ts...)>
    {
        static std::size_t call(R (*f)(Ts...)) HPX_NOEXCEPT
        {
            return reinterpret_cast<std::size_t>(f);
        }
    };

    // For member functions we return the first sizeof(void*) bytes of the
    // storage the member function pointer occupies.
    //
    // BADBAD: This invokes undefined behavior. What it does is to
    //         access the first sizeof(void*) bytes of the member
    //         function pointer. In our case this is not a problem as
    //         the value will be used to resolve a (debug) symbol only.
    //
    //         The value is never dereferenced or used in any other
    //         way. So the worst what can happen is that no matching
    //         symbol is found.
    //
    //         On some systems and for certain types of member function
    //         pointers this might even give the correct value
    //         representing the symbol corresponding to the function.
    //
    template <typename R, typename Obj, typename ... Ts>
    struct get_function_address<R (Obj::*)(Ts...)>
    {
        static std::size_t call(R (Obj::*f)(Ts...)) HPX_NOEXCEPT
        {
#if defined(__clang__)
#  pragma clang diagnostic push
#  pragma clang diagnostic ignored "-Wstrict-aliasing"
#endif

#if defined(__GNUG__) && !defined(__INTEL_COMPILER) && !defined(__NVCC__) && !defined(__CUDACC__)
#  if defined(HPX_GCC_DIAGNOSTIC_PRAGMA_CONTEXTS)
#    pragma GCC diagnostic push
#  endif
#  pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

            return reinterpret_cast<std::size_t>(*reinterpret_cast<void**>(&f));

#if defined(__GNUG__) && !defined(__INTEL_COMPILER) && !defined(__NVCC__) && !defined(__CUDACC__)
#if defined(HPX_GCC_DIAGNOSTIC_PRAGMA_CONTEXTS)
#pragma GCC diagnostic pop
#endif
#endif

#if defined(__clang__)
#  pragma clang diagnostic pop
#endif
        }
    };

    template <typename R, typename Obj, typename ... Ts>
    struct get_function_address<R (Obj::*)(Ts...) const>
    {
        static std::size_t call(R (Obj::*f)(Ts...) const) HPX_NOEXCEPT
        {
#if defined(__clang__)
#  pragma clang diagnostic push
#  pragma clang diagnostic ignored "-Wstrict-aliasing"
#endif

#if defined(__GNUG__) && !defined(__INTEL_COMPILER) && !defined(__NVCC__) && !defined(__CUDACC__)
#  if defined(HPX_GCC_DIAGNOSTIC_PRAGMA_CONTEXTS)
#    pragma GCC diagnostic push
#  endif
#  pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

            return reinterpret_cast<std::size_t>(*reinterpret_cast<void**>(&f));

#if defined(__GNUG__) && !defined(__INTEL_COMPILER) && !defined(__NVCC__) && !defined(__CUDACC__)
#if defined(HPX_GCC_DIAGNOSTIC_PRAGMA_CONTEXTS)
#pragma GCC diagnostic pop
#endif
#endif

#if defined(__clang__)
#  pragma clang diagnostic pop
#endif
        }
    };
}}

#endif
