//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2014-2015 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_DETAIL_VTABLE_CALLABLE_VTABLE_HPP
#define HPX_UTIL_DETAIL_VTABLE_CALLABLE_VTABLE_HPP

#include <hpx/config.hpp>
#include <hpx/traits/get_function_address.hpp>
#include <hpx/traits/get_function_annotation.hpp>
#include <hpx/util/detail/vtable/vtable.hpp>
#include <hpx/util/invoke.hpp>

#include <cstddef>
#include <utility>

namespace hpx { namespace util { namespace detail
{
    struct callable_vtable_base
    {
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
        template <typename T>
        HPX_FORCEINLINE static std::size_t _get_function_address(void** f)
        {
            return traits::get_function_address<T>::call(vtable::get<T>(f));
        }
        std::size_t (*get_function_address)(void**);

        template <typename T>
        HPX_FORCEINLINE static char const* _get_function_annotation(void** f)
        {
            return traits::get_function_annotation<T>::call(vtable::get<T>(f));
        }
        char const* (*get_function_annotation)(void**);

#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
        template <typename T>
        HPX_FORCEINLINE static char const* _get_function_annotation_itt(void** f)
        {
            return traits::get_function_annotation_itt<T>::call(vtable::get<T>(f));
        }
        char const* (*get_function_annotation_itt)(void**);
#endif
#endif

        template <typename T>
        HPX_CONSTEXPR callable_vtable_base(construct_vtable<T>) noexcept
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
          : get_function_address(
                &callable_vtable_base::template _get_function_address<T>)
          , get_function_annotation(
                &callable_vtable_base::template _get_function_annotation<T>)
#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
          , get_function_annotation_itt(
                &callable_vtable_base::template _get_function_annotation_itt<T>)
#endif
#endif
        {}
    };

    template <typename Sig>
    struct callable_vtable;

    template <typename R, typename ...Ts>
    struct callable_vtable<R(Ts...)> : callable_vtable_base
    {
        template <typename T>
        HPX_FORCEINLINE static R _invoke(void** f, Ts&&... vs)
        {
            return util::invoke_r<R>(
                vtable::get<T>(f), std::forward<Ts>(vs)...);
        }
        R (*invoke)(void**, Ts&&...);

        template <typename T>
        HPX_CONSTEXPR callable_vtable(construct_vtable<T>) noexcept
          : callable_vtable_base(construct_vtable<T>())
          , invoke(&callable_vtable::template _invoke<T>)
        {}
    };
}}}

#endif
