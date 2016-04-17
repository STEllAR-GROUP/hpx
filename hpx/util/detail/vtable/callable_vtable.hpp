//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2014 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_DETAIL_VTABLE_CALLABLE_VTABLE_HPP
#define HPX_UTIL_DETAIL_VTABLE_CALLABLE_VTABLE_HPP

#include <hpx/config.hpp>
#include <hpx/traits/get_function_address.hpp>
#include <hpx/util/detail/vtable/vtable.hpp>
#include <hpx/util/invoke.hpp>

#include <typeinfo>
#include <utility>

namespace hpx { namespace util { namespace detail
{
    struct callable_vtable_base
    {
        template <typename T>
        HPX_FORCEINLINE static std::size_t get_function_address(void** f)
        {
            return traits::get_function_address<T>::call(vtable::get<T>(f));
        }
        typedef std::size_t (*get_function_address_t)(void**);
    };

    template <typename Sig>
    struct callable_vtable;

    template <typename R, typename ...Ts>
    struct callable_vtable<R(Ts...)> : callable_vtable_base
    {
        template <typename T>
        HPX_FORCEINLINE static R invoke(void** f, Ts&&... vs)
        {
            return util::invoke<R>(
                vtable::get<T>(f), std::forward<Ts>(vs)...);
        }
        typedef R (*invoke_t)(void**, Ts&&...);
    };
}}}

#endif
