//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2014-2019 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_DETAIL_VTABLE_FUNCTION_VTABLE_HPP
#define HPX_UTIL_DETAIL_VTABLE_FUNCTION_VTABLE_HPP

#include <hpx/config.hpp>
#include <hpx/util/detail/vtable/callable_vtable.hpp>
#include <hpx/util/detail/vtable/copyable_vtable.hpp>
#include <hpx/util/detail/vtable/vtable.hpp>

namespace hpx { namespace util { namespace detail
{
    ///////////////////////////////////////////////////////////////////////
    template <typename Sig, bool Copyable = true>
    struct function_vtable;

    template <typename Sig>
    struct function_vtable<Sig, /*Copyable*/false>
      : vtable, callable_vtable<Sig>
    {
        template <typename T>
        HPX_CONSTEXPR function_vtable(construct_vtable<T>) noexcept
          : vtable(construct_vtable<T>())
          , callable_vtable<Sig>(construct_vtable<T>())
        {}
    };

    template <typename Sig>
    struct function_vtable<Sig, /*Copyable*/true>
      : function_vtable<Sig, false>, copyable_vtable
    {
        template <typename T>
        HPX_CONSTEXPR function_vtable(construct_vtable<T>) noexcept
          : function_vtable<Sig, false>(construct_vtable<T>())
          , copyable_vtable(construct_vtable<T>())
        {}
    };

    template <typename Sig>
    using unique_function_vtable = function_vtable<Sig, false>;
}}}

#endif
