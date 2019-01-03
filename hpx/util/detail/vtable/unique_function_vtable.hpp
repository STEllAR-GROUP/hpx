//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2014-2015 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_DETAIL_VTABLE_UNIQUE_FUNCTION_VTABLE_HPP
#define HPX_UTIL_DETAIL_VTABLE_UNIQUE_FUNCTION_VTABLE_HPP

#include <hpx/config.hpp>
#include <hpx/util/detail/vtable/callable_vtable.hpp>
#include <hpx/util/detail/vtable/vtable.hpp>

namespace hpx { namespace util { namespace detail
{
    ///////////////////////////////////////////////////////////////////////
    template <typename Sig>
    struct unique_function_vtable
      : vtable, callable_vtable<Sig>
    {
        template <typename T>
        HPX_CONSTEXPR unique_function_vtable(construct_vtable<T>) noexcept
          : vtable(construct_vtable<T>())
          , callable_vtable<Sig>(construct_vtable<T>())
        {}
    };
}}}

#endif
