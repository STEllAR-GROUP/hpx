//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2014-2019 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_DETAIL_EMPTY_FUNCTION_HPP
#define HPX_UTIL_DETAIL_EMPTY_FUNCTION_HPP

#include <hpx/config.hpp>
#include <hpx/util/detail/vtable/vtable.hpp>

namespace hpx { namespace util { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    struct empty_function {}; // must be trivial and empty

    HPX_NORETURN HPX_EXPORT void throw_bad_function_call();

    ///////////////////////////////////////////////////////////////////////////
    // make sure the empty table instance is initialized in time, even
    // during early startup
    template <typename Sig, bool Copyable>
    struct function_vtable;

#if defined(HPX_HAVE_CXX11_CONSTEXPR)
    template <typename Sig>
    HPX_CONSTEXPR function_vtable<Sig, true> const*
    get_empty_function_vtable() noexcept
    {
        return &vtables<function_vtable<Sig, true>, empty_function>::instance;
    }
#else
    template <typename Sig>
    function_vtable<Sig, true> const*
    get_empty_function_vtable() noexcept
    {
        static function_vtable<Sig, true> const empty_vtable =
            detail::construct_vtable<empty_function>();
        return &empty_vtable;
    }
#endif
}}}

#endif
