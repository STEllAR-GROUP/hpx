//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2014 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_DETAIL_EMPTY_FUNCTION_HPP
#define HPX_UTIL_DETAIL_EMPTY_FUNCTION_HPP

#include <hpx/config.hpp>
#include <hpx/util/detail/function_registration.hpp>
#include <hpx/util/detail/vtable/vtable.hpp>

namespace hpx { namespace util { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    struct empty_function {}; // must be trivial and empty

    HPX_NORETURN HPX_EXPORT void throw_bad_function_call();

    // Pseudo registration for empty functions.
    // We don't want to serialize empty functions.
    template <typename VTable>
    struct get_function_name_impl<
        VTable
      , hpx::util::detail::empty_function
    >
    {
        HPX_NORETURN static char const* call()
        {
            throw_bad_function_call();
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // make sure the empty table instance is initialized in time, even
    // during early startup
#if defined(HPX_HAVE_CXX11_CONSTEXPR)
    template <typename VTable>
    HPX_CONSTEXPR VTable const* get_empty_function_vtable() noexcept
    {
        return &vtables<VTable, empty_function>::instance;
    }
#else
    template <typename VTable>
    VTable const* get_empty_function_vtable() noexcept
    {
        static VTable const empty_vtable =
            detail::construct_vtable<empty_function>();
        return &empty_vtable;
    }
#endif
}}}

#endif
