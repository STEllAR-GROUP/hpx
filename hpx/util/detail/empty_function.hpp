//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2014 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_DETAIL_EMPTY_FUNCTION_HPP
#define HPX_UTIL_DETAIL_EMPTY_FUNCTION_HPP

#include <hpx/config.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/util/detail/function_registration.hpp>

namespace hpx { namespace util { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    struct empty_function {}; // must be trivial and empty

    HPX_NORETURN inline void throw_bad_function_call()
    {
        hpx::throw_exception(bad_function_call,
            "empty function object should not be used",
            "empty_function::operator()");
    }

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
}}}

#endif
