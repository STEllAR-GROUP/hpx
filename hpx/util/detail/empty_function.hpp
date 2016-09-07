//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2014 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_DETAIL_EMPTY_FUNCTION_HPP
#define HPX_UTIL_DETAIL_EMPTY_FUNCTION_HPP

#include <hpx/throw_exception.hpp>
#include <hpx/util/detail/function_registration.hpp>

namespace hpx { namespace util { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Sig>
    struct empty_function; // must be trivial and empty

    template <typename R, typename ...Ts>
    struct empty_function<R(Ts...)>
    {
        R operator()(Ts...) const
        {
            hpx::throw_exception(bad_function_call,
                "empty function object should not be used",
                "empty_function::operator()");
        }
    };

    // Pseudo registration for empty functions.
    // We don't want to serialize empty functions.
    template <typename VTable, typename Sig>
    struct get_function_name_impl<
        VTable
      , hpx::util::detail::empty_function<Sig>
    >
    {
        static char const* call()
        {
            hpx::throw_exception(bad_function_call,
                "empty function object should not be used",
                "get_function_name<empty_function>");
            return "";
        }
    };
}}}

#endif
