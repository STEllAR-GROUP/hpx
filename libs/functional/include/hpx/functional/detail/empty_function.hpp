//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2014-2019 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_DETAIL_EMPTY_FUNCTION_HPP
#define HPX_UTIL_DETAIL_EMPTY_FUNCTION_HPP

#include <hpx/config.hpp>
#include <hpx/functional/detail/vtable/vtable.hpp>

namespace hpx { namespace util { namespace detail {
    ///////////////////////////////////////////////////////////////////////////
    struct empty_function
    {
    };    // must be trivial and empty

    HPX_NORETURN HPX_EXPORT void throw_bad_function_call();

    template <typename R>
    HPX_NORETURN inline R throw_bad_function_call()
    {
        throw_bad_function_call();
    }

    ///////////////////////////////////////////////////////////////////////////
    // make sure the empty table instance is initialized in time, even
    // during early startup
    template <typename Sig, bool Copyable>
    struct function_vtable;

// NOTE: nvcc (at least CUDA 9.2 and 10.1) fails with an internal compiler error
// ("there was an error in verifying the lgenfe output!") with this enabled, so
// we explicitly use the fallback.
#if defined(HPX_HAVE_CXX11_CONSTEXPR) && !defined(HPX_HAVE_CUDA)
            template <typename Sig>
            HPX_CONSTEXPR function_vtable<Sig, true> const*
            get_empty_function_vtable() noexcept
            {
                return &vtables<function_vtable<Sig, true>,
                    empty_function>::instance;
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
}}}    // namespace hpx::util::detail

#endif
