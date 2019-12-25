// Copyright // Copyright Agustin Berge, 2019.
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_LEXICAL_CAST_DETAIL_CSTRING_WRAPPER_HPP
#define HPX_LEXICAL_CAST_DETAIL_CSTRING_WRAPPER_HPP

#include <hpx/config.hpp>

#include <cstddef>

namespace hpx { namespace util { namespace detail {

    // returns true, if T is one of the character types
    template <typename Char>
    struct cstring_wrapper
    {
        Char const* data;
        std::size_t length;

        cstring_wrapper()
          : data(nullptr)
          , length(0)
        {
        }

        cstring_wrapper(Char const* data, std::size_t length)
          : data(data)
          , length(length)
        {
        }
    };

}}}    // namespace hpx::util::detail

#endif    // HPX_LEXICAL_CAST_DETAIL_CSTRING_WRAPPER_HPP
