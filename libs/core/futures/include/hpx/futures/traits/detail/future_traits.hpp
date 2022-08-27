//  Copyright (c) 2007-2021 Hartmut Kaiser
//  Copyright (c) 2013 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/futures/traits/future_traits.hpp>
#include <hpx/type_support/always_void.hpp>

#include <iterator>

namespace hpx { namespace lcos { namespace detail {
    ///////////////////////////////////////////////////////////////////////////
    template <typename Iter, typename Enable = void>
    struct future_iterator_traits
    {
    };

    template <typename Iterator>
    struct future_iterator_traits<Iterator,
        hpx::util::always_void_t<
            typename std::iterator_traits<Iterator>::value_type>>
    {
        using type = typename std::iterator_traits<Iterator>::value_type;
        using traits_type = hpx::traits::future_traits<type>;
    };

    template <typename Iter>
    using future_iterator_traits_t =
        typename future_iterator_traits<Iter>::type;
}}}    // namespace hpx::lcos::detail
