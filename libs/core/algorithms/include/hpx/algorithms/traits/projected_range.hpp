//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/functional/invoke_result.hpp>
#include <hpx/iterator_support/traits/is_range.hpp>

#include <hpx/algorithms/traits/projected.hpp>

#include <iterator>
#include <type_traits>

namespace hpx { namespace parallel { namespace traits {
    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename Rng, typename Enable = void>
    struct projected_range_result_of
    {
    };

    template <typename Proj, typename Rng>
    struct projected_range_result_of<Proj, Rng,
        typename std::enable_if<hpx::traits::is_range<Rng>::value>::type>
      : detail::projected_result_of<typename std::decay<Proj>::type,
            typename hpx::traits::range_iterator<Rng>::type>
    {
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Proj, typename Rng, typename Enable = void>
    struct is_projected_range : std::false_type
    {
    };

    template <typename Proj, typename Rng>
    struct is_projected_range<Proj, Rng,
        typename std::enable_if<hpx::traits::is_range<Rng>::value>::type>
      : detail::is_projected<typename std::decay<Proj>::type,
            typename hpx::traits::range_iterator<Rng>::type>
    {
    };

    template <typename Proj, typename Rng>
    HPX_INLINE_CONSTEXPR_VARIABLE bool is_projected_range_v =
        is_projected_range<Proj, Rng>::value;

    ///////////////////////////////////////////////////////////////////////////
    template <typename Proj, typename Rng, typename Enable = void>
    struct projected_range
    {
    };

    template <typename Proj, typename Rng>
    struct projected_range<Proj, Rng,
        typename std::enable_if<hpx::traits::is_range<Rng>::value>::type>
    {
        using projector_type = typename std::decay<Proj>::type;
        using iterator_type = typename hpx::traits::range_iterator<Rng>::type;
    };
}}}    // namespace hpx::parallel::traits
