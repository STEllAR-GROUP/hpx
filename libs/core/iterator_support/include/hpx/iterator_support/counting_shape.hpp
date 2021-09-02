//  Copyright (c) 2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/local/config.hpp>
#include <hpx/iterator_support/counting_iterator.hpp>
#include <hpx/iterator_support/iterator_range.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/iterator_support/traits/is_range.hpp>

namespace hpx { namespace util { namespace detail {

    ///////////////////////////////////////////////////////////////////////////
    template <typename Incrementable>
    using counting_shape_type =
        hpx::util::iterator_range<hpx::util::counting_iterator<Incrementable>>;

    template <typename Incrementable>
    HPX_HOST_DEVICE inline counting_shape_type<Incrementable>
    make_counting_shape(Incrementable n)
    {
        return hpx::util::make_iterator_range(
            hpx::util::make_counting_iterator(Incrementable(0)),
            hpx::util::make_counting_iterator(n));
    }
}}}    // namespace hpx::util::detail
