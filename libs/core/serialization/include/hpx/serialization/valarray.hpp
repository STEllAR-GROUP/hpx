//  Copyright (c) 2017 Christopher Taylor
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
#pragma once

#include <hpx/config.hpp>
#include <hpx/serialization/serialization_fwd.hpp>
#include <hpx/serialization/traits/is_bitwise_serializable.hpp>

#include <cstddef>
#include <valarray>

namespace hpx::serialization {

    template <typename T>
    void serialize(input_archive& ar, std::valarray<T>& arr, int /* version */)
    {
        std::size_t sz = 0;
        ar >> sz;
        arr.resize(sz);

        if (sz == 0)
            return;

        for (std::size_t i = 0; i < sz; ++i)
            ar >> arr[i];
    }

    template <typename T>
    void serialize(
        output_archive& ar, std::valarray<T> const& arr, int /* version */)
    {
        std::size_t const sz = arr.size();
        ar << sz;
        for (auto const& v : arr)
            ar << v;
    }
}    // namespace hpx::serialization
