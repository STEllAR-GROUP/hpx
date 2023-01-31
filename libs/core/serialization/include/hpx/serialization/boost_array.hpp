//  Copyright (c) 2015 Anton Bikineev
//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/serialization/config/defines.hpp>

#if defined(HPX_SERIALIZATION_HAVE_BOOST_TYPES)

#include <hpx/serialization/array.hpp>
#include <hpx/serialization/serialization_fwd.hpp>

#include <boost/array.hpp>

#include <cstddef>

namespace hpx::serialization {

    // implement serialization for boost::array
    template <typename Archive, typename T, std::size_t N>
    void serialize(
        Archive& ar, boost::array<T, N>& a, unsigned int const /* version */)
    {
        // clang-format off
        ar & hpx::serialization::make_array(a.begin(), a.size());
        // clang-format on
    }
}    // namespace hpx::serialization

#endif
