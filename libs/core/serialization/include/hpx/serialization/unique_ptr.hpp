//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2014-2015 Anton Bikineev
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/serialization/detail/pointer.hpp>
#include <hpx/serialization/serialization_fwd.hpp>
#include <hpx/serialization/serialize.hpp>

#include <memory>

namespace hpx::serialization {

    template <typename T>
    void load(input_archive& ar, std::unique_ptr<T>& ptr, unsigned)
    {
        detail::serialize_pointer_untracked(ar, ptr);
    }

    template <typename T>
    void save(output_archive& ar, std::unique_ptr<T> const& ptr, unsigned)
    {
        detail::serialize_pointer_untracked(ar, ptr);
    }

    HPX_SERIALIZATION_SPLIT_FREE_TEMPLATE(
        (template <typename T>), (std::unique_ptr<T>) )
}    // namespace hpx::serialization
