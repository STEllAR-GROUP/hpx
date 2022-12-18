// Copyright (c) 2007-2022 Hartmut Kaiser
// Copyright (c) 2015 Anton Bikineev
//
// SPDX-License-Identifier: BSL-1.0
// Use, modification and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/serialization/detail/polymorphic_intrusive_factory.hpp>
#include <hpx/serialization/serialization_fwd.hpp>

#include <cstddef>

namespace hpx::serialization {

    ///////////////////////////////////////////////////////////////////////////
    // Base class for all serialization filters.
    struct binary_filter
    {
        virtual ~binary_filter() = default;

        // compression API
        virtual void set_max_length(std::size_t size) = 0;
        virtual void save(void const* src, std::size_t src_count) = 0;
        virtual bool flush(
            void* dst, std::size_t dst_count, std::size_t& written) = 0;

        // decompression API
        virtual std::size_t init_data(
            void const* buffer, std::size_t size, std::size_t buffer_size) = 0;
        virtual void load(void* dst, std::size_t dst_count) = 0;

        template <typename T>
        constexpr void serialize(T& /*ar*/, unsigned) noexcept
        {
        }
        HPX_SERIALIZATION_POLYMORPHIC_ABSTRACT(binary_filter);
    };
}    // namespace hpx::serialization
