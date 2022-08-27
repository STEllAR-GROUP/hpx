//  Copyright (c) 2007-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/binary_filter/snappy_serialization_filter_registration.hpp>

#if defined(HPX_HAVE_COMPRESSION_SNAPPY)
#include <hpx/modules/serialization.hpp>

#include <cstddef>
#include <memory>
#include <vector>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::plugins::compression {

    struct HPX_LIBRARY_EXPORT snappy_serialization_filter
      : public serialization::binary_filter
    {
        snappy_serialization_filter(bool compress = false,
            serialization::binary_filter* next_filter = nullptr) noexcept
          : current_(0)
          , compress_(compress)
        {
        }

        void load(void* dst, std::size_t dst_count);
        void save(void const* src, std::size_t src_count);
        bool flush(void* dst, std::size_t dst_count, std::size_t& written);

        void set_max_length(std::size_t size);
        std::size_t init_data(
            char const* buffer, std::size_t size, std::size_t buffer_size);

    private:
        // serialization support
        friend class hpx::serialization::access;

        template <typename Archive>
        HPX_FORCEINLINE void serialize(Archive& ar, const unsigned int)
        {
        }

        HPX_SERIALIZATION_POLYMORPHIC(snappy_serialization_filter);

        std::vector<char> buffer_;
        std::size_t current_;
        bool compress_;
    };
}    // namespace hpx::plugins::compression

#include <hpx/config/warnings_suffix.hpp>

#endif
