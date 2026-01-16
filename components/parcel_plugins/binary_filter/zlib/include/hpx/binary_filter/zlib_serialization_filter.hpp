//  Copyright (c) 2007-2026 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/binary_filter/zlib_serialization_filter_registration.hpp>

#if defined(HPX_HAVE_COMPRESSION_ZLIB)
#include <hpx/modules/iostream.hpp>
#include <hpx/modules/serialization.hpp>

#include <cstddef>
#include <memory>
#include <vector>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::plugins::compression {

    namespace detail {

        class zlib_compdecomp;
    }    // namespace detail

    struct HPX_LIBRARY_EXPORT zlib_serialization_filter
      : public serialization::binary_filter
    {
        explicit zlib_serialization_filter(bool compress = false,
            serialization::binary_filter* next_filter = nullptr) noexcept;

        void load(void* dst, std::size_t dst_count) override;
        void save(void const* src, std::size_t src_count) override;
        bool flush(
            void* dst, std::size_t dst_count, std::size_t& written) override;

        void set_max_length(std::size_t size) override;
        std::size_t init_data(void const* buffer, std::size_t size,
            std::size_t buffer_size) override;

    protected:
        std::size_t load_impl(void* dst, std::size_t dst_count, void const* src,
            std::size_t src_count);

    private:
        // serialization support
        friend class hpx::serialization::access;

        template <typename Archive>
        HPX_FORCEINLINE static constexpr void serialize(
            Archive& ar, unsigned int const) noexcept
        {
        }

        HPX_SERIALIZATION_POLYMORPHIC(zlib_serialization_filter, override);

        std::unique_ptr<detail::zlib_compdecomp> compdecomp_;
        std::vector<char> buffer_;
        std::size_t current_;
    };
}    // namespace hpx::plugins::compression

#include <hpx/config/warnings_suffix.hpp>

#endif
