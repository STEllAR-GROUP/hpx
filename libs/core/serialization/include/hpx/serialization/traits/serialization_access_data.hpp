//  Copyright (c) 2017-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/serialization/binary_filter.hpp>
#include <hpx/serialization/serialization_fwd.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace hpx::traits {

    ////////////////////////////////////////////////////////////////////////////
    template <typename Container>
    struct default_serialization_access_data
    {
        using preprocessing_only = std::false_type;

        [[nodiscard]] static constexpr bool is_preprocessing() noexcept
        {
            return false;
        }

        // functions related to output operations
        static constexpr void write(Container& /* cont */,
            std::size_t /* count */, std::size_t /* current */,
            void const* /* address */) noexcept
        {
        }

        static bool flush(serialization::binary_filter* /* filter */,
            Container& /* cont */, std::size_t /* current */, std::size_t size,
            std::size_t& written) noexcept
        {
            written = size;
            return true;
        }

        // functions related to input operations
        static constexpr void read(Container const& /* cont */,
            std::size_t /* count */, std::size_t /* current */,
            void* /* address */) noexcept
        {
        }

        static constexpr std::size_t init_data(Container const& /* cont */,
            serialization::binary_filter* /* filter */,
            std::size_t /* current */, std::size_t decompressed_size) noexcept
        {
            return decompressed_size;
        }

        static constexpr void reset(Container& /* cont */) noexcept {}
    };

    ///////////////////////////////////////////////////////////////////////
    template <typename Container>
    struct serialization_access_data
      : default_serialization_access_data<Container>
    {
        [[nodiscard]] static std::size_t size(Container const& cont) noexcept
        {
            return cont.size();
        }

        static void resize(Container& cont, std::size_t count)
        {
            return cont.resize(cont.size() + count);
        }

        static void write(Container& cont, std::size_t count,
            std::size_t current, void const* address) noexcept
        {
            void* dest = &cont[current];
            switch (count)
            {
            case 16:
                std::memcpy(dest, address, 16);
                break;

            case 8:
                std::memcpy(dest, address, 8);
                break;

            case 4:
                std::memcpy(dest, address, 4);    // -V112
                break;

            case 2:
                std::memcpy(dest, address, 2);
                break;

            case 1:
                *static_cast<std::uint8_t*>(dest) =
                    *static_cast<std::uint8_t const*>(address);
                break;

            default:
                std::memcpy(dest, address, count);
                break;
            }
        }

        static bool flush(serialization::binary_filter* filter, Container& cont,
            std::size_t current, std::size_t size, std::size_t& written)
        {
            return filter->flush(&cont[current], size, written);
        }

        // functions related to input operations
        static void read(Container const& cont, std::size_t count,
            std::size_t current, void* address) noexcept
        {
            void const* src = &cont[current];
            switch (count)
            {
            case 16:
                std::memcpy(address, src, 16);
                break;

            case 8:
                std::memcpy(address, src, 8);
                break;

            case 4:
                std::memcpy(address, src, 4);    // -V112
                break;

            case 2:
                std::memcpy(address, src, 2);
                break;

            case 1:
                *static_cast<std::uint8_t*>(address) =
                    *static_cast<std::uint8_t const*>(src);
                break;

            default:
                std::memcpy(address, src, count);
                break;
            }
        }

        static std::size_t init_data(Container const& cont,
            serialization::binary_filter* filter, std::size_t current,
            std::size_t decompressed_size)
        {
            return filter->init_data(
                &cont[current], cont.size() - current, decompressed_size);
        }
    };

    template <typename Container>
    struct serialization_access_data<Container const>
      : serialization_access_data<Container>
    {
    };
}    // namespace hpx::traits
