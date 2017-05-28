//  Copyright (c) 2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_ACCESS_DATA_HPP
#define HPX_SERIALIZATION_ACCESS_DATA_HPP

#include <hpx/config.hpp>
#include <hpx/lcos/detail/future_data.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/serialization/serialization_fwd.hpp>
#include <hpx/runtime/serialization/binary_filter.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////
    template <typename Container>
    struct default_serialization_access_data
    {
        typedef std::false_type preprocessing_only;

        HPX_CONSTEXPR static bool is_preprocessing() { return false; }

        // functions related to output operations
        static void await_future(
            Container& cont
          , hpx::lcos::detail::future_data_refcnt_base & future_data)
        {}

        static void add_gid(Container& cont,
            naming::gid_type const & gid,
            naming::gid_type const & split_gid)
        {}

        HPX_CONSTEXPR static bool has_gid(Container& cont,
            naming::gid_type const& gid)
        {
            return false;
        }

        static void
        write(Container& cont, std::size_t count,
            std::size_t current, void const* address)
        {
        }

        HPX_CONSTEXPR static bool flush(serialization::binary_filter* filter,
            Container& cont, std::size_t current, std::size_t size,
            std::size_t& written)
        {
            written = size;
            return true;
        }

        // functions related to input operations
        static void read(Container const& cont,
            std::size_t count, std::size_t current, void* address)
        {
        }

        HPX_CONSTEXPR static std::size_t init_data(Container const& cont,
            serialization::binary_filter* filter, std::size_t current,
            std::size_t decompressed_size)
        {
            return decompressed_size;
        }

        static void reset(Container& cont)
        {}
    };

    ///////////////////////////////////////////////////////////////////////
    template <typename Container>
    struct serialization_access_data
      : default_serialization_access_data<Container>
    {
        static std::size_t size(Container const& cont)
        {
            return cont.size();
        }

        static void resize(Container& cont, std::size_t count)
        {
            return cont.resize(cont.size() + count);
        }

        static void write(Container& cont, std::size_t count,
            std::size_t current, void const* address)
        {
            void* dest = &cont[current];
            switch (count)
            {
            case 8:
                *static_cast<std::uint64_t*>(dest) =
                    *static_cast<std::uint64_t const*>(address);
                break;

            case 4:
                *static_cast<std::uint32_t*>(dest) =
                    *static_cast<std::uint32_t const*>(address);
                break;

            case 2:
                *static_cast<std::uint16_t*>(dest) =
                    *static_cast<std::uint16_t const*>(address);
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
            std::size_t current, void* address)
        {
            void const* src = &cont[current];
            switch (count)
            {
            case 8:
                *static_cast<std::uint64_t*>(address) =
                    *static_cast<std::uint64_t const*>(src);
                break;

            case 4:
                *static_cast<std::uint32_t*>(address) =
                    *static_cast<std::uint32_t const*>(src);
                break;

            case 2:
                *static_cast<std::uint16_t*>(address) =
                    *static_cast<std::uint16_t const*>(src);
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
            return filter->init_data(&cont[current], cont.size()-current,
                decompressed_size);
        }
    };
}}

#endif



