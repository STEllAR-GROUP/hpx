//  Copyright (c) 2014 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/config/endian.hpp>
#include <hpx/assert.hpp>
#include <hpx/serialization/array.hpp>
#include <hpx/serialization/detail/serialize_collection.hpp>
#include <hpx/serialization/serialization_fwd.hpp>
#include <hpx/serialization/serialize.hpp>
#include <hpx/serialization/traits/is_bitwise_serializable.hpp>
#include <hpx/serialization/traits/is_not_bitwise_serializable.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace hpx { namespace serialization {

    namespace detail {

        // load vector<T>
        template <typename T, typename Allocator>
        void load_impl(input_archive& ar, std::vector<T, Allocator>& vs,
            std::size_t size, std::false_type)
        {
            // normal load ...
            detail::load_collection(ar, vs, size);
        }

        template <typename T, typename Allocator>
        void load_impl(input_archive& ar, std::vector<T, Allocator>& v,
            std::size_t size, std::true_type)
        {
            bool archive_endianess_differs = endian::native == endian::big ?
                ar.endian_little() :
                ar.endian_big();

#if !defined(HPX_SERIALIZATION_HAVE_ALL_TYPES_ARE_BITWISE_SERIALIZABLE)
            if (ar.disable_array_optimization() || archive_endianess_differs)
            {
                load_impl(ar, v, size, std::false_type());
                return;
            }
#else
            (void) archive_endianess_differs;
            HPX_ASSERT(!(
                ar.disable_array_optimization() || archive_endianess_differs));
#endif

            // bitwise load ...
            if (v.size() < size)
            {
                v.resize(size);
            }

            ar >> hpx::serialization::make_array(v.data(), v.size());
        }
    }    // namespace detail

    template <typename Allocator>
    void serialize(input_archive& ar, std::vector<bool, Allocator>& v, unsigned)
    {
        std::uint64_t size = 0;
        ar >> size;    //-V128

        v.clear();
        if (size == 0)
        {
            return;
        }

        v.reserve(size);
        // normal load ... no chance of doing bitwise here ...
        for (std::size_t i = 0; i != size; ++i)
        {
            bool b = false;
            ar >> b;
            v.push_back(b);
        }
    }

    template <typename T, typename Allocator>
    void serialize(input_archive& ar, std::vector<T, Allocator>& v, unsigned)
    {
        using element_type =
            std::remove_const_t<typename std::vector<T, Allocator>::value_type>;

        using use_optimized = std::integral_constant<bool,
            std::is_default_constructible_v<element_type> &&
                (hpx::traits::is_bitwise_serializable_v<element_type> ||
                    !hpx::traits::is_not_bitwise_serializable_v<element_type>)>;

        v.clear();

        std::uint64_t size;
        ar >> size;    //-V128
        if (size == 0)
        {
            return;
        }

        detail::load_impl(ar, v, size, use_optimized());
    }

    // save vector<T>
    namespace detail {

        template <typename T, typename Allocator>
        void save_impl(output_archive& ar, std::vector<T, Allocator> const& vs,
            std::false_type)
        {
            // normal save ...
            detail::save_collection(ar, vs);
        }

        template <typename T, typename Allocator>
        void save_impl(output_archive& ar, std::vector<T, Allocator> const& v,
            std::true_type)
        {
            bool archive_endianess_differs = endian::native == endian::big ?
                ar.endian_little() :
                ar.endian_big();

#if !defined(HPX_SERIALIZATION_HAVE_ALL_TYPES_ARE_BITWISE_SERIALIZABLE)
            if (ar.disable_array_optimization() || archive_endianess_differs)
            {
                save_impl(ar, v, std::false_type());
                return;
            }
#else
            (void) archive_endianess_differs;
            HPX_ASSERT(!(
                ar.disable_array_optimization() || archive_endianess_differs));
#endif

            // bitwise (zero-copy) save ...
            ar << hpx::serialization::make_array(v.data(), v.size());
        }
    }    // namespace detail

    template <typename Allocator>
    void serialize(
        output_archive& ar, std::vector<bool, Allocator> const& v, unsigned)
    {
        std::uint64_t size = v.size();
        ar << size;
        if (v.empty())
        {
            return;
        }

        // normal save ... no chance of doing bitwise here ...
        for (std::size_t i = 0; i < v.size(); ++i)
        {
            bool b = v[i];
            ar << b;
        }
    }

    template <typename T, typename Allocator>
    void serialize(
        output_archive& ar, std::vector<T, Allocator> const& v, unsigned)
    {
        using element_type =
            std::remove_const_t<typename std::vector<T, Allocator>::value_type>;

        using use_optimized = std::integral_constant<bool,
            std::is_default_constructible_v<element_type> &&
                (hpx::traits::is_bitwise_serializable_v<element_type> ||
                    !hpx::traits::is_not_bitwise_serializable_v<element_type>)>;

        std::uint64_t size = v.size();
        ar << size;
        if (v.empty())
        {
            return;
        }

        detail::save_impl(ar, v, use_optimized());
    }
}}    // namespace hpx::serialization
