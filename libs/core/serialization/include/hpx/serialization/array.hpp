//  (C) Copyright 2005 Matthias Troyer and Dave Abrahams
//  Copyright (c) 2015 Anton Bikineev
//  Copyright (c) 2015 Andreas Schaefer
//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/config/endian.hpp>
#include <hpx/assert.hpp>
#include <hpx/serialization/serialization_fwd.hpp>
#include <hpx/serialization/serialize.hpp>
#include <hpx/serialization/traits/is_bitwise_serializable.hpp>
#include <hpx/serialization/traits/is_not_bitwise_serializable.hpp>

#if defined(HPX_SERIALIZATION_HAVE_BOOST_TYPES)
#include <boost/array.hpp>
#endif

#include <array>
#include <cstddef>
#include <type_traits>

namespace hpx::serialization {

    template <typename T>
    class array
    {
    public:
        using value_type = T;

        constexpr array(value_type* t, std::size_t s) noexcept
          : m_t(t)
          , m_element_count(s)
        {
        }

        constexpr value_type* address() const noexcept
        {
            return m_t;
        }

        constexpr std::size_t count() const noexcept
        {
            return m_element_count;
        }

        template <typename Archive>
        void serialize(Archive& ar, unsigned int)
        {
#if !defined(HPX_SERIALIZATION_HAVE_ALL_TYPES_ARE_BITWISE_SERIALIZABLE)
            // NOLINTNEXTLINE(bugprone-branch-clone)
            if (ar.disable_array_optimization() || ar.endianess_differs())
            {
                // normal serializtion
                for (std::size_t i = 0; i != m_element_count; ++i)
                {
                    // clang-format off
                    ar & m_t[i];
                    // clang-format on
                }
                return;
            }
#else
            HPX_ASSERT(
                !(ar.disable_array_optimization() || ar.endianess_differs()));
#endif
            using element_type = std::remove_const_t<T>;

            constexpr bool use_optimized =
                std::is_default_constructible_v<element_type> &&
                (hpx::traits::is_bitwise_serializable_v<element_type> ||
                    !hpx::traits::is_not_bitwise_serializable_v<element_type>);

            if constexpr (use_optimized)
            {
                // try using chunking
                if constexpr (std::is_same_v<Archive, input_archive>)
                {
                    ar.load_binary_chunk(m_t, m_element_count * sizeof(T));
                }
                else
                {
                    ar.save_binary_chunk(m_t, m_element_count * sizeof(T));
                }
            }
            else
            {
                // normal serialization
                for (std::size_t i = 0; i != m_element_count; ++i)
                {
                    // clang-format off
                    ar & m_t[i];
                    // clang-format on
                }
            }
        }

    private:
        value_type* m_t;
        std::size_t m_element_count;
    };

    // make_array function
    template <typename T>
    HPX_FORCEINLINE constexpr array<T> make_array(
        T* begin, std::size_t size) noexcept
    {
        return array<T>(begin, size);
    }

#if defined(HPX_SERIALIZATION_HAVE_BOOST_TYPES)
    // implement serialization for boost::array
    template <typename Archive, typename T, std::size_t N>
    void serialize(
        Archive& ar, boost::array<T, N>& a, const unsigned int /* version */)
    {
        // clang-format off
        ar & hpx::serialization::make_array(a.begin(), a.size());
        // clang-format on
    }
#endif

    // implement serialization for std::array
    template <typename Archive, typename T, std::size_t N>
    void serialize(
        Archive& ar, std::array<T, N>& a, const unsigned int /* version */)
    {
        // clang-format off
        ar & hpx::serialization::make_array(a.data(), a.size());
        // clang-format on
    }

    // allow our array to be serialized as prvalue
    // compiler should support good ADL implementation
    // but it is rather for all hpx serialization library
    template <typename T>
    HPX_FORCEINLINE output_archive& operator<<(output_archive& ar, array<T> t)
    {
        ar.save(t);
        return ar;
    }

    template <typename T>
    HPX_FORCEINLINE input_archive& operator>>(input_archive& ar, array<T> t)
    {
        ar.load(t);
        return ar;
    }

    template <typename T>
    HPX_FORCEINLINE output_archive& operator&(
        output_archive& ar, array<T> t)    //-V524
    {
        ar.save(t);
        return ar;
    }

    template <typename T>
    HPX_FORCEINLINE input_archive& operator&(
        input_archive& ar, array<T> t)    //-V524
    {
        ar.load(t);
        return ar;
    }

    // serialize plain arrays:
    template <typename T, std::size_t N>
    HPX_FORCEINLINE output_archive& operator<<(output_archive& ar, T (&t)[N])
    {
        array<T> array = make_array(t, N);
        ar.save(array);
        return ar;
    }

    template <typename T, std::size_t N>
    HPX_FORCEINLINE input_archive& operator>>(input_archive& ar, T (&t)[N])
    {
        array<T> array = make_array(t, N);
        ar.load(array);
        return ar;
    }

    template <typename T, std::size_t N>
    HPX_FORCEINLINE output_archive& operator&(
        output_archive& ar, T (&t)[N])    //-V524
    {
        array<T> array = make_array(t, N);
        ar.save(array);
        return ar;
    }

    template <typename T, std::size_t N>
    HPX_FORCEINLINE input_archive& operator&(
        input_archive& ar, T (&t)[N])    //-V524
    {
        array<T> array = make_array(t, N);
        ar.load(array);
        return ar;
    }
}    // namespace hpx::serialization
