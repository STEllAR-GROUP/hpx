//  (C) Copyright 2005 Matthias Troyer and Dave Abrahams
//  Copyright (c) 2015 Anton Bikineev
//  Copyright (c) 2015 Andreas Schaefer
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/config/endian.hpp>
#include <hpx/serialization/serialization_fwd.hpp>
#include <hpx/serialization/serialize.hpp>
#include <hpx/serialization/traits/is_bitwise_serializable.hpp>

#if defined(HPX_SERIALIZATION_HAVE_BOOST_TYPES)
#include <boost/array.hpp>
#endif

#include <array>
#include <cstddef>
#include <type_traits>

namespace hpx { namespace serialization {

    template <class T>
    class array
    {
    public:
        using value_type = T;

        array(value_type* t, std::size_t s)
          : m_t(t)
          , m_element_count(s)
        {
        }

        value_type* address() const
        {
            return m_t;
        }

        std::size_t count() const
        {
            return m_element_count;
        }

        template <class Archive>
        void serialize_optimized(
            Archive& ar, unsigned int /*v*/, std::false_type)
        {
            for (std::size_t i = 0; i != m_element_count; ++i)
                ar& m_t[i];
        }

        void serialize_optimized(
            output_archive& ar, unsigned int, std::true_type)
        {
            // try using chunking
            ar.save_binary_chunk(m_t, m_element_count * sizeof(T));
        }

        void serialize_optimized(
            input_archive& ar, unsigned int, std::true_type)
        {
            // try using chunking
            ar.load_binary_chunk(m_t, m_element_count * sizeof(T));
        }

        template <class Archive>
        void serialize(Archive& ar, unsigned int v)
        {
            using use_optimized = std::integral_constant<bool,
                hpx::traits::is_bitwise_serializable<
                    typename std::remove_const<T>::type>::value>;

            bool archive_endianess_differs = endian::native == endian::big ?
                ar.endian_little() :
                ar.endian_big();

            // NOLINTNEXTLINE(bugprone-branch-clone)
            if (ar.disable_array_optimization() || archive_endianess_differs)
                serialize_optimized(ar, v, std::false_type());
            else
                serialize_optimized(ar, v, use_optimized());
        }

    private:
        value_type* m_t;
        std::size_t m_element_count;
    };

    // make_array function
    template <class T>
    HPX_FORCEINLINE array<T> make_array(T* begin, std::size_t size)
    {
        return array<T>(begin, size);
    }

#if defined(HPX_SERIALIZATION_HAVE_BOOST_TYPES)
    // implement serialization for boost::array
    template <typename Archive, typename T, std::size_t N>
    void serialize(
        Archive& ar, boost::array<T, N>& a, const unsigned int /* version */)
    {
        ar& hpx::serialization::make_array(a.begin(), a.size());
    }
#endif

    // implement serialization for std::array
    template <typename Archive, typename T, std::size_t N>
    void serialize(
        Archive& ar, std::array<T, N>& a, const unsigned int /* version */)
    {
        ar& hpx::serialization::make_array(a.data(), a.size());
    }

    // allow our array to be serialized as prvalue
    // compiler should support good ADL implementation
    // but it is rather for all hpx serialization library
    template <typename T>
    HPX_FORCEINLINE output_archive& operator<<(output_archive& ar, array<T> t)
    {
        ar.invoke(t);
        return ar;
    }

    template <typename T>
    HPX_FORCEINLINE input_archive& operator>>(input_archive& ar, array<T> t)
    {
        ar.invoke(t);
        return ar;
    }

    template <typename T>
    HPX_FORCEINLINE output_archive& operator&(
        output_archive& ar, array<T> t)    //-V524
    {
        ar.invoke(t);
        return ar;
    }

    template <typename T>
    HPX_FORCEINLINE input_archive& operator&(
        input_archive& ar, array<T> t)    //-V524
    {
        ar.invoke(t);
        return ar;
    }

    // serialize plain arrays:
    template <typename T, std::size_t N>
    HPX_FORCEINLINE output_archive& operator<<(output_archive& ar, T (&t)[N])
    {
        array<T> array = make_array(t, N);
        ar.invoke(array);
        return ar;
    }

    template <typename T, std::size_t N>
    HPX_FORCEINLINE input_archive& operator>>(input_archive& ar, T (&t)[N])
    {
        array<T> array = make_array(t, N);
        ar.invoke(array);
        return ar;
    }

    template <typename T, std::size_t N>
    HPX_FORCEINLINE output_archive& operator&(
        output_archive& ar, T (&t)[N])    //-V524
    {
        array<T> array = make_array(t, N);
        ar.invoke(array);
        return ar;
    }

    template <typename T, std::size_t N>
    HPX_FORCEINLINE input_archive& operator&(
        input_archive& ar, T (&t)[N])    //-V524
    {
        array<T> array = make_array(t, N);
        ar.invoke(array);
        return ar;
    }
}}    // namespace hpx::serialization
