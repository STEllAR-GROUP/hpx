//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_VECTOR_HPP
#define HPX_SERIALIZATION_VECTOR_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/serialization/array.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/traits/is_bitwise_serializable.hpp>

#include <type_traits>
#include <vector>

namespace hpx { namespace serialization
{
    namespace detail
    {
        // load vector<T>
        template <typename T, typename Allocator>
        void load_impl(input_archive & ar, std::vector<T, Allocator> & vs,
            std::false_type)
        {
            // normal load ...
            typedef typename std::vector<T, Allocator>::size_type size_type;
            size_type size;
            ar >> size; //-V128
            if(size == 0) return;

            vs.resize(size);
            for(size_type i = 0; i != size; ++i)
            {
                ar >> vs[i];
            }
        }

        template <typename T, typename Allocator>
        void load_impl(input_archive & ar, std::vector<T, Allocator> & v,
            std::true_type)
        {
            if(ar.disable_array_optimization())
            {
                load_impl(ar, v, std::false_type());
            }
            else
            {
                // bitwise load ...
                typedef typename std::vector<T, Allocator>::size_type size_type;
                size_type size;
                ar >> size; //-V128
                if(size == 0) return;

                v.resize(size);
                ar >> hpx::serialization::make_array(v.data(), v.size());
            }
        }
    }

    template <typename Allocator>
    void serialize(input_archive & ar, std::vector<bool, Allocator> & v, unsigned)
    {
        typedef typename std::vector<bool, Allocator>::size_type size_type;
        size_type size = 0;
        ar >> size; //-V128

        v.clear();
        if(size == 0) return;

        v.reserve(size);
        // normal load ... no chance of doing bitwise here ...
        for(size_type i = 0; i != size; ++i)
        {
            bool b = false;
            ar >> b;
            v.push_back(b);
        }
    }

    template <typename T, typename Allocator>
    void serialize(input_archive & ar, std::vector<T, Allocator> & v, unsigned)
    {
        typedef std::integral_constant<bool,
            hpx::traits::is_bitwise_serializable<
                typename std::remove_const<
                    typename std::vector<T, Allocator>::value_type
                >::type
            >::value> use_optimized;

        v.clear();
        detail::load_impl(ar, v, use_optimized());
    }

    // save vector<T>
    namespace detail
    {
        template <typename T, typename Allocator>
        void save_impl(output_archive & ar, const std::vector<T, Allocator> & vs,
            std::false_type)
        {
            // normal save ...
            typedef typename std::vector<T, Allocator>::value_type value_type;
            for(const value_type & v : vs)
            {
                ar << v;
            }
        }

        template <typename T, typename Allocator>
        void save_impl(output_archive & ar, const std::vector<T, Allocator> & v,
            std::true_type)
        {
            if(ar.disable_array_optimization())
            {
                save_impl(ar, v, std::false_type());
            }
            else
            {
                // bitwise (zero-copy) save ...
                ar << hpx::serialization::make_array(v.data(), v.size());
            }
        }
    }

    template <typename Allocator>
    void serialize(output_archive & ar, const std::vector<bool, Allocator> & v,
        unsigned)
    {
        typedef typename std::vector<bool, Allocator>::size_type size_type;
        ar << v.size(); //-V128
        if(v.empty()) return;
        // normal save ... no chance of doing bitwise here ...
        for(size_type i = 0; i < v.size(); ++i)
        {
            bool b = v[i];
            ar << b;
        }
    }

    template <typename T, typename Allocator>
    void serialize(output_archive & ar, const std::vector<T, Allocator> & v,
        unsigned)
    {
        typedef std::integral_constant<bool,
            hpx::traits::is_bitwise_serializable<
                typename std::remove_const<
                    typename std::vector<T, Allocator>::value_type
                >::type
            >::value> use_optimized;

        ar << v.size(); //-V128
        if(v.empty()) return;
        detail::save_impl(ar, v, use_optimized());
    }
}}

#endif
