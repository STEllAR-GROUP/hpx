//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPUTE_SERIALIZATION_VECTOR_HPP
#define HPX_COMPUTE_SERIALIZATION_VECTOR_HPP

#include <hpx/config.hpp>

#if !defined(__CUDA_ARCH__)
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/traits/is_bitwise_serializable.hpp>
#else
#include <hpx/runtime/serialization/serialization_fwd.hpp>
#endif

#include <hpx/compute/vector.hpp>

#include <type_traits>

namespace hpx { namespace serialization
{
#if !defined(__CUDA_ARCH__)
    // load compute::vector<T>
    namespace detail
    {
        template <typename T, typename Allocator>
        void load_impl(input_archive & ar, compute::vector<T, Allocator> & vs,
            std::false_type)
        {
            // normal load ...
            typedef typename compute::vector<T, Allocator>::value_type
                value_type;
            typedef typename compute::vector<T, Allocator>::size_type
                size_type;

            size_type size;
            value_type v;

            ar >> size; //-V128
            if(size == 0) return;

            vs.resize(size);
            for(size_type i = 0; i != size; ++i)
            {
                ar >> v;
                vs[i] = v;
            }
        }

        template <typename T, typename Allocator>
        void load_impl(input_archive & ar, compute::vector<T, Allocator> & v,
            std::true_type)
        {
            if(ar.disable_array_optimization())
            {
                load_impl(ar, v, std::false_type());
            }
            else
            {
                // bitwise load ...
                typedef typename compute::vector<T, Allocator>::value_type
                    value_type;
                typedef typename compute::vector<T, Allocator>::size_type
                    size_type;

                size_type size;
                ar >> size; //-V128
                if(size == 0) return;

                v.resize(size);
                load_binary(ar, v.device_data(), v.size() * sizeof(value_type));
            }
        }
    }

    template <typename T, typename Allocator>
    void serialize(input_archive & ar, compute::vector<T, Allocator> & v,
        unsigned)
    {
        typedef std::integral_constant<bool,
            hpx::traits::is_bitwise_serializable<
                typename std::remove_const<
                    typename compute::vector<T, Allocator>::value_type
                >::type
            >::value> use_optimized;

        v.clear();
        detail::load_impl(ar, v, use_optimized());
    }



    // save compute::vector<T>
    namespace detail
    {
        template <typename T, typename Allocator>
        void save_impl(output_archive & ar,
            compute::vector<T, Allocator> const& vs, std::false_type)
        {
            // normal save ...
            for(auto v : vs)
            {
                ar << v;
            }
        }

        template <typename T, typename Allocator>
        void save_impl(output_archive & ar,
            compute::vector<T, Allocator> const& v, std::true_type)
        {
            if(ar.disable_array_optimization())
            {
                save_impl(ar, v, std::false_type());
            }
            else
            {
                // bitwise save ...
                typedef typename compute::vector<T, Allocator>::value_type
                    value_type;
                save_binary(ar, v.device_data(), v.size() * sizeof(value_type));
            }
        }
    }

    template <typename T, typename Allocator>
    void serialize(output_archive & ar, compute::vector<T, Allocator> const& v,
        unsigned)
    {
        typedef std::integral_constant<bool,
            hpx::traits::is_bitwise_serializable<
                typename std::remove_const<
                    typename compute::vector<T, Allocator>::value_type
                >::type
            >::value> use_optimized;

        ar << v.size(); //-V128
        if(v.empty()) return;

        detail::save_impl(ar, v, use_optimized());
    }
#else
    template <typename T, typename Allocator>
    void serialize(input_archive & ar, compute::vector<T, Allocator> & v,
        unsigned)
    {}

    template <typename T, typename Allocator>
    void serialize(output_archive & ar, compute::vector<T, Allocator> const& v,
        unsigned)
    {}
#endif
}}

#endif
