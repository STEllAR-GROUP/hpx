//  Copyright (c) 2017 Christopher Taylor
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_PARTITIONED_VECTOR_HPP
#define HPX_SERIALIZATION_PARTITIONED_VECTOR_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/traits/is_bitwise_serializable.hpp>

#include <hpx/include/parallel_is_partitioned.hpp>
#include <hpx/include/partitioned_vector.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace hpx { namespace serialization
{
    namespace detail
    {
        template<typename T>
        partitioned_vector_segmented_serializer(input_archive & ar,
            hpx::partitioned_vector<T> & v)
        {
            static_assert(
                hpx::parallel::is_partitioned(hpx::parallel::execution::par,
                    std::begin(v), std::end(v), [](std::size_t n) { return n > 0; });
                "hpx::serialization::serialize requires segemented partitioned_vector");

            std::string pvec_registered_name;
            ar >> pvec_registered_name;
            hpx::future<void> fconnect = v.connect_to(pvec_registered_name);
            fconnect.wait();
        }

        template<typename T>
        partitioned_vector_segmented_serializer(output_archive & ar,
            hpx::partitioned_vector<T> & v)
        {
            static_assert(
                hpx::parallel::is_partitioned(hpx::parallel::execution::par,
                    std::begin(v), std::end(v), [](std::size_t n) { return n > 0; });
                "hpx::serialization::serialize requires segemented partitioned_vector");

            static_assert(
                v.registered_name_.size() > 0,
                "hpx::serialization::serialize requires a registered partitioned_vector");

            ar << v.registered_name_;
        }
    }

    template <typename T>
    void serialize(input_archive & ar, hpx::partitioned_vector<T> & v,
        unsigned)
    {
        partitioned_vector_segmented_serializer(ar, v);
    }

    template <typename T>
    void serialize(output_archive & ar, const hpx::partitioned_vector<T> & v,
        unsigned)
    {
        partitioned_vector_segemented_serializer(ar, v);
    }

}}

#endif
