//  Copyright (c) 2017 Christopher Taylor
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_PARTITIONED_VECTOR_HPP
#define HPX_SERIALIZATION_PARTITIONED_VECTOR_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/traits/is_bitwise_serializable.hpp>

#include <hpx/traits/segmented_iterator_traits.hpp>
#include <hpx/include/partitioned_vector.hpp>

namespace hpx { namespace serialization
{
    namespace detail
    {
        partitioned_vector_segmented_serializer
        {
            template<typename T>
            static void serialize(input_archive & ar,
                hpx::partitioned_vector<T> & v, std::false_type)
            {
                static_assert(
                    hpx::parallel::is_partitioned(hpx::parallel::execution::par,
                        std::begin(v), std::end(v), [](std::size_t n) { return n > 0; });
                    "hpx::serialization::serialize requires segmented partitioned_vector");
            }

            template<typename T>
            static void serialize(input_archive & ar,
                hpx::partitioned_vector<T> & v, std::true_type)
            {
                std::string pvec_registered_name;
                ar >> pvec_registered_name;
                hpx::future<void> fconnect = v.connect_to(pvec_registered_name);
                fconnect.wait();
            }

            template<typename T>
            static void serialize(output_archive & ar,
                const hpx::partitioned_vector<T> & v, std::false_type) const
            {
                static_assert(
                    hpx::parallel::is_partitioned(hpx::parallel::execution::par,
                        std::begin(v), std::end(v), [](std::size_t n) { return n > 0; });
                    "hpx::serialization::serialize requires segmented partitioned_vector");

                static_assert(
                    v.registered_name_.size() > 0,
                    "hpx::serialization::serialize requires a registered partitioned_vector");
            }

            template<typename T>
            static void serialize(output_archive & ar,
                const hpx::partitioned_vector<T> & v, std::true_type) const
            {
                ar << v.registered_name_;
            }
        };
    }

    template <typename T>
    void serialize(input_archive & ar, hpx::partitioned_vector<T> & v,
        unsigned)
    {
        typedef hpx::traits::is_segmented_iterator<FwdIter> is_segmented;

        auto vitr = std::begin(v);
        detail::partitioned_vector_segmented_serializer::serialize<T>(
            ar, v, is_segmented(vitr));
    }

    template <typename T>
    void serialize(output_archive & ar, const hpx::partitioned_vector<T> & v,
        unsigned)
    {
        typedef hpx::traits::is_segmented_iterator<FwdIter> is_segmented;

        auto vitr = std::begin(v);
        detail::partitioned_vector_segmented_serializer::serialize<T>(
            ar, v, is_segmented(vitr));
    }

}}

#endif
