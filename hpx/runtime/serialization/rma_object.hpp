//  Copyright (c) 2017 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_RMA_OBJECT_HPP
#define HPX_SERIALIZATION_RMA_OBJECT_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/serialization/array.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/traits/is_bitwise_serializable.hpp>
//
#include <hpx/traits/is_rma_eligible.hpp>
#include <hpx/runtime/parcelset/rma/memory_region.hpp>
#include <hpx/runtime/parcelset/rma/allocator.hpp>
#include <hpx/runtime/parcelset/rma/rma_object.hpp>
//
#include <type_traits>
#include <vector>

namespace hpx { namespace serialization
{
    namespace detail
    {

        // ----------------------------------------------------------------------
        // load vector<T>, if optimized methods are disabled
        // ----------------------------------------------------------------------
        template <typename T>
        void load_impl(input_archive & ar, hpx::parcelset::rma::rma_vector<T> & vs,
            std::false_type)
        {
            // normal load ...
            typedef typename rma::rma_vector<T>::size_type size_type;
            size_type size;
            ar >> size; //-V128
            if(size == 0) return;

            vs.resize(size);
            for(size_type i = 0; i != size; ++i)
            {
                ar >> vs[i];
            }
        }

        template <typename T>
        void load_impl(input_archive & ar, rma::rma_vector<T> & v, std::true_type)
        {
            // if array optimization is disabled, read each element one by one
            if (ar.disable_array_optimization())
            {
                load_impl(ar, v, std::false_type());
            }
            else
            {
                // read the vector size first
                typedef typename rma::rma_vector<T>::size_type size_type;
                size_type size;
                ar >> size; //-V128
                if (size == 0) {
                    return;
                }
                // if reading chunks is disabled, we must use a normal binary read
                if (size < HPX_ZERO_COPY_SERIALIZATION_THRESHOLD ||
                    ar.disable_data_chunking())
                {
                    v.resize(size/sizeof(T));
                    ar.load_binary(v.data(), size);
                }
                else {
                    // bitwise (zero-copy) load with rma overload...
                    rma::memory_region *region;
                    ar.load_rma_chunk(nullptr, size*sizeof(T), region);
                    v.set_memory_region(region, size);
                }
            }
        }
    }

    template <typename T>
    void serialize(input_archive & ar, rma::rma_vector<T> & v, unsigned)
    {
        typedef std::integral_constant<bool, true> use_optimized;

        v.clear();
        detail::load_impl(ar, v, use_optimized());
    }

    // ----------------------------------------------------------------------
    // save rma::vector<T>
    // ----------------------------------------------------------------------
    namespace detail
    {
        template <typename T>
        void save_impl(
            output_archive & ar, const rma::rma_vector<T> & vs, std::false_type)
        {
            // normal save ...
            typedef typename rma::rma_vector<T>::value_type value_type;
            for (const value_type & v : vs)
            {
                ar << v;
            }
        }

        template <typename T>
        void save_impl(
            output_archive & ar, const rma::rma_vector<T> & v, std::true_type)
        {
            const std::size_t bytes = v.size()*sizeof(T);
            if (bytes < HPX_ZERO_COPY_SERIALIZATION_THRESHOLD)
            {
                // fall back to serialization_chunk-less archive
                ar.save_binary(v.data(), bytes);
            }
            else {
                parcelset::rma::memory_region *region = v.get_region();
                region->set_message_length(bytes);
                // bitwise (zero-copy) save with rma overload...
                ar.save_rma_chunk(v.data(), bytes, region);
            }
        }
    }

    template <typename T>
    void serialize(output_archive & ar, const rma::rma_vector<T> & v, unsigned)
    {
        ar << v.size(); //-V128
        if (v.empty()) return;

        if (ar.disable_array_optimization()) {
            detail::save_impl(ar, v, std::false_type());
        }
        else {
            detail::save_impl(ar, v, std::true_type());
        }
    }
}}

#endif
