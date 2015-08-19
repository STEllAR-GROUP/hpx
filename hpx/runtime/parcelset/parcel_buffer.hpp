//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c)      2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_PARCEL_BUFFER_HPP
#define HPX_PARCELSET_PARCEL_BUFFER_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/serialization/serialization_chunk.hpp>
#include <hpx/performance_counters/parcels/data_point.hpp>
#include <hpx/util/integer/endian.hpp>

#include <boost/atomic.hpp>

#include <vector>

namespace hpx { namespace parcelset
{
    template <typename BufferType,
        typename ChunkType = serialization::serialization_chunk>
    struct parcel_buffer
    {
        typedef std::pair<
            util::integer::ulittle32_t, util::integer::ulittle32_t
        > count_chunks_type;

        typedef typename BufferType::allocator_type allocator_type;

        explicit parcel_buffer(allocator_type allocator = allocator_type())
          : data_(allocator)
          , num_chunks_(count_chunks_type(0, 0))
          , size_(0), data_size_(0)
        {}

        explicit parcel_buffer(BufferType const & data,
                allocator_type allocator = allocator_type())
          : data_(data, allocator)
          , num_chunks_(count_chunks_type(0, 0))
          , size_(0), data_size_(0)
        {}

        explicit parcel_buffer(BufferType && data,
            allocator_type allocator = allocator_type())
          : data_(std::move(data), allocator)
          , num_chunks_(count_chunks_type(0, 0))
          , size_(0), data_size_(0)
        {}

        parcel_buffer(parcel_buffer && other)
          : data_(std::move(other.data_))
          , chunks_(std::move(other.chunks_))
          , transmission_chunks_(std::move(other.transmission_chunks_))
          , num_chunks_(other.num_chunks_)
          , size_(other.size_)
          , data_size_(other.data_size_)
          , data_point_(other.data_point_)
        {
        }

        parcel_buffer &operator=(parcel_buffer && other)
        {
            data_ = std::move(other.data_);
            chunks_ = std::move(other.chunks_);
            transmission_chunks_ = std::move(other.transmission_chunks_);
            num_chunks_ = other.num_chunks_;
            size_ = other.size_;
            data_size_ = other.data_size_;
            data_point_ = other.data_point_;

            return *this;
        }

        void clear()
        {
            data_.clear();
            chunks_.clear();
            transmission_chunks_.clear();
            num_chunks_ = count_chunks_type(0, 0);
            size_ = 0;
            data_size_ = 0;
            data_point_ = performance_counters::parcels::data_point();
        }

        BufferType data_;
        std::vector<ChunkType> chunks_;

        typedef std::pair<
            util::integer::ulittle64_t, util::integer::ulittle64_t
        > transmission_chunk_type;
        std::vector<transmission_chunk_type> transmission_chunks_;

        // pair of (zero-copy, non-zero-copy) chunks
        count_chunks_type num_chunks_;

        util::integer::ulittle64_t size_;
        util::integer::ulittle64_t data_size_;

        /// Counters and their data containers.
        performance_counters::parcels::data_point data_point_;
        HPX_MOVABLE_BUT_NOT_COPYABLE(parcel_buffer)
    };
}}

#endif
