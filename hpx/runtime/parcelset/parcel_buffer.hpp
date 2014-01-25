//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c)      2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_PARCEL_BUFFER_HPP
#define HPX_PARCELSET_PARCEL_BUFFER_HPP

#include <hpx/config.hpp>
#include <hpx/util/portable_binary_archive.hpp>
#include <hpx/performance_counters/parcels/data_point.hpp>

#include <boost/integer/endian.hpp>

#include <vector>

namespace hpx { namespace parcelset
{
    template <typename BufferType, typename ChunkType = util::serialization_chunk>
    struct parcel_buffer
    {
        typedef std::pair<
            boost::integer::ulittle32_t, boost::integer::ulittle32_t
        > count_chunks_type;

        parcel_buffer()
          : num_chunks_(count_chunks_type(0, 0))
          , size_(0), data_size_(0)
        {}

        parcel_buffer(BufferType const & data)
          : data_(data)
          , num_chunks_(count_chunks_type(0, 0))
          , size_(0), data_size_(0)
        {}

        parcel_buffer(BufferType && data)
          : data_(std::move(data))
          , num_chunks_(count_chunks_type(0, 0))
          , size_(0), data_size_(0)
        {}

        BufferType data_;
        std::vector<ChunkType> chunks_;

        typedef std::pair<boost::integer::ulittle64_t, boost::integer::ulittle64_t>
            transmission_chunk_type;
        std::vector<transmission_chunk_type> transmission_chunks_;

        // pair of (zero-copy, non-zero-copy) chunks
        count_chunks_type num_chunks_;

        boost::integer::ulittle64_t size_;
        boost::integer::ulittle64_t data_size_;

        /// Counters and their data containers.
        performance_counters::parcels::data_point data_point_;
    };
}}

#endif
