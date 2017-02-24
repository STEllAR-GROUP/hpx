//  Copyright (c) 2015-2016 John Biddiscombe
//  Copyright (c) 2013-2015 Thomas Heller
//  Copyright (c) 2013-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_LIBFABRIC_HEADER_HPP
#define HPX_PARCELSET_POLICIES_LIBFABRIC_HEADER_HPP

#include <hpx/runtime/parcelset/parcel_buffer.hpp>
#include <hpx/util/assert.hpp>
//
#include <array>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <utility>
#include <vector>

// A generic header structure that can be used by parcelports
// currently, the verbs parcelports make use of it
namespace hpx {
namespace parcelset {
namespace policies {
namespace libfabric
{
    namespace detail {
        struct rdma_region {
            std::size_t  size;
            void       * key;
            const void * addr;
        };

        struct piggy_back {
            uint32_t  offset;
            uint32_t  bytes;
        };

        union rdma_info {
            piggy_back  piggyback;
            rdma_region region;
        };

        typedef std::pair<uint16_t, uint16_t> num_chunks_type;

        struct header_block {
            uint32_t        tag;
            uint32_t        size;
            num_chunks_type num_chunks;
            uint32_t        flags;
            rdma_info       chunk_info;
            rdma_info       message_info;
        };
    }

    template <int SIZE>
    struct header
    {
        static constexpr unsigned int header_block_size = sizeof(detail::header_block);
        static constexpr unsigned int data_size_        = SIZE - header_block_size;
        static const     unsigned int chunk_flag        = 0x01;
        static const     unsigned int message_flag      = 0x02;
        //
    private:
        detail::header_block         message_header;
        std::array<char, data_size_> data_;

    public:
        //
        template <typename Buffer>
        header(Buffer const & buffer, uint32_t tag)
        {
            message_header.flags      = 0;
            message_header.tag        = tag;
            message_header.size       = static_cast<uint32_t>(buffer.size_);
            message_header.num_chunks =
                std::make_pair(buffer.num_chunks_.first, buffer.num_chunks_.second);

            // find out how much space is needed for chunk information
            const std::vector<serialization::serialization_chunk>&
                chunks = buffer.chunks_;
            size_t chunkbytes = chunks.size() *
                sizeof(serialization::serialization_chunk);
            // can we send the chunk info inside the header
            if (chunkbytes <= data_size_) {
              message_header.flags |= chunk_flag;
              message_header.chunk_info.piggyback.offset = 0;
              message_header.chunk_info.piggyback.bytes  = chunkbytes;
              std::memcpy(&data_[0], chunks.data(), chunkbytes);
              LOG_DEBUG_MSG("Chunkbytes is " << decnumber(chunkbytes) <<
                  "header_block_size "
                  << decnumber(sizeof(detail::header_block)));
            }
            else {
              message_header.flags &= ~chunk_flag;
              chunkbytes = 0;
            }

            // the end of header position will be start of piggyback data
            message_header.message_info.piggyback.offset = chunkbytes;

            // can we send main message chunk as well as other information
            if (buffer.data_.size() <= (data_size_ - chunkbytes)) {
                message_header.flags |= message_flag;
            }
            else {
                message_header.flags &= ~message_flag;
            }
        }

        inline char *data() const
        {
            return &data_[0];
        }

        inline uint32_t tag() const
        {
            return message_header.tag;
        }

        inline uint32_t size() const
        {
            return message_header.size;
        }

        inline std::pair<uint32_t, uint32_t> num_chunks() const
        {
            return message_header.num_chunks;
        }

        inline char * chunk_data()
        {
            if ((message_header.flags & chunk_flag) !=0) {
                return &data_[0];
            }
            return 0;
        }

        inline char * piggy_back()
        {
            if ((message_header.flags & message_flag) !=0) {
                return &data_[message_header.message_info.piggyback.offset];
            }
            return 0;
        }

        inline std::size_t header_length() const
        {
            // if chunks are included in header, return header + chunkbytes
            if ((message_header.flags & chunk_flag) !=0)
                return sizeof(detail::header_block)
                    + message_header.chunk_info.piggyback.bytes;
            // otherwise, just end of normal header
            else
                return sizeof(detail::header_block);
        }

        inline void set_message_rdma_key(void* v) {
            message_header.chunk_info.region.key = v;
        }

        inline void* get_message_rdma_key() const {
            return message_header.chunk_info.region.key;
        }

        inline void set_message_rdma_addr(const void *v) {
            message_header.chunk_info.region.addr = v;
        }

        inline const void * get_message_rdma_addr() const {
            return message_header.chunk_info.region.addr;
        }

        inline void set_message_rdma_size(std::size_t v) {
            message_header.chunk_info.region.size = v;
        }

        inline std::size_t get_message_rdma_size() const {
            return message_header.chunk_info.region.size;
        }
    };

}}}}

#endif
