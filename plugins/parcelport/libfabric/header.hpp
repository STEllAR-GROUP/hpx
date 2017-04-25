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
    namespace detail
    {
        // data we send if message is piggybacked
        struct message_info {
            uint64_t message_size;
        };

        // data we send if there are zero copy blocks (or a non piggybacked header)
        struct rma_info {
            uint64_t tag;
        };

        // zerocopy, normal
        typedef std::pair<uint16_t, uint16_t> num_chunks_type;

        struct header_block {
            uint32_t  num_chunks;
            uint32_t  flags;     // for padding to nice boundary
        };
    }

    template <int SIZE>
    struct header
    {
        static constexpr unsigned int header_block_size = sizeof(detail::header_block);
        static constexpr unsigned int data_size_        = SIZE - header_block_size;
        //
        static const unsigned int chunk_flag    = 0x01; // chunks piggybacked
        static const unsigned int message_flag  = 0x02; // message pigybacked
        static const unsigned int normal_flag   = 0x04; // normal chunks present
        static const unsigned int zerocopy_flag = 0x08; // zerocopy chunks present
        //
        typedef serialization::serialization_chunk chunktype;

    private:
        //
        // this is the actual header content
        //
        detail::header_block         message_header;
        std::array<char, data_size_> data_;
        // the data block is laid out as follows for each optional item
        // message_header - always present header_block_size
        // chunk data   : sizeof(chunktype) * numchunks : when chunks included
        // rma_info     : sizeof(rma_info) : only when we have zero_copy chunks
        // message_info : sizeof(message_info) : only when message pigybacked
        // .....
        // message      : buffer.size_ : only when message piggybacked

    public:
        //
        template <typename Buffer>
        header(Buffer const & buffer, void* tag)
        {
            const std::vector<chunktype>& chunks = buffer.chunks_;
            //
            message_header.flags      = 0;
            message_header.num_chunks = chunks.size();
            message_header.flags     |= buffer.num_chunks_.first  ? zerocopy_flag : 0;
            message_header.flags     |= buffer.num_chunks_.second ? normal_flag : 0;

            // space occupied by chunk data
            size_t chunkbytes = chunks.size() * sizeof(chunktype);

            // can we send main message inside the header
            if (buffer.data_.size() <= (data_size_ - chunkbytes)) {
                message_header.flags |= message_flag;
            }
            else {
                message_header.flags = 0;
            }
            // can we send the chunk info inside the header
            // add one chunk size offset just in case the message is not being piggybacked
            if ((chunkbytes+sizeof(chunktype)) <= data_size_)  {
              message_header.flags |= chunk_flag;
              std::memcpy(&data_[0], chunks.data(), chunkbytes);
            }
            else
            {
                // not fitting the chunk info is an error currently
                // an exception will be thrown on the receive end
              message_header.flags &= ~chunk_flag;
              chunkbytes = 0;
            }

            // if the message is not piggybacked and is being sent as a zero copy chunk
            if ((message_header.flags & message_flag) == 0) {
                // add one chunk for the message region
                message_header.num_chunks += 1;
                chunktype message = serialization::create_pointer_chunk(
                    nullptr, buffer.size_, 0);
                std::memcpy(&data_[chunkbytes], &message, sizeof(chunktype));
                // we set zerocopy true for piggybacked data
                message_header.flags |= zerocopy_flag;
            }
            else {
                detail::message_info *info = message_info_ptr();
                info->message_size = buffer.size_;
            }
            // set the rma tag
            if ((message_header.flags & zerocopy_flag) != 0) {
                auto ptr = rma_info_ptr();
                ptr->tag = reinterpret_cast<uint64_t>(tag);
        }

            LOG_DEBUG_MSG("Header : " << *this);
        }

        // --------------------------------------------------------------------
        friend std::ostream & operator<<(std::ostream & os, header<SIZE> & h)
        {
            os  << "Flags " << hexbyte(h.message_header.flags)
                << "chunk_data_offset " << decnumber(h.chunk_data_offset())
                << "rma_info_offset " << decnumber(h.rma_info_offset())
                << "message_info_offset " << decnumber(h.message_info_offset())
                << "message_offset " << decnumber(h.message_offset())
                << "header length " << decnumber(h.header_length())
                << "message length " << hexlength(h.message_size())
                << "chunks " << decnumber(h.num_chunks())
                << "zerocopy ( " << decnumber(h.num_zero_copy_chunks()) << ") "
                << "normal ( " << decnumber(h.num_index_chunks()) << ") "
                << "piggyback " << decnumber((h.message_piggy_back()))
                << "tag " << hexuint64(h.tag());
            return os;
        }

    private:
        // ------------------------------------------------------------------
        inline char *chunk_ptr()
        {
            if ((message_header.flags & chunk_flag) == 0) {
                return nullptr;
            }
            return reinterpret_cast<char *>(&data_[chunk_data_offset()]);
        }

        // ------------------------------------------------------------------
        inline detail::rma_info *rma_info_ptr()
        {
            if ((message_header.flags & zerocopy_flag) == 0) {
                return nullptr;
            }
            return reinterpret_cast<detail::rma_info *>(&data_[rma_info_offset()]);
        }

        // ------------------------------------------------------------------
        inline detail::message_info *message_info_ptr()
        {
            if ((message_header.flags & message_flag) == 0) {
                return nullptr;
            }
            return reinterpret_cast<detail::message_info*>
                (&data_[message_info_offset()]);
        }

        // ------------------------------------------------------------------
        inline char *message_ptr()
        {
            if ((message_header.flags & message_flag) == 0) {
                return nullptr;
            }
            return reinterpret_cast<char*>(&data_[message_offset()]);
        }

        // ------------------------------------------------------------------
        inline uint32_t chunk_data_offset() const
        {
            // just in case we ever add any new stuff
            return 0;
        }

        inline uint32_t rma_info_offset() const
        {
            // add the chunk data offset
            std::uint32_t size = chunk_data_offset();
            if ((message_header.flags & chunk_flag) !=0) {
                size = (message_header.num_chunks * sizeof(chunktype));
            }
            return size;
        }

        inline uint32_t message_info_offset() const
        {
            // add the rma info offset
            std::uint32_t size = rma_info_offset();
            if ((message_header.flags & zerocopy_flag) != 0) {
                size += sizeof(detail::rma_info);
            }
            return size;
        }

        inline uint32_t message_offset() const
        {
            // add the message info offset
            std::uint32_t size = message_info_offset();
            if ((message_header.flags & message_flag) !=0) {
                size += sizeof(detail::message_info);
            }
            return size;
        }

    public:

        // ------------------------------------------------------------------
        // here beginneth the public API
        // ------------------------------------------------------------------
        inline char * chunk_data()
        {
            return chunk_ptr();
        }

        inline char *message_data()
        {
            return message_ptr();
        }

        inline bool message_piggy_back()
        {
            return message_ptr()!=nullptr;
        }

        inline uint64_t tag()
        {
            auto ptr = rma_info_ptr();
            return ptr ? ptr->tag : 0;
        }

        inline uint32_t message_size()
        {
            auto ptr = message_info_ptr();
            if (ptr) {
                return ptr->message_size;
        }
            // if the data is not piggybacked then look at the final chunk
            chunktype *chunks = reinterpret_cast<chunktype *>(chunk_ptr());
            if (!chunks) {
                throw std::runtime_error("Don't call this without chunk data");
            }
            return chunks[message_header.num_chunks-1].size_;
        }

        // the full size of all the header information
        inline std::uint32_t header_length()
        {
            std::uint32_t size = header_block_size + message_offset();
            return size;
        }

        inline void set_message_rdma_info(uint64_t key, const void *addr)
        {
            chunktype *chunks = reinterpret_cast<chunktype *>(chunk_ptr());
            if (!chunks) {
                throw std::runtime_error("Don't call this without chunk data");
            }
            // the last chunk will be our RMA message chunk
            chunks[message_header.num_chunks-1].rkey_       = key;
            chunks[message_header.num_chunks-1].data_.cpos_ = addr;
        }

        std::uint32_t num_chunks()
        {
            return message_header.num_chunks;
        }

        std::uint32_t num_zero_copy_chunks()
        {
            chunktype *chunks = reinterpret_cast<chunktype *>(chunk_ptr());
            if (!chunks) {
                throw std::runtime_error("Don't call this without chunk data");
            }
            uint32_t num=0;
            for (uint32_t i=0; i<message_header.num_chunks; ++i) {
                if (chunks[i].type_ == serialization::chunk_type::chunk_type_pointer) {
                    ++num;
                }
            }
            return num;
        }

        std::uint32_t num_original_zero_copy_chunks()
        {
            if ((message_header.flags & message_flag) == 0) {
                // subtract one chunk for the message region
                return num_zero_copy_chunks() - 1;
            }
            return num_zero_copy_chunks();
        }

        std::uint32_t num_index_chunks()
        {
            chunktype *chunks = reinterpret_cast<chunktype *>(chunk_ptr());
            if (!chunks) {
                throw std::runtime_error("Don't call this without chunk data");
            }
            uint32_t num=0;
            for (uint32_t i=0; i<message_header.num_chunks; ++i) {
                if (chunks[i].type_ == serialization::chunk_type::chunk_type_index) {
                    ++num;
                }
            }
            return num;
        }

        std::vector<chunktype> get_message_chunks()
        {
            chunktype *chunks = reinterpret_cast<chunktype *>(chunk_ptr());
            if (!chunks) {
                throw std::runtime_error("Don't call this without chunk data");
            }
            std::vector<chunktype> new_chunks(message_header.num_chunks);
            std::memcpy(new_chunks.data(), chunks,
                message_header.num_chunks*sizeof(chunktype));
            return new_chunks;
        }
    };

}}}}

#endif
