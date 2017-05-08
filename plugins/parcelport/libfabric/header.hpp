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
// currently, the libfabric parcelport makes use of it
namespace hpx {
namespace parcelset {
namespace policies {
namespace libfabric
{
    namespace detail
    {
        typedef serialization::serialization_chunk chunktype;

        // if chunks are not piggybacked, we must send an rma handle for chunk acccess
        // and state how many other rma chunks need to be retrieved (since this is
        // normally stored in the missing chunk info)
        struct chunk_header {
            uint32_t  num_rma_chunks;
            chunktype chunk_rma;
        };

        // data we send if there are zero copy blocks (or non piggybacked header/chunks)
        struct rma_info {
            uint64_t tag;
        };

        // data we send if message is piggybacked
        struct message_info {
            uint64_t message_size;
        };

        // data we send if both message and chunk data are -not- piggybacked
        // to store the rma information for the message (that otherwise whould be in
        // the chunk data)
        struct message_chunk {
            chunktype message_rma;
        };

        // this header block is always sent
        struct header_block {
            uint32_t  num_chunks;
            uint32_t  flags;     // for padding to nice boundary (only need a few bits)
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

        typedef serialization::serialization_chunk chunktype;

    private:
        //
        // this is the actual header content
        //
        detail::header_block         message_header;
        std::array<char, data_size_> data_;
        // the data block is laid out as follows for each optional item
        // message_header - always present header_block_size
        // chunk data   : sizeof(chunktype) * numchunks : when chunks piggybacked
        //           or : sizeof(chunk_header)  : when chunks not piggybacked
        // rma_info     : sizeof(rma_info)      : when we have anything to be rma'd
        // message_info : sizeof(message_info)  : only when message pigybacked
        //           or : sizeof(message_chunk) : when message+chunk both not piggybacked
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

            // can we send the chunk info inside the header
            // (NB. we add +1 chunk just in case of a non piggybacked message chunk)
            if ((chunkbytes+sizeof(chunktype)) <= data_size_)  {
                message_header.flags |= chunk_flag;
                // copy chunk data directly into the header
                std::memcpy(&data_[chunk_data_offset()], chunks.data(), chunkbytes);
            }
            else
            {
                LOG_DEBUG_MSG("Too many chunks for header "
                    << decnumber(chunks.size())
                    << "requires bytes " << decnumber(chunkbytes));
                message_header.flags &= ~chunk_flag;
                message_header.flags |= zerocopy_flag;
                // send just rma-get information, address and rma key will be added later
                detail::chunk_header *ch =
                    reinterpret_cast<detail::chunk_header*>(&data_[chunk_data_offset()]);
                ch->num_rma_chunks = buffer.num_chunks_.first;
                ch->chunk_rma =
                    serialization::create_pointer_chunk(nullptr, chunkbytes, 0);
                // reset chunkbytes size to size of rma hunk header
                chunkbytes = sizeof(detail::chunk_header);
            }

            // can we send main message inside the header
            if (buffer.data_.size() <= (data_size_ - chunkbytes)) {
                message_header.flags |= message_flag;
                detail::message_info *info = message_info_ptr();
                info->message_size = buffer.size_;
            }
            else {
                message_header.flags &= ~message_flag;
                message_header.flags |= zerocopy_flag;
                if ((message_header.flags & chunk_flag) != 0) {
                    // if chunks are piggybacked, just add one rma chunk for the message
                    message_header.num_chunks += 1;
                    chunktype message =
                        serialization::create_pointer_chunk(nullptr, buffer.size_, 0);
                    std::memcpy(&data_[chunkbytes], &message, sizeof(chunktype));
                }
                else {
                    // the message isn't piggybacked and neither is the chunk data
                    // so we must add rma-get information for the message
                    detail::message_chunk *mc =
                        reinterpret_cast<detail::message_chunk*>(&data_[chunkbytes]);
                    mc->message_rma =
                        serialization::create_pointer_chunk(nullptr, buffer.size_, 0);
                }
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

    public:
        // ------------------------------------------------------------------
        // if chunks are piggybacked, return pointer to list of chunk data
        inline char *chunk_ptr()
        {
            if ((message_header.flags & chunk_flag) == 0) {
                return nullptr;
            }
            return reinterpret_cast<char *>(&data_[chunk_data_offset()]);
        }

        // ------------------------------------------------------------------
        // if chunks are not piggybacked, return pointer to chunk rma info
        inline detail::chunk_header *chunk_header_ptr()
        {
            if ((message_header.flags & chunk_flag) == 0) {
                return reinterpret_cast<detail::chunk_header *>
                    (&data_[chunk_data_offset()]);
            }
            return nullptr;
        }

        // ------------------------------------------------------------------
        // if there are rma blocks, return pointer to the rma tag
        inline detail::rma_info *rma_info_ptr()
        {
            if ((message_header.flags & zerocopy_flag) == 0) {
                return nullptr;
            }
            return reinterpret_cast<detail::rma_info *>(&data_[rma_info_offset()]);
        }

        // ------------------------------------------------------------------
        // if message is piggybacked, return pointer to start of message block
        inline detail::message_info *message_info_ptr()
        {
            if ((message_header.flags & message_flag) == 0) {
                return nullptr;
            }
            return reinterpret_cast<detail::message_info*>
                (&data_[message_info_offset()]);
        }

        // ------------------------------------------------------------------
        // if message+chunk are not piggybacked, return pointer to message chunk
        inline detail::message_chunk *message_chunk_ptr()
        {
            if ((message_header.flags & message_flag) == 0 &&
                (message_header.flags & chunk_flag) == 0)
            {
                return reinterpret_cast<detail::message_chunk*>
                    (&data_[message_info_offset()]);
            }
            return nullptr;
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
            else {
                // chunks are not piggybacked, insert rma details
                size = sizeof(detail::chunk_header);
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
            else if ((message_header.flags & message_flag) == 0 &&
                     (message_header.flags & chunk_flag) == 0)
            {
                size += sizeof(detail::message_chunk);
            }
            return size;
        }

        // ------------------------------------------------------------------
        // here beginneth the main public API
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
                throw std::runtime_error("Don't call message_size without chunk data");
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
                detail::message_chunk *mc = message_chunk_ptr();
                chunks = &mc->message_rma;
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
                throw std::runtime_error("num_zero_copy_chunks without chunk data");
            }
            uint32_t num=0;
            for (uint32_t i=0; i<message_header.num_chunks; ++i) {
                if (chunks[i].type_ == serialization::chunk_type::chunk_type_pointer) {
                    ++num;
                }
            }
            return num;
        }

        std::uint32_t num_index_chunks()
        {
            chunktype *chunks = reinterpret_cast<chunktype *>(chunk_ptr());
            if (!chunks) {
                throw std::runtime_error("num_index_chunks without chunk data");
            }
            uint32_t num=0;
            for (uint32_t i=0; i<message_header.num_chunks; ++i) {
                if (chunks[i].type_ == serialization::chunk_type::chunk_type_index) {
                    ++num;
                }
            }
            return num;
        }

    };

}}}}

#endif
