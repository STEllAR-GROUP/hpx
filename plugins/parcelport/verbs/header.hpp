//  Copyright (c) 2015-2016 John Biddiscombe
//  Copyright (c) 2013-2015 Thomas Heller
//  Copyright (c) 2013-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_VERBS_HEADER_HPP
#define HPX_PARCELSET_POLICIES_VERBS_HEADER_HPP

#include <hpx/runtime/parcelset/parcel_buffer.hpp>

#include <hpx/util/assert.hpp>

#include <boost/array.hpp>

// A generic header structure that can be used by parcelports
// currently, the mpi and verbs parcelports make use of it
namespace hpx { namespace parcelset {
namespace policies { namespace verbs
{
    struct rdma_region {
        std::size_t     size;
        const void *    addr;
        boost::uint32_t key;
    };

    template <int SIZE>
    struct header
    {
        typedef boost::uint32_t value_type;
        typedef char            flag_type;
        enum data_pos
        {
            pos_message_tag       = 0 * sizeof(value_type),
            pos_message_size      = 1 * sizeof(value_type),
            pos_numchunks_first   = 3 * sizeof(value_type),
            pos_numchunks_second  = 4 * sizeof(value_type),
            pos_flags             = 5 * sizeof(value_type),
            pos_chunk_offset      = 6 * sizeof(value_type),
            pos_piggy_back_offset = 7 * sizeof(value_type),
            pos_piggy_back_key    = 8 * sizeof(value_type),
            pos_piggy_back_addr   = 9 * sizeof(value_type),
            pos_data_zone         = 9 * sizeof(value_type) + sizeof(const void *)
        };
        //
        static const unsigned int data_size_      = SIZE;
        static const unsigned int chunk_flag      = 0x01;
        static const unsigned int piggy_back_flag = 0x02;
        //
        template <typename Buffer>
        header(Buffer const & buffer, uint32_t tag)
        {
            boost::int64_t size = static_cast<boost::int64_t>(buffer.size_);

            HPX_ASSERT(size <= (std::numeric_limits<value_type>::max)());

            set<pos_message_tag>(static_cast<value_type>(tag));
            set<pos_message_size>(static_cast<value_type>(size));
            set<pos_numchunks_first>(static_cast<value_type>(buffer.num_chunks_.first));
            set<pos_numchunks_second>(static_cast<value_type>(buffer.num_chunks_.second));

            // find out how much space is needed for chunk information
            const std::vector<serialization::serialization_chunk>& chunks = buffer.chunks_;
            size_t chunkbytes = chunks.size() * sizeof(serialization::serialization_chunk);
            // can we send the chunk info inside the header
            if (chunkbytes <= (data_size_ - pos_data_zone)) {
              set<pos_flags>(get<pos_flags>() | static_cast<value_type>(chunk_flag));
              set<pos_chunk_offset>(static_cast<value_type>(pos_data_zone));
              std::memcpy(&data_[pos_data_zone], chunks.data(), chunkbytes);
              LOG_DEBUG_MSG("Chunkbytes is " << hexnumber(chunkbytes));
            }
            else {
              set<pos_flags>(get<pos_flags>() & ~chunk_flag);
              set<pos_chunk_offset>(static_cast<value_type>(0));
              chunkbytes = 0;
            }

            // the end of header position will be start of piggyback data
            set<pos_piggy_back_offset>(static_cast<value_type>(pos_data_zone + chunkbytes));

            // can we send main message chunk as well as other information
            if (buffer.data_.size() <= (data_size_ - chunkbytes - pos_data_zone)) {
                set<pos_flags>(get<pos_flags>() | static_cast<value_type>(piggy_back_flag));
            }
            else {
                set<pos_flags>(get<pos_flags>() & ~piggy_back_flag);
            }
        }

        header()
        {
            reset();
        }

        void reset()
        {
            std::memset(&data_[0], -1, data_size_);
            set<pos_flags>(0xFF);
        }

        bool valid() const
        {
            return data_[0] != -1;
        }

        char *data()
        {
            return &data_[0];
        }

        value_type tag() const
        {
            return get<pos_message_tag>();
        }

        value_type size() const
        {
            return get<pos_message_size>();
        }

        std::pair<value_type, value_type> num_chunks() const
        {
            return std::make_pair(get<pos_numchunks_first>(),
                get<pos_numchunks_second>());
        }

        char * piggy_back()
        {
            if((get<pos_flags>() & piggy_back_flag) !=0) {
                return &data_[get<pos_piggy_back_offset>()];
            }
            return 0;
        }

        char * chunk_data()
        {
            if((get<pos_flags>() & chunk_flag) !=0) {
                return &data_[get<pos_chunk_offset>()];
            }
            return 0;
        }

        std::size_t header_length()
        {
            // if chunks are included in header, return pos_data_zone + chunkbytes
            if((get<pos_flags>() & chunk_flag) != 0)
                return static_cast<std::size_t>(get<pos_piggy_back_offset>());
            // otherwise, just end of normal header
            else
                return static_cast<std::size_t>(pos_data_zone);
        }

        void setRdmaKey(value_type v) {
            set<pos_piggy_back_key>(v);
        }

        value_type GetRdmaKey() const {
            return get<pos_piggy_back_key>();
        }

        void setRdmaAddr(const void *v) {
            set<pos_piggy_back_addr>(v);
        }

        const void * GetRdmaAddr() const {
            void * res;
            std::memcpy(&res, &data_[pos_piggy_back_addr], sizeof(res));
            return res;
        }

        void setRdmaMessageLength(value_type v) {
            set<pos_piggy_back_offset>(v);
        }

        value_type GetRdmaMessageLength() const {
            return get<pos_piggy_back_offset>();
        }

    private:
        boost::array<char, data_size_> data_;

        template <std::size_t Pos, typename T>
        void set(T const & t)
        {
            std::memcpy(&data_[Pos], &t, sizeof(t));
        }

        template <std::size_t Pos>
        value_type get() const
        {
            value_type res;
            std::memcpy(&res, &data_[Pos], sizeof(res));
            return res;
        }

        friend std::ostream & operator<<(std::ostream & os, header const * h)
        {
            boost::io::ios_flags_saver ifs(os);
            for (int i=0; i<10; i++) {
                value_type res;
                std::memcpy(&res, &h->data_[i*sizeof(value_type)], sizeof(res));
                os << res << ", ";
            }
            return os;
        }
    };

}}}}

#endif
