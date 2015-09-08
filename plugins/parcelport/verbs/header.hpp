//  Copyright (c) 2013-2014 Hartmut Kaiser
//  Copyright (c) 2013-2015 Thomas Heller
//  Copyright (c)      2015 John Biddiscombe
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
    template <int SIZE>
    struct header
    {
        typedef int  value_type;
        typedef char flag_type;
        enum data_pos
        {
            pos_tag               = 0 * sizeof(value_type),
            pos_size              = 1 * sizeof(value_type),
            pos_numbytes          = 2 * sizeof(value_type),
            pos_numchunks_first   = 3 * sizeof(value_type),
            pos_numchunks_second  = 4 * sizeof(value_type),
            pos_chunk_flag        = 5 * sizeof(value_type),
            pos_piggy_back_flag   = 6 * sizeof(value_type),// + (1 * sizeof(flag_type)),
            pos_chunk_offset      = 7 * sizeof(value_type),// + (2 * sizeof(flag_type)),
            pos_piggy_back_offset = 8 * sizeof(value_type),// + (2 * sizeof(flag_type)) + 1,
            pos_data_zone         = 9 * sizeof(value_type),// + (2 * sizeof(flag_type)) + 1
        };

        static int const data_size_ = SIZE;

        template <typename Buffer>
        header(Buffer const & buffer, uint32_t tag)
        {
            boost::int64_t size = static_cast<boost::int64_t>(buffer.size_);
            boost::int64_t numbytes = static_cast<boost::int64_t>(buffer.data_size_);

            HPX_ASSERT(size <= (std::numeric_limits<value_type>::max)());
            HPX_ASSERT(numbytes <= (std::numeric_limits<value_type>::max)());

            // chunk data is not stored by default
            set<pos_chunk_flag>(static_cast<value_type>(0));
            set<pos_piggy_back_flag>(static_cast<value_type>(0));
            set<pos_chunk_offset>(static_cast<value_type>(0));
            set<pos_piggy_back_offset>(static_cast<value_type>(0));

            set<pos_tag>(static_cast<value_type>(tag));
            set<pos_size>(static_cast<value_type>(size));
            set<pos_numbytes>(static_cast<value_type>(numbytes));
            set<pos_numchunks_first>(static_cast<value_type>(buffer.num_chunks_.first));
            set<pos_numchunks_second>(static_cast<value_type>(buffer.num_chunks_.second));
            // find out how much space is needed for chunk information
            const std::vector<serialization::serialization_chunk>& chunks = buffer.chunks_;
            size_t chunkbytes = chunks.size() * sizeof(serialization::serialization_chunk);
            // can we send the chunk info inside the header
            if (chunkbytes <= (data_size_ - pos_data_zone)) {
              set<pos_chunk_flag>(static_cast<value_type>(1));
              set<pos_chunk_offset>(static_cast<value_type>(pos_data_zone));
              std::memcpy(&data_[pos_data_zone], chunks.data(), chunkbytes);
              LOG_DEBUG_MSG("Chunkbytes is " << hexnumber(chunkbytes));
            }
            else {
              chunkbytes = 0;
            }

            // the end of header position will be start of piggyback data
            set<pos_piggy_back_offset>(static_cast<value_type>(pos_data_zone + chunkbytes));

            // can we send main message chunk as well as other information
            if(buffer.data_.size() <= (data_size_ - chunkbytes - pos_data_zone)) {
                set<pos_piggy_back_flag>(static_cast<value_type>(1));
            }
        }

        header()
        {
            reset();
        }

        void reset()
        {
            std::memset(&data_[0], -1, data_size_);
            set<pos_piggy_back_flag>(static_cast<value_type>(1));
        }

        bool valid() const
        {
            return data_[0] != -1;
        }

        void assert_valid() const
        {
            HPX_ASSERT(tag() != -1);
            HPX_ASSERT(size() != -1);
            HPX_ASSERT(numbytes() != -1);
            HPX_ASSERT(num_chunks().first != -1);
            HPX_ASSERT(num_chunks().second != -1);
        }

        char *data()
        {
            return &data_[0];
        }

        value_type tag() const
        {
            return get<pos_tag>();
        }

        value_type size() const
        {
            return get<pos_size>();
        }

        value_type numbytes() const
        {
            return get<pos_numbytes>();
        }

        std::pair<value_type, value_type> num_chunks() const
        {
            return std::make_pair(get<pos_numchunks_first>(),
                get<pos_numchunks_second>());
        }

        char * piggy_back()
        {
            if(get<pos_piggy_back_flag>()!=0) {
                return &data_[get<pos_piggy_back_offset>()];
            }
            return 0;
        }

        char * chunk_data()
        {
            if(get<pos_chunk_flag>()!=0) {
                return &data_[get<pos_chunk_offset>()];
            }
            return 0;
        }

        std::size_t header_length()
        {
            // if chunks are included in header, return pos_data_zone + chunkbytes
            if(get<pos_chunk_flag>())
                return static_cast<std::size_t>(get<pos_piggy_back_offset>());
            // otherwise, just end of normal header
            else
                return static_cast<std::size_t>(pos_data_zone);
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
