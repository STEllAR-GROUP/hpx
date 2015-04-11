//  Copyright (c) 2013-2014 Hartmut Kaiser
//  Copyright (c) 2013-2015 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_HEADER_HPP
#define HPX_PARCELSET_POLICIES_HEADER_HPP

#include <hpx/runtime/parcelset/parcel_buffer.hpp>

#include <hpx/util/assert.hpp>

#include <boost/array.hpp>
#include "plugins/parcelport/verbs/rdmahelper/include/RdmaLogging.h"

namespace hpx { namespace parcelset { namespace policies { namespace mpi
{
    template <int SIZE>
    struct header
    {
        typedef int value_type;
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
            pos_data_zone         = 8 * sizeof(value_type) + 1,// + (2 * sizeof(flag_type)) + 1
        };

        static int const data_size_ = SIZE;

        template <typename Buffer>
        header(Buffer const & buffer, int tag, bool enable_piggyback_copy=true)
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
            set<pos_numchunks_second>(static_cast<value_type>
                (buffer.num_chunks_.second));

            LOG_DEBUG_MSG("Buffer data size is " << buffer.data_.size());

            // find out how much space is needed for chunk information
            const std::vector<serialization::serialization_chunk>& chunks = buffer.chunks_;
            size_t chunkbytes = chunks.size() * sizeof(serialization::serialization_chunk);
            // can we send the chunk info inside the header
            if (chunkbytes <= (data_size_ - pos_data_zone)) {
              set<pos_chunk_flag>(static_cast<value_type>(1));
              set<pos_chunk_offset>(static_cast<value_type>(pos_data_zone));
              std::memcpy(&data_[data_[pos_chunk_offset]], chunks.data(), chunkbytes);
              LOG_DEBUG_MSG("Copied chunk data into header : size " << decnumber(chunkbytes) << " at offset " << decnumber(*(int*)&data_[pos_chunk_offset]));
            }
            else {
              chunkbytes = 0;
            }

            // can we send main chunk as well as other information
            if(buffer.data_.size() <= (data_size_ - chunkbytes - pos_data_zone)) {
                set<pos_piggy_back_flag>(static_cast<value_type>(1));
                set<pos_piggy_back_offset>(static_cast<value_type>(pos_data_zone + chunkbytes));
                LOG_DEBUG_MSG("Piggy back data in header : size "<< buffer.data_.size() << " at offset " << decnumber(*(int*)&data_[pos_piggy_back_offset]));
                if (enable_piggyback_copy) {
                  std::memcpy(&data_[data_[pos_piggy_back_offset]], &buffer.data_[0], buffer.data_.size());
                  LOG_DEBUG_MSG("Copying piggy_back data_[pos_piggy_back_flag] = " << decnumber((int)(data_[pos_piggy_back_flag])));
                }
            }
        }

        header()
        {
            reset();
        }

        void reset()
        {
            std::memset(&data_[0], -1, data_size_);
            data_[pos_piggy_back_flag] = 1;
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
            if(data_[pos_piggy_back_flag])
                return &data_[data_[pos_piggy_back_offset]];
            return 0;
        }

        char * chunk_data()
        {
            if(data_[pos_chunk_flag])
                return &data_[data_[pos_chunk_offset]];
            return 0;
        }

        std::size_t header_length()
        {
            if(data_[pos_chunk_flag])
                return *reinterpret_cast<value_type*>(&data_[pos_piggy_back_offset]);
            else
                return *reinterpret_cast<value_type*>(&data_[pos_chunk_offset]);
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
    };

    struct tag_provider
    {
        typedef lcos::local::spinlock mutex_type;

        struct tag
        {
            tag(tag_provider *provider)
              : provider_(provider)
              , tag_(provider_->acquire())
            {}

            operator int () const
            {
                return tag_;
            }

            ~tag()
            {
                provider_->release(tag_);
            }

            tag_provider *provider_;
            int tag_;
        };

        tag_provider()
          : next_tag_(1)
        {}

        tag operator()()
        {
            return tag(this);
        }

        int acquire()
        {
            mutex_type::scoped_lock l(mtx_);
            if(free_tags_.empty())
                return next_tag_++;

            int tag = free_tags_.front();
            free_tags_.pop_front();
            return tag;
        }

        void release(int tag)
        {
            if(tag == next_tag_) return;

            mutex_type::scoped_lock l(mtx_);
            free_tags_.push_back(tag);
        }

        mutex_type mtx_;
        int next_tag_;
        std::deque<int> free_tags_;
    };

}}}

#endif

#endif
