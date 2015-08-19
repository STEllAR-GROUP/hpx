//  Copyright (c) 2013-2014 Hartmut Kaiser
//  Copyright (c) 2013-2015 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_MPI_HEADER_HPP
#define HPX_PARCELSET_POLICIES_MPI_HEADER_HPP

#include <mpi.h>

#include <hpx/runtime/parcelset/parcel_buffer.hpp>

#include <hpx/util/assert.hpp>
#include <hpx/plugins/parcelport/mpi/mpi_environment.hpp>

#include <boost/array.hpp>

namespace hpx { namespace parcelset { namespace policies { namespace mpi
{
    struct header
    {
        typedef int value_type;
        enum data_pos
        {
            pos_tag              = 0 * sizeof(value_type),
            pos_size             = 1 * sizeof(value_type),
            pos_numbytes         = 2 * sizeof(value_type),
            pos_numchunks_first  = 3 * sizeof(value_type),
            pos_numchunks_second = 4 * sizeof(value_type),
            pos_piggy_back_flag  = 5 * sizeof(value_type),
            pos_piggy_back_data  = 5 * sizeof(value_type) + 1
        };

        static int const data_size_ = 512;

        template <typename Buffer>
        header(Buffer const & buffer, int tag)
        {
            boost::int64_t size = static_cast<boost::int64_t>(buffer.size_);
            boost::int64_t numbytes = static_cast<boost::int64_t>(buffer.data_size_);

            HPX_ASSERT(size <= (std::numeric_limits<value_type>::max)());
            HPX_ASSERT(numbytes <= (std::numeric_limits<value_type>::max)());

            set<pos_tag>(tag);
            set<pos_size>(static_cast<value_type>(size));
            set<pos_numbytes>(static_cast<value_type>(numbytes));
            set<pos_numchunks_first>(static_cast<value_type>(buffer.num_chunks_.first));
            set<pos_numchunks_second>(static_cast<value_type>
                (buffer.num_chunks_.second));

            if(buffer.data_.size() <= (data_size_ - pos_piggy_back_data))
            {
                data_[pos_piggy_back_flag] = 1;
                std::memcpy(&data_[pos_piggy_back_data], &buffer.data_[0],
                    buffer.data_.size());
            }
            else
            {
                data_[pos_piggy_back_flag] = 0;
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
                return &data_[pos_piggy_back_data];
            return 0;
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
}}}}

#endif
