//  Copyright (c) 2013-2014 Hartmut Kaiser
//  Copyright (c) 2013-2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_MPI_HEADER_HPP
#define HPX_PARCELSET_POLICIES_MPI_HEADER_HPP

#include <mpi.h>

#include <hpx/runtime/parcelset/parcel_buffer.hpp>

#include <hpx/util/assert.hpp>
#include <hpx/util/mpi_environment.hpp>

#include <boost/array.hpp>

namespace hpx { namespace parcelset { namespace policies { namespace mpi
{
    struct header
    {
        typedef int value_type;

        static int const data_size_ = 5;

        template <typename BufferType, typename ChunkType>
        header(int rank, value_type tag_,
            parcel_buffer<BufferType, ChunkType> const & buffer)
          : rank_(rank)
        {
            boost::uint64_t size = static_cast<boost::uint64_t>(buffer.size_);
            boost::uint64_t numbytes = static_cast<boost::uint64_t>(buffer.data_size_);

            HPX_ASSERT(rank_ != util::mpi_environment::rank());
            HPX_ASSERT(size <= (std::numeric_limits<value_type>::max)());
            HPX_ASSERT(numbytes <= (std::numeric_limits<value_type>::max)());

            data_[0] = tag_;
            data_[2] = static_cast<value_type>(size);
            data_[1] = static_cast<value_type>(numbytes);
            data_[3] = buffer.num_chunks_.first;
            data_[4] = buffer.num_chunks_.second;
        }


        header()
          : rank_(-1)
        {
            data_[0] = -1;
            data_[1] = -1;
            data_[2] = -1;
            data_[3] = -1;
            data_[4] = -1;
        }

        static header close(int tag, int rank)
        {
            header h;
            h.rank() = rank;
            h.tag() = tag;
            h.size() = -1;
            h.numbytes() = -1;
            return h;
        }

        bool close_request() const
        {
            return (size() == -1) && (numbytes() == -1);
        }

        void assert_valid() const
        {
            HPX_ASSERT(rank_ != util::mpi_environment::rank());
            HPX_ASSERT(rank() != -1);
            HPX_ASSERT(tag() != -1);
            HPX_ASSERT(size() != -1);
            HPX_ASSERT(numbytes() != -1);
            HPX_ASSERT(num_chunks_first() != -1);
            HPX_ASSERT(num_chunks_second() != -1);
        }

        value_type *data()
        {
            return &data_[0];
        }

        value_type const & rank() const
        {
            return rank_;
        }

        value_type const & tag() const
        {
            return data_[0];
        }

        value_type const & size() const
        {
            return data_[1];
        }

        value_type const & numbytes() const
        {
            return data_[2];
        }

        value_type const & num_chunks_first() const
        {
            return data_[3];
        }

        value_type const & num_chunks_second() const
        {
            return data_[4];
        }

        value_type & rank()
        {
            return rank_;
        }

        value_type & tag()
        {
            return data_[0];
        }

        value_type & size()
        {
            return data_[1];
        }

        value_type & numbytes()
        {
            return data_[2];
        }

        value_type & num_chunks_first()
        {
            return data_[3];
        }

        value_type & num_chunks_second()
        {
            return data_[4];
        }

        MPI_Datatype type()
        {
            return MPI_INT;
        }

    private:
        int rank_;
        boost::array<value_type, data_size_> data_;
    };
}}}}

#endif
