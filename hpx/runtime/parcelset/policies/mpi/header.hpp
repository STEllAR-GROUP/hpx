//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_MPI_HEADER_HPP
#define HPX_PARCELSET_POLICIES_MPI_HEADER_HPP

#include <mpi.h>

#include <hpx/util/assert.hpp>
#include <hpx/util/mpi_environment.hpp>

#include <boost/array.hpp>

namespace hpx { namespace parcelset { namespace policies { namespace mpi
{
    struct header
    {
        typedef int value_type;

        static int const data_size_ = 3;

        header(int rank, value_type tag_, std::size_t size_, std::size_t numbytes_)
          : rank_(rank)
        {
            HPX_ASSERT(rank_ != util::mpi_environment::rank());
            HPX_ASSERT(size_ <= (std::numeric_limits<value_type>::max)());
            HPX_ASSERT(numbytes_ <= (std::numeric_limits<value_type>::max)());

            data_[0] = tag_;
            data_[1] = static_cast<value_type>(size_);
            data_[2] = static_cast<value_type>(numbytes_);
        }

        header()
          : rank_(-1)
        {
            data_[0] = -1;
            data_[1] = -1;
            data_[2] = -1;
        }

        void assert_valid() const
        {
            HPX_ASSERT(rank_ != util::mpi_environment::rank());
            HPX_ASSERT(rank() != -1);
            HPX_ASSERT(tag() != -1);
            HPX_ASSERT(size() != -1);
            HPX_ASSERT(numbytes() != -1);
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
