//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_MPI_HEADER_HPP
#define HPX_PARCELSET_MPI_HEADER_HPP

namespace hpx { namespace parcelset { namespace mpi {
    struct header
    {
        typedef int value_type;

        header(int rank, value_type tag_, value_type num_chunks_, value_type size_)
          : rank_(rank)
        {
          data_[0] = tag_;
          data_[1] = num_chunks_;
          data_[2] = size_;
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
            BOOST_ASSERT(rank() != -1);
            BOOST_ASSERT(tag() != -1);
            BOOST_ASSERT(num_chunks() != -1);
            BOOST_ASSERT(size() != -1);
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

        value_type const & num_chunks() const
        {
            return data_[1];
        }

        value_type const & size() const
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

        value_type & num_chunks()
        {
            return data_[1];
        }

        value_type & size()
        {
            return data_[2];
        }

        MPI_Datatype type()
        {
            return MPI_INT;
        }



        private:
            int rank_;
            boost::array<value_type, 3> data_;
    };
}}}

#endif
