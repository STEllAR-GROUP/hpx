//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/runtime/parcelset/mpi/header.hpp>

#include <boost/assert.hpp>
#include <boost/move/move.hpp>

#include <vector>
#include <utility>

#ifndef HPX_PARCELSET_MPI_RECEIVER_HPP
#define HPX_PARCELSET_MPI_RECEIVER_HPP

namespace hpx { namespace parcelset { namespace mpi {

    class parcelport;

    void decode_message(
        std::vector<char> const & parcel_data,
        parcelport& pp);

    struct receiver
      : boost::noncopyable
    {
        receiver(header const & h, MPI_Comm communicator)
          : header_(h)
          , buffer_(h.size(), 0xcd)
        {
            header_.assert_valid();
            BOOST_ASSERT(header_.rank() != util::mpi_environment::rank());
            MPI_Irecv(
                buffer_.data(), // data pointer
                static_cast<int>(buffer_.size()), // number of elements
                MPI_CHAR,     // MPI Datatype
                header_.rank(), // Source
                header_.tag(),  // Tag
                communicator, // Communicator
                &request_);    // Request
        }

        bool done(parcelport & pp)
        {
            MPI_Status status = { 0 };
            int completed = 0;
            MPI_Test(&request_, &completed, &status);
            if(completed)
            {
                BOOST_ASSERT(status.MPI_SOURCE == header_.rank());
                BOOST_ASSERT(status.MPI_SOURCE != util::mpi_environment::rank());
                BOOST_ASSERT(status.MPI_TAG == header_.tag());
#ifdef HPX_DEBUG
                int count = 0;
                MPI_Get_count(&status, MPI_CHAR, &count);
                BOOST_ASSERT(count == header_.size());
                BOOST_ASSERT(static_cast<std::size_t>(count) == buffer_.size());
#endif
                decode_message(buffer_, pp);
                return true;
            }
            return false;
        }

        private:
            header header_;
            std::vector<char> buffer_;

            MPI_Request request_;
    };
}}}

#endif
