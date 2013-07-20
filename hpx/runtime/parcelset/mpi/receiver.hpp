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

    struct parcelport;

    void decode_message(
        std::size_t num_chunks,
        std::vector<char> const & parcel_data,
        parcelport& pp);

    struct receiver
    {
        receiver(header const & h, MPI_Comm communicator)
          : header_(h)
          , buffer_(h.size())
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
                &request);    // Request
        }

        bool done(parcelport & pp)
        {
            MPI_Status status;
            int completed = 0;
            MPI_Test(&request, &completed, &status);
            if(completed)
            {
                BOOST_ASSERT(status.MPI_SOURCE == header_.rank());
                BOOST_ASSERT(status.MPI_TAG == header_.tag());
#ifdef HPX_DEBUG
                int count;
                MPI_Get_count(&status, MPI_CHAR, &count);
                BOOST_ASSERT(count == header_.size());
#endif
                decode_message(header_.num_chunks(), buffer_, pp);
                return true;
            }
            return false;
        }

        private:
            header header_;
            std::vector<char> buffer_;

            MPI_Request request;
    };
}}}

#endif
