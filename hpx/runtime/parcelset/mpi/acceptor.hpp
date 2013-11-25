//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_MPI_ACCEPTOR_HPP
#define HPX_PARCELSET_MPI_ACCEPTOR_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/parcelset/mpi/header.hpp>

#include <hpx/assert.hpp>
#include <boost/noncopyable.hpp>

#include <vector>
#include <utility>

namespace hpx { namespace parcelset { namespace mpi {
    struct acceptor
      : boost::noncopyable
    {
        acceptor()
          : communicator_(0)
        {
        }

        void run(MPI_Comm communicator)
        {
            communicator_ = communicator;
            irecv_header();
        }

        std::pair<bool, header> next_header()
        {
            MPI_Status status;
            int completed = 0;
            MPI_Test(&request_, &completed, &status);
            if(completed)
            {
                HPX_ASSERT(header_.rank() == -1);
                header h = header_;
                h.rank() = status.MPI_SOURCE;
                h.assert_valid();
                irecv_header();
                return std::make_pair(true, h);
            }
            return std::make_pair(false, header());
        }

    private:
        MPI_Comm communicator_;
        MPI_Request request_;
        header header_;

        void irecv_header()
        {
            /*
            HPX_ASSERT(source_ != -1);
            HPX_ASSERT(source_ != util::mpi_environment::rank());
            */
            MPI_Irecv(
                header_.data(), // data pointer
                header_.data_size_,     // number of elements
                header_.type(), // MPI Datatype
                MPI_ANY_SOURCE,         // Source
                0,              // Tag
                communicator_,  // Communicator
                &request_);     // Request
        }
    };
}}}

#endif
