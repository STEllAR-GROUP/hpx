//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/performance_counters/parcels/data_point.hpp>
#include <hpx/runtime/parcelset/mpi/header.hpp>
#include <hpx/runtime/parcelset/mpi/allocator.hpp>
#include <hpx/util/high_resolution_clock.hpp>

#include <boost/assert.hpp>
#include <boost/move/move.hpp>
#include <boost/noncopyable.hpp>

#include <vector>
#include <utility>

#ifndef HPX_PARCELSET_MPI_RECEIVER_HPP
#define HPX_PARCELSET_MPI_RECEIVER_HPP

namespace hpx { namespace parcelset { namespace mpi {

    class parcelport;

    void decode_message(
        std::vector<char, allocator<char> > const & parcel_data, parcelport& pp,
        performance_counters::parcels::data_point& receive_data);

    struct receiver
      : boost::noncopyable
    {
        receiver(header const & h, MPI_Comm communicator)
          : header_(h)
          , buffer_(boost::make_shared<std::vector<char, allocator<char> > >())//h.size()))
        {
            // start collecting statistics for this receive operation
            receive_data_.time_ = util::high_resolution_clock::now();
            buffer_->resize(h.size());
            receive_data_.buffer_allocate_time_ = util::high_resolution_clock::now() - receive_data_.time_;
            receive_data_.serialization_time_ = 0;
            receive_data_.bytes_ = 0;
            receive_data_.num_parcels_ = 0;

            header_.assert_valid();
            BOOST_ASSERT(header_.rank() != util::mpi_environment::rank());
            MPI_Irecv(
                buffer_->data(),    // data pointer
                static_cast<int>(buffer_->size()), // number of elements
                MPI_CHAR,           // MPI Datatype
                header_.rank(),     // Source
                header_.tag(),      // Tag
                communicator,       // Communicator
                &request_);         // Request
        }

        bool done(parcelport & pp)
        {
            int completed = 0;
#ifdef HPX_DEBUG
            MPI_Status status;
            MPI_Test(&request_, &completed, &status);
#else
            MPI_Test(&request_, &completed, MPI_STATUS_IGNORE);
#endif
            if(completed)
            {
                BOOST_ASSERT(status.MPI_SOURCE == header_.rank());
                BOOST_ASSERT(status.MPI_SOURCE != util::mpi_environment::rank());
                BOOST_ASSERT(status.MPI_TAG == header_.tag());
#ifdef HPX_DEBUG
                int count = 0;
                MPI_Get_count(&status, MPI_CHAR, &count);
                BOOST_ASSERT(count == header_.size());
                BOOST_ASSERT(static_cast<std::size_t>(count) == buffer_->size());
#endif
                // take measurement of overall receive time
                receive_data_.time_ = util::high_resolution_clock::now() -
                    receive_data_.time_;

                decode_message(*buffer_, pp, receive_data_);
                return true;
            }
            return false;
        }

    private:
        header header_;
        boost::shared_ptr<std::vector<char, allocator<char> > > buffer_;

        MPI_Request request_;

        performance_counters::parcels::data_point receive_data_;
    };
}}}

#endif
