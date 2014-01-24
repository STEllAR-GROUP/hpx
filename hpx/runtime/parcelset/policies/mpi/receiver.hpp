//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2014 Thomas Heller
//
//  Parts of this code were taken from the Boost.Asio library
//  Copyright (c) 2003-2007 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_MPI_RECEIVER_HPP
#define HPX_PARCELSET_POLICIES_MPI_RECEIVER_HPP

#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/runtime/parcelset/parcelport_connection.hpp>
#include <hpx/runtime/parcelset/parcel_buffer.hpp>
#include <hpx/runtime/parcelset/decode_parcels.hpp>
#include <hpx/performance_counters/parcels/data_point.hpp>
#include <hpx/performance_counters/parcels/gatherer.hpp>

/*
#include <boost/asio/buffer.hpp>
#include <boost/asio/io_service.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/placeholders.hpp>
#include <boost/asio/read.hpp>
#include <boost/asio/write.hpp>
#include <boost/atomic.hpp>
#include <boost/bind.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/integer/endian.hpp>
#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/tuple/tuple.hpp>
*/

#include <sstream>
#include <vector>

namespace hpx { namespace parcelset
{
    template <typename ConnectionHandler>
    class parcelport_impl;
    
    boost::uint64_t get_max_inbound_size(parcelport&);
}}

namespace hpx { namespace parcelset { namespace policies { namespace mpi
{
    class connection_handler;

    class receiver
      : public parcelport_connection<receiver>
    {
        typedef parcel_buffer<std::vector<char>, std::vector<char> > buffer_type;
    public:
        receiver(header const & h, MPI_Comm communicator, connection_handler & parcelport)
          : header_(h)
        {
            in_buffer_.reset(new buffer_type());

            // Store the time of the begin of the read operation
            in_buffer_->data_point_.time_ = timer_.elapsed_nanoseconds();
            in_buffer_->data_point_.serialization_time_ = 0;
            in_buffer_->data_point_.bytes_ = header_.size();
            in_buffer_->data_point_.num_parcels_ = 0;

            if (static_cast<std::size_t>(header_.size()) > get_max_inbound_size(parcelport))
            {
                // report this problem ...
                HPX_THROW_EXCEPTION(boost::asio::error::operation_not_supported,
                    "mpi::receiver::receiver",
                    "The size of this message exceeds the maximum inbound data size");
                return;
            }

            in_buffer_->data_.resize(std::size_t(h.size()));

            header_.assert_valid();
            HPX_ASSERT(header_.rank() != util::mpi_environment::rank());
            MPI_Irecv(
                in_buffer_->data_.data(),    // data pointer
                static_cast<int>(in_buffer_->data_.size()), // number of elements
                MPI_CHAR,           // MPI Datatype
                header_.rank(),     // Source
                header_.tag(),      // Tag
                communicator,       // Communicator
                &request_);         // Request
        }

        bool done(connection_handler & pp)
        {
            int completed = 0;
            MPI_Status status;
#ifdef HPX_DEBUG
            HPX_ASSERT(MPI_Test(&request_, &completed, &status) == MPI_SUCCESS);
#else
            MPI_Test(&request_, &completed, &status);
#endif
            if(completed && status.MPI_ERROR != MPI_ERR_PENDING)
            {
                HPX_ASSERT(status.MPI_SOURCE == header_.rank());
                HPX_ASSERT(status.MPI_SOURCE != util::mpi_environment::rank());
                HPX_ASSERT(status.MPI_TAG == header_.tag());
#ifdef HPX_DEBUG
                int count = 0;
                MPI_Get_count(&status, MPI_CHAR, &count);
                HPX_ASSERT(count == header_.size());
                HPX_ASSERT(static_cast<std::size_t>(count) == in_buffer_->data_.size());
#endif
                // take measurement of overall receive time
                in_buffer_->data_point_.time_ = timer_.elapsed_nanoseconds() -
                    in_buffer_->data_point_.time_;

                // decode the received parcels.
                decode_parcels(pp, shared_from_this(), in_buffer_);
                return true;
            }
            return false;
        }

        ~receiver()
        {
        }

        /// Asynchronously read a data structure from the socket.
        template <typename Handler>
        void async_read(Handler handler)
        {
        }

    private:
        /// buffer for incoming data
        boost::shared_ptr<buffer_type> in_buffer_;
        header header_;
        MPI_Request request_;

        boost::uint64_t max_inbound_size_;

        /// Counters and timers for parcels received.
        util::high_resolution_timer timer_;
    };

    // this makes sure we can store our connections in a set
    inline bool operator<(boost::shared_ptr<receiver> const& lhs,
        boost::shared_ptr<receiver> const& rhs)
    {
        return lhs.get() < rhs.get();
    }
}}}}

#endif
