//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2013-2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_MPI_SENDER_HPP
#define HPX_PARCELSET_POLICIES_MPI_SENDER_HPP

#include <hpx/runtime/naming/locality.hpp>
#include <hpx/runtime/parcelset/parcelport_connection.hpp>
#include <hpx/performance_counters/parcels/data_point.hpp>
#include <hpx/performance_counters/parcels/gatherer.hpp>
#include <hpx/util/high_resolution_timer.hpp>

/*
#include <boost/asio/buffer.hpp>
#include <boost/asio/io_service.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/placeholders.hpp>
#include <boost/asio/read.hpp>
#include <boost/asio/write.hpp>
#include <boost/atomic.hpp>
#include <boost/bind.hpp>
#include <boost/bind/protect.hpp>
#include <boost/cstdint.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/integer/endian.hpp>
#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/tuple/tuple.hpp>
*/

#include <vector>

namespace hpx { namespace parcelset { namespace policies { namespace mpi
{
    class connection_handler;
    void add_sender(connection_handler & handler, boost::shared_ptr<sender> sender_connection);

    class sender
      : public parcelset::parcelport_connection<sender>
    {
    public:

        typedef std::vector<char> buffer_type;
        typedef parcel_buffer<buffer_type> parcel_buffer_type;
        typedef
            HPX_STD_FUNCTION<void(boost::system::error_code const &, std::size_t)>
            handler_function_type;
        typedef
            HPX_STD_FUNCTION<
                void(
                    boost::system::error_code const &
                  , naming::locality const&
                  ,  boost::shared_ptr<sender>
                )
            >
            postprocess_function_type;

        enum sender_state
        {
            invalid,
            start_send,
            sending_header,
            sent_header,
            sending_data,
            sent_data,
            sender_done,
        };

        /// Construct a sending parcelport_connection with the given io_service.
        sender(MPI_Comm communicator,
            int tag,
            hpx::lcos::local::spinlock & tag_mtx,
            std::deque<int> & free_tags,
            naming::locality const& locality_id,
            connection_handler & handler,
            performance_counters::parcels::gatherer& parcels_sent)
          : communicator_(communicator)
          , tag_(tag)
          , tag_mtx_(tag_mtx)
          , free_tags_(free_tags)
          , sender_state_(invalid)
          , parcelport_(handler)
          , there_(locality_id), parcels_sent_(parcels_sent)
        {}
        
        ~sender()
        {
            hpx::lcos::local::spinlock::scoped_lock l(tag_mtx_);
            free_tags_.push_back(tag_);
        }

        naming::locality const& destination() const
        {
            return there_;
        }

        buffer_type get_buffer() const
        {
            return std::vector<char>();
        }

        void verify(naming::locality const & parcel_locality_id) const
        {
            HPX_ASSERT(parcel_locality_id.get_rank() != util::mpi_environment::rank());
        }

        template <typename Handler, typename ParcelPostprocess>
        void async_write(boost::shared_ptr<parcel_buffer<buffer_type> > buffer,
            Handler handler, ParcelPostprocess parcel_postprocess)
        {
            hpx::lcos::local::spinlock::scoped_lock l(mtx_);
            // Check for valid pre conditions
            HPX_ASSERT(!buffer_);
            HPX_ASSERT(!handler_);
            HPX_ASSERT(!postprocess_);
            BOOST_ASSERT(buffer->chunks_.size() == 1);

            /// Increment sends and begin timer.
            buffer->data_point_.time_ = timer_.elapsed_nanoseconds();

            buffer_ = buffer;
            handler_ = handler;
            postprocess_ = parcel_postprocess;

            // make header data structure
            header_ = header(
                there_.get_rank()  // destination rank ...
              , tag_               // tag ...
              , buffer_->size_      // size ...
              , buffer_->data_size_ // data_size_ ...
            );

            sender_state_ = start_send;
            add_sender(parcelport_, shared_from_this());
        }

        bool done()
        {
            hpx::lcos::local::spinlock::scoped_lock l(mtx_);
            switch(sender_state_)
            {
            case start_send:
                {
                    HPX_ASSERT(buffer_);
                    HPX_ASSERT(handler_);
                    HPX_ASSERT(postprocess_);
                    header_.assert_valid();
                    HPX_ASSERT(header_.rank() != util::mpi_environment::rank());
                    
                    MPI_Isend(
                        header_.data(),         // Data pointer
                        header_.data_size_,     // Size
                        header_.type(),         // MPI Datatype
                        header_.rank(),         // Destination
                        0,                      // Tag
                        communicator_,          // Communicator
                        &header_request_        // Request
                    );

                    MPI_Isend(
                        buffer_->data_.data(), // Data pointer
                        static_cast<int>(buffer_->size_), // Size
                        MPI_CHAR,               // MPI Datatype
                        header_.rank(),         // Destination
                        header_.tag(),          // Tag
                        communicator_,          // Communicator
                        &data_request_          // Request
                        );
                    sender_state_ = sending_header;
                    break;
                }
            case sending_header:
                {
                    HPX_ASSERT(buffer_);
                    HPX_ASSERT(handler_);
                    HPX_ASSERT(postprocess_);
                    int completed = 0;
                    MPI_Status status;
                    int ret = MPI_Test(&header_request_, &completed, &status);
                    HPX_ASSERT(ret == MPI_SUCCESS);
                    if(completed && status.MPI_ERROR != MPI_ERR_PENDING)
                    {
                        sender_state_ = sent_header;
                    }
                    break;
                }
            case sent_header:
                {
                    HPX_ASSERT(buffer_);
                    HPX_ASSERT(handler_);
                    HPX_ASSERT(postprocess_);
                    HPX_ASSERT(static_cast<std::size_t>(header_.size()) ==
                        buffer_->data_.size());
                    sender_state_ = sending_data;
                    break;
                }
            case sending_data:
                {
                    HPX_ASSERT(buffer_);
                    HPX_ASSERT(handler_);
                    HPX_ASSERT(postprocess_);
                    int completed = 0;
                    MPI_Status status;
                    int ret = MPI_Test(&data_request_, &completed, &status);
                    HPX_ASSERT(ret == MPI_SUCCESS);
                    if(completed && status.MPI_ERROR != MPI_ERR_PENDING)
                    {
                        sender_state_ = sent_data;
                    }
                    break;
                }
            case sent_data:
                {
                    HPX_ASSERT(buffer_);
                    HPX_ASSERT(handler_);
                    HPX_ASSERT(postprocess_);
                    error_code ec;
                    handler_(ec, header_.size());
                    sender_state_ = sender_done;
                    break;
                }
            case sender_done:
                {
                    HPX_ASSERT(buffer_);
                    HPX_ASSERT(handler_);
                    HPX_ASSERT(postprocess_);
                    // complete data point and push back onto gatherer
                    buffer_->data_point_.time_ = timer_.elapsed_nanoseconds() - buffer_->data_point_.time_;
                    parcels_sent_.add_data(buffer_->data_point_);
                    error_code ec;
                    postprocess_(ec, there_, shared_from_this());
                    // clear our state
                    buffer_.reset();
                    handler_.reset();
                    postprocess_.reset();
                    sender_state_ = invalid;
                    return true;
                }
            case invalid:
                {
                    HPX_ASSERT(!buffer_);
                    HPX_ASSERT(!handler_);
                    HPX_ASSERT(!postprocess_);
                }
                break;
            default:
                {
                    HPX_ASSERT(false);
                }
                break;
            }
            return false;
        }

    private:
        // This mutex protects the data members from possible races due to
        // concurrent access between async_write and done
        hpx::lcos::local::spinlock mtx_;
        MPI_Comm communicator_;

        header header_;
        int tag_;
        hpx::lcos::local::spinlock & tag_mtx_;
        std::deque<int> & free_tags_;
        sender_state sender_state_;

        MPI_Request header_request_;
        MPI_Request data_request_;
        boost::shared_ptr<parcel_buffer_type> buffer_;
        handler_function_type handler_;
        postprocess_function_type postprocess_;

        connection_handler & parcelport_;

        /// the other (receiving) end of this connection
        naming::locality there_;

        /// Counters and their data containers.
        util::high_resolution_timer timer_;
        performance_counters::parcels::gatherer& parcels_sent_;
    };
}}}}

#endif
