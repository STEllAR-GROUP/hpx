//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c)      2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELPORT_IPC_SENDER_HPP
#define HPX_PARCELPORT_IPC_SENDER_HPP

#include <hpx/config/defines.hpp>
#if defined(HPX_HAVE_PARCELPORT_IPC)

#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/runtime/parcelset/parcelport_connection.hpp>
#include <hpx/runtime/parcelset/decode_parcels.hpp>
#include <hpx/plugins/parcelport/ipc/data_window.hpp>
#include <hpx/plugins/parcelport/ipc/data_buffer.hpp>
#include <hpx/plugins/parcelport/ipc/locality.hpp>
#include <hpx/performance_counters/parcels/data_point.hpp>
#include <hpx/performance_counters/parcels/gatherer.hpp>

#include <boost/atomic.hpp>
#include <boost/bind.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/asio/placeholders.hpp>
#include <boost/tuple/tuple.hpp>

#include <string>

namespace hpx { namespace parcelset { namespace policies { namespace ipc
{
    class connection_handler;

    class receiver
      : public parcelport_connection<receiver, data_buffer, std::vector<char> >
    {
    public:
        /// Construct a listening receiver with the given io_service.
        receiver(boost::asio::io_service& io_service,
                parcelset::locality here,
                connection_handler& parcelport)
          : window_(io_service), parcelport_(parcelport)
        {
            std::string fullname(here.get<locality>().address() + "." +
                std::to_string(here.get<locality>().port()));
            window_.set_option(data_window::bound_to(fullname));
        }

        ~receiver()
        {
            // gracefully and portably shutdown the connection
            boost::system::error_code ec;
            window_.close(ec);    // close the socket to give it back to the OS
        }

        /// Get the data window associated with the receiver.
        data_window& window() { return window_; }

        boost::shared_ptr<parcel_buffer_type> get_buffer(parcel const & p = parcel(),
            std::size_t arg_size = 0)
        {
            if(!buffer_ || (buffer_ && !buffer_->parcels_decoded_))
            {
                buffer_ = boost::make_shared<parcel_buffer_type>(buffer_type());
            }
            buffer_->data_.reset();
            return buffer_;
        }

        /// Asynchronously read a data structure from the socket.
        template <typename Handler>
        void async_read(Handler handler)
        {
            buffer_ = get_buffer();
            buffer_->clear();

            // Store the time of the begin of the read operation
            performance_counters::parcels::data_point& data = buffer_->data_point_;
            data.time_ = timer_.elapsed_nanoseconds();
            data.serialization_time_ = 0;
            data.bytes_ = 0;
            data.num_parcels_ = 0;

            // Issue a read operation to read the parcel data.
            void (receiver::*f)(boost::system::error_code const&,
                    boost::tuple<Handler>)
                = &receiver::handle_read_data<Handler>;

            window_.async_read(buffer_->data_,
                boost::bind(f, shared_from_this(),
                    boost::asio::placeholders::error,
                    boost::make_tuple(handler)));
        }

    protected:
        /// Handle a completed read of message data.
        template <typename Handler>
        void handle_read_data(boost::system::error_code const& e,
            boost::tuple<Handler> handler)
        {
            if (e) {
                boost::get<0>(handler)(e);

                // Issue a read operation to read the next parcel.
                async_read(boost::get<0>(handler));
            }
            else {
                // complete data point and pass it along
                buffer_->data_point_.time_ = timer_.elapsed_nanoseconds() -
                    buffer_->data_point_.time_;

                // now send acknowledgment message
                void (receiver::*f)(boost::system::error_code const&,
                          boost::tuple<Handler>)
                    = &receiver::handle_write_ack<Handler>;

                buffer_->data_size_ = buffer_->data_.size();
                buffer_->size_ = buffer_->data_.size();

                // decode the received parcels.
                decode_parcels(parcelport_, *this, buffer_);

                // acknowledge to have received the parcel
                window_.async_write_ack(
                    boost::bind(f, shared_from_this(),
                        boost::asio::placeholders::error, handler));
            }
        }

        template <typename Handler>
        void handle_write_ack(boost::system::error_code const& e,
            boost::tuple<Handler> handler)
        {
            // Inform caller that data has been received ok.
            boost::get<0>(handler)(e);

            // Issue a read operation to handle the next parcel.
            async_read(boost::get<0>(handler));
        }

    private:
        /// Data window for the receiver.
        data_window window_;

        /// The handler used to process the incoming request.
        connection_handler& parcelport_;

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

#endif
