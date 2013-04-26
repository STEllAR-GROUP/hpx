//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARCELSET_SERVER_SHMEM_PARCELPORT_CONNECTION_OV_25_2012_0516PM)
#define HPX_PARCELSET_SERVER_SHMEM_PARCELPORT_CONNECTION_OV_25_2012_0516PM

#include <sstream>
#include <vector>

#include <hpx/runtime/parcelset/server/parcelport_queue.hpp>
#include <hpx/runtime/parcelset/shmem/data_window.hpp>
#include <hpx/runtime/parcelset/shmem/data_buffer.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/performance_counters/parcels/data_point.hpp>
#include <hpx/performance_counters/parcels/gatherer.hpp>

#include <boost/atomic.hpp>
#include <boost/bind.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/tuple/tuple.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset { namespace shmem
{
    // forward declaration only
    class parcelport;

    void decode_message(parcelport&,
        parcelset::shmem::data_buffer buffer,
        performance_counters::parcels::data_point receive_data);
}}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset { namespace server { namespace shmem
{
    /// Represents a single parcelport_connection from a client.
    class parcelport_connection
      : public boost::enable_shared_from_this<parcelport_connection>,
        private boost::noncopyable
    {
    public:
        /// Construct a listening parcelport_connection with the given io_service.
        parcelport_connection(boost::asio::io_service& io_service,
                naming::locality here,
                parcelset::shmem::parcelport& parcelport)
          : window_(io_service), in_buffer_(), parcelport_(parcelport)
        {
            std::string fullname(here.get_address() + "." +
                boost::lexical_cast<std::string>(here.get_port()));
            window_.set_option(parcelset::shmem::data_window::bound_to(fullname));
        }

        ~parcelport_connection()
        {
            // gracefully and portably shutdown the connection
            boost::system::error_code ec;
            window_.close(ec);    // close the socket to give it back to the OS
        }

        /// Get the data window associated with the parcelport_connection.
        parcelset::shmem::data_window& window() { return window_; }

        /// Asynchronously read a data structure from the socket.
        template <typename Handler>
        void async_read(Handler handler)
        {
            // Store the time of the begin of the read operation
            receive_data_.time_ = timer_.elapsed_nanoseconds();
            receive_data_.serialization_time_ = 0;
            receive_data_.bytes_ = 0;
            receive_data_.num_parcels_ = 0;

            // Issue a read operation to read the parcel data.
            void (parcelport_connection::*f)(boost::system::error_code const&,
                    boost::tuple<Handler>)
                = &parcelport_connection::handle_read_data<Handler>;

            in_buffer_.reset();

            window_.async_read(in_buffer_,
                boost::bind(f, shared_from_this(),
                    boost::asio::placeholders::error,
                    boost::make_tuple(handler)));
        }

    protected:
        template <typename Handler>
        void async_restart_read(boost::tuple<Handler> handler)
        {
            async_read(boost::get<0>(handler));
        }

        /// Handle a completed read of message data.
        template <typename Handler>
        void handle_read_data(boost::system::error_code const& e,
            boost::tuple<Handler> handler)
        {
            if (e) {
                boost::get<0>(handler)(e);

                // Issue a read operation to read the next parcel.
                void (parcelport_connection::*f)(boost::tuple<Handler>) = 
                        &parcelport_connection::async_restart_read<Handler>;
                window_.get_io_service().post(
                    boost::bind(f, shared_from_this(), handler));
            }
            else {
                // complete data point and pass it along
                receive_data_.time_ = timer_.elapsed_nanoseconds() -
                    receive_data_.time_;

                // now send acknowledgment message
                void (parcelport_connection::*f)(boost::system::error_code const&, 
                          boost::tuple<Handler>)
                    = &parcelport_connection::handle_write_ack<Handler>;

                // hold on to received data to avoid data races
                parcelset::shmem::data_buffer data(in_buffer_);
                performance_counters::parcels::data_point receive_data = 
                    receive_data_;

                // add parcel data to incoming parcel queue
                decode_message(parcelport_, data, receive_data);

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
            void (parcelport_connection::*f)(boost::tuple<Handler>) = 
                    &parcelport_connection::async_restart_read<Handler>;
            window_.get_io_service().post(
                boost::bind(f, shared_from_this(), handler));
        }

    private:
        /// Data window for the parcelport_connection.
        parcelset::shmem::data_window window_;

        /// buffer for incoming data
        parcelset::shmem::data_buffer in_buffer_;

        /// The handler used to process the incoming request.
        parcelset::shmem::parcelport& parcelport_;

        /// Counters and timers for parcels received.
        util::high_resolution_timer timer_;
        performance_counters::parcels::data_point receive_data_;
    };

    typedef boost::shared_ptr<parcelport_connection> parcelport_connection_ptr;

    // this makes sure we can store our connections in a set
    inline bool operator<(server::shmem::parcelport_connection_ptr const& lhs,
        server::shmem::parcelport_connection_ptr const& rhs)
    {
        return lhs.get() < rhs.get();
    }
}}}}

#endif

