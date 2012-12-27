//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2011 Katelyn Kufahl
//  Copyright (c) 2011 Bryce Lelbach
//
//  Parts of this code were taken from the Boost.Asio library
//  Copyright (c) 2003-2007 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARCELSET_SERVER_TCP_PARCELPORT_CONNECTION_MAR_26_2008_1221PM)
#define HPX_PARCELSET_SERVER_TCP_PARCELPORT_CONNECTION_MAR_26_2008_1221PM

#include <sstream>
#include <vector>

#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/performance_counters/parcels/data_point.hpp>
#include <hpx/performance_counters/parcels/gatherer.hpp>

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

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset { namespace tcp
{
    // forward declaration only
    class parcelport;

    void decode_message(parcelport&,
        boost::shared_ptr<std::vector<char> > buffer,
        performance_counters::parcels::data_point receive_data);
}}}

namespace hpx { namespace parcelset { namespace server { namespace tcp
{
    /// Represents a single parcelport_connection from a client.
    class parcelport_connection
      : public boost::enable_shared_from_this<parcelport_connection>,
        private boost::noncopyable
    {
    public:
        /// Construct a listening parcelport_connection with the given io_service.
        parcelport_connection(boost::asio::io_service& io_service,
                parcelset::tcp::parcelport& parcelport)
          : socket_(io_service), in_priority_(0), in_size_(0),
            in_buffer_(), parcelport_(parcelport)
        {}

        ~parcelport_connection()
        {
            // gracefully and portably shutdown the socket
            boost::system::error_code ec;
            socket_.shutdown(boost::asio::ip::tcp::socket::shutdown_both, ec);
            socket_.close(ec);    // close the socket to give it back to the OS
        }

        /// Get the socket associated with the parcelport_connection.
        boost::asio::ip::tcp::socket& socket() { return socket_; }

        /// Asynchronously read a data structure from the socket.
        template <typename Handler>
        void async_read(Handler handler)
        {
            // Store the time of the begin of the read operation
            receive_data_.time_ = timer_.elapsed_nanoseconds();
            receive_data_.serialization_time_ = 0;
            receive_data_.bytes_ = 0;
            receive_data_.num_parcels_ = 0;

            // Issue a read operation to read the parcel priority and size.
            void (parcelport_connection::*f)(boost::system::error_code const&,
                    boost::tuple<Handler>)
                = &parcelport_connection::handle_read_header<Handler>;

            in_buffer_.reset(new std::vector<char>());
            in_priority_ = 0;
            in_size_ = 0;

            using boost::asio::buffer;
            std::vector<boost::asio::mutable_buffer> buffers;
            buffers.push_back(buffer(&in_priority_, sizeof(in_priority_)));
            buffers.push_back(buffer(&in_size_, sizeof(in_size_)));

#if defined(__linux) || defined(linux) || defined(__linux__)
            boost::asio::detail::socket_option::boolean<
                IPPROTO_TCP, TCP_QUICKACK> quickack(true);
            socket_.set_option(quickack);
#endif

            boost::asio::async_read(socket_, buffers,
                boost::bind(f, shared_from_this(),
                    boost::asio::placeholders::error,
                    boost::make_tuple(handler)));
        }

    protected:
        /// Handle a completed read of the message priority and size from the
        /// message header.
        /// The handler is passed using a tuple since boost::bind seems to have
        /// trouble binding a function object created using boost::bind as a
        /// parameter.
        template <typename Handler>
        void handle_read_header(boost::system::error_code const& e,
            boost::tuple<Handler> handler)
        {
            if (e) {
                boost::get<0>(handler)(e);

                // Issue a read operation to read the next parcel.
                async_read(boost::get<0>(handler));
            }
            else {
                // Determine the length of the serialized data.
                boost::uint64_t inbound_data_size = in_size_;
                receive_data_.bytes_ = std::size_t(inbound_data_size);

                // Start an asynchronous call to receive the data.
                in_buffer_->resize(static_cast<std::size_t>(inbound_data_size));
                void (parcelport_connection::*f)(boost::system::error_code const&,
                        boost::tuple<Handler>)
                    = &parcelport_connection::handle_read_data<Handler>;

#if defined(__linux) || defined(linux) || defined(__linux__)
                boost::asio::detail::socket_option::boolean<
                    IPPROTO_TCP, TCP_QUICKACK> quickack(true);
                socket_.set_option(quickack);
#endif

                boost::asio::async_read(socket_,
                    boost::asio::buffer(*in_buffer_),
                    boost::bind(f, shared_from_this(),
                        boost::asio::placeholders::error, handler));
            }
        }

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
                receive_data_.time_ = timer_.elapsed_nanoseconds() -
                    receive_data_.time_;

                // now send acknowledgment byte
                void (parcelport_connection::*f)(boost::system::error_code const&,
                        boost::tuple<Handler>)
                    = &parcelport_connection::handle_write_ack<Handler>;

                // hold on to received data to avoid data races
                boost::shared_ptr<std::vector<char> > data(in_buffer_);
                performance_counters::parcels::data_point receive_data = 
                    receive_data_;

                ack_ = true;
                boost::asio::async_write(socket_,
                    boost::asio::buffer(&ack_, sizeof(ack_)),
                    boost::bind(f, shared_from_this(), 
                        boost::asio::placeholders::error, handler));

                // add parcel data to incoming parcel queue
                decode_message(parcelport_, data, receive_data);

                // Inform caller that data has been received ok.
                boost::get<0>(handler)(e);
            }
        }

        template <typename Handler>
        void handle_write_ack(boost::system::error_code const& e,
            boost::tuple<Handler> handler)
        {
            // Issue a read operation to read the next parcel.
            async_read(boost::get<0>(handler));
        }

    private:
        /// Socket for the parcelport_connection.
        boost::asio::ip::tcp::socket socket_;

        /// buffer for incoming data
        boost::integer::ulittle8_t in_priority_;
        boost::integer::ulittle64_t in_size_;
        boost::shared_ptr<std::vector<char> > in_buffer_;
        bool ack_;

        /// The handler used to process the incoming request.
        parcelset::tcp::parcelport& parcelport_;

        /// Counters and timers for parcels received.
        util::high_resolution_timer timer_;
        performance_counters::parcels::data_point receive_data_;
    };

    typedef boost::shared_ptr<parcelport_connection> parcelport_connection_ptr;

    // this makes sure we can store our connections in a set
    inline bool operator<(server::tcp::parcelport_connection_ptr const& lhs,
        server::tcp::parcelport_connection_ptr const& rhs)
    {
        return lhs.get() < rhs.get();
    }
}}}}

#endif

