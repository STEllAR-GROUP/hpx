//  Copyright (c) 2007-2011 Hartmut Kaiser, Katelyn Kufahl & Bryce Lelbach
//
//  Parts of this code were taken from the Boost.Asio library
//  Copyright (c) 2003-2007 Christopher M. Kohlhoff (chris at kohlhoff dot com)
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARCELSET_SERVER_PARCELPORT_CONNECTION_MAR_26_2008_1221PM)
#define HPX_PARCELSET_SERVER_PARCELPORT_CONNECTION_MAR_26_2008_1221PM

#include <sstream>
#include <vector>

#include <hpx/runtime/parcelset/server/parcelport_queue.hpp>
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
#include <boost/iostreams/stream.hpp>
#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/tuple/tuple.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset { namespace server 
{
    /// Represents a single parcelport_connection from a client.
    class parcelport_connection
      : public boost::enable_shared_from_this<parcelport_connection>,
        private boost::noncopyable
    {
    public:
        /// Construct a listening parcelport_connection with the given io_service.
        parcelport_connection(boost::asio::io_service& io_service,
            parcelport_queue& handler,
            boost::atomic<boost::int64_t>& receives_started,
            util::high_resolution_timer& receive_timer,
            performance_counters::parcels::data_point& receive_data,
            performance_counters::parcels::gatherer& parcels_received)
          : socket_(io_service), in_priority_(0), in_size_(0), 
            in_buffer_(new std::vector<char>()), parcels_(handler),
            receives_started_(receives_started), receive_timer_(receive_timer),
            receive_data_(receive_data), parcels_received_(parcels_received)
        {
        }

        /// Get the socket associated with the parcelport_connection.
        boost::asio::ip::tcp::socket& socket() { return socket_; }

        /// Asynchronously read a data structure from the socket.
        template <typename Handler>
        void async_read(Handler handler)
        {
            // Increment number of receives started.
            ++receives_started_;
            receive_timer_.restart();
            receive_data_.start = 0;
            receive_data_.parcel = receives_started_;

            // Issue a read operation to read the parcel priority. 
            void (parcelport_connection::*f)(boost::system::error_code const&, 
                    boost::tuple<Handler>)
                = &parcelport_connection::handle_read_priority<Handler>;

            in_buffer_->clear();
            in_priority_ = 0;
            in_size_ = 0;
            boost::asio::async_read(socket_, 
                boost::asio::buffer(&in_priority_, sizeof(in_priority_)),
                boost::bind(f, shared_from_this(), 
                    boost::asio::placeholders::error, boost::make_tuple(handler)));
        }

    protected:
        /// Handle a completed read of the message priority from the message
        /// header. The handler is passed using a tuple since boost::bind seems
        /// to have trouble binding a function object created using boost::bind
        /// as a parameter.
        template <typename Handler>
        void handle_read_priority(boost::system::error_code const& e,
            boost::tuple<Handler> handler)
        {
            if (e) {
                boost::get<0>(handler)(e);
            }
            else {
                // Issue a read operation to read exactly the number of bytes in a 
                // header.
                void (parcelport_connection::*f)(boost::system::error_code const&,
                        boost::tuple<Handler>)
                    = &parcelport_connection::handle_read_size<Handler>;

                boost::asio::async_read(socket_, 
                    boost::asio::buffer(&in_size_, sizeof(in_size_)),
                        boost::bind(f, shared_from_this(), 
                        boost::asio::placeholders::error, handler));
            }
        }

        /// Handle a completed read of the message size from the message header.
        /// The handler is passed using a tuple since boost::bind seems to have
        /// trouble binding a function object created using boost::bind as a
        /// parameter.
        template <typename Handler>
        void handle_read_size(boost::system::error_code const& e,
            boost::tuple<Handler> handler)
        {
            if (e) {
                boost::get<0>(handler)(e);
            }
            else {
                // Determine the length of the serialized data.
                boost::uint64_t inbound_data_size = in_size_;
                receive_data_.bytes = std::size_t(inbound_data_size);

                // Start an asynchronous call to receive the data.
                in_buffer_->resize(inbound_data_size);
                void (parcelport_connection::*f)(boost::system::error_code const&,
                        boost::tuple<Handler>)
                    = &parcelport_connection::handle_read_data<Handler>;

                boost::asio::async_read(socket_, 
                    boost::asio::buffer(*in_buffer_.get()),
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
            }
            else {
                // add parcel data to incoming parcel queue
                parcels_.add_parcel
                    (in_buffer_, static_cast<threads::thread_priority>
                        (boost::integer::ulittle8_t::value_type(in_priority_)));

                // Inform caller that data has been received ok.
                boost::get<0>(handler)(e);

                // Issue a read operation to read the parcel priority. 
                void (parcelport_connection::*f)(boost::system::error_code const&, 
                      boost::tuple<Handler>)
                    = &parcelport_connection::handle_read_priority<Handler>;

                in_buffer_.reset(new std::vector<char>());
                in_priority_ = 0;
                in_size_ = 0;
                boost::asio::async_read(socket_, 
                    boost::asio::buffer(&in_priority_, sizeof(in_priority_)),
                    boost::bind(f, shared_from_this(), 
                        boost::asio::placeholders::error, handler));
            }
        }

    private:
        /// Socket for the parcelport_connection.
        boost::asio::ip::tcp::socket socket_;

        /// buffer for incoming data
        boost::integer::ulittle8_t in_priority_;
        boost::integer::ulittle64_t in_size_;
        boost::shared_ptr<std::vector<char> > in_buffer_;

        /// The handler used to process the incoming request.
        parcelport_queue& parcels_;

        /// Counters and timers for parcels received.
        boost::atomic<boost::int64_t>& receives_started_;
        util::high_resolution_timer& receive_timer_;
        performance_counters::parcels::data_point& receive_data_;
        performance_counters::parcels::gatherer& parcels_received_;
    };

    typedef boost::shared_ptr<parcelport_connection> parcelport_connection_ptr;

///////////////////////////////////////////////////////////////////////////////
}}}

#endif

