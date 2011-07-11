//  Copyright (c) 2007-2011 Hartmut Kaiser
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
            parcelport_queue& handler, boost::atomic<boost::int64_t>& receives_started)
          : socket_(io_service), in_size_(0), 
            in_buffer_(new std::vector<char>()), parcels_(handler),
            receives_started_(receives_started)
        {
        }

        /// Get the socket associated with the parcelport_connection.
        boost::asio::ip::tcp::socket& socket() { return socket_; }

        /// Asynchronously read a data structure from the socket.
        template <typename Handler>
        void async_read(Handler handler)
        {
            // Issue a read operation to read exactly the number of bytes in a 
            // header.
            void (parcelport_connection::*f)(boost::system::error_code const&, 
                    boost::tuple<Handler>)
                = &parcelport_connection::handle_read_header<Handler>;

            in_buffer_->clear();
            in_size_ = 0;
            boost::asio::async_read(socket_, 
                boost::asio::buffer(&in_size_, sizeof(in_size_)),
                boost::bind(f, shared_from_this(), 
                    boost::asio::placeholders::error, boost::make_tuple(handler)));
        }

    protected:
        /// Handle a completed read of a message header. The handler is passed 
        /// using a tuple since boost::bind seems to have trouble binding a 
        /// function object created using boost::bind as a parameter.
        template <typename Handler>
        void handle_read_header(boost::system::error_code const& e,
            boost::tuple<Handler> handler)
        {
            if (e) {
                boost::get<0>(handler)(e);
            }
            else {
                // Increment number of receives started.
                ++receives_started_;

                // Determine the length of the serialized data.
                boost::uint64_t inbound_data_size = in_size_;

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
                parcels_.add_parcel(in_buffer_);

                // Inform caller that data has been received ok.
                boost::get<0>(handler)(e);

                // Issue a new read operation to read exactly the number of 
                // bytes in a header.
                void (parcelport_connection::*f)(boost::system::error_code const&, 
                        boost::tuple<Handler>)
                    = &parcelport_connection::handle_read_header<Handler>;

                in_buffer_.reset(new std::vector<char>());
                in_size_ = 0;
                boost::asio::async_read(socket_, 
                    boost::asio::buffer(&in_size_, sizeof(in_size_)),
                    boost::bind(f, shared_from_this(), 
                        boost::asio::placeholders::error, handler));
            }
        }

    private:
        /// Socket for the parcelport_connection.
        boost::asio::ip::tcp::socket socket_;

        /// buffer for incoming data
        boost::integer::ulittle64_t in_size_;
        boost::shared_ptr<std::vector<char> > in_buffer_;

        /// The handler used to process the incoming request.
        parcelport_queue& parcels_;
        boost::atomic<boost::int64_t>& receives_started_;
    };

    typedef boost::shared_ptr<parcelport_connection> parcelport_connection_ptr;

///////////////////////////////////////////////////////////////////////////////
}}}

#endif
