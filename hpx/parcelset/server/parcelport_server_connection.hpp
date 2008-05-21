//  Copyright (c) 2007-2008 Hartmut Kaiser
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

#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/integer/endian.hpp>

#include <hpx/util/portable_binary_oarchive.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/container_device.hpp>
#include <hpx/parcelset/server/parcelport_queue.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset { namespace server 
{
    /// Represents a single parcelport_connection from a client.
    class parcelport_connection
      : public boost::enable_shared_from_this<parcelport_connection>,
        private boost::noncopyable
    {
        typedef util::container_device<std::vector<char> > io_device_type;

    public:
        /// Construct a listening parcelport_connection with the given io_service.
        parcelport_connection(boost::asio::io_service& io_service,
            parcelport_queue& handler)
          : socket_(io_service), parcels_(handler)
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
            
            in_buffer_.clear();
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
                // Determine the length of the serialized data.
                std::size_t inbound_data_size = in_size_;

                // Start an asynchronous call to receive the data.
                in_buffer_.resize(inbound_data_size);
                void (parcelport_connection::*f)(boost::system::error_code const&,
                        boost::tuple<Handler>)
                    = &parcelport_connection::handle_read_data<Handler>;

                boost::asio::async_read(socket_, boost::asio::buffer(in_buffer_),
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
                // Extract the data structure from the data just received.
                try {
                // create a special io stream on top of in_buffer_
                    boost::iostreams::stream<io_device_type> io(in_buffer_);

                // Deserialize the data
                    parcel p;
                    util::portable_binary_iarchive archive(io);
                    archive >> p;
                
                // add parcel to incoming parcel queue
                    parcels_.add_parcel(p);
                }
                catch (std::exception const& /*e*/) {
                    // Unable to decode data.
                    boost::system::error_code 
                        error(boost::asio::error::invalid_argument);
                    boost::get<0>(handler)(error);
                    return;
                }

                // Inform caller that data has been received ok.
                boost::get<0>(handler)(e);

                // Issue a new read operation to read exactly the number of 
                // bytes in a header.
                void (parcelport_connection::*f)(boost::system::error_code const&, 
                        boost::tuple<Handler>)
                    = &parcelport_connection::handle_read_header<Handler>;
                
                in_buffer_.clear();
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
        std::vector<char> in_buffer_;

        /// The handler used to process the incoming request.
        parcelport_queue& parcels_;
    };

    typedef boost::shared_ptr<parcelport_connection> parcelport_connection_ptr;

///////////////////////////////////////////////////////////////////////////////
}}}

#endif
