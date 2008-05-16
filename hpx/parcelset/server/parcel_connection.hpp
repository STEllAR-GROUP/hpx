//  Copyright (c) 2007-2008 Hartmut Kaiser
//
//  Parts of this code were taken from the Boost.Asio library
//  Copyright (c) 2003-2007 Christopher M. Kohlhoff (chris at kohlhoff dot com)
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARCELSET_SERVER_PARCEL_CONNECTION_MAR_26_2008_1221PM)
#define HPX_PARCELSET_SERVER_PARCEL_CONNECTION_MAR_26_2008_1221PM

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
#include <hpx/parcelset/server/parcel_queue.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset { namespace server 
{
    /// Represents a single connection from a client.
    class connection
      : public boost::enable_shared_from_this<connection>,
        private boost::noncopyable
    {
        typedef util::container_device<std::vector<char> > io_device_type;

    public:
        /// Construct a listening connection with the given io_service.
        connection(boost::asio::io_service& io_service,
            parcel_queue& handler)
          : socket_(io_service), parcels_(&handler)
        {
        }

        /// Construct a sending connection with the given io_service.
        connection(boost::asio::io_service& io_service, parcel const& p)
          : socket_(io_service), parcels_(NULL)
        {
            {
                // create a special io stream on top of out_buffer_
                out_buffer_.clear();
                boost::iostreams::stream<io_device_type> io(out_buffer_);

                // Serialize the data
                util::portable_binary_oarchive archive(io);
                archive << p;
            }
            out_size_ = out_buffer_.size();
        }

        /// Get the socket associated with the connection.
        boost::asio::ip::tcp::socket& socket() { return socket_; }

        /// Asynchronously write a data structure to the socket.
        template <typename Handler>
        void async_write(Handler handler)
        {
            // Write the serialized data to the socket. We use "gather-write" 
            // to send both the header and the data in a single write operation.
            std::vector<boost::asio::const_buffer> buffers;
            buffers.push_back(boost::asio::buffer(&out_size_, sizeof(out_size_)));
            buffers.push_back(boost::asio::buffer(out_buffer_));

            // this additional wrapping of the handler into a bind object is 
            // needed to keep  this connection object alive for the whole
            // write operation
            void (connection::*f)(boost::system::error_code const&, std::size_t,
                    boost::tuple<Handler>)
                = &connection::handle_write<Handler>;
                
            boost::asio::async_write(socket_, buffers,
                boost::bind(f, shared_from_this(), 
                    boost::asio::placeholders::error, _2, boost::make_tuple(handler)));
        }

        /// Asynchronously read a data structure from the socket.
        template <typename Handler>
        void async_read(Handler handler)
        {
            // Issue a read operation to read exactly the number of bytes in a 
            // header.
            void (connection::*f)(boost::system::error_code const&, 
                    boost::tuple<Handler>)
                = &connection::handle_read_header<Handler>;
            
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
                void (connection::*f)(boost::system::error_code const&,
                        boost::tuple<Handler>)
                    = &connection::handle_read_data<Handler>;

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
                    BOOST_ASSERT(NULL != parcels_);   // needs to be set
                    parcels_->add_parcel(p);
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
            }
        }

        /// handle completed write operation
        template <typename Handler>
        void handle_write(boost::system::error_code const& e, std::size_t bytes,
            boost::tuple<Handler> handler)
        {
            boost::get<0>(handler)(e, bytes);    // just call initial handler
        }

    private:
        /// Socket for the connection.
        boost::asio::ip::tcp::socket socket_;

        /// buffer for outgoing data
        boost::integer::ubig64_t out_size_;
        std::vector<char> out_buffer_;

        /// buffer for incoming data
        boost::integer::ubig64_t in_size_;
        std::vector<char> in_buffer_;

        /// The handler used to process the incoming request.
        parcel_queue* parcels_;
    };

    typedef boost::shared_ptr<connection> connection_ptr;

///////////////////////////////////////////////////////////////////////////////
}}}  // namespace hpx::naming::server

#endif
