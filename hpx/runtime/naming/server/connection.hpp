//  Copyright (c) 2007-2008 Hartmut Kaiser
//
//  Parts of this code were taken from the Boost.Asio library
//  Copyright (c) 2003-2007 Christopher M. Kohlhoff (chris at kohlhoff dot com)
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_NAMING_SERVER_CONNECTION_MAR_24_2008_1006AM)
#define HPX_NAMING_SERVER_CONNECTION_MAR_24_2008_1006AM

#include <boost/bind.hpp>
#include <boost/asio.hpp>
#include <boost/array.hpp>
#include <boost/noncopyable.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/integer/endian.hpp>

#include <hpx/util/portable_binary_oarchive.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/container_device.hpp>
#include <hpx/util/high_resolution_timer.hpp>

#include <hpx/runtime/naming/server/reply.hpp>
#include <hpx/runtime/naming/server/request.hpp>
#include <hpx/runtime/naming/server/request_handler.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace naming { namespace server 
{
    /// Represents a single connection from a client.
    class connection
      : public boost::enable_shared_from_this<connection>,
        private boost::noncopyable
    {
    public:
        /// Construct a receiving connection with the given io_service.
        connection(boost::asio::io_service& io_service,
            request_handler& handler)
          : socket_(io_service), request_handler_(handler)
        {
        }
        ~connection()
        {
            socket_.close();
        }

        /// Get the socket associated with the connection.
        boost::asio::ip::tcp::socket& socket() { return socket_; }

        /// Stop all asynchronous operations associated with the connection.
        void stop() { socket_.close(); }

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
                boost::asio::buffer(&size_, sizeof(size_)),
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
                std::size_t inbound_data_size = size_;

                // Start an asynchronous call to receive the data.
                buffer_.resize(inbound_data_size);
                void (connection::*f)(boost::system::error_code const&,
                        boost::tuple<Handler>)
                    = &connection::handle_read_data<Handler>;

                boost::asio::async_read(socket_, boost::asio::buffer(buffer_),
                    boost::bind(f, shared_from_this(), 
                        boost::asio::placeholders::error, handler));
            }
        }

        /// Handle a completed read of message data.
        template <typename Handler>
        void handle_read_data(boost::system::error_code const& e,
            boost::tuple<Handler> handler)
        {
            typedef util::container_device<std::vector<char> > io_device_type;

            if (e) {
                boost::get<0>(handler)(e);

            // send the error reply back to the requesting site
                reply rep (no_success, e.message().c_str());
                async_write(rep, handler);
            }
            else {
            // do some timings
                util::high_resolution_timer t;
                boost::uint8_t command = command_unknown;

            // Extract the data structure from the data just received.
                try {
                // De-serialize the data
                    request req;
                    {
                    // create a special io stream on top of buffer_
                        boost::iostreams::stream<io_device_type> io(buffer_);
#if HPX_USE_PORTABLE_ARCHIVES != 0
                        util::portable_binary_iarchive archive(io);
#else
                        boost::archive::binary_iarchive archive(io);
#endif
                        archive >> req;
                    }

                    command = req.get_command();

                // act on request and generate reply
                    reply rep;
                    request_handler_.handle_request(req, rep);

                // send the reply back to the requesting site
                    async_write(rep, handler);
                }
                catch (std::exception const& e) {
                    // Unable to decode data.
                    boost::system::error_code 
                        error(boost::asio::error::invalid_argument);
                    boost::get<0>(handler)(error);

                // send the error reply back to the requesting site
                    reply rep (no_success, e.what());
                    async_write(rep, handler);
                }

                // gather timings
                request_handler_.add_timing(command, t.elapsed());
            }
        }

        /// Asynchronously write a data structure to the socket.
        template <typename Handler>
        void async_write(reply const& rep, boost::tuple<Handler> handler)
        {
            typedef util::container_device<std::vector<char> > io_device_type;

            {
                // create a special io stream on top of buffer_
                buffer_.clear();
                boost::iostreams::stream<io_device_type> io(buffer_);

                // Serialize the data
#if HPX_USE_PORTABLE_ARCHIVES != 0
                util::portable_binary_oarchive archive(io);
#else
                boost::archive::binary_oarchive archive(io);
#endif
                archive << rep;
            }

            size_ = (boost::uint32_t)buffer_.size();

            // Write the serialized data to the socket. We use "gather-write" 
            // to send both the header and the data in a single write operation.
            std::vector<boost::asio::const_buffer> buffers;
            buffers.push_back(boost::asio::buffer(&size_, sizeof(size_)));
            buffers.push_back(boost::asio::buffer(buffer_));

            void (connection::*f)(boost::system::error_code const&,
                    boost::tuple<Handler>)
                = &connection::handle_write_completion<Handler>;

            boost::asio::async_write(socket_, buffers, 
                    boost::bind(f, shared_from_this(), 
                        boost::asio::placeholders::error, handler));
        }

        template <typename Handler>
        void handle_write_completion(boost::system::error_code const& e,
            boost::tuple<Handler> handler)
        {
            boost::get<0>(handler)(e);

            // listen for the next request
            void (connection::*f)(boost::system::error_code const&, 
                    boost::tuple<Handler>)
                = &connection::handle_read_header<Handler>;

            boost::asio::async_read(socket_, 
                boost::asio::buffer(&size_, sizeof(size_)),
                boost::bind(f, shared_from_this(), 
                    boost::asio::placeholders::error, handler));
        }

    private:
        /// Socket for the connection.
        boost::asio::ip::tcp::socket socket_;

        /// The handler used to process the incoming request.
        request_handler& request_handler_;

        /// buffer for incoming and outgoing data
        boost::integer::ulittle32_t size_;
        std::vector<char> buffer_;
    };

    typedef boost::shared_ptr<connection> connection_ptr;

///////////////////////////////////////////////////////////////////////////////
}}}  // namespace hpx::naming::server

#endif
