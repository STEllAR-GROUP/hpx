//  Copyright (c) 2007-2008 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_NAMING_RESOLVERCLIENT_CONNECTION_MAY_27_2008_0317PM)
#define HPX_NAMING_RESOLVERCLIENT_CONNECTION_MAY_27_2008_0317PM

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
#include <hpx/runtime/naming/server/reply.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace naming 
{
    namespace detail
    {
        template <typename T>
        struct handler;
        
        template <>
        struct handler<bool>
        {
            static void call(boost::system::error_code const& e, 
                util::promise<bool>* promise_, server::request const& req, 
                server::reply* rep) 
            {
                if (!e) {
                    // everything is fine
                    promise_->set_value(rep->get_status() == success);
                }
                else {
                    // propagate exception
                    promise_->set_exception(
                        boost::copy_exception(boost::system::system_error(e)));
                }
            }
        };

        template <>
        struct handler<std::pair<bool, address> >
        {
            static void call(boost::system::error_code const& e, 
                util::promise<std::pair<bool, address> >* promise_, 
                server::request const& req, server::reply* rep) 
            {
                if (!e) {
                    // everything is fine
                    if (rep->get_status() == success)
                        promise_->set_value(std::make_pair(true, rep->get_address()));
                    else
                        promise_->set_value(std::make_pair(false, address()));
                }
                else {
                    // propagate exception
                    promise_->set_exception(
                        boost::copy_exception(boost::system::system_error(e)));
                }
            }
        };
    }
    
    /// Represents a single resolver_client_connection from a client.
    template <typename T>
    struct resolver_client_connection
      : public boost::enable_shared_from_this<resolver_client_connection<T> >,
        private boost::noncopyable
    {    
    private:
        typedef util::container_device<std::vector<char> > io_device_type;

    public:
        /// Construct a sending resolver_client_connection (for the \a
        /// naming#server#command_bind_range command)
        resolver_client_connection(boost::asio::ip::tcp::socket& socket,
                server::dgas_server_command c, id_type lower_id, 
                std::size_t count, address const& addr, std::ptrdiff_t offset)
          : socket_(socket), req_(c, lower_id, count, addr, offset)
        {}
        
        /// Construct a sending resolver_client_connection (for the \a
        /// naming#server#command_unbind_range command)
        resolver_client_connection(boost::asio::ip::tcp::socket& socket,
                server::dgas_server_command c, id_type lower_id, 
                std::size_t count)
          : socket_(socket), req_(c, lower_id, count)
        {}
        
        /// Construct a sending resolver_client_connection (for the \a
        /// naming#server#command_resolve command)
        resolver_client_connection(boost::asio::ip::tcp::socket& socket,
                server::dgas_server_command c, id_type id)
          : socket_(socket), req_(c, id)
        {}
        
        /// Asynchronously write a data structure to the socket.
        void execute()
        {
            // serialize the request into the buffer
            {
                // create a special io stream on top of out_buffer_
                buffer_.clear();
                boost::iostreams::stream<io_device_type> io(buffer_);

                // Serialize the data
                util::portable_binary_oarchive archive(io);
                archive << req_;
            }
            size_ = buffer_.size();

            // execute the requested action
            execute(boost::bind(&detail::handler<T>::call, _1, _2, _3, _4));
        }
        
        /// Get the socket associated with the resolver_client_connection.
        boost::asio::ip::tcp::socket& socket() { return socket_; }

        /// return future usable to retrieve the result of the current operation
        util::unique_future<T> get_future()
        {
            return promise_.get_future();
        }
        
    protected:
        template <typename Handler>
        void execute(Handler handler)
        {
            namespace asio = boost::asio;
            namespace system = boost::system;
            
            // Write the serialized data to the socket. We use "gather-write" 
            // to send both the header and the data in a single write operation.
            std::vector<asio::const_buffer> buffers;
            buffers.push_back(asio::buffer(&size_, sizeof(size_)));
            buffers.push_back(asio::buffer(buffer_));

            // this additional wrapping of the handler into a bind object is 
            // needed to keep  this resolver_client_connection object alive for the whole
            // write operation
            void (resolver_client_connection::*f)(system::error_code const&, 
                    boost::tuple<Handler>) =
                &resolver_client_connection::handle_write;
                
            asio::async_write(socket_, buffers, 
                boost::bind(f, this->shared_from_this(), 
                    asio::placeholders::error, boost::make_tuple(handler)));
        }

        /// handle completed write operation
        template <typename Handler>
        void handle_write(boost::system::error_code const& e, 
            boost::tuple<Handler> handler)
        {
            namespace asio = boost::asio;
            namespace system = boost::system;
            
            if (e) {
                // propagate error
                boost::get<0>(handler)(e, &promise_, req_, (server::reply *)NULL);
            }
            else {
                // Issue a read operation to read exactly the number of bytes 
                // in a header.
                void (resolver_client_connection::*f)(system::error_code const&, 
                        boost::tuple<Handler>)
                    = &resolver_client_connection::handle_read_header;
                
                buffer_.clear();
                size_ = 0;
                boost::asio::async_read(socket_, 
                    boost::asio::buffer(&size_, sizeof(size_)),
                    boost::bind(f, this->shared_from_this(), 
                        asio::placeholders::error, handler));
            }
        }

        /// Handle a completed read of a message header. The handler is passed 
        /// using a tuple since boost::bind seems to have trouble binding a 
        /// function object created using boost::bind as a parameter.
        template <typename Handler>
        void handle_read_header(boost::system::error_code const& e,
            boost::tuple<Handler> handler)
        {
            namespace asio = boost::asio;
            namespace system = boost::system;
            
            if (e) {
                // propagate error
                boost::get<0>(handler)(e, &promise_, req_, (server::reply *)NULL);
            }
            else {
                // Determine the length of the serialized data.
                std::size_t inbound_data_size = size_;

                // Start an asynchronous call to receive the data.
                buffer_.resize(inbound_data_size);
                void (resolver_client_connection::*f)(system::error_code const&,
                        boost::tuple<Handler>)
                    = &resolver_client_connection::handle_read_data;

                boost::asio::async_read(socket_, boost::asio::buffer(buffer_),
                    boost::bind(f, this->shared_from_this(), 
                        asio::placeholders::error, handler));
            }
        }

        /// Handle a completed read of message data.
        template <typename Handler>
        void handle_read_data(boost::system::error_code const& e,
            boost::tuple<Handler> handler)
        {
            namespace asio = boost::asio;
            namespace system = boost::system;
            
            if (e) {
                // propagate error
                boost::get<0>(handler)(e, &promise_, req_, (server::reply *)NULL);
            }
            else {
                // Extract the data structure from the data just received.
                server::reply rep;
                try {
                // create a special io stream on top of in_buffer_
                    boost::iostreams::stream<io_device_type> io(buffer_);

                // de-serialize the data
                    util::portable_binary_iarchive archive(io);
                    archive >> rep;
                }
                catch (std::exception const& /*e*/) {
                    // Unable to decode data.
                    system::error_code error(asio::error::invalid_argument);
                    boost::get<0>(handler)(error, &promise_, req_, (server::reply *)NULL);
                    return;
                }

                // Inform caller that data has been received ok.
                boost::get<0>(handler)(e, &promise_, req_, &rep);
            }
        }

    private:
        /// promise for the result to return
        util::promise<T> promise_;
        
        /// Socket for the resolver_client_connection.
        boost::asio::ip::tcp::socket& socket_;

        /// Request sent by this connection
        server::request req_;
        
        /// buffer for outgoing and incoming data
        boost::integer::ulittle32_t size_;
        std::vector<char> buffer_;
    };

///////////////////////////////////////////////////////////////////////////////
}}

#endif
