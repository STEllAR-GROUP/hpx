//  Copyright (c) 2007-2008 Hartmut Kaiser
//
//  Parts of this code were taken from the Boost.Asio library
//  Copyright (c) 2003-2007 Christopher M. Kohlhoff (chris at kohlhoff dot com)
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_NAMING_SERVER_RESOLVER_SERVER_MAR_24_2008_1005AM)
#define HPX_NAMING_SERVER_RESOLVER_SERVER_MAR_24_2008_1005AM

#include <string>
#include <vector>

#include <boost/asio.hpp>
#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>

#include <hpx/config.hpp>
#include <hpx/util/io_service_pool.hpp>
#include <hpx/naming/server/connection.hpp>
#include <hpx/naming/server/request_handler.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace naming 
{
    /// The top-level class of the GAC server.
    class resolver_server
      : private boost::noncopyable
    {
    public:
        /// Construct the server to listen on the specified TCP address and port, 
        /// and serve up requests to the address translation service.
        explicit resolver_server (
            std::string const& address = "localhost", 
            unsigned short port = HPX_NAME_RESOLVER_PORT, 
            bool start_service_asynchronously = false, 
            std::size_t io_service_pool_size = 1);

        /// Construct the server to listen to the endpoint given by the 
        /// locality and serve up requests to the address translation service.
        resolver_server (locality l, 
            bool start_service_asynchronously = false, 
            std::size_t io_service_pool_size = 1);

        /// Destruct the object. Stops the service if it has not been stopped
        /// already.
        ~resolver_server();
        
        /// Run the server's io_service loop.
        void run (bool blocking = true);

        /// Stop the server.
        void stop();

    private:
        /// Handle completion of an asynchronous accept and read operations.
        void handle_accept(boost::system::error_code const& e);
        void handle_completion(boost::system::error_code const& e);

        /// The pool of io_service objects used to perform asynchronous operations.
        util::io_service_pool io_service_pool_;

        /// Acceptor used to listen for incoming connections.
        boost::asio::ip::tcp::acceptor acceptor_;

        /// The next connection to be accepted.
        server::connection_ptr new_connection_;

        /// The handler for all incoming requests.
        server::request_handler request_handler_;
        
        /// this represents the locality this server is running on
        naming::locality here_;
    };

///////////////////////////////////////////////////////////////////////////////
}}  // namespace hpx::naming

#endif 
