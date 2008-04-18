//  Copyright (c) 2007-2008 Hartmut Kaiser
//
//  Parts of this code were taken from the Boost.Asio library
//  Copyright (c) 2003-2007 Christopher M. Kohlhoff (chris at kohlhoff dot com)
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include<iostream>

#include <boost/bind.hpp>
#include <boost/lexical_cast.hpp>
#include <hpx/util/dgas_logging.hpp>
#include <hpx/naming/resolver_server.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace naming 
{
    resolver_server::resolver_server (locality l, 
        bool start_service_async, std::size_t io_service_pool_size)
      : io_service_pool_(io_service_pool_size),
        acceptor_(io_service_pool_.get_io_service()),
        new_connection_(new server::connection(
              io_service_pool_.get_io_service(), request_handler_)),
        request_handler_(), here_(l)
   {
        util::init_dgas_logs();
        
        // Open the acceptor with the option to reuse the address (i.e. SO_REUSEADDR).
        using boost::asio::ip::tcp;
        acceptor_.open(l.get_endpoint().protocol());
        acceptor_.set_option(tcp::acceptor::reuse_address(true));
        acceptor_.bind(l.get_endpoint());
        acceptor_.listen();
        acceptor_.async_accept(new_connection_->socket(),
            boost::bind(&resolver_server::handle_accept, this,
                boost::asio::placeholders::error));

        if (start_service_async) 
            run(false);
    }

    resolver_server::resolver_server(std::string const& address, 
          unsigned short port, bool start_service_async, 
          std::size_t io_service_pool_size)
      : io_service_pool_(io_service_pool_size),
        acceptor_(io_service_pool_.get_io_service()),
        new_connection_(new server::connection(
              io_service_pool_.get_io_service(), request_handler_)),
        request_handler_(), here_(address, port)
    {
        util::init_dgas_logs();
        
        // Open the acceptor with the option to reuse the address (i.e. SO_REUSEADDR).
        using boost::asio::ip::tcp;

        // Open the acceptor with the option to reuse the address (i.e. SO_REUSEADDR).
        tcp::resolver resolver(acceptor_.io_service());
        tcp::resolver::query query(address, 
            boost::lexical_cast<std::string>(port));
        tcp::endpoint endpoint = *resolver.resolve(query);

        acceptor_.open(endpoint.protocol());
        acceptor_.set_option(tcp::acceptor::reuse_address(true));
        acceptor_.bind(endpoint);
        acceptor_.listen();
        acceptor_.async_accept(new_connection_->socket(),
            boost::bind(&resolver_server::handle_accept, this,
                boost::asio::placeholders::error));

        if (start_service_async) 
            run(false);
    }

    resolver_server::~resolver_server()
    {
    }

    void resolver_server::run(bool blocking)
    {
        LDGAS_(info) << "startup: listening at: " << here_;
        io_service_pool_.run(blocking);
    }

    void resolver_server::stop()
    {
        io_service_pool_.stop();
        LDGAS_(info) << "shutdown: stopped listening at: " << here_;
    }

    void resolver_server::handle_accept(boost::system::error_code const& e)
    {
        if (!e) {
        // handle incoming request
            new_connection_->async_read(
                boost::bind(&resolver_server::handle_completion, this,
                boost::asio::placeholders::error));
            
        // create new connection waiting for next incoming request
            new_connection_.reset(new server::connection(
                  io_service_pool_.get_io_service(), request_handler_));
            acceptor_.async_accept(new_connection_->socket(),
                boost::bind(&resolver_server::handle_accept, this,
                  boost::asio::placeholders::error));
        }
    }

    /// Handle completion of a read operation.
    void resolver_server::handle_completion(boost::system::error_code const& e)
    {
        if (e && e != boost::asio::error::operation_aborted)
        {
            // FIXME: add error handling
        }
    }

///////////////////////////////////////////////////////////////////////////////
}}  // namespace hpx::naming
