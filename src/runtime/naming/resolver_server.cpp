//  Copyright (c) 2007-2008 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include<iostream>

#include <hpx/hpx_fwd.hpp>

#include <boost/bind.hpp>
#include <boost/lexical_cast.hpp>

#include <hpx/exception_list.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/asio_util.hpp>
#include <hpx/util/runtime_configuration.hpp>
#include <hpx/runtime/naming/resolver_server.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/export.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace naming 
{
    resolver_server::resolver_server (util::io_service_pool& io_service_pool, 
            locality l)
      : io_service_pool_(io_service_pool),
        acceptor_(io_service_pool_.get_io_service()),
        request_handler_(), here_(l)
   {
        util::init_dgas_logs();

        // start the io_service
        run(false);

        // Open the acceptor with the option to reuse the address (i.e. SO_REUSEADDR).
        using boost::asio::ip::tcp;

        int tried = 0;
        exception_list errors;
        locality::iterator_type end = here_.accept_end();
        for (locality::iterator_type it = 
                here_.accept_begin(io_service_pool_.get_io_service()); 
             it != end; ++it, ++tried)
        {
            try {
                server::connection_ptr conn(new server::connection(
                    io_service_pool_.get_io_service(), request_handler_));

                tcp::endpoint ep = *it;
                acceptor_.open(ep.protocol());
                acceptor_.set_option(tcp::acceptor::reuse_address(true));
                acceptor_.bind(ep);
                acceptor_.listen();
                acceptor_.async_accept(conn->socket(),
                    boost::bind(&resolver_server::handle_accept, this,
                        boost::asio::placeholders::error, conn));
            }
            catch (boost::system::system_error const& e) {
                errors.add(e);   // store all errors
                continue;
            }
        }

        if (errors.get_error_count() == tried) {
            // all tries failed
            boost::throw_exception(
                hpx::exception(network_error, errors.get_message()));
        }
    }

    resolver_server::resolver_server(util::io_service_pool& io_service_pool, 
            std::string const& address, boost::uint16_t port)
      : io_service_pool_(io_service_pool),
        acceptor_(io_service_pool_.get_io_service()),
        request_handler_(), 
        here_(util::runtime_configuration().get_dgas_locality(address, port))
    {
        util::init_dgas_logs();

        // start the io_service
        run(false);

        // Open the acceptor with the option to reuse the address (i.e. SO_REUSEADDR).
        using boost::asio::ip::tcp;

        int tried = 0;
        exception_list errors;
        locality::iterator_type end = here_.accept_end();
        for (locality::iterator_type it = 
                here_.accept_begin(io_service_pool_.get_io_service()); 
             it != end; ++it, ++tried)
        {
            try {
                server::connection_ptr conn(new server::connection(
                    io_service_pool_.get_io_service(), request_handler_));

                tcp::endpoint ep = *it;
                acceptor_.open(ep.protocol());
                acceptor_.set_option(tcp::acceptor::reuse_address(true));
                acceptor_.bind(ep);
                acceptor_.listen();
                acceptor_.async_accept(conn->socket(),
                    boost::bind(&resolver_server::handle_accept, this,
                        boost::asio::placeholders::error, conn));
            }
            catch (boost::system::system_error const& e) {
                errors.add(e);   // store all errors
                continue;
            }
        }

        if (errors.get_error_count() == tried) {
            // all tries failed
            boost::throw_exception(
                hpx::exception(network_error, errors.get_message()));
        }
    }

    resolver_server::~resolver_server()
    {
        stop();       // stop services if not stopped already
    }

    void resolver_server::run(bool blocking)
    {
        if (!io_service_pool_.is_running()) {
            LDGAS_(info) << "startup: listening at: " << here_;
            io_service_pool_.run(blocking);
        }
    }

    void resolver_server::stop()
    {
        if (io_service_pool_.is_running()) {
            io_service_pool_.stop();
            LDGAS_(info) << "shutdown: stopped listening at: " << here_;
        }
    }

    void resolver_server::handle_accept(boost::system::error_code const& e,
        server::connection_ptr conn)
    {
        if (!e) {
        // handle incoming request
            conn->async_read(boost::bind(&resolver_server::handle_completion, 
                this, boost::asio::placeholders::error));

        // create new connection waiting for next incoming request
            conn.reset(new server::connection(
                io_service_pool_.get_io_service(), request_handler_));
            acceptor_.async_accept(conn->socket(),
                boost::bind(&resolver_server::handle_accept, this,
                  boost::asio::placeholders::error, conn));
        }
    }

    /// Handle completion of a read operation.
    void resolver_server::handle_completion(boost::system::error_code const& e)
    {
        if (e && e != boost::asio::error::operation_aborted)
        {
            // FIXME: add error handling
            LDGAS_(error) << "handle read operation completion: error: " 
                          << e.message();
        }
    }

///////////////////////////////////////////////////////////////////////////////
}}  // namespace hpx::naming
