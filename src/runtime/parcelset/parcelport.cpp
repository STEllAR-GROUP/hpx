//  Copyright (c) 2007-2008 Hartmut Kaiser
//  Copyright (c) 2007 Richard D Guidry Jr
// 
//  Parts of this code were taken from the Boost.Asio library
//  Copyright (c) 2003-2007 Christopher M. Kohlhoff (chris at kohlhoff dot com)
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <string>

#include <boost/version.hpp>
#include <boost/asio.hpp>
#include <boost/bind.hpp>

#include <hpx/runtime/naming/locality.hpp>
#include <hpx/runtime/parcelset/parcelport.hpp>
#include <hpx/util/io_service_pool.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset
{
    parcelport::parcelport(util::io_service_pool& io_service_pool, 
            naming::locality here)
      : io_service_pool_(io_service_pool),
        acceptor_(io_service_pool_.get_io_service()),
        parcels_(This()),
        new_server_connection_(new server::parcelport_connection(
              io_service_pool_.get_io_service(), parcels_)),
        connection_cache_(HPX_MAX_CONNECTION_CACHE_SIZE), here_(here)
    {
        // initialize network
        using boost::asio::ip::tcp;
        
        acceptor_.open(here.get_endpoint().protocol());
        acceptor_.set_option(tcp::acceptor::reuse_address(true));
        acceptor_.bind(here.get_endpoint());
        acceptor_.listen();
        acceptor_.async_accept(new_server_connection_->socket(),
            boost::bind(&parcelport::handle_accept, this,
                boost::asio::placeholders::error));
    }

    parcelport::parcelport(util::io_service_pool& io_service_pool, 
            std::string const& address, unsigned short port)
      : io_service_pool_(io_service_pool),
        acceptor_(io_service_pool_.get_io_service()),
        parcels_(This()),
        new_server_connection_(new server::parcelport_connection(
              io_service_pool_.get_io_service(), parcels_)),
        connection_cache_(HPX_MAX_CONNECTION_CACHE_SIZE), here_(address, port)
    {
        // initialize network
        using boost::asio::ip::tcp;
        
        acceptor_.open(here_.get_endpoint().protocol());
        acceptor_.set_option(tcp::acceptor::reuse_address(true));
        acceptor_.bind(here_.get_endpoint());
        acceptor_.listen();
        acceptor_.async_accept(new_server_connection_->socket(),
            boost::bind(&parcelport::handle_accept, this,
                boost::asio::placeholders::error));
    }

    bool parcelport::run(bool blocking)
    {
        return io_service_pool_.run(blocking);
    }

    void parcelport::stop(bool blocking)
    {
        io_service_pool_.stop();
        if (blocking)
            io_service_pool_.join();
    }
    
    /// accepted new incoming connection
    void parcelport::handle_accept(boost::system::error_code const& e)
    {
        if (!e) {
        // handle this incoming parcel
            new_server_connection_->async_read(
                boost::bind(&parcelport::handle_read_completion, this,
                boost::asio::placeholders::error));

        // create new connection waiting for next incoming parcel
            new_server_connection_.reset(new server::parcelport_connection(
                io_service_pool_.get_io_service(), parcels_));
            acceptor_.async_accept(new_server_connection_->socket(),
                boost::bind(&parcelport::handle_accept, this,
                    boost::asio::placeholders::error));
        }
    }

    /// Handle completion of a read operation.
    void parcelport::handle_read_completion(boost::system::error_code const& e)
    {
        if (e && e != boost::asio::error::operation_aborted)
        {
            // FIXME: add error handling
            std::cerr << "Error: " << e.message() << std::endl;
        }
    }

///////////////////////////////////////////////////////////////////////////////
}}
