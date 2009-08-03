//  Copyright (c) 2007-2009 Hartmut Kaiser
//  Copyright (c) 2007 Richard D Guidry Jr
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <string>

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception_list.hpp>
#include <hpx/runtime/naming/locality.hpp>
#include <hpx/runtime/parcelset/parcelport.hpp>
#include <hpx/util/io_service_pool.hpp>

#include <boost/version.hpp>
#include <boost/asio/io_service.hpp>
#include <boost/asio/buffer.hpp>
#include <boost/asio/read.hpp>
#include <boost/asio/write.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/bind.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset
{
    parcelport::parcelport(util::io_service_pool& io_service_pool, 
            naming::locality here)
      : io_service_pool_(io_service_pool),
        acceptor_(NULL),
        parcels_(This()),
        connection_cache_(HPX_MAX_PARCEL_CONNECTION_CACHE_SIZE, "  [PT] "), 
        here_(here)
    {}

    parcelport::~parcelport()
    {
        delete acceptor_;
    }

    bool parcelport::run(bool blocking)
    {
        io_service_pool_.run(false);    // start pool

        using boost::asio::ip::tcp;
        if (NULL == acceptor_)
            acceptor_ = new boost::asio::ip::tcp::acceptor(io_service_pool_.get_io_service());

        // initialize network
        int tried = 0;
        exception_list errors;
        naming::locality::iterator_type end = here_.accept_end();
        for (naming::locality::iterator_type it = 
                here_.accept_begin(io_service_pool_.get_io_service()); 
             it != end; ++it, ++tried)
        {
            try {
                server::parcelport_connection_ptr conn(
                    new server::parcelport_connection(
                        io_service_pool_.get_io_service(), parcels_)
                );

                tcp::endpoint ep = *it;
                acceptor_->open(ep.protocol());
                acceptor_->set_option(tcp::acceptor::reuse_address(true));
                acceptor_->bind(ep);
                acceptor_->listen();
                acceptor_->async_accept(conn->socket(),
                    boost::bind(&parcelport::handle_accept, this,
                        boost::asio::placeholders::error, conn));
            }
            catch (boost::system::system_error const& e) {
                errors.add(e);   // store all errors
                continue;
            }
        }

        if (errors.get_error_count() == tried) {
            // all attempts failed
            HPX_THROW_EXCEPTION(network_error, 
                "parcelport::parcelport", errors.get_message());
        }

        return io_service_pool_.run(blocking);
    }

    void parcelport::stop(bool blocking)
    {
        delete acceptor_;
        acceptor_ = NULL;

        io_service_pool_.stop();
        if (blocking) {
            io_service_pool_.join();
            io_service_pool_.clear();
        }
    }

    /// accepted new incoming connection
    void parcelport::handle_accept(boost::system::error_code const& e,
        server::parcelport_connection_ptr conn)
    {
        if (!e) {
        // handle this incoming parcel
            server::parcelport_connection_ptr c(conn);    // hold on to conn

        // create new connection waiting for next incoming parcel
            conn.reset(new server::parcelport_connection(
                io_service_pool_.get_io_service(), parcels_));
            acceptor_->async_accept(conn->socket(),
                boost::bind(&parcelport::handle_accept, this,
                    boost::asio::placeholders::error, conn));

        // now accept the incoming connection by starting to read from the 
        // socket
            c->async_read(
                boost::bind(&parcelport::handle_read_completion, this,
                boost::asio::placeholders::error));
        }
    }

    /// Handle completion of a read operation.
    void parcelport::handle_read_completion(boost::system::error_code const& e)
    {
        if (e && e != boost::asio::error::operation_aborted)
        {
            LPT_(error) << "handle read operation completion: error: " 
                        << e.message();
        }
    }

///////////////////////////////////////////////////////////////////////////////
}}
