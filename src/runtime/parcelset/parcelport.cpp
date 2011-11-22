//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c) 2007      Richard D Guidry Jr
//  Copyright (c) 2011      Bryce Lelbach & Katelyn Kufahl
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <string>

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception_list.hpp>
#include <hpx/runtime/naming/locality.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/runtime/parcelset/parcelport.hpp>
#include <hpx/util/io_service_pool.hpp>
#include <hpx/util/stringstream.hpp>

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
        connection_cache_(HPX_MAX_PARCEL_CONNECTION_CACHE_SIZE),
        here_(here)
    {}

    parcelport::~parcelport()
    {
        // make sure all existing connections get destroyed first
        connection_cache_.clear();
        if (NULL != acceptor_)
            delete acceptor_;
    }

    bool parcelport::run(bool blocking)
    {
        io_service_pool_.run(false);    // start pool

        using boost::asio::ip::tcp;
        if (NULL == acceptor_)
            acceptor_ = new boost::asio::ip::tcp::acceptor(io_service_pool_.get_io_service());

        // initialize network
        std::size_t tried = 0;
        exception_list errors;
        naming::locality::iterator_type end = here_.accept_end();
        for (naming::locality::iterator_type it =
                here_.accept_begin(io_service_pool_.get_io_service());
             it != end; ++it, ++tried)
        {
            try {
                server::parcelport_connection_ptr conn(
                    new server::parcelport_connection(
                        io_service_pool_.get_io_service(), parcels_,
                        timer_, parcels_received_)
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
        // make sure no more work is pending, wait for service pool to get empty
        io_service_pool_.stop();
        if (blocking) {
            io_service_pool_.join();

            // now it's safe to take everything down
            connection_cache_.clear();

            delete acceptor_;
            acceptor_ = NULL;

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
                io_service_pool_.get_io_service(), parcels_,
                timer_, parcels_received_));
            acceptor_->async_accept(conn->socket(),
                boost::bind(&parcelport::handle_accept, this,
                    boost::asio::placeholders::error, conn));

        // now accept the incoming connection by starting to read from the
        // socket
            c->async_read(
                boost::bind(&parcelport::handle_read_completion, this,
                boost::asio::placeholders::error, c));
        }
    }

    /// Handle completion of a read operation.
    void parcelport::handle_read_completion(boost::system::error_code const& e,
        server::parcelport_connection_ptr c)
    {
        if (e && e != boost::asio::error::operation_aborted) {
            LPT_(error) << "handle read operation completion: error: "
                        << e.message();
        }
        else {
            // complete data point and push back
            performance_counters::parcels::data_point& data = c->get_receive_data();
            data.timer_ = timer_.elapsed_microseconds() - data.timer_;
            parcels_received_.push_back(data);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    void parcelport::send_parcel(parcel const& p, naming::address const& addr,
        write_handler_type f)
    {
        const boost::uint32_t prefix =
            naming::get_prefix_from_gid(p.get_destination());

        parcelport_connection_ptr client_connection(connection_cache_.get(prefix));

//        if (!client_connection) {
//            if (threads::get_self_ptr()) 
//                hpx::threads::suspend(
//                    boost::posix_time::milliseconds(500));
//            else
//                boost::this_thread::sleep(boost::get_system_time() +
//                    boost::posix_time::milliseconds(500));
           
            // Try again. 
//            client_connection = connection_cache_.get(prefix);
//        }                

        if (!client_connection) {
//                 LPT_(info) << "parcelport: creating new connection to: "
//                            << addr.locality_;

        // The parcel gets serialized inside the connection constructor, no
        // need to keep the original parcel alive after this call returned.
            client_connection.reset(new parcelport_connection(
                io_service_pool_.get_io_service(), prefix,
                connection_cache_, timer_, parcels_sent_));

            client_connection->set_parcel(p);

        // connect to the target locality, retry if needed
            boost::system::error_code error = boost::asio::error::try_again;
            for (int i = 0; i < HPX_MAX_NETWORK_RETRIES; ++i)
            {
                try {
                    naming::locality::iterator_type end = addr.locality_.connect_end();
                    for (naming::locality::iterator_type it =
                            addr.locality_.connect_begin(io_service_pool_.get_io_service());
                         it != end; ++it)
                    {
//                         boost::system::error_code ec;
//                         client_connection->socket().shutdown(
//                             boost::asio::socket_base::shutdown_both, ec);
                        client_connection->socket().close();
                        client_connection->socket().connect(*it, error);
                        if (!error)
                            break;
                    }
                    if (!error)
                        break;

                    // we wait for a really short amount of time
                    boost::this_thread::sleep(boost::get_system_time() +
                        boost::posix_time::milliseconds(HPX_NETWORK_RETRIES_SLEEP));
                }
                catch (boost::system::error_code const& e) {
                    HPX_THROW_EXCEPTION(network_error,
                        "parcelport::send_parcel", e.message());
                }
            }
            if (error) {
//                 boost::system::error_code ec;
//                 client_connection->socket().shutdown(
//                     boost::asio::socket_base::shutdown_both, ec);
                client_connection->socket().close();

                hpx::util::osstream strm;
                strm << error.message() << " (while trying to connect to: "
                     << addr.locality_ << ")";
                HPX_THROW_EXCEPTION(network_error,
                    "parcelport::send_parcel",
                    hpx::util::osstream_get_string(strm));
            }

        // Start an asynchronous write operation.
            client_connection->async_write(f);
        }
        else {
//                 LPT_(info) << "parcelport: reusing existing connection to: "
//                            << addr.locality_;

        // reuse an existing connection
            client_connection->set_parcel(p);
            client_connection->async_write(f);
        }
    }
}}
