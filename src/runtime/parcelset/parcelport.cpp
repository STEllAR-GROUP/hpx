//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2007 Richard D Guidry Jr
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2011 Katelyn Kufahl
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <string>
#include <map>
#include <set>

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception_list.hpp>
#include <hpx/runtime/naming/locality.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/runtime/parcelset/parcelport.hpp>
#include <hpx/util/runtime_configuration.hpp>
#include <hpx/util/io_service_pool.hpp>
#include <hpx/util/stringstream.hpp>

#include <boost/version.hpp>
#include <boost/asio/io_service.hpp>
#include <boost/asio/buffer.hpp>
#include <boost/asio/read.hpp>
#include <boost/asio/write.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/bind.hpp>
#include <boost/foreach.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        struct call_for_each
        {
            typedef void result_type;

            typedef std::vector<parcelport::write_handler_type> data_type;
            data_type fv_;

            call_for_each(data_type const& fv)
              : fv_(fv)
            {}

            result_type operator()(
                boost::system::error_code const& e,
                std::size_t bytes_written) const
            {
                BOOST_FOREACH(parcelport::write_handler_type f, fv_)
                {
                    f(e, bytes_written);
                }
            }
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    parcelport::parcelport(util::io_service_pool& io_service_pool,
            util::runtime_configuration const& ini)
      : io_service_pool_(io_service_pool),
        acceptor_(NULL),
        parcels_(This()),
        connection_cache_(ini.get_max_connections(), ini.get_max_connections_per_loc()),
        here_(ini.get_parcelport_address())
    {
        if (here_.get_type() != connection_tcpip) {
            HPX_THROW_EXCEPTION(network_error, "parcelport::parcelport",
                "this parcelport was instantiated to represent a unexpected "
                "locality type: " + get_connection_type_name(here_.get_type()));
        }
    }

    parcelport::~parcelport()
    {
        // make sure all existing connections get destroyed first
        connection_cache_.clear();
        if (NULL != acceptor_) {
            boost::system::error_code ec;
            acceptor_->close(ec);
            delete acceptor_;
        }
    }

    bool parcelport::run(bool blocking)
    {
        io_service_pool_.run(false);    // start pool

        using boost::asio::ip::tcp;
        if (NULL == acceptor_)
            acceptor_ = new tcp::acceptor(io_service_pool_.get_io_service());

        // initialize network
        std::size_t tried = 0;
        exception_list errors;
        naming::locality::iterator_type end = accept_end(here_);
        for (naming::locality::iterator_type it =
                accept_begin(here_, io_service_pool_.get_io_service());
             it != end; ++it, ++tried)
        {
            try {
                server::parcelport_connection_ptr conn(
                    new server::parcelport_connection(
                        io_service_pool_.get_io_service(), parcels_));

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
            return false;
        }

        if (blocking)
            io_service_pool_.join();

        return true;
    }

    void parcelport::stop(bool blocking)
    {
        // make sure no more work is pending, wait for service pool to get empty
        io_service_pool_.stop();
        if (blocking) {
            io_service_pool_.join();

            // now it's safe to take everything down
            connection_cache_.clear();

            {
                // cancel all pending read operations, close those sockets
                util::spinlock::scoped_lock l(mtx_);
                BOOST_FOREACH(server::parcelport_connection_ptr c,
                    accepted_connections_)
                {
                    boost::system::error_code ec;
                    c->socket().shutdown(boost::asio::ip::tcp::socket::shutdown_both, ec);
                    c->socket().close(ec);    // close the socket to give it back to the OS
                }
                accepted_connections_.clear();
            }

            // cancel all pending accept operations
            if (NULL != acceptor_)
            {
                boost::system::error_code ec;
                acceptor_->close(ec);
                delete acceptor_;
                acceptor_ = NULL;
            }

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

            {
                // keep track of all the accepted connections
                util::spinlock::scoped_lock l(mtx_);
                accepted_connections_.insert(c);
            }

            // disable Nagle algorithm, disable lingering on close
            c->socket().set_option(boost::asio::ip::tcp::no_delay(true));
            c->socket().set_option(boost::asio::socket_base::linger(true, 0));

            // now accept the incoming connection by starting to read from the
            // socket
            c->async_read(
                boost::bind(&parcelport::handle_read_completion, this,
                    boost::asio::placeholders::error, c));
        }
    }

    /// Handle completion of a read operation.
    void parcelport::handle_read_completion(boost::system::error_code const& e,
        server::parcelport_connection_ptr)
    {
        if (e && e != boost::asio::error::operation_aborted
              && e != boost::asio::error::eof)
        {
            LPT_(error)
                << "handle read operation completion: error: "
                << e.message();
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    void parcelport::put_parcel(parcel const& p, write_handler_type f)
    {
        typedef pending_parcels_map::iterator iterator;
        typedef pending_parcels_map::mapped_type mapped_type;

        naming::locality locality_id = p.get_destination_locality();

        // enqueue the incoming parcel ...
        {
            util::spinlock::scoped_lock l(mtx_);

            mapped_type& e = pending_parcels_[locality_id];
            pending_parcels_[locality_id].first.push_back(p);
            pending_parcels_[locality_id].second.push_back(f);
        }

        parcelport_connection_ptr client_connection = get_connection(locality_id);
        std::vector<parcel> parcels;
        std::vector<write_handler_type> handlers;

        {
            util::spinlock::scoped_lock l(mtx_);
            iterator it = pending_parcels_.find(locality_id);

            if (it != pending_parcels_.end())
            {
                BOOST_ASSERT(it->first == locality_id);
                std::swap(parcels, it->second.first);
                std::swap(handlers, it->second.second);
            }
        }

        // If the parcels didn't get sent by another connection ...
        if (!parcels.empty() && !handlers.empty())
        {
            send_pending_parcels(client_connection, parcels, handlers);
        }
        else
        {
            // ... or re-add the stuff to the cache
            BOOST_ASSERT(locality_id == client_connection->destination());
            connection_cache_.reclaim(locality_id, client_connection);
        }
    }

    void parcelport::send_pending_parcels_trampoline(
        naming::locality const& locality_id,
        parcelport_connection_ptr client_connection)
    {
        std::vector<parcel> parcels;
        std::vector<write_handler_type> handlers;

        typedef pending_parcels_map::iterator iterator;

        util::spinlock::scoped_lock l(mtx_);
        iterator it = pending_parcels_.find(locality_id);

        if (it != pending_parcels_.end())
        {
            std::swap(parcels, it->second.first);
            std::swap(handlers, it->second.second);
        }

        if (!parcels.empty() && !handlers.empty())
        {
            // Create a new thread which sends parcels that might still be
            // pending.
            hpx::applier::register_thread_nullary(
                HPX_STD_BIND(&parcelport::send_pending_parcels, this,
                    client_connection, boost::move(parcels),
                    boost::move(handlers)), "send_pending_parcels");
        }
        else
        {
            // Give this connection back to the cache as it's not needed
            // anymore.
            BOOST_ASSERT(locality_id == client_connection->destination());
            connection_cache_.reclaim(locality_id, client_connection);
        }
    }

    void parcelport::send_pending_parcels(
        parcelport_connection_ptr client_connection,
        std::vector<parcel> const & parcels,
        std::vector<write_handler_type> const & handlers)
    {
#if defined(HPX_DEBUG)
        // verify the connection points to the right destination
        BOOST_FOREACH(parcel const& p, parcels)
        {
            naming::locality const parcel_locality_id = p.get_destination_locality();
            BOOST_ASSERT(parcel_locality_id == client_connection->destination());
            BOOST_ASSERT(parcel_locality_id.get_address() ==
                client_connection->socket().remote_endpoint().address().to_string());
            BOOST_ASSERT(parcel_locality_id.get_port() ==
                client_connection->socket().remote_endpoint().port());
        }
#endif
        // store parcels in connection
        // The parcel gets serialized inside set_parcel, no
        // need to keep the original parcel alive after this call returned.
        client_connection->set_parcel(parcels);

        // ... start an asynchronous write operation now.
        client_connection->async_write(
            detail::call_for_each(handlers),
            boost::bind(&parcelport::send_pending_parcels_trampoline, this,
                ::_1, ::_2));
    }

    ///////////////////////////////////////////////////////////////////////////
    parcelport_connection_ptr parcelport::get_connection(naming::locality const& l)
    {
        parcelset::parcelport_connection_ptr client_connection;
        bool got_cache_space = false;

        for (std::size_t i = 0; i < HPX_MAX_NETWORK_RETRIES; ++i)
        {
            // Get a connection or reserve space for a new connection.
            if (connection_cache_.get_or_reserve(l, client_connection))
            {
                got_cache_space = true;
                break;
            }

            // Wait for a really short amount of time (usually 100 ms).
            boost::this_thread::sleep(boost::get_system_time() +
                boost::posix_time::milliseconds(HPX_NETWORK_RETRIES_SLEEP));
        }

        // If we didn't get a connection or permission to create one (which is
        // unlikely), bail.
        if (!got_cache_space)
        {
            HPX_THROW_EXCEPTION(network_error,
                "parcelport::get_connection",
                "timed out while trying to find room in the connection cache");
            return client_connection;
        }

        // Check if we need to create the new connection.
        if (!client_connection)
        {
            // The parcel gets serialized inside the connection constructor, no
            // need to keep the original parcel alive after this call returned.
            client_connection.reset(new parcelport_connection(
                io_service_pool_.get_io_service(), l,
                connection_cache_, parcels_sent_));

            // Connect to the target locality, retry if needed
            boost::system::error_code error = boost::asio::error::try_again;
            for (std::size_t i = 0; i < HPX_MAX_NETWORK_RETRIES; ++i)
            {
                try {
                    naming::locality::iterator_type end = connect_end(l);
                    for (naming::locality::iterator_type it =
                            connect_begin(l, io_service_pool_.get_io_service());
                         it != end; ++it)
                    {
                        client_connection->socket().close();
                        client_connection->socket().connect(*it, error);
                        if (!error)
                            break;
                    }
                    if (!error)
                        break;

                    // wait for a really short amount of time (usually 100 ms)
                    boost::this_thread::sleep(boost::get_system_time() +
                        boost::posix_time::milliseconds(
                            HPX_NETWORK_RETRIES_SLEEP));
                }
                catch (boost::system::error_code const& e) {
                    HPX_THROW_EXCEPTION(network_error,
                        "parcelport::get_connection", e.message());
                    return client_connection;
                }
            }

            if (error) {
                client_connection->socket().close();

                hpx::util::osstream strm;
                strm << error.message() << " (while trying to connect to: "
                     << l << ")";
                HPX_THROW_EXCEPTION(network_error,
                    "parcelport::get_connection",
                    hpx::util::osstream_get_string(strm));
                return client_connection;
            }

            // make sure the Nagle algorithm is disabled for this socket,
            // disable lingering on close
            client_connection->socket().set_option(
                boost::asio::ip::tcp::no_delay(true));
            client_connection->socket().set_option(
                boost::asio::socket_base::linger(true, 0));
        }

#if defined(HPX_DEBUG)
        BOOST_ASSERT(l == client_connection->destination());

        std::string connection_addr = client_connection->socket().remote_endpoint().address().to_string();
        boost::uint16_t connection_port = client_connection->socket().remote_endpoint().port();
        BOOST_ASSERT(l.get_address() == connection_addr);
        BOOST_ASSERT(l.get_port() == connection_port);
#endif

        return client_connection;
    }
}}
