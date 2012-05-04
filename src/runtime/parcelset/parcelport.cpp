//  Copyright (c) 2007-2012 Hartmut Kaiser
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
#if defined(HPX_DEBUG)
#include <boost/foreach.hpp>
#endif

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
            naming::locality h
          , std::size_t max_cache_size
          , std::size_t max_connections_per_loc)
      : io_service_pool_(io_service_pool),
        acceptor_(NULL),
        parcels_(This()),
        connection_cache_(max_cache_size, max_connections_per_loc),
        here_(h)
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
                        io_service_pool_.get_io_service(), parcels_, timer_));

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

            if (NULL != acceptor_)
            {
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
                io_service_pool_.get_io_service(), parcels_, timer_));

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
    void parcelport::send_parcel(parcel const& p, naming::address const& addr,
        write_handler_type f)
    {
        BOOST_ASSERT(p.get_destination_addr() == addr);

        typedef pending_parcels_map::iterator iterator;
        const boost::uint32_t locality_id =
            naming::get_locality_id_from_gid(p.get_destination());

        parcelport_connection_ptr client_connection(
            connection_cache_.get(locality_id));

        // enqueue the incoming parcel ...
        {
            util::spinlock::scoped_lock l(mtx_);
            pending_parcels_[locality_id].first.push_back(p);
            pending_parcels_[locality_id].second.push_back(f);
        }

        if (!client_connection)
        {
            if (connection_cache_.full(locality_id))
                return;

            client_connection.reset(new parcelport_connection(
                io_service_pool_.get_io_service(), locality_id,
                connection_cache_, timer_, parcels_sent_));

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
                        client_connection->socket().close();
                        client_connection->socket().connect(*it, error);
                        if (!error)
                            break;
                    }
                    if (!error)
                        break;

                    // we wait for a really short amount of time
                    // TODO: Should this be an hpx::threads::suspend?
                    boost::this_thread::sleep(boost::get_system_time() +
                        boost::posix_time::milliseconds(HPX_NETWORK_RETRIES_SLEEP));
                }
                catch (boost::system::error_code const& e) {
                    HPX_THROW_EXCEPTION(network_error,
                        "parcelport::send_parcel", e.message());
                }
            }

            if (error) {
                client_connection->socket().close();

                hpx::util::osstream strm;
                strm << error.message() << " (while trying to connect to: "
                     << addr.locality_ << ")";
                HPX_THROW_EXCEPTION(network_error,
                    "parcelport::send_parcel",
                    hpx::util::osstream_get_string(strm));
            }
#if defined(HPX_DEBUG)
            else {
                client_connection->set_locality(addr.locality_);

                std::string connection_addr = client_connection->socket().remote_endpoint().address().to_string();
                boost::uint16_t connection_port = client_connection->socket().remote_endpoint().port();
                BOOST_ASSERT(addr.locality_.get_address() == connection_addr);
                BOOST_ASSERT(addr.locality_.get_port() == connection_port);
            }
#endif
        }
#if defined(HPX_DEBUG)
        else {
//                 LPT_(info) << "parcelport: reusing existing connection to: "
//                            << addr.locality_;
            BOOST_ASSERT(addr.locality_ == client_connection->get_locality());
            BOOST_ASSERT(locality_id == client_connection->destination());

            std::string connection_addr = client_connection->socket().remote_endpoint().address().to_string();
            boost::uint16_t connection_port = client_connection->socket().remote_endpoint().port();
            BOOST_ASSERT(addr.locality_.get_address() == connection_addr);
            BOOST_ASSERT(addr.locality_.get_port() == connection_port);
        }
#endif

        std::vector<parcel> parcels;
        std::vector<write_handler_type> handlers;
        {
            util::spinlock::scoped_lock l(mtx_);
            std::swap(parcels, pending_parcels_[locality_id].first);
            std::swap(handlers, pending_parcels_[locality_id].second);
        }

        // if the parcels didn't get sent by another connection ...
        if(!parcels.empty() && !handlers.empty())
        {
#if defined(HPX_DEBUG)
            // verify the connection points to the right destination
            BOOST_FOREACH(parcel const& pp, parcels)
            {
                const boost::uint32_t parcel_locality_id =
                    naming::get_locality_id_from_gid(pp.get_destination());
                BOOST_ASSERT(parcel_locality_id == locality_id);
                BOOST_ASSERT(parcel_locality_id == client_connection->destination());
                BOOST_ASSERT(pp.get_destination_addr() == addr);
                BOOST_ASSERT(addr.locality_.get_address() ==
                    client_connection->socket().remote_endpoint().address().to_string());
                BOOST_ASSERT(addr.locality_.get_port() ==
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
                boost::bind(&parcelport::send_pending_parcels_trampoline, this, ::_1)
            );
        }
        else
        {
            // ... or re-add the stuff to the cache
            BOOST_ASSERT(locality_id == client_connection->destination());
            connection_cache_.add(locality_id, client_connection);
        }
    }

    void parcelport::send_pending_parcels_trampoline(boost::uint32_t locality_id)
    {
        parcelport_connection_ptr client_connection(
            connection_cache_.get(locality_id));

        // If another thread was faster ... try again
        if (!client_connection)
            return;

        std::vector<parcel> parcels;
        std::vector<write_handler_type> handlers;
        {
            typedef pending_parcels_map::iterator iterator;

            util::spinlock::scoped_lock l(mtx_);
            iterator it = pending_parcels_.find(locality_id);

            if(it != pending_parcels_.end())
            {
                std::swap(parcels, it->second.first);
                std::swap(handlers, it->second.second);
            }
        }

        if (!parcels.empty() && !handlers.empty())
        {
#if defined(HPX_DEBUG)
            // verify the connection points to the right destination
            BOOST_FOREACH(parcel const& p, parcels)
            {
                const boost::uint32_t parcel_locality_id =
                    naming::get_locality_id_from_gid(p.get_destination());
                BOOST_ASSERT(parcel_locality_id == locality_id);
                BOOST_ASSERT(parcel_locality_id == client_connection->destination());
                BOOST_ASSERT(p.get_destination_addr().locality_.get_address() ==
                    client_connection->socket().remote_endpoint().address().to_string());
                BOOST_ASSERT(p.get_destination_addr().locality_.get_port() ==
                    client_connection->socket().remote_endpoint().port());
            }
#endif

            // create a new thread which sends parcels that might still be pending
            hpx::applier::register_thread_nullary(
                HPX_STD_BIND(&parcelport::send_pending_parcels, this,
                    client_connection, boost::move(parcels),
                    boost::move(handlers)), "send_pending_parcels");
        }
        else
        {
            BOOST_ASSERT(locality_id == client_connection->destination());
            connection_cache_.add(locality_id, client_connection);
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
            const boost::uint32_t parcel_locality_id =
                naming::get_locality_id_from_gid(p.get_destination());
            BOOST_ASSERT(parcel_locality_id == client_connection->destination());
            BOOST_ASSERT(p.get_destination_addr().locality_.get_address() ==
                client_connection->socket().remote_endpoint().address().to_string());
            BOOST_ASSERT(p.get_destination_addr().locality_.get_port() ==
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
            boost::bind(&parcelport::send_pending_parcels_trampoline, this, ::_1)
        );
    }
}}
