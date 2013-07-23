//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2007 Richard D Guidry Jr
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2011 Katelyn Kufahl
//  Copyright (c) 2011-2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception_list.hpp>
#include <hpx/runtime/naming/locality.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/runtime/parcelset/tcp/parcelport.hpp>
#include <hpx/runtime/parcelset/detail/call_for_each.hpp>
#include <hpx/util/runtime_configuration.hpp>
#include <hpx/util/io_service_pool.hpp>
#include <hpx/util/stringstream.hpp>
#include <hpx/util/logging.hpp>

#if defined(HPX_HAVE_SECURITY)
#include <hpx/components/security/hash.hpp>
#include <hpx/components/security/parcel_suffix.hpp>
#include <hpx/components/security/certificate.hpp>
#include <hpx/components/security/signed_type.hpp>
#endif

#include <boost/version.hpp>
#include <boost/asio/buffer.hpp>
#include <boost/asio/read.hpp>
#include <boost/asio/write.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/bind.hpp>
#include <boost/foreach.hpp>

namespace hpx
{
    /// \brief Verify the certificate in the given byte sequence
    ///
    /// \param data      The full received message buffer, assuming that it
    ///                  has a parcel_suffix appended.
    /// \param parcel_id The parcel id of the first parcel in side the message
    ///
    HPX_API_EXPORT bool verify_parcel_suffix(std::vector<char> const& data,
        naming::gid_type& parcel_id, error_code& ec = throws);
}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset { namespace tcp
{
    ///////////////////////////////////////////////////////////////////////////
    parcelport::parcelport(util::runtime_configuration const& ini,
            HPX_STD_FUNCTION<void(std::size_t, char const*)> const& on_start_thread,
            HPX_STD_FUNCTION<void()> const& on_stop_thread)
      : parcelset::parcelport(ini),
        io_service_pool_(ini.get_thread_pool_size("parcel_pool"),
            on_start_thread, on_stop_thread, "parcel_pool_tcp", "-tcp"),
        acceptor_(NULL),
        connection_cache_(ini.get_max_connections(), ini.get_max_connections_per_loc())
    {
        /*
        if (here_.get_type() != connection_tcpip) {
            HPX_THROW_EXCEPTION(network_error, "tcp::parcelport::parcelport",
                "this parcelport was instantiated to represent an unexpected "
                "locality type: " + get_connection_type_name(here_.get_type()));
        }
        */
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

    util::io_service_pool* parcelport::get_thread_pool(char const* name)
    {
        if (0 == std::strcmp(name, io_service_pool_.get_name()))
            return &io_service_pool_;
        return 0;
    }

    bool parcelport::run(bool blocking)
    {
        io_service_pool_.run(false);    // start pool

        using boost::asio::ip::tcp;
        boost::asio::io_service& io_service = io_service_pool_.get_io_service();
        if (NULL == acceptor_)
            acceptor_ = new tcp::acceptor(io_service);

        // initialize network
        std::size_t tried = 0;
        exception_list errors;
        naming::locality::iterator_type end = accept_end(here_);
        for (naming::locality::iterator_type it = accept_begin(here_, io_service);
             it != end; ++it, ++tried)
        {
            try {
                server::tcp::parcelport_connection_ptr conn(
                    new server::tcp::parcelport_connection(io_service, *this));

                tcp::endpoint ep = *it;
                acceptor_->open(ep.protocol());
                acceptor_->set_option(tcp::acceptor::reuse_address(true));
                acceptor_->bind(ep);
                acceptor_->listen();
                acceptor_->async_accept(conn->socket(),
                    boost::bind(&parcelport::handle_accept,
                        this->shared_from_this(),
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
                "tcp::parcelport::run", errors.get_message());
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
            connection_cache_.shutdown();

            {
                // cancel all pending read operations, close those sockets
                lcos::local::spinlock::scoped_lock l(connections_mtx_);
                BOOST_FOREACH(server::tcp::parcelport_connection_ptr c,
                    accepted_connections_)
                {
                    boost::system::error_code ec;
                    boost::asio::ip::tcp::socket& s = c->socket();
                    if (s.is_open()) {
                        s.shutdown(boost::asio::ip::tcp::socket::shutdown_both, ec);
                        s.close(ec);    // close the socket to give it back to the OS
                    }
                }

                accepted_connections_.clear();
#if defined(HPX_HOLDON_TO_OUTGOING_CONNECTIONS)
                write_connections_.clear();
#endif
            }

            connection_cache_.clear();

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
        server::tcp::parcelport_connection_ptr conn)
    {
        if (!e) {
            // handle this incoming parcel
            server::tcp::parcelport_connection_ptr c(conn);    // hold on to conn

            // create new connection waiting for next incoming parcel
            conn.reset(new server::tcp::parcelport_connection(
                io_service_pool_.get_io_service(), *this));

            acceptor_->async_accept(conn->socket(),
                boost::bind(&parcelport::handle_accept,
                    this->shared_from_this(),
                    boost::asio::placeholders::error, conn));

            {
                // keep track of all the accepted connections
                lcos::local::spinlock::scoped_lock l(connections_mtx_);
                accepted_connections_.insert(c);
            }

            // disable Nagle algorithm, disable lingering on close
            boost::asio::ip::tcp::socket& s = c->socket();
            s.set_option(boost::asio::ip::tcp::no_delay(true));
            s.set_option(boost::asio::socket_base::linger(true, 0));

            // now accept the incoming connection by starting to read from the
            // socket
            c->async_read(
                boost::bind(&parcelport::handle_read_completion,
                    this->shared_from_this(),
                    boost::asio::placeholders::error, c));
        }
        else {
            // remove this connection from the list of known connections
            lcos::local::spinlock::scoped_lock l(mtx_);
            accepted_connections_.erase(conn);
        }
    }

    // Handle completion of a read operation.
    void parcelport::handle_read_completion(boost::system::error_code const& e,
        server::tcp::parcelport_connection_ptr c)
    {
        if (!e) return;

        if (e != boost::asio::error::operation_aborted &&
            e != boost::asio::error::eof)
        {
            LPT_(error)
                << "handle read operation completion: error: "
                << e.message();

            // remove this connection from the list of known connections
            lcos::local::spinlock::scoped_lock l(connections_mtx_);
            accepted_connections_.erase(c);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    void parcelport::put_parcels(std::vector<parcel> const & parcels,
        std::vector<write_handler_type> const& handlers)
    {
        typedef pending_parcels_map::mapped_type mapped_type;

        if (parcels.size() != handlers.size())
        {
            HPX_THROW_EXCEPTION(bad_parameter, "parcelport::put_parcels",
                "mismatched number of parcels and handlers");
            return;
        }

        naming::locality locality_id = parcels[0].get_destination_locality();
        naming::gid_type parcel_id = parcels[0].get_parcel_id();

#if defined(HPX_DEBUG)
        // make sure all parcels go to the same locality
        for (std::size_t i = 1; i != parcels.size(); ++i)
        {
            BOOST_ASSERT(locality_id == parcels[i].get_destination_locality());
        }
#endif

        // enqueue the outgoing parcels ...
        {
            lcos::local::spinlock::scoped_lock l(mtx_);

            mapped_type& e = pending_parcels_[locality_id];
            for (std::size_t i = 0; i != parcels.size(); ++i)
            {
                e.first.push_back(parcels[i]);
                e.second.push_back(handlers[i]);
            }
        }

        get_connection_and_send_parcels(locality_id, parcel_id);
    }

    void parcelport::put_parcel(parcel const& p, write_handler_type f)
    {
        typedef pending_parcels_map::mapped_type mapped_type;

        naming::locality locality_id = p.get_destination_locality();
        naming::gid_type parcel_id = p.get_parcel_id();

        // enqueue the outgoing parcel ...
        {
            lcos::local::spinlock::scoped_lock l(mtx_);

            mapped_type& e = pending_parcels_[locality_id];
            e.first.push_back(p);
            e.second.push_back(f);
        }

        get_connection_and_send_parcels(locality_id, parcel_id);
    }

    ///////////////////////////////////////////////////////////////////////////
    void parcelport::get_connection_and_send_parcels(
        naming::locality const& locality_id, naming::gid_type const& parcel_id)
    {
        error_code ec;
        parcelport_connection_ptr client_connection =
            get_connection(locality_id, ec);

        if (!client_connection)
        {
            if (ec)
                report_potential_connection_error(locality_id, parcel_id, ec);

            // We can safely return if no connection is available at this point.
            // As soon as a connection becomes available it checks for pending
            // parcels and sends those out.
            return;
        }

        send_parcels_or_reclaim_connection(locality_id, client_connection);
    }

    // This function is scheduled in an HPX thread by send_pending_parcels_trampoline
    // below if more parcels are waiting to be sent.
    void parcelport::retry_sending_parcels(naming::locality const& locality_id)
    {
        naming::gid_type parcel_id;

        // do nothing if parcels have already been picked up by another thread
        {
            lcos::local::spinlock::scoped_lock l(mtx_);
            pending_parcels_map::iterator it = pending_parcels_.find(locality_id);
            if (it == pending_parcels_.end() || it->second.first.empty())
                return;

            parcel_id = it->second.first.front().get_parcel_id();
        }

        get_connection_and_send_parcels(locality_id, parcel_id);
    }

    ///////////////////////////////////////////////////////////////////////////
    void parcelport::send_parcels_or_reclaim_connection(
        naming::locality const& locality_id,
        parcelport_connection_ptr const& client_connection)
    {
        typedef pending_parcels_map::iterator iterator;

        std::vector<parcel> parcels;
        std::vector<write_handler_type> handlers;

        {
            lcos::local::spinlock::scoped_lock l(mtx_);
            iterator it = pending_parcels_.find(locality_id);

            if (it != pending_parcels_.end() && !it->second.first.empty())
            {
                BOOST_ASSERT(it->first == locality_id);
                std::swap(parcels, it->second.first);
                std::swap(handlers, it->second.second);

                if (parcels.empty())
                {
                    // if no parcels are pending re-add the connection to
                    // the cache
                    BOOST_ASSERT(handlers.empty());
                    BOOST_ASSERT(locality_id == client_connection->destination());
                    connection_cache_.reclaim(locality_id, client_connection);
                    return;
                }
            }
            else
            {
                // Give this connection back to the cache as it's not
                // needed anymore.
                BOOST_ASSERT(locality_id == client_connection->destination());
                connection_cache_.reclaim(locality_id, client_connection);
                return;
            }
        }

        // send parcels if they didn't get sent by another connection
        send_pending_parcels(client_connection, parcels, handlers);
    }

    void parcelport::send_pending_parcels_trampoline(
        boost::system::error_code const& ec,
        naming::locality const& locality_id,
        parcelport_connection_ptr client_connection)
    {
#if defined(HPX_TRACK_STATE_OF_OUTGOING_TCP_CONNECTION)
        client_connection->set_state(parcelport_connection::state_scheduled_thread);
#endif
        {
            lcos::local::spinlock::scoped_lock l(mtx_);

            // Give this connection back to the cache as it's not  needed anymore.
            BOOST_ASSERT(locality_id == client_connection->destination());
            connection_cache_.reclaim(locality_id, client_connection);

            pending_parcels_map::iterator it = pending_parcels_.find(locality_id);
            if (it == pending_parcels_.end() || it->second.first.empty())
                return;
        }

        // Create a new HPX thread which sends parcels that are still pending.
        hpx::applier::register_thread_nullary(
            HPX_STD_BIND(&parcelport::retry_sending_parcels,
                this->shared_from_this(), locality_id), "retry_sending_parcels",
                threads::pending, true, threads::thread_priority_critical);
    }

    void parcelport::send_pending_parcels(
        parcelport_connection_ptr client_connection,
        std::vector<parcel> const & parcels,
        std::vector<write_handler_type> const & handlers)
    {
#if defined(HPX_TRACK_STATE_OF_OUTGOING_TCP_CONNECTION)
        client_connection->set_state(parcelport_connection::state_send_pending);
#endif

#if defined(HPX_DEBUG)
        // verify the connection points to the right destination
        boost::asio::ip::tcp::socket& s = client_connection->socket();
        BOOST_FOREACH(parcel const& p, parcels)
        {
            naming::locality const parcel_locality_id = p.get_destination_locality();
            BOOST_ASSERT(parcel_locality_id == client_connection->destination());
            BOOST_ASSERT(parcel_locality_id.get_address() ==
                s.remote_endpoint().address().to_string());
            BOOST_ASSERT(parcel_locality_id.get_port() ==
                s.remote_endpoint().port());
        }
#endif
        // store parcels in connection
        // The parcel gets serialized inside set_parcel, no
        // need to keep the original parcel alive after this call returned.
        client_connection->set_parcel(parcels);

        // ... start an asynchronous write operation now.
        client_connection->async_write(
            hpx::parcelset::detail::call_for_each(handlers),
            boost::bind(&parcelport::send_pending_parcels_trampoline,
                this->shared_from_this(),
                boost::asio::placeholders::error, ::_2, ::_3));
    }

    parcelport_connection_ptr parcelport::get_connection(
        naming::locality const& l, error_code& ec)
    {
        parcelport_connection_ptr client_connection;

        // Get a connection or reserve space for a new connection.
        if (!connection_cache_.get_or_reserve(l, client_connection))
        {
            // if no slot is available it's not a problem as the parcel will
            // sent out whenever the next connection is returned to the cache
            if (&ec != &throws)
                ec = make_success_code();
            return client_connection;
        }

        // Check if we need to create the new connection.
        if (!client_connection)
            return create_connection(l, ec);

        if (&ec != &throws)
            ec = make_success_code();

        return client_connection;
    }

    parcelport_connection_ptr parcelport::create_connection(
        naming::locality const& l, error_code& ec)
    {
        boost::asio::io_service& io_service = io_service_pool_.get_io_service();

        // The parcel gets serialized inside the connection constructor, no
        // need to keep the original parcel alive after this call returned.
        parcelport_connection_ptr client_connection(new parcelport_connection(
            io_service, l, parcels_sent_, this->get_max_message_size()));

        // Connect to the target locality, retry if needed
        boost::system::error_code error = boost::asio::error::try_again;
        for (std::size_t i = 0; i < HPX_MAX_NETWORK_RETRIES; ++i)
        {
            try {
                naming::locality::iterator_type end = connect_end(l);
                for (naming::locality::iterator_type it =
                        connect_begin(l, io_service);
                      it != end; ++it)
                {
                    boost::asio::ip::tcp::socket& s = client_connection->socket();
                    s.close();
                    s.connect(*it, error);
                    if (!error)
                        break;
                }
                if (!error)
                    break;

                // wait for a really short amount of time
                if (hpx::threads::get_self_ptr()) {
                    this_thread::suspend();
                }
                else {
                    boost::this_thread::sleep(boost::get_system_time() +
                        boost::posix_time::milliseconds(
                            HPX_NETWORK_RETRIES_SLEEP));
                }
            }
            catch (boost::system::system_error const& e) {
                client_connection->socket().close();
                client_connection.reset();

                HPX_THROWS_IF(ec, network_error,
                    "tcp::parcelport::get_connection", e.what());
                return client_connection;
            }
        }

        if (error) {
            client_connection->socket().close();
            client_connection.reset();

            hpx::util::osstream strm;
            strm << error.message() << " (while trying to connect to: "
                  << l << ")";

            HPX_THROWS_IF(ec, network_error,
                "tcp::parcelport::get_connection",
                hpx::util::osstream_get_string(strm));
            return client_connection;
        }

        // make sure the Nagle algorithm is disabled for this socket,
        // disable lingering on close
        client_connection->socket().set_option(
            boost::asio::ip::tcp::no_delay(true));
        client_connection->socket().set_option(
            boost::asio::socket_base::linger(true, 0));

#if defined(HPX_HOLDON_TO_OUTGOING_CONNECTIONS)
        {
            lcos::local::spinlock::scoped_lock lock(connections_mtx_);
            write_connections_.insert(client_connection);
        }
#endif
#if defined(HPX_DEBUG)
        BOOST_ASSERT(l == client_connection->destination());

        std::string connection_addr = client_connection->socket().remote_endpoint().address().to_string();
        boost::uint16_t connection_port = client_connection->socket().remote_endpoint().port();
        BOOST_ASSERT(l.get_address() == connection_addr);
        BOOST_ASSERT(l.get_port() == connection_port);
#endif

        if (&ec != &throws)
            ec = make_success_code();

        return client_connection;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Return the given connection cache statistic
    boost::int64_t parcelport::get_connection_cache_statistics(
        connection_cache_statistics_type t, bool reset)
    {
        switch (t) {
        case connection_cache_insertions:
            return connection_cache_.get_cache_insertions(reset);

        case connection_cache_evictions:
            return connection_cache_.get_cache_evictions(reset);

        case connection_cache_hits:
            return connection_cache_.get_cache_hits(reset);

        case connection_cache_misses:
            return connection_cache_.get_cache_misses(reset);

        case connection_cache_reclaims:
            return connection_cache_.get_cache_reclaims(reset);

        default:
            break;
        }

        HPX_THROW_EXCEPTION(bad_parameter,
            "tcp::parcelport::get_connection_cache_statistics",
            "invalid connection cache statistics type");
        return 0;
    }

    ///////////////////////////////////////////////////////////////////////////
#if defined(HPX_HAVE_SECURITY)
    // read the certificate, if available, and add it to the local certificate
    // store
    template <typename Archive>
    bool deserialize_certificate(Archive& archive, bool first_message)
    {
        bool has_certificate = false;
        archive >> has_certificate;

        components::security::signed_certificate certificate;
        if (has_certificate) {
            archive >> certificate;
            add_locality_certificate(certificate);
            if (first_message)
                first_message = false;
        }
        return first_message;
    }

    // calculate and verify the hash
    void verify_message_suffix(
        std::vector<char> const& parcel_data,
        performance_counters::parcels::data_point& receive_data,
        naming::gid_type& parcel_id)
    {
        // mark start of security work
        util::high_resolution_timer timer_sec;

        if (!verify_parcel_suffix(parcel_data, parcel_id)) {
            // all hell breaks loose!
            HPX_THROW_EXCEPTION(security_error,
                "decode_message(tcp)",
                "verify_message_suffix failed");
            return;
        }

        // store the time required for security
        receive_data.security_time_ = timer_sec.elapsed_nanoseconds();
    }
#endif

    bool decode_message(parcelport& pp,
        boost::shared_ptr<std::vector<char> > parcel_data,
        boost::uint64_t inbound_data_size,
        performance_counters::parcels::data_point receive_data,
        bool first_message)
    {
        unsigned archive_flags = boost::archive::no_header;
        if (!pp.allow_array_optimizations())
            archive_flags |= util::disable_array_optimization;

        // protect from un-handled exceptions bubbling up
        try {
            try {
                // mark start of serialization
                util::high_resolution_timer timer;
                boost::int64_t overall_add_parcel_time = 0;

                {
                    // De-serialize the parcel data
                    util::portable_binary_iarchive archive(*parcel_data,
                        inbound_data_size, archive_flags);

                    std::size_t parcel_count = 0;
                    archive >> parcel_count;
                    for(std::size_t i = 0; i != parcel_count; ++i)
                    {
#if defined(HPX_HAVE_SECURITY)
                        // handle certificate and verify parcel suffix once
                        naming::gid_type parcel_id;
                        first_message = deserialize_certificate(archive, first_message);
                        if (!first_message && i == 0)
                            verify_message_suffix(*parcel_data, receive_data, parcel_id);

                        // de-serialize parcel and add it to incoming parcel queue
                        parcel p;
                        archive >> p;

                        // verify parcel id, but only once while handling the first parcel
                        if (!first_message && i == 0 && parcel_id != p.get_parcel_id()) {
                            // again, all hell breaks loose
                            HPX_THROW_EXCEPTION(security_error,
                                "decode_message(tcp)",
                                "parcel id mismatch");
                            return false;
                        }
#else
                        // de-serialize parcel and add it to incoming parcel queue
                        parcel p;
                        archive >> p;
#endif
                        // make sure this parcel ended up on the right locality
                        BOOST_ASSERT(p.get_destination_locality() == pp.here());

                        // be sure not to measure add_parcel as serialization time
                        boost::int64_t add_parcel_time = timer.elapsed_nanoseconds();
                        pp.add_received_parcel(p);
                        overall_add_parcel_time += timer.elapsed_nanoseconds() -
                            add_parcel_time;
                    }

                    // complete received data with parcel count
                    receive_data.num_parcels_ = parcel_count;
                    receive_data.raw_bytes_ = archive.bytes_read();
                }

                // store the time required for serialization
                receive_data.serialization_time_ = timer.elapsed_nanoseconds() -
                    overall_add_parcel_time;

                pp.add_received_data(receive_data);
            }
            catch (hpx::exception const& e) {
                LPT_(error)
                    << "decode_message(tcp): caught hpx::exception: "
                    << e.what();
                hpx::report_error(boost::current_exception());
            }
            catch (boost::system::system_error const& e) {
                LPT_(error)
                    << "decode_message(tcp): caught boost::system::error: "
                    << e.what();
                hpx::report_error(boost::current_exception());
            }
            catch (boost::exception const&) {
                LPT_(error)
                    << "decode_message(tcp): caught boost::exception.";
                hpx::report_error(boost::current_exception());
            }
            catch (std::exception const& e) {
                // We have to repackage all exceptions thrown by the
                // serialization library as otherwise we will loose the
                // e.what() description of the problem, due to slicing.
                boost::throw_exception(boost::enable_error_info(
                    hpx::exception(serialization_error, e.what())));
            }
        }
        catch (...) {
            LPT_(error)
                << "decode_message(tcp): caught unknown exception.";
            hpx::report_error(boost::current_exception());
        }

        return first_message;
    }

    ///////////////////////////////////////////////////////////////////////////
    boost::uint64_t get_max_inbound_size(parcelport& pp)
    {
        return pp.get_max_message_size();
    }

    ///////////////////////////////////////////////////////////////////////////
    // the code below is needed to bootstrap the parcel layer
    void early_write_handler(boost::system::error_code const& ec,
        std::size_t size)
    {
        if (ec) {
            // all errors during early parcel handling are fatal
            try {
                HPX_THROW_EXCEPTION(network_error, "early_write_handler",
                    "error while handling early parcel: " +
                        ec.message() + "(" +
                        boost::lexical_cast<std::string>(ec.value())+ ")");
            }
            catch (hpx::exception const& e) {
                hpx::detail::report_exception_and_terminate(e);
            }
            return;
        }
    }

    void early_pending_parcel_handler(boost::system::error_code const& ec,
        naming::locality const&, parcelport_connection_ptr const&)
    {
        if (ec) {
            // all errors during early parcel handling are fatal
            try {
                HPX_THROW_EXCEPTION(network_error, "early_write_handler",
                    "error while handling early parcel: " +
                        ec.message() + "(" +
                        boost::lexical_cast<std::string>(ec.value())+ ")");
            }
            catch (hpx::exception const& e) {
                hpx::detail::report_exception_and_terminate(e);
            }
            return;
        }
    }

    void parcelport::send_early_parcel(parcel& p)
    {
        naming::locality const& l = p.get_destination_locality();
        error_code ec;
        parcelport_connection_ptr client_connection = get_connection_wait(l, ec);

        if (ec) {
            // all errors during early parcel handling are fatal
            hpx::detail::report_exception_and_terminate(
                hpx::detail::access_exception(ec));
            return;
        }

        BOOST_ASSERT(client_connection );
        client_connection->set_parcel(p);
        client_connection->async_write(early_write_handler,
            early_pending_parcel_handler);
    }

    parcelport_connection_ptr parcelport::get_connection_wait(
        naming::locality const& l, error_code& ec)
    {
        parcelport_connection_ptr client_connection;
        bool got_cache_space = false;

        for (std::size_t i = 0; i < HPX_MAX_NETWORK_RETRIES; ++i)
        {
            // Get a connection or reserve space for a new connection.
            if (connection_cache_.get_or_reserve(l, client_connection))
            {
                got_cache_space = true;
                break;
            }

            // Wait for a really short amount of time.
            boost::this_thread::sleep(boost::get_system_time() +
                boost::posix_time::milliseconds(HPX_NETWORK_RETRIES_SLEEP));
        }

        // If we didn't get a connection or permission to create one (which is
        // unlikely), bail.
        if (!got_cache_space)
        {
            HPX_THROWS_IF(ec, invalid_status, "parcelport::get_connection_wait",
                "didn't get a connection slot from connection cache, bailing out");
            return client_connection;
        }

        // Check if we need to create the new connection.
        if (!client_connection)
            return create_connection(l, ec);

        if (&ec != &throws)
            ec = make_success_code();

        return client_connection;
    }
}}}
