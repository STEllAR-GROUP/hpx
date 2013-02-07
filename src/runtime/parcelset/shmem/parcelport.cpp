//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>

#if defined(HPX_USE_SHMEM_PARCELPORT)
#include <hpx/exception_list.hpp>
#include <hpx/runtime/naming/locality.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/runtime/parcelset/shmem/parcelport.hpp>
#include <hpx/util/runtime_configuration.hpp>
#include <hpx/util/io_service_pool.hpp>
#include <hpx/util/stringstream.hpp>
#include <hpx/util/logging.hpp>

#include <boost/version.hpp>
#include <boost/bind.hpp>
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/asio/placeholders.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset { namespace shmem
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
    parcelport::parcelport(util::runtime_configuration const& ini,
            HPX_STD_FUNCTION<void(std::size_t, char const*)> const& on_start_thread,
            HPX_STD_FUNCTION<void()> const& on_stop_thread)
      : parcelset::parcelport(naming::locality(ini.get_parcelport_address())),
        io_service_pool_(ini.get_thread_pool_size("parcel_pool"),
            on_start_thread, on_stop_thread, "parcel_pool_shmem", "-shmem"),
        acceptor_(NULL), connection_count_(0),
        connection_cache_(ini.get_max_connections(), ini.get_max_connections_per_loc()),
        data_buffer_cache_(ini.get_max_connections() * 4)
    {
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

        if (NULL == acceptor_)
            acceptor_ = new acceptor(io_service_pool_.get_io_service(0));

        // initialize network
        std::size_t tried = 0;
        exception_list errors;
        naming::locality::iterator_type end = accept_end(here_);
        for (naming::locality::iterator_type it =
                accept_begin(here_, io_service_pool_.get_io_service(0));
             it != end; ++it, ++tried)
        {
            try {
                server::shmem::parcelport_connection_ptr conn(
                    new server::shmem::parcelport_connection(
                        io_service_pool_.get_io_service(1), here(), *this));

                boost::asio::ip::tcp::endpoint ep = *it;

                std::string fullname(ep.address().to_string() + "." +
                    boost::lexical_cast<std::string>(ep.port()));

                acceptor_->set_option(acceptor::msg_num(10));
                acceptor_->set_option(acceptor::manage(true));
                acceptor_->bind(fullname);
                acceptor_->open();

                acceptor_->async_accept(conn->window(),
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
                "shmem::parcelport::parcelport", errors.get_message());
            return false;
        }

        if (blocking)
            io_service_pool_.join();

        return true;
    }

    void parcelport::stop(bool blocking)
    {
        // now it's safe to take everything down
        connection_cache_.shutdown();

        {
            // cancel all pending read operations, close those sockets
            util::spinlock::scoped_lock l(mtx_);
            BOOST_FOREACH(server::shmem::parcelport_connection_ptr c,
                accepted_connections_)
            {
                boost::system::error_code ec;
                parcelset::shmem::data_window& w = c->window();
                w.shutdown(ec); // shut down connection
                w.close(ec);    // close the data window to give it back to the OS
            }
            accepted_connections_.clear();
        }

        connection_cache_.clear();
        data_buffer_cache_.clear();

        // cancel all pending accept operations
        if (NULL != acceptor_)
        {
            boost::system::error_code ec;
            acceptor_->close(ec);
            delete acceptor_;
            acceptor_ = NULL;
        }

        // make sure no more work is pending, wait for service pool to get empty
        io_service_pool_.stop();
        if (blocking) {
            io_service_pool_.join();
            io_service_pool_.clear();
        }
    }

    /// accepted new incoming connection
    void parcelport::handle_accept(boost::system::error_code const& e,
        server::shmem::parcelport_connection_ptr conn)
    {
        if (!e) {
            // handle this incoming parcel
            server::shmem::parcelport_connection_ptr c(conn);    // hold on to conn

            // create new connection waiting for next incoming parcel
            conn.reset(new server::shmem::parcelport_connection(
                io_service_pool_.get_io_service(1), here(), *this));

            acceptor_->async_accept(conn->window(),
                boost::bind(&parcelport::handle_accept, this,
                    boost::asio::placeholders::error, conn));

            {
                // keep track of all the accepted connections
                util::spinlock::scoped_lock l(mtx_);
                accepted_connections_.insert(c);
            }

            // now accept the incoming connection by starting to read from the
            // data window
            c->async_read(boost::bind(&parcelport::handle_read_completion,
                this, boost::asio::placeholders::error, c));
        }
        else {
            // remove this connection from the list of known connections
            util::spinlock::scoped_lock l(mtx_);
            accepted_connections_.erase(conn);
        }
    }

    /// Handle completion of a read operation.
    void parcelport::handle_read_completion(boost::system::error_code const& e,
        server::shmem::parcelport_connection_ptr c)
    {
        if (!e) return;

        if (e != boost::asio::error::operation_aborted &&
            e != boost::asio::error::eof)
        {
            LPT_(error)
                << "handle read operation completion: error: "
                << e.message();

            // remove this connection from the list of known connections
            util::spinlock::scoped_lock l(mtx_);
            accepted_connections_.erase(c);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    void parcelport::put_parcel(parcel const& p, write_handler_type f)
    {
        typedef pending_parcels_map::iterator iterator;
        typedef pending_parcels_map::mapped_type mapped_type;

        naming::locality locality_id = p.get_destination_locality();
        naming::gid_type parcel_id = p.get_parcel_id();

        // enqueue the incoming parcel ...
        {
            util::spinlock::scoped_lock l(mtx_);

            mapped_type& e = pending_parcels_[locality_id];
            e.first.push_back(p);
            e.second.push_back(f);
        }

        error_code ec;
        parcelport_connection_ptr client_connection =
            get_connection(locality_id, ec);

        if (!client_connection)
        {
            // If there was an error, we might be safe if there are no parcels
            // to be sent anymore (some other thread already picked them up)
            // or if there are parcels, but the parcel we were about to sent
            // has been already processed.
            util::spinlock::scoped_lock l(mtx_);

            iterator it = pending_parcels_.find(locality_id);
            if (it != pending_parcels_.end())
            {
                map_second_type& data = it->second;

                std::vector<parcel>::iterator end = data.first.end();
                std::vector<write_handler_type>::iterator fit = data.second.begin();
                for (std::vector<parcel>::iterator pit = data.first.begin();
                     pit != end; ++pit, ++fit)
                {
                    if ((*pit).get_parcel_id() == parcel_id)
                    {
                        // remove this parcel from pending parcel queue
                        data.first.erase(pit);
                        data.second.erase(fit);

                        // re-schedule this function call and bail out
                        threads::register_thread_nullary(
                            util::bind(&parcelport::put_parcel, this, p, f));
                    }
                }
            }
            return;
        }

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
            // ... or re-add the connection to the cache
            BOOST_ASSERT(locality_id == client_connection->destination());
            connection_cache_.reclaim(locality_id, client_connection);
        }
    }

    void parcelport::send_pending_parcels_trampoline(
        boost::system::error_code const& ec,
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

        if (!ec && !parcels.empty() && !handlers.empty())
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
        // store parcels in connection
        // The parcel gets serialized inside set_parcel, no
        // need to keep the original parcel alive after this call returned.
        client_connection->set_parcel(parcels);

        // ... start an asynchronous write operation now.
        client_connection->async_write(
            detail::call_for_each(handlers),
            boost::bind(&parcelport::send_pending_parcels_trampoline, this,
                boost::asio::placeholders::error, ::_2, ::_3));
    }

    ///////////////////////////////////////////////////////////////////////////
    parcelport_connection_ptr parcelport::get_connection(
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
            this_thread::suspend();
        }

        // If we didn't get a connection or permission to create one (which is
        // unlikely), bail.
        if (!got_cache_space)
        {
            HPX_THROWS_IF(ec, network_error,
                "shmem::parcelport::get_connection",
                "timed out while trying to find room in the connection cache");
            return client_connection;
        }

        // Check if we need to create the new connection.
        if (!client_connection)
        {
            // The parcel gets serialized inside the connection constructor, no
            // need to keep the original parcel alive after this call returned.
            client_connection.reset(new parcelport_connection(
                io_service_pool_.get_io_service(1), here_, l,
                data_buffer_cache_, parcels_sent_, ++connection_count_));

            // Connect to the target locality, retry if needed
            boost::system::error_code error = boost::asio::error::try_again;
            for (std::size_t i = 0; i < HPX_MAX_NETWORK_RETRIES; ++i)
            {
                try {
                    naming::locality::iterator_type end = connect_end(l);
                    for (naming::locality::iterator_type it =
                            connect_begin(l, io_service_pool_.get_io_service(0));
                         it != end; ++it)
                    {
                        boost::asio::ip::tcp::endpoint const& ep = *it;
                        std::string fullname(ep.address().to_string() + "." +
                            boost::lexical_cast<std::string>(ep.port()));

                        parcelset::shmem::data_window& w = client_connection->window();
                        w.close();
                        w.connect(fullname, error);
                        if (!error)
                            break;
                    }
                    if (!error)
                        break;

                    // wait for a really short amount of time (usually 100 ms)
                    this_thread::suspend(HPX_NETWORK_RETRIES_SLEEP);
                }
                catch (boost::system::system_error const& e) {
                    client_connection->window().close();
                    client_connection.reset();

                    HPX_THROWS_IF(ec, network_error,
                        "shmem::parcelport::get_connection", e.what());
                    return client_connection;
                }
            }

            if (error) {
                client_connection->window().close();
                client_connection.reset();

                hpx::util::osstream strm;
                strm << error.message() << " (while trying to connect to: "
                     << l << ")";
                HPX_THROWS_IF(ec, network_error,
                    "shmem::parcelport::get_connection",
                    hpx::util::osstream_get_string(strm));
                return client_connection;
            }
        }

        return client_connection;
    }

    ///////////////////////////////////////////////////////////////////////////
    void decode_message(parcelport& pp,
        parcelset::shmem::data_buffer parcel_data,
        performance_counters::parcels::data_point receive_data)
    {
        // protect from un-handled exceptions bubbling up
        try {
            try {
                // mark start of serialization
                util::high_resolution_timer timer;
                boost::int64_t overall_add_parcel_time = 0;

                {
                    // De-serialize the parcel data
                    util::portable_binary_iarchive archive(
                        parcel_data.get_buffer(), boost::archive::no_header);

                    std::size_t parcel_count = 0;
                    std::size_t arg_size = 0;

                    archive >> parcel_count;
                    for(std::size_t i = 0; i < parcel_count; ++i)
                    {
                        // de-serialize parcel and add it to incoming parcel queue
                        parcel p;
                        archive >> p;

                        // make sure this parcel ended up on the right locality
                        BOOST_ASSERT(p.get_destination_locality() == pp.here());

                        // incoming argument's size
                        arg_size += traits::get_type_size(p);

                        // be sure not to measure add_parcel as serialization time
                        boost::int64_t add_parcel_time = timer.elapsed_nanoseconds();
                        pp.add_received_parcel(p);
                        overall_add_parcel_time += timer.elapsed_nanoseconds() -
                            add_parcel_time;
                    }

                    // complete received data with parcel count
                    receive_data.num_parcels_ = parcel_count;
                    receive_data.type_bytes_ = arg_size;
                }

                // store the time required for serialization
                receive_data.serialization_time_ = timer.elapsed_nanoseconds() -
                    overall_add_parcel_time;

                pp.add_received_data(receive_data);
            }
            catch (hpx::exception const& e) {
                LPT_(error)
                    << "decode_message: caught hpx::exception: "
                    << e.what();
                hpx::report_error(boost::current_exception());
            }
            catch (boost::system::system_error const& e) {
                LPT_(error)
                    << "decode_message: caught boost::system::error: "
                    << e.what();
                hpx::report_error(boost::current_exception());
            }
            catch (boost::exception const&) {
                LPT_(error)
                    << "decode_message: caught boost::exception.";
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
                << "decode_message: caught unknown exception.";
            hpx::report_error(boost::current_exception());
        }
    }
}}}

#endif
