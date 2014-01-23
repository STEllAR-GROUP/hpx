//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2007 Richard D Guidry Jr
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2011 Katelyn Kufahl
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_PARCELPORT_IMPL_HPP
#define HPX_PARCELSET_PARCELPORT_IMPL_HPP

#include <hpx/runtime/parcelset/parcelport.hpp>
#include <hpx/runtime/parcelset/encode_parcels.hpp>
#include <hpx/runtime/parcelset/detail/call_for_each.hpp>
#include <hpx/util/io_service_pool.hpp>
#include <hpx/util/connection_cache.hpp>
#include <hpx/util/runtime_configuration.hpp>

#include <boost/asio/placeholders.hpp>

namespace hpx { namespace parcelset
{

    template <typename ConnectionHandler>
    struct connection_handler_traits;

    template <typename ConnectionHandler>
    class HPX_EXPORT parcelport_impl
      : public parcelport
    {
        typedef
            typename connection_handler_traits<ConnectionHandler>::connection_type
            connection;
    
        static const char * connection_handler_name()
        {
            return connection_handler_traits<ConnectionHandler>::name();
        }


    public:
        /// Construct the parcelport on the given locality.
        parcelport_impl(util::runtime_configuration const& ini,
            HPX_STD_FUNCTION<void(std::size_t, char const*)> const& on_start_thread,
            HPX_STD_FUNCTION<void()> const& on_stop_thread)
          : parcelport(ini, connection_handler_name())

            // FIXME: add connection handler specific configuration
          , io_service_pool_(ini.get_thread_pool_size("parcel_pool"),
                on_start_thread, on_stop_thread, "parcel_pool_tcp", "-tcp")
          , connection_cache_(ini.get_max_connections(), ini.get_max_connections_per_loc())
          , archive_flags_(boost::archive::no_header)
        {
            // FIXME: adapt for other parcelport types ...
#ifdef BOOST_BIG_ENDIAN
            std::string endian_out = get_config_entry("hpx.parcel.endian_out", "big");
#else
            std::string endian_out = get_config_entry("hpx.parcel.endian_out", "little");
#endif
            if (endian_out == "little")
                archive_flags_ |= util::endian_little;
            else if (endian_out == "big")
                archive_flags_ |= util::endian_big;
            else {
                HPX_ASSERT(endian_out =="little" || endian_out == "big");
            }

            std::string array_optimization =
                get_config_entry("hpx.parcel.tcpip.array_optimization", "1");

            if (boost::lexical_cast<int>(array_optimization) == 0) {
                archive_flags_ |= util::disable_array_optimization;
                archive_flags_ |= util::disable_data_chunking;
            }
            else {
                std::string zero_copy_optimization =
                    get_config_entry("hpx.parcel.tcpip.zero_copy_optimization", "1");
                if (boost::lexical_cast<int>(zero_copy_optimization) == 0)
                    archive_flags_ |= util::disable_data_chunking;
            }
        }

        ~parcelport_impl()
        {
            connection_cache_.clear();
        }
        
        bool run(bool blocking = true)
        {
            io_service_pool_.run(false);    // start pool

            bool success = connection_handler().run();
            
            if (blocking)
                io_service_pool_.join();

            return success;
        }
        
        void stop(bool blocking = true)
        {
            // make sure no more work is pending, wait for service pool to get empty
            io_service_pool_.stop();
            if (blocking) {
                connection_cache_.shutdown();

                io_service_pool_.join();
                connection_handler().stop();
                connection_cache_.clear();
                io_service_pool_.clear();
            }
        }
        
        void put_parcel(parcel const& p, write_handler_type const& f)
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
        
        void put_parcels(std::vector<parcel> const & parcels,
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
                HPX_ASSERT(locality_id == parcels[i].get_destination_locality());
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
        
        void send_early_parcel(parcel& p)
        {
            send_early_parcel_impl<ConnectionHandler>(p);
        }
        
        util::io_service_pool* get_thread_pool(char const* name)
        {
            if (0 == std::strcmp(name, io_service_pool_.get_name()))
                return &io_service_pool_;
            return 0;
        }

        void do_background_work()
        {
            do_background_work_impl<ConnectionHandler>();
        }

        /// support enable_shared_from_this
        boost::shared_ptr<parcelport_impl> shared_from_this()
        {
            return boost::static_pointer_cast<parcelport_impl>(
                parcelset::parcelport::shared_from_this());
        }

        boost::shared_ptr<parcelport_impl const> shared_from_this() const
        {
            return boost::static_pointer_cast<parcelport_impl const>(
                parcelset::parcelport::shared_from_this());
        }
        
        virtual std::string get_locality_name() const
        {
            return connection_handler().get_locality_name();
        }
        
        /// Cache specific functionality
        void remove_from_connection_cache(naming::locality const& loc)
        {
            connection_cache_.clear(loc);
        }
    
        /////////////////////////////////////////////////////////////////////////
        // Return the given connection cache statistic
        boost::int64_t get_connection_cache_statistics(
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
            
            // FIXME: add correct pp name
            HPX_THROW_EXCEPTION(bad_parameter,
                "tcp::parcelport::get_connection_cache_statistics",
                "invalid connection cache statistics type");
            return 0;
        }

    private:
        ConnectionHandler & connection_handler()
        {
            return static_cast<ConnectionHandler &>(*this);
        }
        
        ConnectionHandler const & connection_handler() const
        {
            return static_cast<ConnectionHandler const &>(*this);
        }

        ///////////////////////////////////////////////////////////////////////////
        // the code below is needed to bootstrap the parcel layer
        static void early_write_handler(boost::system::error_code const& ec,
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

        static void early_pending_parcel_handler(boost::system::error_code const& ec,
            naming::locality const&, boost::shared_ptr<connection> const&)
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

        template <typename ConnectionHandler_>
        typename boost::enable_if<
            typename connection_handler_traits<
                ConnectionHandler_
            >::send_early_parcel
        >::type
        send_early_parcel_impl(parcel& p)
        {
            naming::locality const& l = p.get_destination_locality();
            error_code ec;
            boost::shared_ptr<connection> sender_connection
                = get_connection_wait(l, ec);

            if (ec) {
                // all errors during early parcel handling are fatal
                hpx::detail::report_exception_and_terminate(
                    hpx::detail::access_exception(ec));
                return;
            }

            HPX_ASSERT(sender_connection.get() != 0);
            boost::shared_ptr<parcel_buffer<typename connection::buffer_type> >
                buffer = encode_parcels(p, *sender_connection, archive_flags_);

            sender_connection->async_write(
                buffer
              , early_write_handler
              , early_pending_parcel_handler);
        }

        template <typename ConnectionHandler_>
        typename boost::disable_if<
            typename connection_handler_traits<
                ConnectionHandler_
            >::send_early_parcel
        >::type
        send_early_parcel_impl(parcel& p)
        {
            HPX_THROW_EXCEPTION(network_error, "send_early_parcel",
                "This parcelport does not support sending early parcels");
        }

        template <typename ConnectionHandler_>
        typename boost::enable_if<
            typename connection_handler_traits<
                ConnectionHandler_
            >::do_background_work
        >::type
        do_background_work_impl()
        {
            connection_handler().do_background_work();
        }

        template <typename ConnectionHandler_>
        typename boost::disable_if<
            typename connection_handler_traits<
                ConnectionHandler_
            >::do_background_work
        >::type
        do_background_work_impl()
        {}

        void get_connection_and_send_parcels(
            naming::locality const& locality_id, naming::gid_type const& parcel_id)
        {
            error_code ec;
            boost::shared_ptr<connection> sender_connection
                = get_connection(locality_id, ec);
            
            if (!sender_connection)
            {
                if (ec)
                    report_potential_connection_error(locality_id, parcel_id, ec);

                // We can safely return if no connection is available at this point.
                // As soon as a connection becomes available it checks for pending
                // parcels and sends those out.
                return;
            }

            send_parcels_or_reclaim_connection(locality_id, sender_connection);
        }

        boost::shared_ptr<connection> get_connection(
            naming::locality const& l, error_code& ec)
        {
            boost::shared_ptr<connection> sender_connection;

            // Get a connection or reserve space for a new connection.
            if (!connection_cache_.get_or_reserve(l, sender_connection))
            {
                // if no slot is available it's not a problem as the parcel will
                // sent out whenever the next connection is returned to the cache
                if (&ec != &throws)
                    ec = make_success_code();
                return sender_connection;
            }

            // Check if we need to create the new connection.
            if (!sender_connection)
                return connection_handler().create_connection(l, ec);

            if (&ec != &throws)
                ec = make_success_code();

            return sender_connection;
        }

        boost::shared_ptr<connection> get_connection_wait(
            naming::locality const& l, error_code& ec)
        {
            boost::shared_ptr<connection> sender_connection;
            bool got_cache_space = false;

            for (std::size_t i = 0; i < HPX_MAX_NETWORK_RETRIES; ++i)
            {
                // Get a connection or reserve space for a new connection.
                if (connection_cache_.get_or_reserve(l, sender_connection))
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
                return sender_connection;
            }

            // Check if we need to create the new connection.
            if (!sender_connection)
                return connection_handler().create_connection(l, ec);

            if (&ec != &throws)
                ec = make_success_code();

            return sender_connection;
        }
    
        void retry_sending_parcels(naming::locality const& locality_id)
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

        void send_parcels_or_reclaim_connection(
            naming::locality const& locality_id,
            boost::shared_ptr<connection> const& sender_connection)
        {
            typedef pending_parcels_map::iterator iterator;

            std::vector<parcel> parcels;
            std::vector<write_handler_type> handlers;

            {
                lcos::local::spinlock::scoped_lock l(mtx_);
                iterator it = pending_parcels_.find(locality_id);

                if (it != pending_parcels_.end() && !it->second.first.empty())
                {
                    HPX_ASSERT(it->first == locality_id);
                    std::swap(parcels, it->second.first);
                    std::swap(handlers, it->second.second);

                    if (parcels.empty())
                    {
                        // if no parcels are pending re-add the connection to
                        // the cache
                        HPX_ASSERT(handlers.empty());
                        HPX_ASSERT(locality_id == sender_connection->destination());
                        connection_cache_.reclaim(locality_id, sender_connection);
                        return;
                    }
                }
                else
                {
                    // Give this connection back to the cache as it's not
                    // needed anymore.
                    HPX_ASSERT(locality_id == sender_connection->destination());
                    connection_cache_.reclaim(locality_id, sender_connection);
                    return;
                }
            }

            // send parcels if they didn't get sent by another connection
            send_pending_parcels(sender_connection, parcels, handlers);
        }
    
        void send_pending_parcels_trampoline(
            boost::system::error_code const& ec,
            naming::locality const& locality_id,
            boost::shared_ptr<connection> sender_connection)
        {
#if defined(HPX_TRACK_STATE_OF_OUTGOING_TCP_CONNECTION)
            client_connection->set_state(parcelport_connection::state_scheduled_thread);
#endif
            {
                lcos::local::spinlock::scoped_lock l(mtx_);

                // Give this connection back to the cache as it's not  needed anymore.
                HPX_ASSERT(locality_id == sender_connection->destination());
                connection_cache_.reclaim(locality_id, sender_connection);

                pending_parcels_map::iterator it = pending_parcels_.find(locality_id);
                if (it == pending_parcels_.end() || it->second.first.empty())
                    return;
            }

            // Create a new HPX thread which sends parcels that are still pending.
            hpx::applier::register_thread_nullary(
                HPX_STD_BIND(&parcelport_impl::retry_sending_parcels,
                    this, locality_id), "retry_sending_parcels",
                    threads::pending, true, threads::thread_priority_critical);
        }

        void send_pending_parcels(
            boost::shared_ptr<connection> sender_connection,
            std::vector<parcel> const & parcels,
            std::vector<write_handler_type> const & handlers)
        {
#if defined(HPX_TRACK_STATE_OF_OUTGOING_TCP_CONNECTION)
            sender_connection->set_state(parcelport_connection::state_send_pending);
#endif

#if defined(HPX_DEBUG)
            // verify the connection points to the right destination
            BOOST_FOREACH(parcel const& p, parcels)
            {
                naming::locality const parcel_locality_id = p.get_destination_locality();
                HPX_ASSERT(parcel_locality_id == sender_connection->destination());
                sender_connection->verify(parcel_locality_id);
            }
#endif
            // encode the parcels
            boost::shared_ptr<parcel_buffer<typename connection::buffer_type> >
                buffer = encode_parcels(parcels, *sender_connection, archive_flags_);
            
            // send them asynchronously
            sender_connection->async_write(
                buffer,
                hpx::parcelset::detail::call_for_each(handlers),
                boost::bind(&parcelport_impl::send_pending_parcels_trampoline,
                    this,
                    ::_1, ::_2, ::_3));
        }

    protected:
        /// The pool of io_service objects used to perform asynchronous operations.
        util::io_service_pool io_service_pool_;

        /// The connection cache for sending connections
        util::connection_cache<connection, naming::locality> connection_cache_;

        int archive_flags_;
    };
}}

#endif
