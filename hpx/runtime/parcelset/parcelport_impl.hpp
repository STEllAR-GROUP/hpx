//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2007-2014 Hartmut Kaiser
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
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/util/io_service_pool.hpp>
#include <hpx/util/connection_cache.hpp>
#include <hpx/util/runtime_configuration.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    bool is_starting();
}

namespace hpx { namespace parcelset
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename ConnectionHandler>
    struct connection_handler_traits;

    template <typename ConnectionHandler>
    class HPX_EXPORT parcelport_impl
      : public parcelport
    {
        typedef
            typename connection_handler_traits<ConnectionHandler>::connection_type
            connection;
    public:
        static const char * connection_handler_name()
        {
            return connection_handler_traits<ConnectionHandler>::name();
        }

        static std::size_t thread_pool_size(util::runtime_configuration const& ini)
        {
            std::string key("hpx.parcel.");
            key += connection_handler_name();

            std::string thread_pool_size =
                ini.get_entry(key + ".io_pool_size", "2");
            return boost::lexical_cast<std::size_t>(thread_pool_size);
        }

        static const char *pool_name()
        {
            return connection_handler_traits<ConnectionHandler>::pool_name();
        }

        static const char *pool_name_postfix()
        {
            return connection_handler_traits<ConnectionHandler>::pool_name_postfix();
        }

        static std::size_t max_connections(util::runtime_configuration const& ini)
        {
            std::string key("hpx.parcel.");
            key += connection_handler_name();

            std::string max_connections =
                ini.get_entry(key + ".max_connections",
                    HPX_PARCEL_MAX_CONNECTIONS);
            return boost::lexical_cast<std::size_t>(max_connections);
        }

        static std::size_t max_connections_per_loc(util::runtime_configuration const& ini)
        {
            std::string key("hpx.parcel.");
            key += connection_handler_name();

            std::string max_connections_per_locality =
                ini.get_entry(key + ".max_connections_per_locality",
                    HPX_PARCEL_MAX_CONNECTIONS_PER_LOCALITY);
            return boost::lexical_cast<std::size_t>(max_connections_per_locality);
        }

    public:
        /// Construct the parcelport on the given locality.
        parcelport_impl(util::runtime_configuration const& ini,
            HPX_STD_FUNCTION<void(std::size_t, char const*)> const& on_start_thread,
            HPX_STD_FUNCTION<void()> const& on_stop_thread)
          : parcelport(ini, connection_handler_name())
          , io_service_pool_(thread_pool_size(ini),
                on_start_thread, on_stop_thread, pool_name(), pool_name_postfix())
          , connection_cache_(max_connections(ini), max_connections_per_loc(ini))
          , archive_flags_(boost::archive::no_header)
        {
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

            if (!this->allow_array_optimizations()) {
                archive_flags_ |= util::disable_array_optimization;
                archive_flags_ |= util::disable_data_chunking;
            }
            else {
                if (!this->allow_zero_copy_optimizations())
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

            bool success = connection_handler().do_run();

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
                connection_handler().do_stop();
                io_service_pool_.join();
                connection_cache_.clear();
                io_service_pool_.clear();
            }
        }

        void put_parcel(parcel p, write_handler_type f)
        {
            naming::locality const& locality_id = p.get_destination_locality();

            // enqueue the outgoing parcel ...
            enqueue_parcel(locality_id, std::move(p), std::move(f));

            if (enable_parcel_handling_)
            {
                if (hpx::is_running() && async_serialization())
                {
                    trigger_sending_parcels(locality_id);
                }
                else
                {
                    get_connection_and_send_parcels(locality_id);
                }
            }
        }

        void put_parcels(std::vector<parcel> parcels,
            std::vector<write_handler_type> handlers)
        {
            if (parcels.size() != handlers.size())
            {
                HPX_THROW_EXCEPTION(bad_parameter, "parcelport::put_parcels",
                    "mismatched number of parcels and handlers");
                return;
            }

            naming::locality const& locality_id =
                parcels[0].get_destination_locality();

#if defined(HPX_DEBUG)
            // make sure all parcels go to the same locality
            for (std::size_t i = 1; i != parcels.size(); ++i)
            {
                HPX_ASSERT(locality_id == parcels[i].get_destination_locality());
            }
#endif

            // enqueue the outgoing parcels ...
            HPX_ASSERT(parcels.size() == handlers.size());
            enqueue_parcels(locality_id, std::move(parcels), std::move(handlers));

            if (enable_parcel_handling_)
            {
                if (hpx::is_running() && async_serialization())
                {
                    trigger_sending_parcels(locality_id);
                }
                else
                {
                    get_connection_and_send_parcels(locality_id);
                }
            }
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
            trigger_pending_work();
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

        /// Temporarily enable/disable all parcel handling activities in the
        /// parcelport
        void enable(bool new_state)
        {
            enable_parcel_handling_ = new_state;
            do_enable_parcel_handling_impl<ConnectionHandler>(new_state);
            if (new_state)
                trigger_pending_work();
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

            HPX_THROW_EXCEPTION(bad_parameter,
                "parcelport_impl::get_connection_cache_statistics",
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
        void early_pending_parcel_handler(
            boost::system::error_code const& ec, std::size_t size, parcel const & p)
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
            put_parcel(
                p
              , boost::bind(
                    &parcelport_impl::early_pending_parcel_handler
                  , this
                  , ::_1
                  , ::_2
                  , p
                )
            );
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
            connection_handler().background_work();
        }

        template <typename ConnectionHandler_>
        typename boost::disable_if<
            typename connection_handler_traits<
                ConnectionHandler_
            >::do_background_work
        >::type
        do_background_work_impl()
        {
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename ConnectionHandler_>
        typename boost::enable_if<
            typename connection_handler_traits<
                ConnectionHandler_
            >::do_enable_parcel_handling
        >::type
        do_enable_parcel_handling_impl(bool new_state)
        {
            connection_handler().enable_parcel_handling(new_state);
        }

        template <typename ConnectionHandler_>
        typename boost::disable_if<
            typename connection_handler_traits<
                ConnectionHandler_
            >::do_enable_parcel_handling
        >::type
        do_enable_parcel_handling_impl(bool new_state)
        {
        }

        ///////////////////////////////////////////////////////////////////////
        boost::shared_ptr<connection> get_connection(
            naming::locality const& l, bool force, error_code& ec)
        {
            // Request new connection from connection cache.
            boost::shared_ptr<connection> sender_connection;

            // Get a connection or reserve space for a new connection.
            if (!connection_cache_.get_or_reserve(l, sender_connection, force))
            {
                // If no slot is available it's not a problem as the parcel
                // will be sent out whenever the next connection is returned
                // to the cache.
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

        ///////////////////////////////////////////////////////////////////////
        void enqueue_parcel(naming::locality const& locality_id,
            parcel&& p, write_handler_type&& f)
        {
            typedef pending_parcels_map::mapped_type mapped_type;

            lcos::local::spinlock::scoped_lock l(mtx_);

            mapped_type& e = pending_parcels_[locality_id];
            e.first.push_back(std::move(p));
            e.second.push_back(std::move(f));

            parcel_destinations_.insert(locality_id);
        }

        void enqueue_parcels(naming::locality const& locality_id,
            std::vector<parcel>&& parcels,
            std::vector<write_handler_type>&& handlers)
        {
            typedef pending_parcels_map::mapped_type mapped_type;

            lcos::local::spinlock::scoped_lock l(mtx_);

            HPX_ASSERT(parcels.size() == handlers.size());

            mapped_type& e = pending_parcels_[locality_id];
            if (e.first.empty())
            {
                HPX_ASSERT(e.second.empty());
#if HPX_GCC_VERSION >= 40600 && HPX_GCC_VERSION < 40700
                // GCC4.6 gets incredibly confused
                std::swap(e.first, static_cast<std::vector<parcel>&>(parcels));
                std::swap(e.second, static_cast<std::vector<write_handler_type>&>(handlers));
#else
                std::swap(e.first, parcels);
                std::swap(e.second, handlers);
#endif
            }
            else
            {
                HPX_ASSERT(e.first.size() == e.second.size());
                std::size_t new_size = e.first.size() + parcels.size();
                e.first.reserve(new_size);
                e.second.reserve(new_size);

                std::move(parcels.begin(), parcels.end(),
                    std::back_inserter(e.first));
                std::move(handlers.begin(), handlers.end(),
                    std::back_inserter(e.second));
            }

            parcel_destinations_.insert(locality_id);
        }

        bool dequeue_parcels(naming::locality const& locality_id,
            std::vector<parcel>& parcels,
            std::vector<write_handler_type>& handlers)
        {
            typedef pending_parcels_map::iterator iterator;

            if (!enable_parcel_handling_)
                return false;

            {
                lcos::local::spinlock::scoped_lock l(mtx_);

                iterator it = pending_parcels_.find(locality_id);

                // do nothing if parcels have already been picked up by
                // another thread
                if (it != pending_parcels_.end() && !it->second.first.empty())
                {
                    HPX_ASSERT(it->first == locality_id);
                    HPX_ASSERT(handlers.size() == parcels.size());
                    std::swap(parcels, it->second.first);
                    std::swap(handlers, it->second.second);

                    HPX_ASSERT(!handlers.empty());
                }
                else
                {
                    HPX_ASSERT(it->second.second.empty());
                    return false;
                }

                parcel_destinations_.erase(locality_id);

                return true;
            }
        }

        ///////////////////////////////////////////////////////////////////////
        bool trigger_sending_parcels(naming::locality const& loc,
            bool background = false)
        {
            if (!enable_parcel_handling_)
                return true;         // do not schedule sending parcels

            error_code ec(lightweight);
            hpx::applier::register_thread_nullary(
                util::bind(
                    &parcelport_impl::get_connection_and_send_parcels,
                    this, loc, background),
                "get_connection_and_send_parcels",
                threads::pending, true, threads::thread_priority_critical,
                std::size_t(-1), threads::thread_stacksize_default, ec);
            return ec ? false : true;
        }

        bool trigger_pending_work()
        {
            std::vector<naming::locality> destinations;
            destinations.reserve(parcel_destinations_.size());

            {
                lcos::local::spinlock::scoped_lock l(mtx_);
                if (parcel_destinations_.empty())
                    return true;

                destinations.reserve(parcel_destinations_.size());
                BOOST_FOREACH(naming::locality const& loc, parcel_destinations_)
                {
                    destinations.push_back(loc);
                }
            }

            // Create new HPX threads which send the parcels that are still
            // pending.
            BOOST_FOREACH(naming::locality const& loc, destinations)
            {
                if (!trigger_sending_parcels(loc, true))
                    return false;
            }

            return true;
        }

        ///////////////////////////////////////////////////////////////////////
        void get_connection_and_send_parcels(
            naming::locality const& locality_id, bool background = false)
        {
            // repeat until no more parcels are to be sent
            while (true)
            {
                std::vector<parcel> parcels;
                std::vector<write_handler_type> handlers;

                if (!dequeue_parcels(locality_id, parcels, handlers))
                    break;

                HPX_ASSERT(!parcels.empty() && !handlers.empty());
                HPX_ASSERT(parcels.size() == handlers.size());

                // If none of the parcels may require id-splitting we're safe to
                // force a new connection from the connection cache.
                bool force_connection = true;
                BOOST_FOREACH(parcel const& p, parcels)
                {
                    if (p.may_require_id_splitting())
                    {
                        force_connection = false;
                        break;
                    }
                }

                error_code ec;
                boost::shared_ptr<connection> sender_connection =
                    get_connection(locality_id, force_connection, ec);

                if (!sender_connection)
                {
                    if (!force_connection && background)
                    {
                        // retry getting a connection, this time enforcing a
                        // new connection to be created (if needed)
                        sender_connection = get_connection(locality_id, true, ec);
                    }

                    if (!sender_connection)
                    {
                        // give the parcels back to the queues for later
                        enqueue_parcels(locality_id, std::move(parcels),
                            std::move(handlers));

                        // We can safely return if no connection is available
                        // at this point. As soon as a connection becomes
                        // available it checks for pending parcels and sends
                        // those out.
                        return;
                    }
                }

                // send parcels if they didn't get sent by another connection
                if (!hpx::is_starting() && threads::get_self_ptr() == 0)
                {
                    // Re-schedule if this is not executed by an HPX thread
                    std::size_t thread_num = get_worker_thread_num();
                    hpx::applier::register_thread_nullary(
                        hpx::util::bind(
                            hpx::util::one_shot(&parcelport_impl::send_pending_parcels)
                          , this
                          , sender_connection
                          , std::move(parcels)
                          , std::move(handlers)
                        )
                      , "parcelport_impl::send_pending_parcels"
                      , threads::pending, true, threads::thread_priority_critical,
                        thread_num, threads::thread_stacksize_default
                    );
                }
                else
                {
                    send_pending_parcels(sender_connection, std::move(parcels),
                        std::move(handlers));
                }

                // We yield here for a short amount of time to give another
                // HPX thread the chance to put a subsequent parcel which
                // leads to a more effective parcel buffering
                if (hpx::threads::get_self_ptr())
                    hpx::this_thread::yield();
            }
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

                HPX_ASSERT(locality_id == sender_connection->destination());
                if (!ec)
                {
                    // Give this connection back to the cache as it's not
                    // needed anymore.
                    connection_cache_.reclaim(locality_id, sender_connection);
                }
                else
                {
                    // remove this connection from cache
                    connection_cache_.clear(locality_id, sender_connection);
                }

                pending_parcels_map::iterator it = pending_parcels_.find(locality_id);
                if (it == pending_parcels_.end() || it->second.first.empty())
                    return;
            }

            // Create a new HPX thread which sends parcels that are still
            // pending.
            trigger_sending_parcels(locality_id);
        }

        void send_pending_parcels(
            boost::shared_ptr<connection> sender_connection,
            std::vector<parcel>&& parcels,
            std::vector<write_handler_type>&& handlers)
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
                buffer = encode_parcels(parcels, *sender_connection,
                    archive_flags_, this->enable_security());

            // send them asynchronously
            sender_connection->async_write(
                hpx::parcelset::detail::call_for_each(std::move(handlers)),
                boost::bind(&parcelport_impl::send_pending_parcels_trampoline,
                    this,
                    ::_1, ::_2, ::_3));

            do_background_work_impl<ConnectionHandler>();
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
