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
#include <hpx/runtime/serialization/detail/future_await_container.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/io_service_pool.hpp>
#include <hpx/util/connection_cache.hpp>
#include <hpx/util/runtime_configuration.hpp>
#include <hpx/util/safe_lexical_cast.hpp>

#include <boost/thread/locks.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    bool is_starting();
    bool is_stopped();
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
        static const char * connection_handler_type()
        {
            return connection_handler_traits<ConnectionHandler>::type();
        }

        static std::size_t thread_pool_size(util::runtime_configuration const& ini)
        {
            std::string key("hpx.parcel.");
            key += connection_handler_type();

            return hpx::util::get_entry_as<std::size_t>(
                ini, key + ".io_pool_size", "2");
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
            key += connection_handler_type();

            return hpx::util::get_entry_as<std::size_t>(
                ini, key + ".max_connections", HPX_PARCEL_MAX_CONNECTIONS);
        }

        static
            std::size_t max_connections_per_loc(util::runtime_configuration const& ini)
        {
            std::string key("hpx.parcel.");
            key += connection_handler_type();

            return hpx::util::get_entry_as<std::size_t>(
                ini, key + ".max_connections_per_locality",
                HPX_PARCEL_MAX_CONNECTIONS_PER_LOCALITY);
        }

    public:
        /// Construct the parcelport on the given locality.
        parcelport_impl(util::runtime_configuration const& ini,
            locality const & here,
            util::function_nonser<void(std::size_t, char const*)> const& on_start_thread,
            util::function_nonser<void()> const& on_stop_thread)
          : parcelport(ini, here, connection_handler_type())
          , io_service_pool_(thread_pool_size(ini),
                on_start_thread, on_stop_thread, pool_name(), pool_name_postfix())
          , connection_cache_(max_connections(ini), max_connections_per_loc(ini))
          , archive_flags_(0)
          , operations_in_flight_(0)
        {
#ifdef BOOST_BIG_ENDIAN
            std::string endian_out = get_config_entry("hpx.parcel.endian_out", "big");
#else
            std::string endian_out = get_config_entry("hpx.parcel.endian_out", "little");
#endif
            if (endian_out == "little")
                archive_flags_ |= serialization::endian_little;
            else if (endian_out == "big")
                archive_flags_ |= serialization::endian_big;
            else {
                HPX_ASSERT(endian_out =="little" || endian_out == "big");
            }

            if (!this->allow_array_optimizations()) {
                archive_flags_ |= serialization::disable_array_optimization;
                archive_flags_ |= serialization::disable_data_chunking;
            }
            else {
                if (!this->allow_zero_copy_optimizations())
                    archive_flags_ |= serialization::disable_data_chunking;
            }
        }

        ~parcelport_impl()
        {
            connection_cache_.clear();
        }

        bool can_bootstrap() const
        {
            return
                connection_handler_traits<
                    ConnectionHandler
                >::send_early_parcel::value;
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
            do_background_work(0);

            // make sure no more work is pending, wait for service pool to get
            // empty
            while(operations_in_flight_ != 0)
            {
                if(threads::get_self_ptr())
                    hpx::this_thread::suspend(hpx::threads::pending,
                        "parcelport_impl::stop");
            }

            io_service_pool_.stop();
            if (blocking) {
                connection_cache_.shutdown();
                connection_handler().do_stop();
                io_service_pool_.join();
                connection_cache_.clear();
                io_service_pool_.clear();
            }
        }

        void put_parcel(locality const & dest, parcel p, write_handler_type f)
        {
            put_parcel(dest, std::move(p), std::move(f), true);
        }

        void put_parcel(locality const & dest, parcel p, write_handler_type f,
            bool trigger)
        {
            HPX_ASSERT(dest.type() == type());

            boost::shared_ptr<hpx::serialization::detail::future_await_container>
                future_await(new hpx::serialization::detail::future_await_container());
            boost::shared_ptr<hpx::serialization::output_archive>
                archive(
                    new hpx::serialization::output_archive(
                        *future_await, 0, 0, 0, 0, &future_await->new_gids_)
                );
            (*archive) << p;

            if(future_await->has_futures())
            {
                void (parcelport_impl::*awaiter)(
                    locality const &, parcel, write_handler_type, bool
                  , boost::shared_ptr<hpx::serialization::output_archive> const &
                  , boost::shared_ptr<
                        hpx::serialization::detail::future_await_container> const &
                )
                    = &parcelport_impl::put_parcel_impl;
                (*future_await)(
                    util::bind(
                        util::one_shot(awaiter), this,
                        dest, std::move(p), std::move(f), true,
                        archive, future_await)
                );
                return;
            }
            else
            {
                put_parcel_impl(
                    dest, std::move(p), std::move(f), trigger, archive, future_await);
            }
        }

        void put_parcel_impl(
            locality const & dest, parcel p, write_handler_type f, bool trigger
          , boost::shared_ptr<hpx::serialization::output_archive> const &
          , boost::shared_ptr<
                hpx::serialization::detail::future_await_container
            > const & future_await)
        {
            // enqueue the outgoing parcel ...
            enqueue_parcel(
                dest, std::move(p), std::move(f), std::move(future_await->new_gids_));

            if (trigger && enable_parcel_handling_)
            {
                if (hpx::is_running() && async_serialization())
                {
                    trigger_sending_parcels(dest);
                }
                else
                {
                    get_connection_and_send_parcels(dest);
                }
            }
        }

        void put_parcels(std::vector<locality> dests, std::vector<parcel> parcels,
            std::vector<write_handler_type> handlers)
        {
            if (dests.size() == parcels.size() && parcels.size() != handlers.size())
            {
                HPX_THROW_EXCEPTION(bad_parameter, "parcelport::put_parcels",
                    "mismatched number of parcels and handlers");
                return;
            }

            locality const& locality_id = dests[0];

#if defined(HPX_DEBUG)
            // make sure all parcels go to the same locality
            for (std::size_t i = 1; i != dests.size(); ++i)
            {
                HPX_ASSERT(locality_id == dests[i]);
                HPX_ASSERT(parcels[0].destination_locality() ==
                    parcels[i].destination_locality());
            }
#endif

            // enqueue the outgoing parcels ...
            HPX_ASSERT(parcels.size() == handlers.size());
            for(std::size_t i = 0; i < parcels.size(); ++i)
            {
                put_parcel(
                    locality_id, std::move(parcels[i]), std::move(handlers[i]),
                    false);
            }

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

        void send_early_parcel(locality const & dest, parcel p)
        {
            send_early_parcel_impl<ConnectionHandler>(dest, std::move(p));
        }

        util::io_service_pool* get_thread_pool(char const* name)
        {
            if (0 == std::strcmp(name, io_service_pool_.get_name()))
                return &io_service_pool_;
            return 0;
        }

        bool do_background_work(std::size_t num_thread)
        {
            bool did_some_work = false;
            did_some_work = do_background_work_impl<ConnectionHandler>(num_thread);
            if(num_thread == 0)
            {
                trigger_pending_work();
            }
            return did_some_work;
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

        /// Cache specific functionality
        void remove_from_connection_cache_delayed(locality const& loc)
        {
            if (operations_in_flight_ != 0)
            {
                error_code ec(lightweight);
                hpx::applier::register_thread_nullary(
                    util::bind(
                        &parcelport_impl::remove_from_connection_cache,
                        this, loc),
                    "remove_from_connection_cache",
                    threads::pending, true, threads::thread_priority_normal,
                    std::size_t(-1), threads::thread_stacksize_default, ec);
                if (!ec) return;
            }

            connection_cache_.clear(loc);
        }

        void remove_from_connection_cache(locality const& loc)
        {
            error_code ec(lightweight);
            threads::thread_id_type id =
                hpx::applier::register_thread_nullary(
                    util::bind(
                        &parcelport_impl::remove_from_connection_cache_delayed,
                        this, loc),
                    "remove_from_connection_cache",
                    threads::suspended, true, threads::thread_priority_normal,
                    std::size_t(-1), threads::thread_stacksize_default, ec);
            if (ec) return;

            threads::set_thread_state(id,
                boost::chrono::milliseconds(100), threads::pending,
                threads::wait_signaled, threads::thread_priority_boost, ec);
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

        /// Return the name of this locality
        std::string get_locality_name() const
        {
            return connection_handler().get_locality_name();
        }

        ////////////////////////////////////////////////////////////////////////
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

        ////////////////////////////////////////////////////////////////////////
        // the code below is needed to bootstrap the parcel layer
        void early_pending_parcel_handler(
            boost::system::error_code const& ec, parcel const & p)
        {
            if (ec) {
                // all errors during early parcel handling are fatal
                boost::exception_ptr exception =
                    hpx::detail::get_exception(hpx::exception(ec),
                        "early_pending_parcel_handler", __FILE__, __LINE__,
                        "error while handling early parcel: " +
                            ec.message() + "(" +
                            boost::lexical_cast<std::string>(ec.value()) +
                            ")" + parcelset::dump_parcel(p));

                hpx::report_error(exception);
                return;
            }
        }

        template <typename ConnectionHandler_>
        typename boost::enable_if<
            typename connection_handler_traits<
                ConnectionHandler_
            >::send_early_parcel
        >::type
        send_early_parcel_impl(locality const & dest, parcel p)
        {
            put_parcel(
                dest
              , std::move(p)
              , util::bind(
                    &parcelport_impl::early_pending_parcel_handler
                  , this
                  , util::placeholders::_1
                  , util::placeholders::_2
                )
            );
        }

        template <typename ConnectionHandler_>
        typename boost::disable_if<
            typename connection_handler_traits<
                ConnectionHandler_
            >::send_early_parcel
        >::type
        send_early_parcel_impl(locality const & dest, parcel p)
        {
            HPX_THROW_EXCEPTION(network_error, "send_early_parcel",
                "This parcelport does not support sending early parcels");
        }

        template <typename ConnectionHandler_>
        typename boost::enable_if<
            typename connection_handler_traits<
                ConnectionHandler_
            >::do_background_work,
            bool
        >::type
        do_background_work_impl(std::size_t num_thread)
        {
            return connection_handler().background_work(num_thread);
        }

        template <typename ConnectionHandler_>
        typename boost::disable_if<
            typename connection_handler_traits<
                ConnectionHandler_
            >::do_background_work,
            bool
        >::type
        do_background_work_impl(std::size_t)
        {
            return false;
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
            locality const& l, bool force, error_code& ec)
        {
            // Request new connection from connection cache.
            boost::shared_ptr<connection> sender_connection;

            // Get a connection or reserve space for a new connection.
            if (!connection_cache_.get_or_reserve(l, sender_connection))
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

        void merge_gids(new_gids_map & gids, new_gids_map && new_gids)
        {
            for(auto & v : new_gids)
            {
                new_gids_map::iterator it = gids.find(v.first);
                if(it == gids.end())
                {
                    gids.insert(v);
                }
                else
                {
                    it->second.insert(
                        it->second.end(), v.second.begin(), v.second.end());
                }
            }
        }

        ///////////////////////////////////////////////////////////////////////
        void enqueue_parcel(locality const& locality_id,
            parcel&& p, write_handler_type&& f, new_gids_map && new_gids)
        {
            typedef pending_parcels_map::mapped_type mapped_type;

            boost::unique_lock<lcos::local::spinlock> l(mtx_);
            // We ignore the lock here. It might happen that while enqueuing,
            // we need to acquire a lock. This should not cause any problems
            // (famous last words)
            util::ignore_while_checking<
                boost::unique_lock<lcos::local::spinlock>
            > il(&l);

            mapped_type& e = pending_parcels_[locality_id];
#if defined(HPX_PARCELSET_PENDING_PARCELS_WORKAROUND)
            if(!util::get<0>(e))
                util::get<0>(e) = boost::make_shared<std::vector<parcel> >();
            util::get<0>(e)->push_back(std::move(p));
#else
            util::get<0>(e).push_back(std::move(p));
#endif
            util::get<1>(e).push_back(std::move(f));

            merge_gids(util::get<2>(e), std::move(new_gids));

            parcel_destinations_.insert(locality_id);
        }

        void enqueue_parcels(locality const& locality_id,
            std::vector<parcel>&& parcels,
            std::vector<write_handler_type>&& handlers, new_gids_map && new_gids)
        {
            typedef pending_parcels_map::mapped_type mapped_type;

            boost::unique_lock<lcos::local::spinlock> l(mtx_);
            // We ignore the lock here. It might happen that while enqueuing,
            // we need to acquire a lock. This should not cause any problems
            // (famous last words)
            util::ignore_while_checking<
                boost::unique_lock<lcos::local::spinlock>
            > il(&l);

            HPX_ASSERT(parcels.size() == handlers.size());

            mapped_type& e = pending_parcels_[locality_id];
#if defined(HPX_PARCELSET_PENDING_PARCELS_WORKAROUND)
            if(!util::get<0>(e))
            {
                util::get<0>(e) = boost::make_shared<std::vector<parcel> >();
                HPX_ASSERT(util::get<1>(e).empty());
#if HPX_GCC_VERSION < 40700
                // GCC4.6 gets incredibly confused
                std::swap(
                    *util::get<0>(e),
                    static_cast<std::vector<parcel>&>(parcels));
                std::swap(
                    util::get<1>(e),
                    static_cast<std::vector<write_handler_type>&>(handlers));
#else
                std::swap(*util::get<0>(e), parcels);
                std::swap(util::get<1>(e), handlers);
#endif
            }
#else
            if (util::get<0>(e).empty())
            {
                HPX_ASSERT(util::get<1>(e).empty());
#if HPX_GCC_VERSION < 40700
                // GCC4.6 gets incredibly confused
                std::swap(
                    util::get<0>(e),
                    static_cast<std::vector<parcel>&>(parcels));
                std::swap(
                    util::get<1>(e),
                    static_cast<std::vector<write_handler_type>&>(handlers));
#else
                std::swap(util::get<0>(e), parcels);
                std::swap(util::get<1>(e), handlers);
#endif
            }
#endif
            else
            {
#if defined(HPX_PARCELSET_PENDING_PARCELS_WORKAROUND)
                HPX_ASSERT(util::get<0>(e)->size() == util::get<1>(e).size());
                std::size_t new_size = util::get<0>(e)->size() + parcels.size();
                util::get<0>(e)->reserve(new_size);

                std::move(parcels.begin(), parcels.end(),
                    std::back_inserter(*util::get<0>(e)));
#else
                HPX_ASSERT(util::get<0>(e).size() == util::get<1>(e).size());
                std::size_t new_size = util::get<0>(e).size() + parcels.size();
                util::get<0>(e).reserve(new_size);

                std::move(parcels.begin(), parcels.end(),
                    std::back_inserter(util::get<0>(e)));
#endif
                util::get<1>(e).reserve(new_size);
                std::move(handlers.begin(), handlers.end(),
                    std::back_inserter(util::get<1>(e)));
            }

            merge_gids(util::get<2>(e), std::move(new_gids));

            parcel_destinations_.insert(locality_id);
        }

        bool dequeue_parcels(locality const& locality_id,
            std::vector<parcel>& parcels,
            std::vector<write_handler_type>& handlers,
            new_gids_map & new_gids)
        {
            typedef pending_parcels_map::iterator iterator;

            if (!enable_parcel_handling_)
                return false;

            {
                boost::lock_guard<lcos::local::spinlock> l(mtx_);

                iterator it = pending_parcels_.find(locality_id);

                // do nothing if parcels have already been picked up by
                // another thread
#if defined(HPX_PARCELSET_PENDING_PARCELS_WORKAROUND)
                if (it != pending_parcels_.end() && !util::get<0>(it->second)->empty())
#else
                if (it != pending_parcels_.end() && !util::get<0>(it->second).empty())
#endif
                {
                    HPX_ASSERT(it->first == locality_id);
                    HPX_ASSERT(handlers.size() == parcels.size());
#if defined(HPX_PARCELSET_PENDING_PARCELS_WORKAROUND)
                    std::swap(parcels, *util::get<0>(it->second));
#else
                    std::swap(parcels, util::get<0>(it->second));
#endif
                    std::swap(handlers, util::get<1>(it->second));

                    std::swap(new_gids, util::get<2>(it->second));

                    HPX_ASSERT(!handlers.empty());
                }
                else
                {
                    HPX_ASSERT(util::get<1>(it->second).empty());
                    return false;
                }

                parcel_destinations_.erase(locality_id);

                return true;
            }
        }


        ///////////////////////////////////////////////////////////////////////
        bool trigger_sending_parcels(locality const& loc,
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
                threads::pending, true, threads::thread_priority_boost,
                std::size_t(-1), threads::thread_stacksize_default, ec);
            return ec ? false : true;
        }

        bool trigger_pending_work()
        {
            if(hpx::is_stopped()) return true;

            std::vector<locality> destinations;

            {
                boost::unique_lock<lcos::local::spinlock> l(mtx_, boost::try_to_lock);
                if(l.owns_lock())
                {
                    if (parcel_destinations_.empty())
                        return true;

                    destinations.reserve(parcel_destinations_.size());
                    for (locality const& loc : parcel_destinations_)
                    {
                        destinations.push_back(loc);
                    }
                }
            }

            // Create new HPX threads which send the parcels that are still
            // pending.
            for (locality const& loc : destinations)
            {
                if (!trigger_sending_parcels(loc, true))
                    return false;
            }

            return true;
        }

        ///////////////////////////////////////////////////////////////////////
        void get_connection_and_send_parcels(
            locality const& locality_id, bool background = false)
        {
            // repeat until no more parcels are to be sent
            while (!hpx::is_stopped())
            {
                std::vector<parcel> parcels;
                std::vector<write_handler_type> handlers;
                new_gids_map new_gids;

                if(!dequeue_parcels(locality_id, parcels, handlers, new_gids))
                    break;

                // If one of the sending threads are in suspended state, we
                // need to force a new connection to avoid deadlocks.
                bool force_connection = true;

                error_code ec;
                boost::shared_ptr<connection> sender_connection =
                    get_connection(locality_id, force_connection, ec);

                if (!sender_connection)
                {
                    // give the parcels back to the queues for later
                    enqueue_parcels(locality_id, std::move(parcels),
                        std::move(handlers), std::move(new_gids));

                    // We can safely return if no connection is available
                    // at this point. As soon as a connection becomes
                    // available it checks for pending parcels and sends
                    // those out.
                    return;
                }

                threads::thread_id_type new_send_thread;
                // send parcels if they didn't get sent by another connection
                if (!hpx::is_starting() && threads::get_self_ptr() == 0)
                {
                    // Re-schedule if this is not executed by an HPX thread
                    std::size_t thread_num = get_worker_thread_num();
                    new_send_thread =
                        hpx::applier::register_thread_nullary(
                            hpx::util::bind(
                                hpx::util::one_shot(&parcelport_impl
                                    ::send_pending_parcels)
                              , this
                              , locality_id
                              , sender_connection
                              , std::move(parcels)
                              , std::move(handlers)
                              , std::move(new_gids)
                            )
                          , "parcelport_impl::send_pending_parcels"
                          , threads::pending, true, threads::thread_priority_boost,
                            thread_num, threads::thread_stacksize_default
                        );
                }
                else
                {
                    send_pending_parcels(
                        locality_id,
                        sender_connection, std::move(parcels),
                        std::move(handlers), std::move(new_gids));
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
            locality const& locality_id,
            boost::shared_ptr<connection> sender_connection)
        {
            --operations_in_flight_;
#if defined(HPX_TRACK_STATE_OF_OUTGOING_TCP_CONNECTION)
            client_connection->set_state(parcelport_connection::state_scheduled_thread);
#endif
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
            {
                boost::lock_guard<lcos::local::spinlock> l(mtx_);

                HPX_ASSERT(locality_id == sender_connection->destination());
                pending_parcels_map::iterator it = pending_parcels_.find(locality_id);
#if defined(HPX_PARCELSET_PENDING_PARCELS_WORKAROUND)
                if (it == pending_parcels_.end() ||
                    (util::get<0>(it->second) && util::get<0>(it->second)->empty()))
#else
                if (it == pending_parcels_.end() || util::get<0>(it->second).empty())
#endif
                    return;
            }

            // Create a new HPX thread which sends parcels that are still
            // pending.
            trigger_sending_parcels(locality_id);
        }

        void send_pending_parcels(
            parcelset::locality const & parcel_locality_id,
            boost::shared_ptr<connection> sender_connection,
            std::vector<parcel>&& parcels,
            std::vector<write_handler_type>&& handlers,
            new_gids_map new_gids)
        {
            // If we are stopped already, discard the remaining pending parcels
            if (hpx::is_stopped()) return;

#if defined(HPX_TRACK_STATE_OF_OUTGOING_TCP_CONNECTION)
            sender_connection->set_state(parcelport_connection::state_send_pending);
#endif

#if defined(HPX_DEBUG)
            // verify the connection points to the right destination
            HPX_ASSERT(parcel_locality_id == sender_connection->destination());
            sender_connection->verify(parcel_locality_id);
#endif
            // encode the parcels
            std::size_t num_parcels = encode_parcels(&parcels[0],
                    parcels.size(), sender_connection->buffer_,
                    archive_flags_,
                    this->get_max_outbound_message_size(),
                    &new_gids);

            using hpx::parcelset::detail::call_for_each;
            using namespace hpx::util::placeholders;
            if (num_parcels == parcels.size())
            {
                ++operations_in_flight_;
                // send all of the parcels
                sender_connection->async_write(
                    call_for_each(std::move(handlers), std::move(parcels)),
                    util::bind(&parcelport_impl::send_pending_parcels_trampoline,
                        this, _1, _2, _3));
            }
            else
            {
                ++operations_in_flight_;
                HPX_ASSERT(num_parcels < parcels.size());

                std::vector<write_handler_type> handled_handlers;
                handled_handlers.reserve(num_parcels);

                std::move(handlers.begin(), handlers.begin()+num_parcels,
                    std::back_inserter(handled_handlers));

                std::vector<parcel> handled_parcels;
                handled_parcels.reserve(num_parcels);

                std::move(parcels.begin(), parcels.begin()+num_parcels,
                    std::back_inserter(handled_parcels));

                // send only part of the parcels
                sender_connection->async_write(
                    call_for_each(
                        std::move(handled_handlers), std::move(handled_parcels)),
                    util::bind(&parcelport_impl::send_pending_parcels_trampoline,
                        this, _1, _2, _3));

                // give back unhandled parcels
                parcels.erase(parcels.begin(), parcels.begin()+num_parcels);
                handlers.erase(handlers.begin(), handlers.begin()+num_parcels);

                enqueue_parcels(parcel_locality_id, std::move(parcels),
                    std::move(handlers), std::move(new_gids));
            }

            std::size_t num_thread(0);
            if(threads::get_self_ptr())
                num_thread = hpx::get_worker_thread_num();
            do_background_work_impl<ConnectionHandler>(num_thread);
        }

    protected:
        /// The pool of io_service objects used to perform asynchronous operations.
        util::io_service_pool io_service_pool_;

        /// The connection cache for sending connections
        util::connection_cache<connection, locality> connection_cache_;

        typedef hpx::lcos::local::spinlock mutex_type;

        int archive_flags_;
        boost::atomic<std::size_t> operations_in_flight_;
    };
}}

#endif
