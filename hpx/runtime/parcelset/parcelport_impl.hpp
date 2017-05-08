//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c) 2007 Richard D Guidry Jr
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2011 Katelyn Kufahl
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_PARCELPORT_IMPL_HPP
#define HPX_PARCELSET_PARCELPORT_IMPL_HPP

#include <hpx/config.hpp>
#include <hpx/error_code.hpp>
#include <hpx/runtime/config_entry.hpp>
#include <hpx/runtime/parcelset/detail/call_for_each.hpp>
#include <hpx/runtime/parcelset/detail/parcel_await.hpp>
#include <hpx/runtime/parcelset/encode_parcels.hpp>
#include <hpx/runtime/parcelset/parcelport.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/util/atomic_count.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/connection_cache.hpp>
#include <hpx/util/io_service_pool.hpp>
#include <hpx/util/runtime_configuration.hpp>
#include <hpx/util/safe_lexical_cast.hpp>

#include <boost/atomic.hpp>
#include <boost/detail/endian.hpp>
#include <boost/exception_ptr.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
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
          , num_thread_(0)
          , max_background_thread_(hpx::util::safe_lexical_cast<std::size_t>(
                hpx::get_config_entry(
                    "hpx.max_background_threads",
                        (std::numeric_limits<std::size_t>::max)()
                )
            ))
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

        void flush_parcels()
        {
            // We suspend our thread, which will make progress on the network
            if(threads::get_self_ptr())
            {
                hpx::this_thread::suspend(hpx::threads::pending_boost,
                    "parcelport_impl::flush_parcels");
            }

            // make sure no more work is pending, wait for service pool to get
            // empty

            while(operations_in_flight_ != 0 || get_pending_parcels_count(false) != 0)
            {
                if(threads::get_self_ptr())
                {
                    hpx::this_thread::suspend(hpx::threads::pending_boost,
                        "parcelport_impl::flush_parcels");
                }
            }
        }

        void stop(bool blocking = true)
        {
            flush_parcels();

            io_service_pool_.stop();
            if (blocking) {
                connection_cache_.shutdown();
                connection_handler().do_stop();
                io_service_pool_.join();
                connection_cache_.clear();
                io_service_pool_.clear();
            }

        }

    public:
        // this is the handler used by put_parcel - it deals with a single parcel
        // at a time
        struct parcel_await_handler
        {
            parcelport_impl& this_;
            locality dest_;
            write_handler_type f_;

            void operator()(parcel&& p)
            {
                if (connection_handler_traits<ConnectionHandler>::
                        send_immediate_parcels::value &&
                    this_.can_send_immediate_impl<ConnectionHandler>())
                {
                    this_.send_parcel_immediate(dest_, std::move(p), std::move(f_));
                }
                else
                {
                    // enqueue the outgoing parcel ...
                    this_.enqueue_parcel(dest_, std::move(p), std::move(f_));

                    this_.get_connection_and_send_parcels(dest_);
                }
            }
        };

        // this is the handler used by put_parcels - this version handles a vector
        // of parcels rather than just a single one
        struct parcel_await_handlers
        {
            parcelport_impl* this_;
            locality dest_;
            std::vector<write_handler_type> handler_;
            std::vector<parcel> parcels_;

            parcel_await_handlers(
                parcelport_impl& pp,
                locality dest,
                std::vector<write_handler_type>&& handler)
              : this_(&pp),
                dest_(std::move(dest)),
                handler_(std::move(handler))
            {}

            parcel_await_handlers(parcel_await_handlers&& other)
              : this_(other.this_),
                dest_(std::move(other.dest_)),
                handler_(std::move(other.handler_)),
                parcels_(std::move(other.parcels_))
            {}

            parcel_await_handlers& operator=(parcel_await_handlers&& other)
            {
                this_ = other.this_;
                dest_ = std::move(other.dest_);
                handler_ = std::move(other.handler_);
                parcels_ = std::move(other.parcels_);
                return *this;
            }

            HPX_MOVABLE_ONLY(parcel_await_handlers);

            void operator()(parcel&& p)
            {
                parcels_.push_back(std::move(p));
                // enqueue the outgoing parcels ...

                if (parcels_.size() == handler_.size())
                {
                    this_->enqueue_parcels(
                        dest_, std::move(parcels_), std::move(handler_));

                    this_->get_connection_and_send_parcels(dest_);
                }
            }
        };

        void put_parcel(locality const & dest, parcel p, write_handler_type f)
        {
            HPX_ASSERT(dest.type() == type());

            std::make_shared<detail::parcel_await>(
                std::move(p), archive_flags_,
                parcel_await_handler{*this, dest, std::move(f)})->apply();
        }

        void put_parcels(locality const& dest, std::vector<parcel> parcels,
            std::vector<write_handler_type> handlers)
        {
            if (parcels.size() != handlers.size())
            {
                HPX_THROW_EXCEPTION(bad_parameter, "parcelport::put_parcels",
                    "mismatched number of parcels and handlers");
                return;
            }

#if defined(HPX_DEBUG)
            // make sure all parcels go to the same locality
            HPX_ASSERT(dest.type() == type());
            for (std::size_t i = 1; i != parcels.size(); ++i)
            {
                HPX_ASSERT(parcels[0].destination_locality() ==
                    parcels[i].destination_locality());
            }
#endif
            parcel_await_handlers handler(
                *this, dest, std::move(handlers));
            if (!connection_handler_traits<ConnectionHandler>::
                    send_immediate_parcels::value)
            {
                handler.parcels_.reserve(parcels.size());
            }

            std::make_shared<detail::parcel_await>(
                std::move(parcels), archive_flags_, std::move(handler))->apply();
        }

        void send_early_parcel(locality const & dest, parcel p)
        {
            send_early_parcel_impl<ConnectionHandler>(dest, std::move(p));
        }

        util::io_service_pool* get_thread_pool(char const* name)
        {
            if (0 == std::strcmp(name, io_service_pool_.get_name()))
                return &io_service_pool_;
            return nullptr;
        }

        bool do_background_work(std::size_t num_thread)
        {
            if (!connection_handler_traits<ConnectionHandler>::
                    send_immediate_parcels::value)
            {
                trigger_pending_work();
            }
            return do_background_work_impl<ConnectionHandler>(num_thread);
        }

        /// support enable_shared_from_this
        std::shared_ptr<parcelport_impl> shared_from_this()
        {
            return std::static_pointer_cast<parcelport_impl>(
                parcelset::parcelport::shared_from_this());
        }

        std::shared_ptr<parcelport_impl const> shared_from_this() const
        {
            return std::static_pointer_cast<parcelport_impl const>(
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
                    "remove_from_connection_cache_delayed",
                    threads::pending, true, threads::thread_priority_normal,
                    get_next_num_thread(), threads::thread_stacksize_default,
                    ec);
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
                    get_next_num_thread(), threads::thread_stacksize_default,
                    ec);
            if (ec) return;

            threads::set_thread_state(id,
                std::chrono::milliseconds(100), threads::pending,
                threads::wait_signaled, threads::thread_priority_boost, ec);
        }

        /// Return the name of this locality
        std::string get_locality_name() const
        {
            return connection_handler().get_locality_name();
        }

        ////////////////////////////////////////////////////////////////////////
        // Return the given connection cache statistic
        std::int64_t get_connection_cache_statistics(
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

        template <typename ConnectionHandler_>
        typename std::enable_if<
            connection_handler_traits<
                ConnectionHandler_
            >::send_early_parcel::value
        >::type
        send_early_parcel_impl(locality const & dest, parcel p)
        {
            put_parcel(
                dest
              , std::move(p)
              , util::bind(
                    &parcelport::early_pending_parcel_handler
                  , this
                  , util::placeholders::_1
                  , util::placeholders::_2
                )
            );
        }

        template <typename ConnectionHandler_>
        typename std::enable_if<
            !connection_handler_traits<
                ConnectionHandler_
            >::send_early_parcel::value
        >::type
        send_early_parcel_impl(locality const & dest, parcel p)
        {
            HPX_THROW_EXCEPTION(network_error, "send_early_parcel",
                "This parcelport does not support sending early parcels");
        }

        template <typename ConnectionHandler_>
        typename std::enable_if<
            connection_handler_traits<
                ConnectionHandler_
            >::do_background_work::value,
            bool
        >::type
        do_background_work_impl(std::size_t num_thread)
        {
            return connection_handler().background_work(num_thread);
        }

        template <typename ConnectionHandler_>
        typename std::enable_if<
           !connection_handler_traits<
                ConnectionHandler_
            >::do_background_work::value,
            bool
        >::type
        do_background_work_impl(std::size_t)
        {
            return false;
        }

        template <typename ConnectionHandler_>
        typename std::enable_if<
            connection_handler_traits<
                ConnectionHandler_
            >::send_immediate_parcels::value,
            bool
        >::type
        can_send_immediate_impl()
        {
            return connection_handler().can_send_immediate();
        }

        template <typename ConnectionHandler_>
        typename std::enable_if<
           !connection_handler_traits<
                ConnectionHandler_
            >::send_immediate_parcels::value,
            bool
        >::type
        can_send_immediate_impl()
        {
            return false;
        }

        ///////////////////////////////////////////////////////////////////////
        std::shared_ptr<connection> get_connection(
            locality const& l, bool force, error_code& ec)
        {
            // Request new connection from connection cache.
            std::shared_ptr<connection> sender_connection;

            if (connection_handler_traits<ConnectionHandler>::
                    send_immediate_parcels::value)
            {
                std::terminate();
            }
            else {
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
            }

            // Check if we need to create the new connection.
            if (!sender_connection)
                return connection_handler().create_connection(l, ec);

            if (&ec != &throws)
                ec = make_success_code();

            return sender_connection;
        }

        ///////////////////////////////////////////////////////////////////////
        void enqueue_parcel(locality const& locality_id,
            parcel&& p, write_handler_type&& f)
        {
            typedef pending_parcels_map::mapped_type mapped_type;

            std::unique_lock<lcos::local::spinlock> l(mtx_);
            // We ignore the lock here. It might happen that while enqueuing,
            // we need to acquire a lock. This should not cause any problems
            // (famous last words)
            util::ignore_while_checking<
                std::unique_lock<lcos::local::spinlock>
            > il(&l);

            mapped_type& e = pending_parcels_[locality_id];
#if defined(HPX_PARCELSET_PENDING_PARCELS_WORKAROUND)
            if(!util::get<0>(e))
                util::get<0>(e) = std::make_shared<std::vector<parcel> >();
            util::get<0>(e)->push_back(std::move(p));
#else
            util::get<0>(e).push_back(std::move(p));
#endif
            util::get<1>(e).push_back(std::move(f));

            parcel_destinations_.insert(locality_id);
            ++num_parcel_destinations_;
        }

        void enqueue_parcels(locality const& locality_id,
            std::vector<parcel>&& parcels,
            std::vector<write_handler_type>&& handlers)
        {
            typedef pending_parcels_map::mapped_type mapped_type;

            std::unique_lock<lcos::local::spinlock> l(mtx_);
            // We ignore the lock here. It might happen that while enqueuing,
            // we need to acquire a lock. This should not cause any problems
            // (famous last words)
            util::ignore_while_checking<
                std::unique_lock<lcos::local::spinlock>
            > il(&l);

            HPX_ASSERT(parcels.size() == handlers.size());

            mapped_type& e = pending_parcels_[locality_id];
#if defined(HPX_PARCELSET_PENDING_PARCELS_WORKAROUND)
            if(!util::get<0>(e))
            {
                util::get<0>(e) = std::make_shared<std::vector<parcel> >();
                HPX_ASSERT(util::get<1>(e).empty());
                std::swap(*util::get<0>(e), parcels);
                std::swap(util::get<1>(e), handlers);
            }
#else
            if (util::get<0>(e).empty())
            {
                HPX_ASSERT(util::get<1>(e).empty());
                std::swap(util::get<0>(e), parcels);
                std::swap(util::get<1>(e), handlers);
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

            parcel_destinations_.insert(locality_id);
            ++num_parcel_destinations_;
        }

        bool dequeue_parcels(locality const& locality_id,
            std::vector<parcel>& parcels,
            std::vector<write_handler_type>& handlers)
        {
            typedef pending_parcels_map::iterator iterator;

            {
                std::unique_lock<lcos::local::spinlock> l(mtx_, std::try_to_lock);

                if (!l) return false;

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

                    HPX_ASSERT(!handlers.empty());
                }
                else
                {
                    HPX_ASSERT(util::get<1>(it->second).empty());
                    return false;
                }

                parcel_destinations_.erase(locality_id);

                HPX_ASSERT(0 != num_parcel_destinations_.load());
                --num_parcel_destinations_;

                return true;
            }
        }

        bool trigger_pending_work()
        {
            if (0 == num_parcel_destinations_.load(boost::memory_order_relaxed))
                return true;

            std::vector<locality> destinations;

            {
                std::unique_lock<lcos::local::spinlock> l(mtx_, std::try_to_lock);
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
                get_connection_and_send_parcels(loc);
            }

            return true;
        }

        ///////////////////////////////////////////////////////////////////////
        void send_parcel_immediate(locality const& locality_id, parcel p,
            write_handler_type handler)
        {
            error_code ec;
            std::shared_ptr<connection> sender_connection =
                connection_handler().create_connection(locality_id, ec);

#if defined(HPX_DEBUG)
            // verify the connection points to the right destination
            HPX_ASSERT(locality_id == sender_connection->destination());
            sender_connection->verify(locality_id);
#endif

#if defined(HPX_TRACK_STATE_OF_OUTGOING_TCP_CONNECTION)
            sender_connection->set_state(parcelport_connection::state_send_pending);
#endif
            // encode the parcels
            std::size_t num_parcels = encode_parcels(*this, &p, 1,
                    sender_connection->buffer_,
                    archive_flags_,
                    this->get_max_outbound_message_size());

            using hpx::util::placeholders::_1;
            using hpx::util::placeholders::_2;
            using hpx::util::placeholders::_3;
            HPX_ASSERT(num_parcels == 1);
            {
                ++operations_in_flight_;

                // send the parcel
                sender_connection->async_write(
                    util::bind(util::one_shot(handler),
                        _1,  std::move(p)),
                    util::bind(&parcelport_impl::immediate_send_done,
                        this, _1, _2, _3));
            }
        }

        ///////////////////////////////////////////////////////////////////////
        void get_connection_and_send_parcels(
            locality const& locality_id, bool background = false)
        {
            // repeat until no more parcels are to be sent
            std::vector<parcel> parcels;
            std::vector<write_handler_type> handlers;

            if(!dequeue_parcels(locality_id, parcels, handlers))
            {
                return;
            }

            // If one of the sending threads are in suspended state, we
            // need to force a new connection to avoid deadlocks.
            bool force_connection = true;

            error_code ec;
            std::shared_ptr<connection> sender_connection =
                get_connection(locality_id, force_connection, ec);

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

            // send parcels if they didn't get sent by another connection
            send_pending_parcels(
                locality_id,
                sender_connection, std::move(parcels),
                std::move(handlers));

            // We yield here for a short amount of time to give another
            // HPX thread the chance to put a subsequent parcel which
            // leads to a more effective parcel buffering
//                 if (hpx::threads::get_self_ptr())
//                     hpx::this_thread::yield();
        }

        void immediate_send_done(
            boost::system::error_code const& ec,
            locality const& locality_id,
            std::shared_ptr<connection> sender_connection)
        {
            HPX_ASSERT(operations_in_flight_ != 0);
            --operations_in_flight_;

#if defined(HPX_TRACK_STATE_OF_OUTGOING_TCP_CONNECTION)
            sender_connection->set_state(parcelport_connection::state_scheduled_thread);
#endif
        }


        void send_pending_parcels_trampoline(
            boost::system::error_code const& ec,
            locality const& locality_id,
            std::shared_ptr<connection> sender_connection)
        {
            HPX_ASSERT(operations_in_flight_ != 0);
            --operations_in_flight_;

#if defined(HPX_TRACK_STATE_OF_OUTGOING_TCP_CONNECTION)
            sender_connection->set_state(parcelport_connection::state_scheduled_thread);
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
                std::lock_guard<lcos::local::spinlock> l(mtx_);

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
            get_connection_and_send_parcels(locality_id);
        }

        void send_pending_parcels(
            parcelset::locality const & parcel_locality_id,
            std::shared_ptr<connection> sender_connection,
            std::vector<parcel>&& parcels,
            std::vector<write_handler_type>&& handlers)
        {
#if defined(HPX_TRACK_STATE_OF_OUTGOING_TCP_CONNECTION)
            sender_connection->set_state(parcelport_connection::state_send_pending);
#endif

#if defined(HPX_DEBUG)
            // verify the connection points to the right destination
            HPX_ASSERT(parcel_locality_id == sender_connection->destination());
            sender_connection->verify(parcel_locality_id);
#endif
            // encode the parcels
            std::size_t num_parcels = encode_parcels(*this, &parcels[0],
                    parcels.size(), sender_connection->buffer_,
                    archive_flags_,
                    this->get_max_outbound_message_size());

            using hpx::parcelset::detail::call_for_each;
            using hpx::util::placeholders::_1;
            using hpx::util::placeholders::_2;
            using hpx::util::placeholders::_3;
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
                    std::move(handlers));
            }

            if(threads::get_self_ptr())
            {
                // We suspend our thread, which will make progress on the network
                hpx::this_thread::suspend(hpx::threads::pending_boost,
                    "parcelport_impl::send_pending_parcels");
            }
        }

    public:
        std::size_t get_next_num_thread()
        {
            return ++num_thread_ % max_background_thread_;
        }

    protected:
        /// The pool of io_service objects used to perform asynchronous operations.
        util::io_service_pool io_service_pool_;

        /// The connection cache for sending connections
        util::connection_cache<connection, locality> connection_cache_;

        typedef hpx::lcos::local::spinlock mutex_type;

        int archive_flags_;
        hpx::util::atomic_count operations_in_flight_;

        boost::atomic<std::size_t> num_thread_;
        std::size_t const max_background_thread_;
    };
}}

#endif
