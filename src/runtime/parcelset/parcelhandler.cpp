//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c) 2013-2014 Thomas Heller
//  Copyright (c) 2007      Richard D Guidry Jr
//  Copyright (c) 2011      Bryce Lelbach & Katelyn Kufahl
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING)
#include <hpx/config/asio.hpp>
#include <hpx/config/endian.hpp>
#include <hpx/config/detail/compat_error_code.hpp>
#include <hpx/actions/continuation.hpp>
#include <hpx/assert.hpp>
#include <hpx/async_distributed/applier/applier.hpp>
#include <hpx/functional/bind.hpp>
#include <hpx/functional/bind_front.hpp>
#include <hpx/functional/deferred_call.hpp>
#include <hpx/io_service/io_service_pool.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/format.hpp>
#include <hpx/modules/futures.hpp>
#include <hpx/modules/itt_notify.hpp>
#include <hpx/modules/logging.hpp>
#include <hpx/modules/string_util.hpp>
#include <hpx/modules/threadmanager.hpp>
#include <hpx/performance_counters/counter_creators.hpp>
#include <hpx/performance_counters/counters.hpp>
#include <hpx/performance_counters/manage_counter_type.hpp>
#include <hpx/preprocessor/stringize.hpp>
#include <hpx/runtime/message_handler_fwd.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>
#include <hpx/runtime/parcelset/parcelhandler.hpp>
#include <hpx/runtime/parcelset/policies/message_handler.hpp>
#include <hpx/runtime/parcelset/static_parcelports.hpp>
#include <hpx/runtime_configuration/runtime_configuration.hpp>
#include <hpx/runtime_local/config_entry.hpp>
#include <hpx/state.hpp>
#include <hpx/synchronization/counting_semaphore.hpp>
#include <hpx/thread_support/unlock_guard.hpp>
#include <hpx/threading_base/external_timer.hpp>
#include <hpx/threading_base/thread_helpers.hpp>
#include <hpx/util/from_string.hpp>
#include <hpx/util/get_entry_as.hpp>

#include <hpx/plugins/parcelport_factory_base.hpp>

#include <boost/asio/error.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>
#if defined(HPX_HAVE_PARCEL_PROFILING)
#include <chrono>
#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    bool is_stopped_or_shutting_down();
}

namespace hpx { namespace detail
{
    void dijkstra_make_black();     // forward declaration only
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset
{
    ///////////////////////////////////////////////////////////////////////////
    policies::message_handler* get_message_handler(
        parcelhandler* ph, char const* action, char const* type, std::size_t num,
        std::size_t interval, locality const& loc,
        error_code& ec)
    {
        return ph->get_message_handler(action, type, num, interval, loc, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    // A parcel is submitted for transport at the source locality site to
    // the parcel set of the locality with the put-parcel command
    // This function is synchronous.
    void parcelhandler::sync_put_parcel(parcel p) //-V669
    {
        lcos::local::promise<void> promise;
        future<void> sent_future = promise.get_future();
        put_parcel(
            std::move(p), [&promise](std::error_code const&, parcel const&) {
                promise.set_value();
            });               // schedule parcel send
        sent_future.get();    // wait for the parcel to be sent
    }

    parcelhandler::parcelhandler(util::runtime_configuration& cfg,
        threads::threadmanager* tm,
        threads::policies::callback_notifier const& notifier)
      : resolver_(nullptr)
      , tm_(tm)
      , use_alternative_parcelports_(false)
      , enable_parcel_handling_(true)
      , load_message_handlers_(util::get_entry_as<int>(cfg,
                                   "hpx.parcel.message_handlers", 0) != 0)
      , count_routed_(0)
      , write_handler_(&default_write_handler)
      , is_networking_enabled_(hpx::is_networking_enabled())
    {
        LPROGRESS_;

#if defined(HPX_HAVE_NETWORKING)
        if (is_networking_enabled_ &&
            cfg.get_entry("hpx.parcel.enable", "1") != "0")
        {
            for (plugins::parcelport_factory_base* factory :
                    get_parcelport_factories())
            {
                std::shared_ptr<parcelport> pp(factory->create(cfg, notifier));
                attach_parcelport(pp);
            }
        }
#endif
    }

    std::shared_ptr<parcelport> parcelhandler::get_bootstrap_parcelport() const
    {
        if(!pports_.empty())
        {
            std::string cfgkey("hpx.parcel.bootstrap");
            pports_type::const_iterator it =
                pports_.find(get_priority(get_config_entry(cfgkey, "tcp")));
            if(it != pports_.end() && it->first > 0) return it->second;
        }
        for (pports_type::value_type const& pp : pports_)
        {
            if(pp.first > 0 && pp.second->can_bootstrap())
                return pp.second;
        }
        return std::shared_ptr<parcelport>();
    }


    void parcelhandler::initialize(naming::resolver_client &resolver,
        applier::applier *applier)
    {
        resolver_ = &resolver;

        for (pports_type::value_type& pp : pports_)
        {
            pp.second->set_applier(applier);
            if(pp.second != get_bootstrap_parcelport())
            {
                if(pp.first > 0)
                    pp.second->run(false);
            }
        }
    }

    void parcelhandler::list_parcelport(std::ostringstream& strm,
        std::string const& ppname, int priority, bool bootstrap) const
    {
        strm << "parcel port: " << ppname;

        std::string cfgkey("hpx.parcel." + ppname + ".enable");
        std::string enabled = get_config_entry(cfgkey, "0");
        strm << ", "
             << (hpx::util::from_string<int>(enabled, 0) ? "" : "not ")
             << "enabled";

        if (bootstrap)
            strm << ", bootstrap";

        strm << ", priority " << priority;

        strm << '\n';
    }

    // list available parcel ports
    void parcelhandler::list_parcelports(std::ostringstream& strm) const
    {
        for (pports_type::value_type const& pp : pports_)
        {
            list_parcelport(
                strm
              , pp.second->type()
              , pp.second->priority()
              , pp.second == get_bootstrap_parcelport()
            );
        }
        strm << '\n';
    }

    void parcelhandler::attach_parcelport(std::shared_ptr<parcelport> const& pp)
    {
#if defined(HPX_HAVE_NETWORKING)
        if (!hpx::is_networking_enabled() || !pp)
            return;

        // add the new parcelport to the list of parcel-ports we care about
        int priority = pp->priority();
        std::string cfgkey(std::string("hpx.parcel.") + pp->type() + ".enable");
        if (get_config_entry(cfgkey, "0") != "1")
        {
            priority = -priority;
        }
        pports_[priority] = pp;
        priority_[pp->type()] = priority;

        // add the endpoint of the new parcelport
        HPX_ASSERT(pp->type() == pp->here().type());
        if(priority > 0)
            endpoints_[pp->type()] = pp->here();
#endif
    }

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Make sure the specified locality is not held by any
    /// connection caches anymore
    void parcelhandler::remove_from_connection_cache(
        naming::gid_type const& gid, endpoints_type const& endpoints)
    {
        for (endpoints_type::value_type const& loc : endpoints)
        {
            for (pports_type::value_type& pp : pports_)
            {
                if(std::string(pp.second->type()) == loc.second.type())
                {
                    pp.second->remove_from_connection_cache(loc.second);
                }
            }
        }

        HPX_ASSERT(resolver_);
        resolver_->remove_resolved_locality(gid);
    }

    ///////////////////////////////////////////////////////////////////////////
    bool parcelhandler::do_background_work(std::size_t num_thread,
        bool stop_buffering, parcelport_background_mode mode)
    {
        bool did_some_work = false;

#if defined(HPX_HAVE_NETWORKING)
        // flush all parcel buffers
        if (is_networking_enabled_ && 0 == num_thread &&
            (mode & parcelport_background_mode_flush_buffers))
        {
            std::unique_lock<mutex_type> l(handlers_mtx_, std::try_to_lock);

            if(l.owns_lock())
            {
                using parcelset::policies::message_handler;
                message_handler::flush_mode flush_mode =
                    message_handler::flush_mode_background_work;

                message_handler_map::iterator end = handlers_.end();
                for (message_handler_map::iterator it = handlers_.begin();
                     it != end; ++it)
                {
                    if ((*it).second)
                    {
                        std::shared_ptr<policies::message_handler> p((*it).second);
                        util::unlock_guard<std::unique_lock<mutex_type> > ul(l);
                        did_some_work =
                            p->flush(flush_mode, stop_buffering) || did_some_work;
                    }
                }
            }
        }

        // make sure all pending parcels are being handled
        for (pports_type::value_type& pp : pports_)
        {
            if(pp.first > 0)
            {
                did_some_work =
                    pp.second->do_background_work(num_thread, mode) ||
                    did_some_work;
            }
        }
#endif
        return did_some_work;
    }

    void parcelhandler::flush_parcels()
    {
        // now flush all parcel ports to be shut down
        for (pports_type::value_type& pp : pports_)
        {
            if(pp.first > 0)
            {
                pp.second->flush_parcels();
            }
        }
    }

    void parcelhandler::stop(bool blocking)
    {
        // now stop all parcel ports
        for (pports_type::value_type& pp : pports_)
        {
            if(pp.first > 0)
            {
                pp.second->stop(blocking);
            }
        }

        // release all message handlers
        handlers_.clear();
    }

    naming::resolver_client& parcelhandler::get_resolver()
    {
        return *resolver_;
    }

    bool parcelhandler::get_raw_remote_localities(
        std::vector<naming::gid_type>& locality_ids,
        components::component_type type, error_code& ec) const
    {
        std::vector<naming::gid_type> allprefixes;

        HPX_ASSERT(resolver_);
        bool result = resolver_->get_localities(allprefixes, type, ec);
        if (ec || !result) return false;

        std::remove_copy(allprefixes.begin(), allprefixes.end(),
            std::back_inserter(locality_ids), get_locality());

        return !locality_ids.empty();
    }

    bool parcelhandler::get_raw_localities(
        std::vector<naming::gid_type>& locality_ids,
        components::component_type type, error_code& ec) const
    {
        HPX_ASSERT(resolver_);
        bool result = resolver_->get_localities(locality_ids, type, ec);
        if (ec || !result) return false;

        return !locality_ids.empty();
    }

    std::pair<std::shared_ptr<parcelport>, locality>
    parcelhandler::find_appropriate_destination(
        naming::gid_type const& dest_gid)
    {
        HPX_ASSERT(resolver_);
        endpoints_type const & dest_endpoints =
            resolver_->resolve_locality(dest_gid);

        for (pports_type::value_type& pp : pports_)
        {
            if(pp.first > 0)
            {
                locality const& dest = find_endpoint(dest_endpoints, pp.second->type());
                if(dest && pp.second->can_connect(dest, use_alternative_parcelports_))
                    return std::make_pair(pp.second, dest);
            }
        }

        std::ostringstream strm;
        strm << "target locality: " << dest_gid << "\n";
        strm << "available destination endpoints:\n" << dest_endpoints << "\n";
        strm << "available partcelports:\n";
        for (auto const& pp : pports_)
        {
            list_parcelport(strm, pp.second->type(), pp.second->priority(),
                pp.second == get_bootstrap_parcelport());
            strm << "\t [" << pp.second->here() << "]\n";
        }

        HPX_THROW_EXCEPTION(network_error,
            "parcelhandler::find_appropriate_destination",
            "The locality gid cannot be resolved to a valid endpoint. "
            "No valid parcelport configured. Detailed information:\n" +
            strm.str());
        return std::pair<std::shared_ptr<parcelport>, locality>();
    }

    locality parcelhandler::find_endpoint(endpoints_type const & eps,
        std::string const & name)
    {
        endpoints_type::const_iterator it = eps.find(name);
        if(it != eps.end()) return it->second;
        return locality();
    }

    /// Return the reference to an existing io_service
    util::io_service_pool* parcelhandler::get_thread_pool(char const* name)
    {
        util::io_service_pool* result = nullptr;
        for (pports_type::value_type& pp : pports_)
        {
            result = pp.second->get_thread_pool(name);
            if (result) return result;
        }
        return result;
    }

    namespace detail {
        void parcel_sent_handler(
            parcelhandler::write_handler_type& f,    //-V669
            std::error_code const& ec, parcel const& p)
        {
            // inform termination detection of a sent message
            if (!p.does_termination_detection())
            {
                hpx::detail::dijkstra_make_black();
            }

            // invoke the original handler
            f(ec, p);

#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
            static util::itt::event parcel_send("send_parcel");
            util::itt::event_tick(parcel_send);
#endif

#if defined(HPX_HAVE_APEX) && defined(HPX_HAVE_PARCEL_PROFILING)
            // tell APEX about the sent parcel
            util::external_timer::send(p.parcel_id().get_lsb(), p.size(),
                p.destination_locality_id());
#endif
        }
    }

    void parcelhandler::put_parcel(parcel p)
    {
        auto handler = [this](std::error_code const& ec,
                           parcel const& p) -> void {
            invoke_write_handler(ec, p);
        };

        put_parcel_impl(std::move(p), std::move(handler));
    }

    void parcelhandler::put_parcel(parcel p, write_handler_type f)
    {
        auto handler = [this, f = std::move(f)](
                           std::error_code const& ec,
                           parcel const& p) -> void {
            invoke_write_handler(ec, p);
            f(ec, p);
        };

        put_parcel_impl(std::move(p), std::move(handler));
    }

    void parcelhandler::put_parcel_impl(parcel&& p, write_handler_type&& f)
    {
#if defined(HPX_HAVE_NETWORKING)
        HPX_ASSERT(is_networking_enabled_);
        HPX_ASSERT(resolver_);

        naming::gid_type const& gid = p.destination();
        naming::address& addr = p.addr();

        // During bootstrap this is handled separately (see
        // addressing_service::resolve_locality.

        // if this isn't an HPX thread, the stack space check will return false
        if (!this_thread::has_sufficient_stack_space() &&
            hpx::threads::threadmanager_is(hpx::state::state_running))
        {
//             naming::gid_type locality =
//                 naming::get_locality_from_gid(id.get_gid());
//             if (!resolver_->has_resolved_locality(locality))
            {
                // reschedule request as an HPX thread to avoid hangs
                void (parcelhandler::*put_parcel_ptr) (
                        parcel p, write_handler_type f
                    ) = &parcelhandler::put_parcel;

                threads::thread_init_data data(
                    threads::make_thread_function_nullary(util::deferred_call(
                        put_parcel_ptr, this, std::move(p), std::move(f))),
                    "parcelhandler::put_parcel",
                    threads::thread_priority::boost,
                    threads::thread_schedule_hint(),
                    threads::thread_stacksize::medium,
                    threads::thread_schedule_state::pending, true);
                threads::register_thread(data);
                return;
            }
        }

        // properly initialize parcel
        init_parcel(p);

        bool resolved_locally = true;

        if (!addr)
        {
            resolved_locally = resolver_->resolve_local(gid, addr);
        }

#if defined(HPX_HAVE_PARCEL_PROFILING)
        if (!p.parcel_id())
        {
            p.parcel_id() = parcelset::parcel::generate_unique_id();
        }
#endif

        write_handler_type wrapped_f =
            util::bind_front(&detail::parcel_sent_handler, std::move(f));

        // If we were able to resolve the address(es) locally we send the
        // parcel directly to the destination.
        if (resolved_locally)
        {
            // dispatch to the message handler which is associated with the
            // encapsulated action
            typedef std::pair<std::shared_ptr<parcelport>, locality> destination_pair;
            destination_pair dest = find_appropriate_destination(addr.locality_);

            if (load_message_handlers_ && !hpx::is_stopped_or_shutting_down())
            {
                policies::message_handler* mh =
                    p.get_message_handler(this, dest.second);

                if (mh) {
                    mh->put_parcel(dest.second, std::move(p), std::move(wrapped_f));
                    return;
                }
            }

            dest.first->put_parcel(dest.second, std::move(p), std::move(wrapped_f));
            return;
        }

        // At least one of the addresses is locally unknown, route the parcel
        // to the AGAS managing the destination.
        ++count_routed_;

        resolver_->route(std::move(p), std::move(wrapped_f));
#else
        HPX_THROW_EXCEPTION(invalid_status,
            "Networking was disabled at configuration time. Please "
            "reconfigure HPX using -DHPX_WITH_NETWORKING=On.",
            "parcelhandler::put_parcel");
#endif
    }

    void parcelhandler::put_parcels(std::vector<parcel> parcels)
    {
        std::vector<write_handler_type> handlers(parcels.size(),
            [this](std::error_code const& ec, parcel const& p)
                -> void { return invoke_write_handler(ec, p); });

        put_parcels_impl(std::move(parcels), std::move(handlers));
    }

    void parcelhandler::put_parcels(
        std::vector<parcel> parcels, std::vector<write_handler_type> funcs)
    {
        std::vector<write_handler_type> handlers;

        handlers.reserve(parcels.size());
        for (std::size_t i = 0; i != parcels.size(); ++i)
        {
            handlers.push_back([this, f = std::move(funcs[i])](
                                   std::error_code const& ec,
                                   parcel const& p) -> void {
                invoke_write_handler(ec, p);
                f(ec, p);
            });
        }

        put_parcels_impl(std::move(parcels), std::move(handlers));
    }

    void parcelhandler::put_parcels_impl(std::vector<parcel>&& parcels,
        std::vector<write_handler_type>&& handlers)
    {
#if defined(HPX_HAVE_NETWORKING)
        HPX_ASSERT(is_networking_enabled_);
        HPX_ASSERT(resolver_);

        if (parcels.size() != handlers.size())
        {
            HPX_THROW_EXCEPTION(bad_parameter,
                "parcelhandler::put_parcels",
                "mismatched number of parcels and handlers");
            return;
        }

        // if this isn't an HPX thread, the stack space check will return false
        if (!this_thread::has_sufficient_stack_space() &&
            hpx::threads::threadmanager_is(hpx::state::state_running))
        {
//             naming::gid_type locality = naming::get_locality_from_gid(
//                 parcels[0].destination());
//             if (!resolver_->has_resolved_locality(locality))
            {
                // reschedule request as an HPX thread to avoid hangs
                void (parcelhandler::*put_parcels_ptr) (
                        std::vector<parcel>, std::vector<write_handler_type>
                    ) = &parcelhandler::put_parcels;

                threads::thread_init_data data(
                    threads::make_thread_function_nullary(
                        util::deferred_call(put_parcels_ptr, this,
                            std::move(parcels), std::move(handlers))),
                    "parcelhandler::put_parcels",
                    threads::thread_priority::boost,
                    threads::thread_schedule_hint(),
                    threads::thread_stacksize::medium,
                    threads::thread_schedule_state::pending, true);
                threads::register_thread(data);
                return;
            }
        }

        // partition parcels depending on whether their destination can be
        // resolved locally
        std::size_t num_parcels = parcels.size();

        std::vector<parcel> resolved_parcels;
        resolved_parcels.reserve(num_parcels);
        std::vector<write_handler_type> resolved_handlers;
        resolved_handlers.reserve(num_parcels);

        typedef std::pair<std::shared_ptr<parcelport>, locality>
            destination_pair;

        destination_pair resolved_dest;

        std::vector<parcel> nonresolved_parcels;
        nonresolved_parcels.reserve(num_parcels);
        std::vector<write_handler_type> nonresolved_handlers;
        nonresolved_handlers.reserve(num_parcels);

        for (std::size_t i = 0; i != num_parcels; ++i)
        {
            parcel& p = parcels[i];

            // make sure all parcels go to the same locality
            if (parcels[0].destination_locality() !=
                p.destination_locality())
            {
                HPX_THROW_EXCEPTION(bad_parameter,
                    "parcelhandler::put_parcels",
                    "mismatched destinations, all parcels are expected to "
                    "target the same locality");
                return;
            }

            // properly initialize parcel
            init_parcel(p);
#if defined(HPX_HAVE_PARCEL_PROFILING)
            if (!p.parcel_id())
            {
                p.parcel_id() = parcelset::parcel::generate_unique_id();
            }
#endif

            bool resolved_locally = true;
            naming::address& addr = p.addr();

            if (!addr)
            {
                resolved_locally = resolver_->resolve_local(
                    p.destination(), addr);
            }

            write_handler_type f = util::bind_front(&detail::parcel_sent_handler,
                std::move(handlers[i]));

            // If we were able to resolve the address(es) locally we would send
            // the parcel directly to the destination.
            if (resolved_locally)
            {
                // dispatch to the message handler which is associated with the
                // encapsulated action
                destination_pair dest = find_appropriate_destination(
                    addr.locality_);

                if (load_message_handlers_)
                {
                    policies::message_handler* mh = p.get_message_handler(
                        this, dest.second);

                    if (mh) {
                        mh->put_parcel(dest.second, std::move(p), std::move(f));
                        continue;
                    }
                }

                resolved_parcels.push_back(std::move(p));
                resolved_handlers.push_back(std::move(f));
                if (!resolved_dest.second)
                {
                    resolved_dest = dest;
                }
                else
                {
                    HPX_ASSERT(resolved_dest == dest);
                }
            }
            else
            {
                nonresolved_parcels.push_back(std::move(p));
                nonresolved_handlers.push_back(std::move(f));
            }
        }

        // handle parcel which have been locally resolved
        if (!resolved_parcels.empty())
        {
            HPX_ASSERT(!!resolved_dest.first && !!resolved_dest.second);
            resolved_dest.first->put_parcels(resolved_dest.second,
                std::move(resolved_parcels),
                std::move(resolved_handlers));
        }

        // At least one of the addresses is locally unknown, route the
        // parcel to the AGAS managing the destination.
        for (std::size_t i = 0; i != nonresolved_parcels.size(); ++i)
        {
            ++count_routed_;
            resolver_->route(std::move(nonresolved_parcels[i]),
                std::move(nonresolved_handlers[i]));
        }
#else
        HPX_THROW_EXCEPTION(invalid_status,
            "Networking was disabled at configuration time. Please "
            "reconfigure HPX using -DHPX_WITH_NETWORKING=On.",
            "parcelhandler::put_parcels");
#endif
    }

    void parcelhandler::invoke_write_handler(
        std::error_code const& ec, parcel const& p) const
    {
        write_handler_type f;
        {
            std::lock_guard<mutex_type> l(mtx_);
            f = write_handler_;
        }
        f(ec, p);
    }

    ///////////////////////////////////////////////////////////////////////////
    std::int64_t parcelhandler::get_outgoing_queue_length(bool reset) const
    {
        std::int64_t parcel_count = 0;
        for (pports_type::value_type const& pp : pports_)
        {
            parcel_count += pp.second->get_pending_parcels_count(reset);
        }
        return parcel_count;
    }

    ///////////////////////////////////////////////////////////////////////////
    // default callback for put_parcel
    void default_write_handler(std::error_code const& ec, parcel const& p)
    {
#if defined(HPX_HAVE_NETWORKING)
        if (ec)
        {
            // If we are in a stopped state, ignore some errors
            if (hpx::is_stopped_or_shutting_down())
            {
                if (compat_error_code::equal(
                        ec, boost::asio::error::connection_aborted) ||
                    compat_error_code::equal(
                        ec, boost::asio::error::connection_reset) ||
                    compat_error_code::equal(
                        ec, boost::asio::error::broken_pipe) ||
                    compat_error_code::equal(
                        ec, boost::asio::error::not_connected) ||
                    compat_error_code::equal(ec, boost::asio::error::eof))
                {
                    return;
                }
            }
            else if (hpx::tolerate_node_faults())
            {
                if (compat_error_code::equal(
                        ec, boost::asio::error::connection_reset))
                    return;
            }

            // all unhandled exceptions terminate the whole application
            std::exception_ptr exception =
                hpx::detail::get_exception(hpx::exception(ec),
                    "default_write_handler", __FILE__,
                    __LINE__, parcelset::dump_parcel(p));

            hpx::report_error(exception);
        }
#endif
    }

    ///////////////////////////////////////////////////////////////////////////
    policies::message_handler* parcelhandler::get_message_handler(
        char const* action, char const* message_handler_type,
        std::size_t num_messages, std::size_t interval,
        locality const& loc, error_code& ec)
    {
#if defined(HPX_HAVE_NETWORKING)
        if (!is_networking_enabled_)
            return nullptr;

        std::unique_lock<mutex_type> l(handlers_mtx_);
        handler_key_type key(loc, action);
        message_handler_map::iterator it = handlers_.find(key);

        if (it == handlers_.end()) {
            std::shared_ptr<policies::message_handler> p;

            {
                // Just ignore the handlers_mtx_ while checking. We need to hold
                // the lock here to avoid multiple registrations that happens
                // right now in the parcel coalescing plugin
                hpx::util::ignore_while_checking<std::unique_lock<mutex_type>> il(&l);
                p.reset(hpx::create_message_handler(message_handler_type,
                    action, find_parcelport(loc.type()), num_messages, interval, ec));
            }

            it = handlers_.find(key);
            if (it != handlers_.end()) {
                // if some other thread has created the entry in the meantime
                l.unlock();
                if (&ec != &throws) {
                    if ((*it).second.get())
                        ec = make_success_code();
                    else
                        ec = make_error_code(bad_parameter, lightweight);
                }
                return (*it).second.get();
            }

            if (ec || !p.get()) {
                // insert an empty entry into the map to avoid trying to
                // create this handler again
                p.reset();
                std::pair<message_handler_map::iterator, bool> r =
                    handlers_.insert(message_handler_map::value_type(key, p));

                l.unlock();
                if (!r.second) {
                    HPX_THROWS_IF(ec, internal_server_error,
                        "parcelhandler::get_message_handler",
                        "could not store empty message handler");
                    return nullptr;
                }
                return nullptr;           // no message handler available
            }

            std::pair<message_handler_map::iterator, bool> r =
                handlers_.insert(message_handler_map::value_type(key, p));

            l.unlock();
            if (!r.second) {
                HPX_THROWS_IF(ec, internal_server_error,
                    "parcelhandler::get_message_handler",
                    "could not store newly created message handler");
                return nullptr;
            }
            it = r.first;
        }
        else if (!(*it).second.get()) {
            l.unlock();
            if (&ec != &throws)
            {
                ec = make_error_code(bad_parameter, lightweight);
            }
            else
            {
                HPX_THROW_EXCEPTION(bad_parameter,
                    "parcelhandler::get_message_handler",
                    "couldn't find an appropriate message handler");
            }
            return nullptr;           // no message handler available
        }

        if (&ec != &throws)
            ec = make_success_code();

        return (*it).second.get();
#else
        HPX_THROW_EXCEPTION(invalid_status,
            "Networking was disabled at configuration time. Please "
            "reconfigure HPX using -DHPX_WITH_NETWORKING=On.",
            "parcelhandler::get_message_handler");
        return nullptr;
#endif
    }

    ///////////////////////////////////////////////////////////////////////////
    std::string parcelhandler::get_locality_name() const
    {
        for (pports_type::value_type const& pp : pports_)
        {
            if(pp.first > 0)
            {
                std::string name = pp.second->get_locality_name();
                if(!name.empty())
                    return name;
            }
        }
        return "<unknown>";
    }

    ///////////////////////////////////////////////////////////////////////////
    // Performance counter data

    // number of parcels sent
    std::int64_t parcelhandler::get_parcel_send_count(
        std::string const& pp_type, bool reset) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_parcel_send_count(reset) : 0;
    }

    // number of parcels routed
    std::int64_t parcelhandler::get_parcel_routed_count(bool reset)
    {
        return util::get_and_reset_value(count_routed_, reset);
    }

    // number of messages sent
    std::int64_t parcelhandler::get_message_send_count(
        std::string const& pp_type, bool reset) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_message_send_count(reset) : 0;
    }

    // number of parcels received
    std::int64_t parcelhandler::get_parcel_receive_count(
        std::string const& pp_type, bool reset) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_parcel_receive_count(reset) : 0;
    }

    // number of messages received
    std::int64_t parcelhandler::get_message_receive_count(
        std::string const& pp_type, bool reset) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_message_receive_count(reset) : 0;
    }

    // the total time it took for all sends, from async_write to the
    // completion handler (nanoseconds)
    std::int64_t parcelhandler::get_sending_time(
        std::string const& pp_type, bool reset) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_sending_time(reset) : 0;
    }

    // the total time it took for all receives, from async_read to the
    // completion handler (nanoseconds)
    std::int64_t parcelhandler::get_receiving_time(
        std::string const& pp_type, bool reset) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_receiving_time(reset) : 0;
    }

    // the total time it took for all sender-side serialization operations
    // (nanoseconds)
    std::int64_t parcelhandler::get_sending_serialization_time(
        std::string const& pp_type, bool reset) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_sending_serialization_time(reset) : 0;
    }

    // the total time it took for all receiver-side serialization
    // operations (nanoseconds)
    std::int64_t parcelhandler::get_receiving_serialization_time(
        std::string const& pp_type, bool reset) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_receiving_serialization_time(reset) : 0;
    }

    // total data sent (bytes)
    std::int64_t parcelhandler::get_data_sent(std::string const& pp_type,
        bool reset) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_data_sent(reset) : 0;
    }

    // total data (uncompressed) sent (bytes)
    std::int64_t parcelhandler::get_raw_data_sent(std::string const& pp_type,
        bool reset) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_raw_data_sent(reset) : 0;
    }

    // total data received (bytes)
    std::int64_t parcelhandler::get_data_received(std::string const& pp_type,
        bool reset) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_data_received(reset) : 0;
    }

    // total data (uncompressed) received (bytes)
    std::int64_t parcelhandler::get_raw_data_received(std::string const& pp_type,
        bool reset) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_raw_data_received(reset) : 0;
    }

    std::int64_t parcelhandler::get_buffer_allocate_time_sent(
        std::string const& pp_type, bool reset) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_buffer_allocate_time_sent(reset) : 0;
    }
    std::int64_t parcelhandler::get_buffer_allocate_time_received(
        std::string const& pp_type, bool reset) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_buffer_allocate_time_received(reset) : 0;
    }

    // connection stack statistics
    std::int64_t parcelhandler::get_connection_cache_statistics(
        std::string const& pp_type,
        parcelport::connection_cache_statistics_type stat_type, bool reset) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_connection_cache_statistics(stat_type, reset) : 0;
    }

#if defined(HPX_HAVE_PARCELPORT_ACTION_COUNTERS)
    // same as above, just separated data for each action
    // number of parcels sent
    std::int64_t parcelhandler::get_action_parcel_send_count(
        std::string const& pp_type, std::string const& action, bool reset) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_action_parcel_send_count(action, reset) : 0;
    }

    // number of parcels received
    std::int64_t parcelhandler::get_action_parcel_receive_count(
        std::string const& pp_type, std::string const& action, bool reset) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_action_parcel_receive_count(action, reset) : 0;
    }

    // the total time it took for all sender-side serialization operations
    // (nanoseconds)
    std::int64_t parcelhandler::get_action_sending_serialization_time(
        std::string const& pp_type, std::string const& action, bool reset) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_action_sending_serialization_time(action, reset) : 0;
    }

    // the total time it took for all receiver-side serialization
    // operations (nanoseconds)
    std::int64_t parcelhandler::get_action_receiving_serialization_time(
        std::string const& pp_type, std::string const& action, bool reset) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_action_receiving_serialization_time(action, reset) : 0;
    }

    // total data sent (bytes)
    std::int64_t parcelhandler::get_action_data_sent(
        std::string const& pp_type, std::string const& action, bool reset) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_action_data_sent(action, reset) : 0;
    }

    // total data received (bytes)
    std::int64_t parcelhandler::get_action_data_received(
        std::string const& pp_type, std::string const& action, bool reset) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_action_data_received(action, reset) : 0;
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    void parcelhandler::register_counter_types()
    {
        // register connection specific counters
        for (pports_type::value_type const & pp : pports_)
        {
            register_counter_types(pp.second->type());
            register_connection_cache_counter_types(pp.second->type());
        }

        using util::placeholders::_1;
        using util::placeholders::_2;

        // register common counters
        util::function_nonser<std::int64_t(bool)> incoming_queue_length(
            util::bind_front(&parcelhandler::get_incoming_queue_length, this));
        util::function_nonser<std::int64_t(bool)> outgoing_queue_length(
            util::bind_front(&parcelhandler::get_outgoing_queue_length, this));
        util::function_nonser<std::int64_t(bool)> outgoing_routed_count(
            util::bind_front(&parcelhandler::get_parcel_routed_count, this));

        performance_counters::generic_counter_type_data const counter_types[] =
        {
            { "/parcelqueue/length/receive",
              performance_counters::counter_raw,
              "returns the number current length of the queue of incoming "
                  "parcels",
              HPX_PERFORMANCE_COUNTER_V1,
              util::bind(&performance_counters::locality_raw_counter_creator,
                  _1, incoming_queue_length, _2),
              &performance_counters::locality_counter_discoverer,
              ""
            },
            { "/parcelqueue/length/send",
              performance_counters::counter_raw,
              "returns the number current length of the queue of outgoing "
                  "parcels",
              HPX_PERFORMANCE_COUNTER_V1,
              util::bind(&performance_counters::locality_raw_counter_creator,
                  _1, outgoing_queue_length, _2),
              &performance_counters::locality_counter_discoverer,
              ""
            },
            { "/parcels/count/routed",
              performance_counters::counter_monotonically_increasing,
              "returns the number of (outbound) parcel routed through the "
                  "responsible AGAS service",
              HPX_PERFORMANCE_COUNTER_V1,
              util::bind(&performance_counters::locality_raw_counter_creator,
                  _1, outgoing_routed_count, _2),
              &performance_counters::locality_counter_discoverer,
              ""
            }
        };
        performance_counters::install_counter_types(
            counter_types, sizeof(counter_types)/sizeof(counter_types[0]));
    }

    void parcelhandler::register_counter_types(std::string const& pp_type)
    {
#if defined(HPX_HAVE_NETWORKING)
        if (!is_networking_enabled_)
            return;

        using util::placeholders::_1;
        using util::placeholders::_2;

#if defined(HPX_HAVE_PARCELPORT_ACTION_COUNTERS)
        util::function_nonser<std::int64_t(std::string const&, bool)>
            num_parcel_sends(util::bind_front(
                &parcelhandler::get_action_parcel_send_count, this,
                pp_type));
        util::function_nonser<std::int64_t(std::string const&, bool)>
            num_parcel_receives(util::bind_front(
                &parcelhandler::get_action_parcel_receive_count, this,
                pp_type));
#else
        util::function_nonser<std::int64_t(bool)>
            num_parcel_sends(util::bind_front(
                &parcelhandler::get_parcel_send_count, this,
                pp_type));
        util::function_nonser<std::int64_t(bool)>
            num_parcel_receives(util::bind_front(
                &parcelhandler::get_parcel_receive_count, this,
                pp_type));
#endif

        util::function_nonser<std::int64_t(bool)> num_message_sends(
            util::bind_front(&parcelhandler::get_message_send_count, this,
                pp_type));
        util::function_nonser<std::int64_t(bool)> num_message_receives(
            util::bind_front(&parcelhandler::get_message_receive_count, this,
                pp_type));

        util::function_nonser<std::int64_t(bool)> sending_time(
            util::bind_front(&parcelhandler::get_sending_time, this,
                pp_type));
        util::function_nonser<std::int64_t(bool)> receiving_time(
            util::bind_front(&parcelhandler::get_receiving_time, this,
                pp_type));

#if defined(HPX_HAVE_PARCELPORT_ACTION_COUNTERS)
        util::function_nonser<std::int64_t(std::string const&, bool)>
            sending_serialization_time(util::bind_front(
                &parcelhandler::get_action_sending_serialization_time, this,
                pp_type));
        util::function_nonser<std::int64_t(std::string const&, bool)>
            receiving_serialization_time(util::bind_front(
                &parcelhandler::get_action_receiving_serialization_time, this,
                pp_type));
#else
        util::function_nonser<std::int64_t(bool)>
            sending_serialization_time(util::bind_front(
                &parcelhandler::get_sending_serialization_time, this,
                pp_type));
        util::function_nonser<std::int64_t(bool)>
            receiving_serialization_time(util::bind_front(
                &parcelhandler::get_receiving_serialization_time, this,
                pp_type));
#endif

#if defined(HPX_HAVE_PARCELPORT_ACTION_COUNTERS)
        util::function_nonser<std::int64_t(std::string const&, bool)>
            data_sent(util::bind_front(
                &parcelhandler::get_action_data_sent, this,
                pp_type));
        util::function_nonser<std::int64_t(std::string const&, bool)>
            data_received(util::bind_front(
                &parcelhandler::get_action_data_received, this,
                pp_type));
#else
        util::function_nonser<std::int64_t(bool)>
            data_sent(util::bind_front(
                &parcelhandler::get_data_sent, this, pp_type));
        util::function_nonser<std::int64_t(bool)>
            data_received(util::bind_front(
                &parcelhandler::get_data_received, this, pp_type));
#endif

        util::function_nonser<std::int64_t(bool)> data_raw_sent(
            util::bind_front(&parcelhandler::get_raw_data_sent, this,
                pp_type));
        util::function_nonser<std::int64_t(bool)> data_raw_received(
            util::bind_front(&parcelhandler::get_raw_data_received, this,
                pp_type));

        util::function_nonser<std::int64_t(bool)> buffer_allocate_time_sent(
            util::bind_front(&parcelhandler::get_buffer_allocate_time_sent, this,
                pp_type));
        util::function_nonser<std::int64_t(bool)> buffer_allocate_time_received(
            util::bind_front(&parcelhandler::get_buffer_allocate_time_received, this,
                pp_type));

        performance_counters::generic_counter_type_data const counter_types[] =
        {
            { hpx::util::format("/parcels/count/{}/sent", pp_type),
              performance_counters::counter_monotonically_increasing,
              hpx::util::format(
                  "returns the number of parcels sent using the {} "
                  "connection type for the referenced locality", pp_type),
              HPX_PERFORMANCE_COUNTER_V1,
#if defined(HPX_HAVE_PARCELPORT_ACTION_COUNTERS)
              util::bind(
                  &performance_counters::per_action_data_counter_creator,
                  _1, std::move(num_parcel_sends), _2),
              &performance_counters::per_action_data_counter_discoverer,
#else
              util::bind(
                  &performance_counters::locality_raw_counter_creator,
                  _1, std::move(num_parcel_sends), _2),
              &performance_counters::locality_counter_discoverer,
#endif
              ""
            },
            { hpx::util::format("/parcels/count/{}/received", pp_type),
              performance_counters::counter_monotonically_increasing,
              hpx::util::format(
                  "returns the number of parcels received using the {} "
                  "connection type for the referenced locality", pp_type),
              HPX_PERFORMANCE_COUNTER_V1,
#if defined(HPX_HAVE_PARCELPORT_ACTION_COUNTERS)
              util::bind(
                  &performance_counters::per_action_data_counter_creator,
                  _1, std::move(num_parcel_receives), _2),
              &performance_counters::per_action_data_counter_discoverer,
#else
              util::bind(
                  &performance_counters::locality_raw_counter_creator,
                  _1, std::move(num_parcel_receives), _2),
              &performance_counters::locality_counter_discoverer,
#endif
              ""
            },
            { hpx::util::format("/messages/count/{}/sent", pp_type),
              performance_counters::counter_monotonically_increasing,
              hpx::util::format(
                  "returns the number of messages sent using the {} "
                  "connection type for the referenced locality", pp_type),
              HPX_PERFORMANCE_COUNTER_V1,
              util::bind(&performance_counters::locality_raw_counter_creator,
                  _1, std::move(num_message_sends), _2),
              &performance_counters::locality_counter_discoverer,
              ""
            },
            { hpx::util::format("/messages/count/{}/received", pp_type),
              performance_counters::counter_monotonically_increasing,
              hpx::util::format(
                  "returns the number of messages received using the {} "
                  "connection type for the referenced locality", pp_type),
              HPX_PERFORMANCE_COUNTER_V1,
              util::bind(&performance_counters::locality_raw_counter_creator,
                  _1, std::move(num_message_receives), _2),
              &performance_counters::locality_counter_discoverer,
              ""
            },

            { hpx::util::format("/data/time/{}/sent", pp_type),
              performance_counters::counter_elapsed_time,
              hpx::util::format(
                  "returns the total time between the start of each "
                  "asynchronous write and the invocation of the write callback "
                  "using the {} connection type for the referenced locality",
                      pp_type),
              HPX_PERFORMANCE_COUNTER_V1,
              util::bind(&performance_counters::locality_raw_counter_creator,
                  _1, std::move(sending_time), _2),
              &performance_counters::locality_counter_discoverer,
              "ns"
            },
            { hpx::util::format("/data/time/{}/received", pp_type),
              performance_counters::counter_elapsed_time,
              hpx::util::format(
                  "returns the total time between the start of each "
                  "asynchronous read and the invocation of the read callback "
                  "using the {} connection type for the referenced locality",
                      pp_type),
              HPX_PERFORMANCE_COUNTER_V1,
              util::bind(&performance_counters::locality_raw_counter_creator,
                  _1, std::move(receiving_time), _2),
              &performance_counters::locality_counter_discoverer,
              "ns"
            },
            { hpx::util::format("/serialize/time/{}/sent", pp_type),
              performance_counters::counter_elapsed_time,
              hpx::util::format(
                  "returns the total time required to serialize all sent "
                  "parcels using the {} connection type for the referenced "
                  "locality", pp_type),
              HPX_PERFORMANCE_COUNTER_V1,
#if defined(HPX_HAVE_PARCELPORT_ACTION_COUNTERS)
              util::bind(
                  &performance_counters::per_action_data_counter_creator,
                  _1, std::move(sending_serialization_time), _2),
              &performance_counters::per_action_data_counter_discoverer,
#else
              util::bind(
                  &performance_counters::locality_raw_counter_creator,
                  _1, std::move(sending_serialization_time), _2),
              &performance_counters::locality_counter_discoverer,
#endif
              "ns"
            },
            { hpx::util::format("/serialize/time/{}/received", pp_type),
              performance_counters::counter_elapsed_time,
              hpx::util::format(
                  "returns the total time required to de-serialize all "
                  "received parcels using the {} connection type for the "
                  "referenced locality", pp_type),
              HPX_PERFORMANCE_COUNTER_V1,
#if defined(HPX_HAVE_PARCELPORT_ACTION_COUNTERS)
              util::bind(
                  &performance_counters::per_action_data_counter_creator,
                  _1, std::move(receiving_serialization_time), _2),
              &performance_counters::per_action_data_counter_discoverer,
#else
              util::bind(
                  &performance_counters::locality_raw_counter_creator,
                  _1, std::move(receiving_serialization_time), _2),
              &performance_counters::locality_counter_discoverer,
#endif
              "ns"
            },

            { hpx::util::format("/data/count/{}/sent", pp_type),
              performance_counters::counter_monotonically_increasing,
              hpx::util::format(
                  "returns the amount of (uncompressed) parcel argument data "
                  "sent using the {} connection type by the referenced "
                  "locality", pp_type),
              HPX_PERFORMANCE_COUNTER_V1,
              util::bind(&performance_counters::locality_raw_counter_creator,
                  _1, std::move(data_raw_sent), _2),
              &performance_counters::locality_counter_discoverer,
              "bytes"
            },
            { hpx::util::format("/data/count/{}/received", pp_type),
              performance_counters::counter_monotonically_increasing,
              hpx::util::format(
                  "returns the amount of (uncompressed) parcel argument data "
                  "received using the {} connection type by the referenced "
                  "locality", pp_type),
              HPX_PERFORMANCE_COUNTER_V1,
              util::bind(&performance_counters::locality_raw_counter_creator,
                  _1, std::move(data_raw_received), _2),
              &performance_counters::locality_counter_discoverer,
              "bytes"
            },
            { hpx::util::format(
                  "/serialize/count/{}/sent", pp_type),
              performance_counters::counter_monotonically_increasing,
              hpx::util::format(
                  "returns the amount of parcel data (including headers, "
                  "possibly compressed) sent using the {} connection type "
                  "by the referenced locality", pp_type),
              HPX_PERFORMANCE_COUNTER_V1,
#if defined(HPX_HAVE_PARCELPORT_ACTION_COUNTERS)
              util::bind(
                  &performance_counters::per_action_data_counter_creator,
                  _1, std::move(data_sent), _2),
              &performance_counters::per_action_data_counter_discoverer,
#else
              util::bind(
                  &performance_counters::locality_raw_counter_creator,
                  _1, std::move(data_sent), _2),
              &performance_counters::locality_counter_discoverer,
#endif
              "bytes"
            },
            { hpx::util::format(
                  "/serialize/count/{}/received", pp_type),
              performance_counters::counter_monotonically_increasing,
              hpx::util::format(
                  "returns the amount of parcel data (including headers, "
                  "possibly compressed) received using the {} connection type "
                  "by the referenced locality", pp_type),
              HPX_PERFORMANCE_COUNTER_V1,
#if defined(HPX_HAVE_PARCELPORT_ACTION_COUNTERS)
              util::bind(
                  &performance_counters::per_action_data_counter_creator,
                  _1, std::move(data_received), _2),
              &performance_counters::per_action_data_counter_discoverer,
#else
              util::bind(
                  &performance_counters::locality_raw_counter_creator,
                  _1, std::move(data_received), _2),
              &performance_counters::locality_counter_discoverer,
#endif
              "bytes"
            },
            { hpx::util::format(
                "/parcels/time/{}/buffer_allocate/received", pp_type),
              performance_counters::counter_elapsed_time,
              hpx::util::format(
                  "returns the time needed to allocate the buffers for "
                  "serializing using the {} connection type", pp_type),
              HPX_PERFORMANCE_COUNTER_V1,
              util::bind(&performance_counters::locality_raw_counter_creator,
                  _1, std::move(buffer_allocate_time_received), _2),
              &performance_counters::locality_counter_discoverer,
              "ns"
            },
            { hpx::util::format(
                "/parcels/time/{}/buffer_allocate/sent", pp_type),
              performance_counters::counter_elapsed_time,
              hpx::util::format(
                  "returns the time needed to allocate the buffers for "
                  "serializing using the {} connection type", pp_type),
              HPX_PERFORMANCE_COUNTER_V1,
              util::bind(&performance_counters::locality_raw_counter_creator,
                  _1, std::move(buffer_allocate_time_sent), _2),
              &performance_counters::locality_counter_discoverer,
              "ns"
            },
        };
        performance_counters::install_counter_types(
            counter_types, sizeof(counter_types)/sizeof(counter_types[0]));
#endif
    }

    // register connection specific performance counters related to connection
    // caches
    void parcelhandler::register_connection_cache_counter_types(
        std::string const& pp_type)
    {
#if defined(HPX_HAVE_NETWORKING)
        if (!is_networking_enabled_)
            return;

        using hpx::util::placeholders::_1;
        using hpx::util::placeholders::_2;

        util::function_nonser<std::int64_t(bool)> cache_insertions(
            util::bind_front(&parcelhandler::get_connection_cache_statistics,
                this, pp_type, parcelport::connection_cache_insertions));
        util::function_nonser<std::int64_t(bool)> cache_evictions(
            util::bind_front(&parcelhandler::get_connection_cache_statistics,
                this, pp_type, parcelport::connection_cache_evictions));
        util::function_nonser<std::int64_t(bool)> cache_hits(
            util::bind_front(&parcelhandler::get_connection_cache_statistics,
                this, pp_type, parcelport::connection_cache_hits));
        util::function_nonser<std::int64_t(bool)> cache_misses(
            util::bind_front(&parcelhandler::get_connection_cache_statistics,
                this, pp_type, parcelport::connection_cache_misses));
        util::function_nonser<std::int64_t(bool)> cache_reclaims(
            util::bind_front(&parcelhandler::get_connection_cache_statistics,
                this, pp_type, parcelport::connection_cache_reclaims));

        performance_counters::generic_counter_type_data const
            connection_cache_types[] =
        {
            { hpx::util::format(
                  "/parcelport/count/{}/cache-insertions", pp_type),
              performance_counters::counter_raw,
              hpx::util::format(
                  "returns the number of cache insertions while accessing the "
                  "connection cache for the {} connection type on the "
                  "referenced locality", pp_type),
              HPX_PERFORMANCE_COUNTER_V1,
              util::bind(&performance_counters::locality_raw_counter_creator,
                  _1, std::move(cache_insertions), _2),
              &performance_counters::locality_counter_discoverer,
              ""
            },
            { hpx::util::format(
                  "/parcelport/count/{}/cache-evictions", pp_type),
              performance_counters::counter_raw,
              hpx::util::format(
                  "returns the number of cache evictions while accessing the "
                  "connection cache for the {} connection type on the "
                  "referenced locality", pp_type),
              HPX_PERFORMANCE_COUNTER_V1,
              util::bind(&performance_counters::locality_raw_counter_creator,
                  _1, std::move(cache_evictions), _2),
              &performance_counters::locality_counter_discoverer,
              ""
            },
            { hpx::util::format(
                  "/parcelport/count/{}/cache-hits", pp_type),
              performance_counters::counter_raw,
              hpx::util::format(
                  "returns the number of cache hits while accessing the "
                  "connection cache for the {} connection type on the "
                  "referenced locality", pp_type),
              HPX_PERFORMANCE_COUNTER_V1,
              util::bind(&performance_counters::locality_raw_counter_creator,
                  _1, std::move(cache_hits), _2),
              &performance_counters::locality_counter_discoverer,
              ""
            },
            { hpx::util::format(
                  "/parcelport/count/{}/cache-misses", pp_type),
              performance_counters::counter_raw,
              hpx::util::format(
                  "returns the number of cache misses while accessing the "
                  "connection cache for the {} connection type on the "
                  "referenced locality", pp_type),
              HPX_PERFORMANCE_COUNTER_V1,
              util::bind(&performance_counters::locality_raw_counter_creator,
                  _1, std::move(cache_misses), _2),
              &performance_counters::locality_counter_discoverer,
              ""
            },
            { hpx::util::format(
                  "/parcelport/count/{}/cache-reclaims", pp_type),
              performance_counters::counter_raw,
              hpx::util::format(
                  "returns the number of cache reclaims while accessing the "
                  "connection cache for the {} connection type on the "
                  "referenced locality", pp_type),
              HPX_PERFORMANCE_COUNTER_V1,
              util::bind(&performance_counters::locality_raw_counter_creator,
                  _1, std::move(cache_reclaims), _2),
              &performance_counters::locality_counter_discoverer,
              ""
            }
        };
        performance_counters::install_counter_types(connection_cache_types,
            sizeof(connection_cache_types)/sizeof(connection_cache_types[0]));
#endif
    }

    std::vector<plugins::parcelport_factory_base *> &
    parcelhandler::get_parcelport_factories()
    {
        static std::vector<plugins::parcelport_factory_base *> factories;
#if defined(HPX_HAVE_NETWORKING)
        if(factories.empty() && hpx::is_networking_enabled())
        {
            init_static_parcelport_factories(factories);
        }
#endif
        return factories;
    }

    void parcelhandler::add_parcelport_factory(
        plugins::parcelport_factory_base *factory)
    {
#if defined(HPX_HAVE_NETWORKING)
        if (hpx::is_networking_enabled())
        {
            auto & factories = get_parcelport_factories();
            if (std::find(factories.begin(), factories.end(), factory) !=
                factories.end())
            {
                return;
            }
            factories.push_back(factory);
        }
#endif
    }

    void parcelhandler::init(int *argc, char ***argv,
        util::command_line_handling &cfg)
    {
#if defined(HPX_HAVE_NETWORKING)
        HPX_ASSERT(hpx::is_networking_enabled());

        for (plugins::parcelport_factory_base* factory :
            get_parcelport_factories())
        {
            factory->init(argc, argv, cfg);
        }
#endif
    }

    std::vector<std::string> load_runtime_configuration()
    {
        // TODO: properly hide this in plugins ...
        std::vector<std::string> ini_defs;

#if defined(HPX_HAVE_NETWORKING)
        ini_defs.push_back("[hpx.parcel]");
        ini_defs.push_back(
            "address = ${HPX_PARCEL_SERVER_ADDRESS:" HPX_INITIAL_IP_ADDRESS
            "}");
        ini_defs.push_back("port = ${HPX_PARCEL_SERVER_PORT:" HPX_PP_STRINGIZE(
            HPX_INITIAL_IP_PORT) "}");
        ini_defs.push_back(
            "bootstrap = ${HPX_PARCEL_BOOTSTRAP:" HPX_PARCEL_BOOTSTRAP "}");
        ini_defs.push_back(
            "max_connections = ${HPX_PARCEL_MAX_CONNECTIONS:" HPX_PP_STRINGIZE(
                HPX_PARCEL_MAX_CONNECTIONS) "}");
        ini_defs.push_back(
            "max_connections_per_locality = "
            "${HPX_PARCEL_MAX_CONNECTIONS_PER_LOCALITY:" HPX_PP_STRINGIZE(
                HPX_PARCEL_MAX_CONNECTIONS_PER_LOCALITY) "}");
        ini_defs.push_back("max_message_size = "
                           "${HPX_PARCEL_MAX_MESSAGE_SIZE:" HPX_PP_STRINGIZE(
                               HPX_PARCEL_MAX_MESSAGE_SIZE) "}");
        ini_defs.push_back(
            "max_outbound_message_size = "
            "${HPX_PARCEL_MAX_OUTBOUND_MESSAGE_SIZE:" HPX_PP_STRINGIZE(
                HPX_PARCEL_MAX_OUTBOUND_MESSAGE_SIZE) "}");
        ini_defs.push_back(endian::native == endian::big ?
                "endian_out = ${HPX_PARCEL_ENDIAN_OUT:big}" :
                "endian_out = ${HPX_PARCEL_ENDIAN_OUT:little}");
        ini_defs.push_back(
            "array_optimization = ${HPX_PARCEL_ARRAY_OPTIMIZATION:1}");
        ini_defs.push_back(
            "zero_copy_optimization = ${HPX_PARCEL_ZERO_COPY_OPTIMIZATION:"
            "$[hpx.parcel.array_optimization]}");
        ini_defs.push_back(
            "async_serialization = ${HPX_PARCEL_ASYNC_SERIALIZATION:1}");
#if defined(HPX_HAVE_PARCEL_COALESCING)
        ini_defs.push_back(
            "message_handlers = ${HPX_PARCEL_MESSAGE_HANDLERS:1}");
#else
        ini_defs.push_back(
            "message_handlers = ${HPX_PARCEL_MESSAGE_HANDLERS:0}");
#endif

        for (plugins::parcelport_factory_base* f :
            parcelhandler::get_parcelport_factories())
        {
            f->get_plugin_info(ini_defs);
        }
#endif
        return ini_defs;
    }

    ///////////////////////////////////////////////////////////////////////////
    void parcelhandler::init_parcel(parcel& p)
    {
        // ensure the source locality id is set (if no component id is given)
        if (!p.source_id())
            p.set_source_id(naming::id_type(get_locality(),
                naming::id_type::unmanaged));

#if defined(HPX_HAVE_PARCEL_PROFILING)
        // set the current local time for this locality
        p.set_start_time(hpx::chrono::high_resolution_timer::now());
#endif
    }
}}

#endif
