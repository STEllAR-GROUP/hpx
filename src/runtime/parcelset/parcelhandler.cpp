//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2013-2014 Thomas Heller
//  Copyright (c) 2007      Richard D Guidry Jr
//  Copyright (c) 2011      Bryce Lelbach & Katelyn Kufahl
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/state.hpp>
#include <hpx/exception.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/io_service_pool.hpp>
#include <hpx/util/safe_lexical_cast.hpp>
#include <hpx/util/runtime_configuration.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>
#include <hpx/runtime/parcelset/parcelhandler.hpp>
#include <hpx/runtime/parcelset/static_parcelports.hpp>
#include <hpx/runtime/threads/threadmanager.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/lcos/local/counting_semaphore.hpp>
#include <hpx/include/performance_counters.hpp>
#include <hpx/performance_counters/counter_creators.hpp>

#include <hpx/plugins/parcelport_factory_base.hpp>

#include <boost/assign/std/vector.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/asio/io_service.hpp>
#include <boost/thread.hpp>
#include <boost/format.hpp>
#include <boost/foreach.hpp>

#include <algorithm>
#include <sstream>
#include <string>

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
    void parcelhandler::sync_put_parcel(parcel& p) //-V669
    {
        lcos::local::promise<void> promise;
        put_parcel(
            p
          , [&promise](boost::system::error_code const&, parcel const&)
            {
                promise.set_value();
            }
        );  // schedule parcel send
        promise.get_future().wait(); // wait for the parcel to be sent
    }

    void parcelhandler::parcel_sink(parcel const& p)
    {
        // wait for thread-manager to become active
        while (tm_->status() & starting)
        {
            boost::this_thread::sleep(boost::get_system_time() +
                boost::posix_time::milliseconds(HPX_NETWORK_RETRIES_SLEEP));
        }

        // Give up if we're shutting down.
        if (tm_->status() & stopping)
        {
            LPT_(debug) << "parcel_sink: dropping late parcel";
            return;
        }

        parcels_->add_parcel(p);
    }

    parcelhandler::parcelhandler(
            util::runtime_configuration & cfg,
            threads::threadmanager_base* tm, parcelhandler_queue_base* policy,
            util::function_nonser<void(std::size_t, char const*)> const& on_start_thread,
            util::function_nonser<void()> const& on_stop_thread)
      : tm_(tm),
        parcels_(policy),
        use_alternative_parcelports_(false),
        enable_parcel_handling_(true),
        count_routed_(0),
        write_handler_(&parcelhandler::default_write_handler)
    {
        BOOST_FOREACH(plugins::parcelport_factory_base *factory,
            get_parcelport_factories())
        {
            boost::shared_ptr<parcelport> pp;
            pp.reset(
                factory->create(
                    cfg
                  , on_start_thread
                  , on_stop_thread
                )
            );
            attach_parcelport(pp);
        }
    }

    boost::shared_ptr<parcelport> parcelhandler::get_bootstrap_parcelport() const
    {
        if(!pports_.empty())
        {
            std::string cfgkey("hpx.parcel.bootstrap");
            pports_type::const_iterator it =
                pports_.find(get_priority(get_config_entry(cfgkey, "tcp")));
            if(it != pports_.end() && it->first > 0) return it->second;
        }
        BOOST_FOREACH(pports_type::value_type const & pp, pports_)
        {
            if(pp.first > 0 && pp.second->can_bootstrap())
                return pp.second;
        }
        return boost::shared_ptr<parcelport>();
    }


    void parcelhandler::initialize(naming::resolver_client &resolver)
    {
        resolver_ = &resolver;
        HPX_ASSERT(parcels_);

        parcels_->set_parcelhandler(this);
        BOOST_FOREACH(pports_type::value_type & pp, pports_)
        {
            if(pp.second != get_bootstrap_parcelport())
            {
                if(pp.first > 0)
                    pp.second->run(false);
            }
            else
            {
                using util::placeholders::_1;
                pp.second->register_event_handler(
                    util::bind(&parcelhandler::parcel_sink, this, _1));
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
             << (hpx::util::safe_lexical_cast<int>(enabled, 0) ? "" : "not ")
             << "enabled";

        if (bootstrap)
            strm << ", bootstrap";

        strm << ", priority " << priority;

        strm << '\n';
    }

    // list available parcel ports
    void parcelhandler::list_parcelports(std::ostringstream& strm) const
    {
        BOOST_FOREACH(pports_type::value_type const & pp, pports_)
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

    void parcelhandler::attach_parcelport(boost::shared_ptr<parcelport> const& pp)
    {
        using util::placeholders::_1;

        if(!pp) return;

        // register our callback function with the parcelport
        pp->register_event_handler(util::bind(&parcelhandler::parcel_sink, this, _1));

        // add the new parcelport to the list of parcel-ports we care about
        int priority = pp->priority();
        std::string cfgkey(std::string("hpx.parcel.") + pp->type() + ".enable");
        if(get_config_entry(cfgkey, "0") != "1")
        {
            priority = -priority;
        }
        pports_[priority] = pp;
        priority_[pp->type()] = priority;

        // add the endpoint of the new parcelport
        HPX_ASSERT(pp->type() == pp->here().type());
        if(priority > 0)
            endpoints_[pp->type()] = pp->here();
    }

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Make sure the specified locality is not held by any
    /// connection caches anymore
    void parcelhandler::remove_from_connection_cache(
        endpoints_type const& endpoints)
    {
        BOOST_FOREACH(endpoints_type::value_type const & loc, endpoints)
        {
            BOOST_FOREACH(pports_type::value_type & pp, pports_)
            {
                if(std::string(pp.second->type()) == loc.second.type())
                {
                    pp.second->remove_from_connection_cache(loc.second);
                }
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    bool parcelhandler::do_background_work(std::size_t num_thread,
        bool stop_buffering)
    {
        bool did_some_work = false;

        // flush all parcel buffers
        if(0 == num_thread)
        {
            mutex_type::scoped_try_lock l(handlers_mtx_);

            if(l)
            {
                message_handler_map::iterator end = handlers_.end();
                for (message_handler_map::iterator it = handlers_.begin();
                     it != end; ++it)
                {
                    if ((*it).second)
                    {
                        boost::shared_ptr<policies::message_handler> p((*it).second);
                        util::scoped_unlock<mutex_type::scoped_try_lock> ul(l);
                        did_some_work = p->flush(stop_buffering) || did_some_work;
                    }
                }
            }
        }

        // make sure all pending parcels are being handled
        BOOST_FOREACH(pports_type::value_type & pp, pports_)
        {
            if(pp.first > 0)
            {
                did_some_work = pp.second->do_background_work(num_thread) ||
                    did_some_work;
            }
        }

        return did_some_work;
    }

    void parcelhandler::stop(bool blocking)
    {
        // now stop all parcel ports
        BOOST_FOREACH(pports_type::value_type & pp, pports_)
        {
            if(pp.first > 0)
            {
                pp.second->stop(blocking);
            }
        }
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

    std::pair<boost::shared_ptr<parcelport>, locality>
    parcelhandler::find_appropriate_destination(
        naming::gid_type const& dest_gid)
    {
        HPX_ASSERT(resolver_);
        endpoints_type const & dest_endpoints = resolver_->resolve_locality(dest_gid);

        BOOST_FOREACH(pports_type::value_type & pp, pports_)
        {
            if(pp.first > 0)
            {
                locality const& dest = find_endpoint(dest_endpoints, pp.second->type());
                if(dest && pp.second->can_connect(dest, use_alternative_parcelports_))
                    return std::make_pair(pp.second, dest);
            }
        }

        HPX_THROW_EXCEPTION(network_error,
            "parcelhandler::find_appropriate_destination",
            "The locality gid cannot be resolved to a valid endpoint. "
            "No valid parcelport configured.");
        return std::pair<boost::shared_ptr<parcelport>, locality>();
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
        util::io_service_pool* result = 0;
        BOOST_FOREACH(pports_type::value_type & pp, pports_)
        {
            result = pp.second->get_thread_pool(name);
            if (result) return result;
        }
        return result;
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        // The original parcel-sent handler is wrapped to keep the parcel alive
        // until after the data has been reliably sent (which is needed for zero
        // copy serialization).
        void parcel_sent_handler(boost::system::error_code const& ec,
            parcelhandler::write_handler_type const& f, parcel const& p)
        {
            // invoke the original handler
            f(ec, p);

            // inform termination detection of a sent message
            if (!p.does_termination_detection())
                hpx::detail::dijkstra_make_black();
        }
    }

    void parcelhandler::put_parcel(parcel& p, write_handler_type const& f)
    {
        HPX_ASSERT(resolver_);

        // properly initialize parcel
        init_parcel(p);

        naming::id_type const* ids = p.get_destinations();
        naming::address* addrs = p.get_destination_addrs();

        bool resolved_locally = true;

#if !defined(HPX_SUPPORT_MULTIPLE_PARCEL_DESTINATIONS)
        if (!addrs[0])
        {
            resolved_locally = resolver_->resolve_local(ids[0], addrs[0]);
        }
#else
        std::size_t size = p.size();

        if (0 == size) {
            HPX_THROW_EXCEPTION(network_error, "parcelhandler::put_parcel",
                "no destination address given");
            return;
        }

        if (1 == size) {
            if (!addrs[0])
                resolved_locally = resolver_->resolve_local(ids[0], addrs[0]);
        }
        else {
            boost::dynamic_bitset<> locals;
            resolved_locally = resolver_->resolve_local(ids, addrs, size, locals);
        }
#endif

        if (!p.get_parcel_id())
            p.set_parcel_id(parcel::generate_unique_id());

        // If we were able to resolve the address(es) locally we send the
        // parcel directly to the destination.
        if (resolved_locally) {

            // re-wrap the given parcel-sent handler
            using util::placeholders::_1;
            write_handler_type wrapped_f =
                util::bind(&detail::parcel_sent_handler, _1, f, p);

            // dispatch to the message handler which is associated with the
            // encapsulated action
            typedef std::pair<boost::shared_ptr<parcelport>, locality> destination_pair;
            destination_pair dest = find_appropriate_destination(addrs[0].locality_);
            policies::message_handler* mh =
                p.get_message_handler(this, dest.second);

            if (mh) {
                mh->put_parcel(dest.second, p, wrapped_f);
                return;
            }

            dest.first->put_parcel(dest.second, p, wrapped_f);
            return;
        }

        // At least one of the addresses is locally unknown, route the parcel
        // to the AGAS managing the destination.
        ++count_routed_;

        resolver_->route(p, f);
    }

    boost::int64_t parcelhandler::get_outgoing_queue_length(bool reset) const
    {
        boost::int64_t parcel_count = 0;
        BOOST_FOREACH(pports_type::value_type const& pp, pports_)
        {
            parcel_count += pp.second->get_pending_parcels_count(reset);
        }
        return parcel_count;
    }

    ///////////////////////////////////////////////////////////////////////////
    // default callback for put_parcel
    void parcelhandler::default_write_handler(
        boost::system::error_code const& ec, parcel const& p)
    {
        if (ec) {
            // If we are in a stopped state, ignore some errors
            if (hpx::is_stopped_or_shutting_down())
            {
                if (ec == boost::asio::error::connection_aborted ||
                    ec == boost::asio::error::connection_reset ||
                    ec == boost::asio::error::broken_pipe ||
                    ec == boost::asio::error::not_connected ||
                    ec == boost::asio::error::eof)
                {
                    return;
                }
            }

            // all unhandled exceptions terminate the whole application
            boost::exception_ptr exception =
                hpx::detail::get_exception(hpx::exception(ec),
                    "parcelhandler::default_write_handler", __FILE__,
                    __LINE__, parcelset::dump_parcel(p));

            hpx::report_error(exception);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    policies::message_handler* parcelhandler::get_message_handler(
        char const* action, char const* message_handler_type,
        std::size_t num_messages, std::size_t interval,
        locality const& loc, error_code& ec)
    {
        mutex_type::scoped_lock l(handlers_mtx_);
        handler_key_type key(loc, action);
        message_handler_map::iterator it = handlers_.find(key);
        if (it == handlers_.end()) {
            boost::shared_ptr<policies::message_handler> p;

            {
                util::scoped_unlock<mutex_type::scoped_lock> ul(l);
                p.reset(hpx::create_message_handler(message_handler_type,
                    action, find_parcelport(loc.type()), num_messages, interval, ec));
            }

            it = handlers_.find(key);
            if (it != handlers_.end()) {
                // if some other thread has created the entry in the mean time
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
                    return 0;
                }
                return 0;           // no message handler available
            }

            std::pair<message_handler_map::iterator, bool> r =
                handlers_.insert(message_handler_map::value_type(key, p));

            l.unlock();
            if (!r.second) {
                HPX_THROWS_IF(ec, internal_server_error,
                    "parcelhandler::get_message_handler",
                    "could not store newly created message handler");
                return 0;
            }
            it = r.first;
        }
        else if (!(*it).second.get()) {
            l.unlock();
            if (&ec != &throws)
                ec = make_error_code(bad_parameter, lightweight);
            return 0;           // no message handler available
        }

        if (&ec != &throws)
            ec = make_success_code();

        return (*it).second.get();
    }

    ///////////////////////////////////////////////////////////////////////////
    std::string parcelhandler::get_locality_name() const
    {
        BOOST_FOREACH(pports_type::value_type const & pp, pports_)
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
    bool parcelhandler::enable(bool new_state)
    {
        new_state = enable_parcel_handling_.exchange(
            new_state, boost::memory_order_acquire);

        BOOST_FOREACH(pports_type::value_type & pp, pports_)
        {
            if(pp.first > 0)
                pp.second->enable(enable_parcel_handling_);
        }

        return new_state;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Performance counter data

    // number of parcels sent
    boost::int64_t parcelhandler::get_parcel_send_count(
        std::string const& pp_type, bool reset) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_parcel_send_count(reset) : 0;
    }

    // number of parcels routed
    boost::int64_t parcelhandler::get_parcel_routed_count(bool reset)
    {
        return util::get_and_reset_value(count_routed_, reset);
    }

    // number of messages sent
    boost::int64_t parcelhandler::get_message_send_count(
        std::string const& pp_type, bool reset) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_message_send_count(reset) : 0;
    }

    // number of parcels received
    boost::int64_t parcelhandler::get_parcel_receive_count(
        std::string const& pp_type, bool reset) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_parcel_receive_count(reset) : 0;
    }

    // number of messages received
    boost::int64_t parcelhandler::get_message_receive_count(
        std::string const& pp_type, bool reset) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_message_receive_count(reset) : 0;
    }

    // the total time it took for all sends, from async_write to the
    // completion handler (nanoseconds)
    boost::int64_t parcelhandler::get_sending_time(
        std::string const& pp_type, bool reset) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_sending_time(reset) : 0;
    }

    // the total time it took for all receives, from async_read to the
    // completion handler (nanoseconds)
    boost::int64_t parcelhandler::get_receiving_time(
        std::string const& pp_type, bool reset) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_receiving_time(reset) : 0;
    }

    // the total time it took for all sender-side serialization operations
    // (nanoseconds)
    boost::int64_t parcelhandler::get_sending_serialization_time(
        std::string const& pp_type, bool reset) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_sending_serialization_time(reset) : 0;
    }

    // the total time it took for all receiver-side serialization
    // operations (nanoseconds)
    boost::int64_t parcelhandler::get_receiving_serialization_time(
        std::string const& pp_type, bool reset) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_receiving_serialization_time(reset) : 0;
    }

#if defined(HPX_HAVE_SECURITY)
    // the total time it took for all sender-side security operations
    // (nanoseconds)
    boost::int64_t parcelhandler::get_sending_security_time(
        std::string const& pp_type, bool reset) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_sending_security_time(reset) : 0;
    }

    // the total time it took for all receiver-side security
    // operations (nanoseconds)
    boost::int64_t parcelhandler::get_receiving_security_time(
        std::string const& pp_type, bool reset) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_receiving_security_time(reset) : 0;
    }
#endif

    // total data sent (bytes)
    boost::int64_t parcelhandler::get_data_sent(std::string const& pp_type,
        bool reset) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_data_sent(reset) : 0;
    }

    // total data (uncompressed) sent (bytes)
    boost::int64_t parcelhandler::get_raw_data_sent(std::string const& pp_type,
        bool reset) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_raw_data_sent(reset) : 0;
    }

    // total data received (bytes)
    boost::int64_t parcelhandler::get_data_received(std::string const& pp_type,
        bool reset) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_data_received(reset) : 0;
    }

    // total data (uncompressed) received (bytes)
    boost::int64_t parcelhandler::get_raw_data_received(std::string const& pp_type,
        bool reset) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_raw_data_received(reset) : 0;
    }

    boost::int64_t parcelhandler::get_buffer_allocate_time_sent(
        std::string const& pp_type, bool reset) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_buffer_allocate_time_sent(reset) : 0;
    }
    boost::int64_t parcelhandler::get_buffer_allocate_time_received(
        std::string const& pp_type, bool reset) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_buffer_allocate_time_received(reset) : 0;
    }

    // connection stack statistics
    boost::int64_t parcelhandler::get_connection_cache_statistics(
        std::string const& pp_type,
        parcelport::connection_cache_statistics_type stat_type, bool reset) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_connection_cache_statistics(stat_type, reset) : 0;
    }

    ///////////////////////////////////////////////////////////////////////////
    void parcelhandler::register_counter_types()
    {
        // register connection specific counters
        BOOST_FOREACH(pports_type::value_type const & pp, pports_)
        {
            register_counter_types(pp.second->type());
        }

        using util::placeholders::_1;
        using util::placeholders::_2;

        // register common counters
        util::function_nonser<boost::int64_t(bool)> incoming_queue_length(
            util::bind(&parcelhandler::get_incoming_queue_length, this, _1));
        util::function_nonser<boost::int64_t(bool)> outgoing_queue_length(
            util::bind(&parcelhandler::get_outgoing_queue_length, this, _1));
        util::function_nonser<boost::int64_t(bool)> outgoing_routed_count(
            util::bind(&parcelhandler::get_parcel_routed_count, this, _1));

        performance_counters::generic_counter_type_data const counter_types[] =
        {
            { "/parcelqueue/length/receive",
              performance_counters::counter_raw,
              "returns the number current length of the queue of incoming parcels",
              HPX_PERFORMANCE_COUNTER_V1,
              util::bind(&performance_counters::locality_raw_counter_creator,
                  _1, incoming_queue_length, _2),
              &performance_counters::locality_counter_discoverer,
              ""
            },
            { "/parcelqueue/length/send",
              performance_counters::counter_raw,
              "returns the number current length of the queue of outgoing parcels",
              HPX_PERFORMANCE_COUNTER_V1,
              util::bind(&performance_counters::locality_raw_counter_creator,
                  _1, outgoing_queue_length, _2),
              &performance_counters::locality_counter_discoverer,
              ""
            },
            { "/parcels/count/routed",
              performance_counters::counter_raw,
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
        using util::placeholders::_1;
        using util::placeholders::_2;

        util::function_nonser<boost::int64_t(bool)> num_parcel_sends(
            util::bind(&parcelhandler::get_parcel_send_count, this, pp_type, _1));
        util::function_nonser<boost::int64_t(bool)> num_parcel_receives(
            util::bind(&parcelhandler::get_parcel_receive_count, this, pp_type, _1));

        util::function_nonser<boost::int64_t(bool)> num_message_sends(
            util::bind(&parcelhandler::get_message_send_count, this, pp_type, _1));
        util::function_nonser<boost::int64_t(bool)> num_message_receives(
            util::bind(&parcelhandler::get_message_receive_count, this, pp_type, _1));

        util::function_nonser<boost::int64_t(bool)> sending_time(
            util::bind(&parcelhandler::get_sending_time, this, pp_type, _1));
        util::function_nonser<boost::int64_t(bool)> receiving_time(
            util::bind(&parcelhandler::get_receiving_time, this, pp_type, _1));

        util::function_nonser<boost::int64_t(bool)> sending_serialization_time(
            util::bind(&parcelhandler::get_sending_serialization_time, this, pp_type, _1));
        util::function_nonser<boost::int64_t(bool)> receiving_serialization_time(
            util::bind(&parcelhandler::get_receiving_serialization_time, this, pp_type, _1));

#if defined(HPX_HAVE_SECURITY)
        util::function_nonser<boost::int64_t(bool)> sending_security_time(
            util::bind(&parcelhandler::get_sending_security_time, this, pp_type, _1));
        util::function_nonser<boost::int64_t(bool)> receiving_security_time(
            util::bind(&parcelhandler::get_receiving_security_time, this, pp_type, _1));
#endif
        util::function_nonser<boost::int64_t(bool)> data_sent(
            util::bind(&parcelhandler::get_data_sent, this, pp_type, _1));
        util::function_nonser<boost::int64_t(bool)> data_received(
            util::bind(&parcelhandler::get_data_received, this, pp_type, _1));

        util::function_nonser<boost::int64_t(bool)> data_raw_sent(
            util::bind(&parcelhandler::get_raw_data_sent, this, pp_type, _1));
        util::function_nonser<boost::int64_t(bool)> data_raw_received(
            util::bind(&parcelhandler::get_raw_data_received, this, pp_type, _1));

        util::function_nonser<boost::int64_t(bool)> buffer_allocate_time_sent(
            util::bind(&parcelhandler::get_buffer_allocate_time_sent, this, pp_type, _1));
        util::function_nonser<boost::int64_t(bool)> buffer_allocate_time_received(
            util::bind(&parcelhandler::get_buffer_allocate_time_received, this, pp_type, _1));

        performance_counters::generic_counter_type_data const counter_types[] =
        {
            { boost::str(boost::format("/parcels/count/%s/sent") % pp_type),
              performance_counters::counter_raw,
              boost::str(boost::format("returns the number of parcels sent using the %s "
                  "connection type for the referenced locality") % pp_type),
              HPX_PERFORMANCE_COUNTER_V1,
              util::bind(&performance_counters::locality_raw_counter_creator,
                  _1, num_parcel_sends, _2),
              &performance_counters::locality_counter_discoverer,
              ""
            },
            { boost::str(boost::format("/parcels/count/%s/received") % pp_type),
               performance_counters::counter_raw,
              boost::str(boost::format("returns the number of parcels received using the %s "
                  "connection type for the referenced locality") % pp_type),
              HPX_PERFORMANCE_COUNTER_V1,
              util::bind(&performance_counters::locality_raw_counter_creator,
                  _1, num_parcel_receives, _2),
              &performance_counters::locality_counter_discoverer,
              ""
            },
            { boost::str(boost::format("/messages/count/%s/sent") % pp_type),
              performance_counters::counter_raw,
              boost::str(boost::format("returns the number of messages sent using the %s "
                  "connection type for the referenced locality") % pp_type),
              HPX_PERFORMANCE_COUNTER_V1,
              util::bind(&performance_counters::locality_raw_counter_creator,
                  _1, num_message_sends, _2),
              &performance_counters::locality_counter_discoverer,
              ""
            },
            { boost::str(boost::format("/messages/count/%s/received") % pp_type),
              performance_counters::counter_raw,
              boost::str(boost::format("returns the number of messages received using the %s "
                  "connection type for the referenced locality") % pp_type),
              HPX_PERFORMANCE_COUNTER_V1,
              util::bind(&performance_counters::locality_raw_counter_creator,
                  _1, num_message_receives, _2),
              &performance_counters::locality_counter_discoverer,
              ""
            },

            { boost::str(boost::format("/data/time/%s/sent") % pp_type),
              performance_counters::counter_raw,
              boost::str(boost::format("returns the total time between the start of "
                  "each asynchronous write and the invocation of the write callback "
                  "using the %s connection type for the referenced locality") %
                      pp_type),
              HPX_PERFORMANCE_COUNTER_V1,
              util::bind(&performance_counters::locality_raw_counter_creator,
                  _1, sending_time, _2),
              &performance_counters::locality_counter_discoverer,
              "ns"
            },
            { boost::str(boost::format("/data/time/%s/received") % pp_type),
              performance_counters::counter_raw,
              boost::str(boost::format("returns the total time between the start of "
                  "each asynchronous read and the invocation of the read callback "
                  "using the %s connection type for the referenced locality") %
                      pp_type),
              HPX_PERFORMANCE_COUNTER_V1,
              util::bind(&performance_counters::locality_raw_counter_creator,
                  _1, receiving_time, _2),
              &performance_counters::locality_counter_discoverer,
              "ns"
            },
            { boost::str(boost::format("/serialize/time/%s/sent") % pp_type),
              performance_counters::counter_raw,
              boost::str(boost::format("returns the total time required to serialize "
                  "all sent parcels using the %s connection type for the referenced "
                  "locality") % pp_type),
              HPX_PERFORMANCE_COUNTER_V1,
              util::bind(&performance_counters::locality_raw_counter_creator,
                  _1, sending_serialization_time, _2),
              &performance_counters::locality_counter_discoverer,
              "ns"
            },
            { boost::str(boost::format("/serialize/time/%s/received") % pp_type),
              performance_counters::counter_raw,
              boost::str(boost::format("returns the total time required to de-serialize "
                  "all received parcels using the %s connection type for the referenced "
                  "locality") % pp_type),
              HPX_PERFORMANCE_COUNTER_V1,
              util::bind(&performance_counters::locality_raw_counter_creator,
                  _1, receiving_serialization_time, _2),
              &performance_counters::locality_counter_discoverer,
              "ns"
            },

#if defined(HPX_HAVE_SECURITY)
            { boost::str(boost::format("/security/time/%s/sent") % pp_type),
              performance_counters::counter_raw,
              boost::str(boost::format("returns the total time required to perform "
                  "tasks related to security in the parcel layer for all sent parcels "
                  "using the %s connection type for the referenced locality") %
                        pp_type),
              HPX_PERFORMANCE_COUNTER_V1,
              util::bind(&performance_counters::locality_raw_counter_creator,
                  _1, sending_security_time, _2),
              &performance_counters::locality_counter_discoverer,
              "ns"
            },
            { boost::str(boost::format("/security/time/%s/received") % pp_type),
              performance_counters::counter_raw,
              boost::str(boost::format("returns the total time required to perform "
                  "tasks related to security in the parcel layer for all received parcels "
                  "using the %s connection type for the referenced locality") %
                        pp_type),
              HPX_PERFORMANCE_COUNTER_V1,
              util::bind(&performance_counters::locality_raw_counter_creator,
                  _1, receiving_security_time, _2),
              &performance_counters::locality_counter_discoverer,
              "ns"
            },
#endif
            { boost::str(boost::format("/data/count/%s/sent") % pp_type),
              performance_counters::counter_raw,
              boost::str(boost::format("returns the amount of (uncompressed) parcel "
                  "argument data sent using the %s connection type by the referenced "
                  "locality") % pp_type),
              HPX_PERFORMANCE_COUNTER_V1,
              util::bind(&performance_counters::locality_raw_counter_creator,
                  _1, data_raw_sent, _2),
              &performance_counters::locality_counter_discoverer,
              "bytes"
            },
            { boost::str(boost::format("/data/count/%s/received") % pp_type),
              performance_counters::counter_raw,
              boost::str(boost::format("returns the amount of (uncompressed) parcel "
                  "argument data received using the %s connection type by the referenced "
                  "locality") % pp_type),
              HPX_PERFORMANCE_COUNTER_V1,
              util::bind(&performance_counters::locality_raw_counter_creator,
                  _1, data_raw_received, _2),
              &performance_counters::locality_counter_discoverer,
              "bytes"
            },
            { boost::str(boost::format("/serialize/count/%s/sent") % pp_type),
              performance_counters::counter_raw,
              boost::str(boost::format("returns the amount of parcel data (including "
                  "headers, possibly compressed) sent using the %s connection type "
                  "by the referenced locality") % pp_type),
              HPX_PERFORMANCE_COUNTER_V1,
              util::bind(&performance_counters::locality_raw_counter_creator,
                  _1, data_sent, _2),
              &performance_counters::locality_counter_discoverer,
              "bytes"
            },
            { boost::str(boost::format("/serialize/count/%s/received") % pp_type),
              performance_counters::counter_raw,
              boost::str(boost::format("returns the amount of parcel data (including "
                  "headers, possibly compressed) received using the %s connection type "
                  "by the referenced locality") % pp_type),
              HPX_PERFORMANCE_COUNTER_V1,
              util::bind(&performance_counters::locality_raw_counter_creator,
                  _1, data_received, _2),
              &performance_counters::locality_counter_discoverer,
              "bytes"
            },
            { boost::str(boost::format("/parcels/time/%s/buffer_allocate/received") % pp_type),
              performance_counters::counter_raw,
              boost::str(boost::format("returns the time needed to allocate the "
                "buffers for serializing using the %s connection type") % pp_type),
              HPX_PERFORMANCE_COUNTER_V1,
              util::bind(&performance_counters::locality_raw_counter_creator,
                  _1, buffer_allocate_time_received, _2),
              &performance_counters::locality_counter_discoverer,
              "ns"
            },
            { boost::str(boost::format("/parcels/time/%s/buffer_allocate/sent") % pp_type),
              performance_counters::counter_raw,
              boost::str(boost::format("returns the time needed to allocate the "
                "buffers for serializing using the %s connection type") % pp_type),
              HPX_PERFORMANCE_COUNTER_V1,
              util::bind(&performance_counters::locality_raw_counter_creator,
                  _1, buffer_allocate_time_sent, _2),
              &performance_counters::locality_counter_discoverer,
              "ns"
            },
        };
        performance_counters::install_counter_types(
            counter_types, sizeof(counter_types)/sizeof(counter_types[0]));

        // register connection specific performance counters related to connection
        // caches
        util::function_nonser<boost::int64_t(bool)> cache_insertions(
            util::bind(&parcelhandler::get_connection_cache_statistics,
                this, pp_type, parcelport::connection_cache_insertions, _1));
        util::function_nonser<boost::int64_t(bool)> cache_evictions(
            util::bind(&parcelhandler::get_connection_cache_statistics,
                this, pp_type, parcelport::connection_cache_evictions, _1));
        util::function_nonser<boost::int64_t(bool)> cache_hits(
            util::bind(&parcelhandler::get_connection_cache_statistics,
                this, pp_type, parcelport::connection_cache_hits, _1));
        util::function_nonser<boost::int64_t(bool)> cache_misses(
            util::bind(&parcelhandler::get_connection_cache_statistics,
                this, pp_type, parcelport::connection_cache_misses, _1));
        util::function_nonser<boost::int64_t(bool)> cache_reclaims(
            util::bind(&parcelhandler::get_connection_cache_statistics,
                this, pp_type, parcelport::connection_cache_reclaims, _1));

        performance_counters::generic_counter_type_data const connection_cache_types[] =
        {
            { boost::str(boost::format("/parcelport/count/%s/cache-insertions") % pp_type),
              performance_counters::counter_raw,
              boost::str(boost::format("returns the number of cache insertions while accessing "
                  "the connection cache for the %s connection type on the referenced "
                  "locality") % pp_type),
              HPX_PERFORMANCE_COUNTER_V1,
              util::bind(&performance_counters::locality_raw_counter_creator,
                  _1, cache_insertions, _2),
              &performance_counters::locality_counter_discoverer,
              ""
            },
            { boost::str(boost::format("/parcelport/count/%s/cache-evictions") % pp_type),
              performance_counters::counter_raw,
              boost::str(boost::format("returns the number of cache evictions while accessing "
                  "the connection cache for the %s connection type on the referenced "
                  "locality") % pp_type),
              HPX_PERFORMANCE_COUNTER_V1,
              util::bind(&performance_counters::locality_raw_counter_creator,
                  _1, cache_evictions, _2),
              &performance_counters::locality_counter_discoverer,
              ""
            },
            { boost::str(boost::format("/parcelport/count/%s/cache-hits") % pp_type),
              performance_counters::counter_raw,
              boost::str(boost::format("returns the number of cache hits while accessing "
                  "the connection cache for the %s connection type on the referenced "
                  "locality") % pp_type),
              HPX_PERFORMANCE_COUNTER_V1,
              util::bind(&performance_counters::locality_raw_counter_creator,
                  _1, cache_hits, _2),
              &performance_counters::locality_counter_discoverer,
              ""
            },
            { boost::str(boost::format("/parcelport/count/%s/cache-misses") % pp_type),
              performance_counters::counter_raw,
              boost::str(boost::format("returns the number of cache misses while accessing "
                  "the connection cache for the %s connection type on the referenced "
                  "locality") % pp_type),
              HPX_PERFORMANCE_COUNTER_V1,
              util::bind(&performance_counters::locality_raw_counter_creator,
                  _1, cache_misses, _2),
              &performance_counters::locality_counter_discoverer,
              ""
            },
            { boost::str(boost::format("/parcelport/count/%s/cache-reclaims") % pp_type),
              performance_counters::counter_raw,
              boost::str(boost::format("returns the number of cache reclaims while accessing "
                  "the connection cache for the %s connection type on the referenced "
                  "locality") % pp_type),
              HPX_PERFORMANCE_COUNTER_V1,
              util::bind(&performance_counters::locality_raw_counter_creator,
                  _1, cache_reclaims, _2),
              &performance_counters::locality_counter_discoverer,
              ""
            }
        };
        performance_counters::install_counter_types(connection_cache_types,
            sizeof(connection_cache_types)/sizeof(connection_cache_types[0]));
    }

    std::vector<plugins::parcelport_factory_base *> &
    parcelhandler::get_parcelport_factories()
    {
        static std::vector<plugins::parcelport_factory_base *> factories;
        if(factories.empty())
        {
            init_static_parcelport_factories();
        }

        return factories;
    }

    void parcelhandler::add_parcelport_factory(plugins::parcelport_factory_base *factory)
    {
        get_parcelport_factories().push_back(factory);
    }

    void parcelhandler::init(int *argc, char ***argv, util::command_line_handling &cfg)
    {
        BOOST_FOREACH(plugins::parcelport_factory_base *factory, get_parcelport_factories())
        {
            factory->init(argc, argv, cfg);
        }
    }

    std::vector<std::string> parcelhandler::load_runtime_configuration()
    {
        /// TODO: properly hide this in plugins ...
        std::vector<std::string> ini_defs;

        using namespace boost::assign;
        ini_defs +=
            "[hpx.parcel]",
            "address = ${HPX_PARCEL_SERVER_ADDRESS:" HPX_INITIAL_IP_ADDRESS "}",
            "port = ${HPX_PARCEL_SERVER_PORT:"
                BOOST_PP_STRINGIZE(HPX_INITIAL_IP_PORT) "}",
            "bootstrap = ${HPX_PARCEL_BOOTSTRAP:" HPX_PARCEL_BOOTSTRAP "}",
            "max_connections = ${HPX_PARCEL_MAX_CONNECTIONS:"
                BOOST_PP_STRINGIZE(HPX_PARCEL_MAX_CONNECTIONS) "}",
            "max_connections_per_locality = ${HPX_PARCEL_MAX_CONNECTIONS_PER_LOCALITY:"
                BOOST_PP_STRINGIZE(HPX_PARCEL_MAX_CONNECTIONS_PER_LOCALITY) "}",
            "max_message_size = ${HPX_PARCEL_MAX_MESSAGE_SIZE:"
                BOOST_PP_STRINGIZE(HPX_PARCEL_MAX_MESSAGE_SIZE) "}",
            "max_outbound_message_size = ${HPX_PARCEL_MAX_OUTBOUND_MESSAGE_SIZE:"
                BOOST_PP_STRINGIZE(HPX_PARCEL_MAX_OUTBOUND_MESSAGE_SIZE) "}",
#ifdef BOOST_BIG_ENDIAN
            "endian_out = ${HPX_PARCEL_ENDIAN_OUT:big}",
#else
            "endian_out = ${HPX_PARCEL_ENDIAN_OUT:little}",
#endif
            "array_optimization = ${HPX_PARCEL_ARRAY_OPTIMIZATION:1}",
            "zero_copy_optimization = ${HPX_PARCEL_ZERO_COPY_OPTIMIZATION:"
                "$[hpx.parcel.array_optimization]}",
            "enable_security = ${HPX_PARCEL_ENABLE_SECURITY:0}",
            "async_serialization = ${HPX_PARCEL_ASYNC_SERIALIZATION:1}"
            ;

        BOOST_FOREACH(plugins::parcelport_factory_base *factory, get_parcelport_factories())
        {
            factory->get_plugin_info(ini_defs);
        }

        return ini_defs;
    }
}}

