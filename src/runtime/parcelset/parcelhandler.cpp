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
#include <hpx/util/mpi_environment.hpp>
#include <hpx/util/stringstream.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>
#include <hpx/runtime/parcelset/parcelhandler.hpp>
#include <hpx/runtime/threads/threadmanager.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/lcos/local/counting_semaphore.hpp>
#include <hpx/include/performance_counters.hpp>
#include <hpx/performance_counters/counter_creators.hpp>

#include <string>
#include <algorithm>

#include <boost/version.hpp>
#include <boost/assign/std/vector.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/asio/io_service.hpp>
#include <boost/asio/buffer.hpp>
#include <boost/asio/read.hpp>
#include <boost/asio/write.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/ref.hpp>
#include <boost/bind.hpp>
#include <boost/thread.hpp>
#include <boost/thread/condition.hpp>
#include <boost/format.hpp>
#include <boost/foreach.hpp>

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
    std::string get_connection_type_name(connection_type t)
    {
        switch(t) {
        case connection_tcp:
            return "tcp";

        case connection_ipc:
          return "ipc";

        case connection_ibverbs:
          return "ibverbs";

        case connection_portals4:
            return "portals4";

        case connection_mpi:
            return "mpi";

        default:
            break;
        }
        return "<unknown>";
    }

    ///////////////////////////////////////////////////////////////////////////
    bool connection_type_available(connection_type t)
    {
        switch(t) {
#if defined(HPX_PARCELPORT_TCP)
        case connection_tcp:
            return true;
#endif
#if defined(HPX_PARCELPORT_IPC)
        case connection_ipc:
            return true;
#endif
#if defined(HPX_PARCELPORT_IBVERBS)
        case connection_ibverbs:
            return true;
#endif
#if defined(HPX_PARCELPORT_PORTALS4)
        case connection_portals4:
            return true;
#endif
#if defined(HPX_PARCELPORT_MPI)
        case connection_mpi:
            return true;
#endif
        default:
            break;
        }
        return false;
    }

    connection_type get_connection_type_from_name(std::string const& t)
    {
        if (!std::strcmp(t.c_str(), "tcp"))
            return connection_tcp;

        if (!std::strcmp(t.c_str(), "ipc"))
            return connection_ipc;

        if (!std::strcmp(t.c_str(), "ibverbs"))
            return connection_ibverbs;

        if (!std::strcmp(t.c_str(), "portals4"))
            return connection_portals4;

        if (!std::strcmp(t.c_str(), "mpi"))
            return connection_mpi;

        return connection_unknown;
    }

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

    struct wait_for_put_parcel
    {
        wait_for_put_parcel() : sema_(new lcos::local::counting_semaphore) {}

        void operator()(boost::system::error_code const&, parcel const&)
        {
            sema_->signal();
        }

        void wait()
        {
            sema_->wait();
        }

        boost::shared_ptr<lcos::local::counting_semaphore> sema_;
    };

    void parcelhandler::sync_put_parcel(parcel& p) //-V669
    {
        wait_for_put_parcel wfp;
        put_parcel(p, wfp);  // schedule parcel send
        wfp.wait();          // wait for the parcel to be sent
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

    parcelhandler::parcelhandler(naming::resolver_client& resolver,
            threads::threadmanager_base* tm, parcelhandler_queue_base* policy,
            HPX_STD_FUNCTION<void(std::size_t, char const*)> const& on_start_thread,
            HPX_STD_FUNCTION<void()> const& on_stop_thread)
      : resolver_(resolver),
        pports_(connection_last),
        endpoints_(connection_last),
        tm_(tm),
        parcels_(policy),
        use_alternative_parcelports_(false),
        enable_parcel_handling_(true),
        count_routed_(0)
    {
#if defined(HPX_PARCELPORT_IPC)
        std::string enable_ipc =
            get_config_entry("hpx.parcel.ipc.enable", "0");

        if (hpx::util::safe_lexical_cast<int>(enable_ipc, 0))
        {
            attach_parcelport(parcelport::create(
                connection_ipc, hpx::get_config(),
                on_start_thread, on_stop_thread), false);
        }
#endif
#if defined(HPX_PARCELPORT_IBVERBS)
        std::string enable_ibverbs =
            get_config_entry("hpx.parcel.ibverbs.enable", "0");

        if (hpx::util::safe_lexical_cast<int>(enable_ibverbs, 0))
        {
            attach_parcelport(parcelport::create(
                connection_ibverbs, hpx::get_config(),
                on_start_thread, on_stop_thread), false);
        }
#endif
#if defined(HPX_PARCELPORT_MPI)
        if (util::mpi_environment::enabled()) {
            attach_parcelport(parcelport::create(
                connection_mpi, hpx::get_config(),
                on_start_thread, on_stop_thread), false);
        }
#endif

#if defined(HPX_PARCELPORT_TCP)
        std::string enable_tcp =
            get_config_entry("hpx.parcel.tcp.enable", "1");
        if (hpx::util::safe_lexical_cast<int>(enable_tcp, 1)) {
            attach_parcelport(parcelport::create(
                connection_tcp, hpx::get_config(),
                on_start_thread, on_stop_thread), false);
        }
#endif
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

        for(int i = 0; i < connection_type::connection_last; ++i)
        {
            std::pair<std::vector<std::string>, bool> pp_ini_defs =
                parcelport::runtime_configuration(i);
            std::string name = get_connection_type_name(connection_type(i));
            std::string name_uc = boost::to_upper_copy(name);
            std::string enable = pp_ini_defs.second ? "1" : "0";

            // Load some defaults
            ini_defs += "[hpx.parcel." + name + "]";
            if (!connection_type_available(connection_type(i)))
            {
                // skip this configuration if the parcelport is not available
                ini_defs += "enable = 0";
            }
            else
            {
                ini_defs +=
                    "enable = ${HPX_PARCELPORT_" + name_uc + ":" + enable + "}",
                    "parcel_pool_size = ${HPX_PARCEL_" + name_uc + "_PARCEL_POOL_SIZE:"
                        "$[hpx.threadpools.parcel_pool_size]}",
                    "max_connections =  ${HPX_PARCEL_" + name_uc + "_MAX_CONNECTIONS:"
                        "$[hpx.parcel.max_connections]}",
                    "max_connections_per_locality = "
                        "${HPX_PARCEL_" + name_uc + "_MAX_CONNECTIONS_PER_LOCALITY:"
                        "$[hpx.parcel.max_connections_per_locality]}",
                    "max_message_size =  ${HPX_PARCEL_" + name_uc +
                        "_MAX_MESSAGE_SIZE:$[hpx.parcel.max_message_size]}",
                    "max_outbound_message_size =  ${HPX_PARCEL_" + name_uc +
                        "_MAX_OUTBOUND_MESSAGE_SIZE:$[hpx.parcel.max_outbound_message_size]}",
                    "array_optimization = ${HPX_PARCEL_" + name_uc +
                        "_ARRAY_OPTIMIZATION:$[hpx.parcel.array_optimization]}",
                    "zero_copy_optimization = ${HPX_PARCEL_" + name_uc +
                        "_ZERO_COPY_OPTIMIZATION:"
                        "$[hpx.parcel.zero_copy_optimization]}",
                    "enable_security = ${HPX_PARCEL_" + name_uc +
                        "_ENABLE_SECURITY:"
                        "$[hpx.parcel.enable_security]}",
                    "async_serialization = ${HPX_PARCEL_" + name_uc +
                        "_ASYNC_SERIALIZATION:"
                        "$[hpx.parcel.async_serialization]}"
                    ;
            }

            // add the pp specific configuration parameter
            ini_defs.insert(ini_defs.end(),
                pp_ini_defs.first.begin(), pp_ini_defs.first.end());
        }

        return ini_defs;
    }

    boost::shared_ptr<parcelport> parcelhandler::get_bootstrap_parcelport() const
    {
        std::string pptype = get_config_entry("hpx.parcel.bootstrap", "tcp");

        int type = get_connection_type_from_name(pptype);
        if (type == connection_unknown)
        {
#if defined(HPX_PARCELPORT_MPI)
            if (util::mpi_environment::enabled())
                type = connection_mpi;
            else
#endif
            type = connection_tcp;
        }

        return pports_[type];
    }


    void parcelhandler::initialize()
    {
        HPX_ASSERT(parcels_);

        parcels_->set_parcelhandler(this);
        for(int i = 0; i < connection_type::connection_last; ++i)
        {
            if(pports_[i])
            {
                if(pports_[i] != get_bootstrap_parcelport())
                    pports_[i]->run(false);
                else
                {
                    pports_[i]->register_event_handler(boost::bind(&parcelhandler::parcel_sink, this, _1));
                }
            }
        }
    }

    void parcelhandler::list_parcelport(util::osstream& strm, connection_type t,
        bool available)
    {
        std::string ppname = get_connection_type_name(t);
        strm << "parcel port: " << ppname
             << " (" << (available ? "" : "not ") << "available)";

        if (available)
        {
            std::string cfgkey("hpx.parcel." + ppname + ".enable");
            std::string enabled = get_config_entry(cfgkey, "0");
            strm << ", " << (hpx::util::safe_lexical_cast<int>(enabled, 0) ? "" : "not ")
                 << "enabled";

            std::string bootstrap = get_config_entry("hpx.parcel.bootstrap", "tcp");
            if (bootstrap == ppname)
                strm << ", bootstrap";
        }

        strm << '\n';
    }

    // list available parcel ports
    void parcelhandler::list_parcelports(util::osstream& strm)
    {
        list_parcelport(strm, connection_tcp);

#if defined(HPX_PARCELPORT_IPC)
        list_parcelport(strm, connection_ipc);
#else
        list_parcelport(strm, connection_ipc, false);
#endif
// #if defined(HPX_PARCELPORT_PORTALS4)
//         list_parcelport(strm, connection_portals4);
// #else
//         list_parcelport(strm, connection_portals4, false);
// #endif
#if defined(HPX_PARCELPORT_IBVERBS)
        list_parcelport(strm, connection_ibverbs);
#else
        list_parcelport(strm, connection_ibverbs, false);
#endif
#if defined(HPX_PARCELPORT_MPI)
        list_parcelport(strm, connection_mpi);
#else
        list_parcelport(strm, connection_mpi, false);
#endif
    }

    // find and return the specified parcelport
    parcelport* parcelhandler::find_parcelport(connection_type type,
        error_code& ec) const
    {
        if (HPX_UNLIKELY(!pports_[type])) { //-V108
            HPX_THROWS_IF(ec, bad_parameter, "parcelhandler::find_parcelport",
                "cannot find parcelport for connection type " +
                    get_connection_type_name(type));
            return 0;
        }

        if (&ec != &throws)
            ec = make_success_code();

        return pports_[type].get(); //-V108
    }

    void parcelhandler::attach_parcelport(boost::shared_ptr<parcelport> const& pp,
        bool run)
    {
        // register our callback function with the parcelport
        pp->register_event_handler(boost::bind(&parcelhandler::parcel_sink, this, _1));

        // start the parcelport's thread pool
        if (run) pp->run(false);

        // add the new parcelport to the list of parcel-ports we care about
        pports_[pp->get_type()] = pp;

        // add the endpoint of the new parcelport
        HPX_ASSERT(pp->get_type() == pp->here().get_type());
        endpoints_[pp->get_type()] = pp->here();
    }

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Make sure the specified locality is not held by any
    /// connection caches anymore
    void parcelhandler::remove_from_connection_cache(endpoints_type const& endpoints)
    {
        BOOST_FOREACH(locality const & loc, endpoints)
        {
            boost::shared_ptr<parcelport> pp = pports_[loc.get_type()];
            if (!pp) {
                HPX_THROW_EXCEPTION(network_error,
                    "parcelhandler::remove_from_connection_cache",
                    "cannot find parcelport for connection type " +
                        get_connection_type_name(loc.get_type()));
                return;
            }
            pp->remove_from_connection_cache(loc);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    void parcelhandler::do_background_work(bool stop_buffering)
    {
        // flush all parcel buffers
        {
            mutex_type::scoped_lock l(handlers_mtx_);

            message_handler_map::iterator end = handlers_.end();
            for (message_handler_map::iterator it = handlers_.begin(); it != end; ++it)
            {
                if ((*it).second)
                {
                    boost::shared_ptr<policies::message_handler> p((*it).second);
                    util::scoped_unlock<mutex_type::scoped_lock> ul(l);
                    p->flush(stop_buffering);
                }
            }
        }

        // make sure all pending parcels are being handled
        BOOST_FOREACH(boost::shared_ptr<parcelport> pp, pports_)
        {
            if (pp) pp->do_background_work();
        }
    }

    void parcelhandler::stop(bool blocking)
    {
        // now stop all parcel ports
        BOOST_FOREACH(boost::shared_ptr<parcelport> pp, pports_)
        {
            if (pp) pp->stop(blocking);
        }
    }

    naming::resolver_client& parcelhandler::get_resolver()
    {
        return resolver_;
    }

    naming::gid_type const& parcelhandler::get_locality() const
    {
        return resolver_.get_local_locality();
    }

    bool parcelhandler::get_raw_remote_localities(
        std::vector<naming::gid_type>& locality_ids,
        components::component_type type, error_code& ec) const
    {
        std::vector<naming::gid_type> allprefixes;

        bool result = resolver_.get_localities(allprefixes, type, ec);
        if (ec || !result) return false;

        std::remove_copy(allprefixes.begin(), allprefixes.end(),
            std::back_inserter(locality_ids), get_locality());

        return !locality_ids.empty();
    }

    bool parcelhandler::get_raw_localities(
        std::vector<naming::gid_type>& locality_ids,
        components::component_type type, error_code& ec) const
    {
        bool result = resolver_.get_localities(locality_ids, type, ec);
        if (ec || !result) return false;

        return !locality_ids.empty();
    }

    locality parcelhandler::find_appropriate_destination(
        naming::gid_type const& dest_gid)
    {
        mutex_type::scoped_lock l(resolved_endpoints_mtx_);
        resolved_endpoints_type::iterator lit = resolved_endpoints_.find(dest_gid);

        if(lit == resolved_endpoints_.end())
        {
            hpx::util::osstream oss;
            oss << "The locality gid cannot be resolved to a valid endpoint.\n"
                << "Got locality " << dest_gid << ". Available endpoints:\n";
            BOOST_FOREACH(resolved_endpoints_type::value_type const & endpoints, resolved_endpoints_)
            {
                oss << "    " << endpoints.first << ": " << endpoints.second << "\n";
            }
            HPX_THROW_EXCEPTION(network_error, "parcelhandler::find_appropriate_destination",
                hpx::util::osstream_get_string(oss));
            return locality();
        }
        endpoints_type const & dest_endpoints = lit->second;

#if defined(HPX_PARCELPORT_IPC)
        std::string enable_ipc =
            get_config_entry("hpx.parcel.ipc.enable", "0");
        if(use_alternative_parcelports_ && hpx::util::safe_lexical_cast<int>(enable_ipc, 0))
        {
            // Find ipc parcelport endpoints ...
            locality here = find_endpoint(endpoints_, connection_ipc);
            locality dest = find_endpoint(dest_endpoints, connection_ipc);
            if(here == dest && pports_[connection_ipc])
            {
                return dest;
            }
        }
#endif
#if defined(HPX_PARCELPORT_IBVERBS)
        // FIXME: add check if ibverbs is really available for this destination.

        std::string enable_ibverbs =
            get_config_entry("hpx.parcel.ibverbs.enable", "0");
        if (use_alternative_parcelports_ && hpx::util::safe_lexical_cast<int>(enable_ibverbs, 0))
        {
            // Find ibverbs parcelport endpoints ...
            locality dest = find_endpoint(dest_endpoints, connection_ibverbs);
            if(dest && pports_[connection_ibverbs])
            {
                return dest;
            }
        }
#endif
#if defined(HPX_PARCELPORT_MPI)
        // FIXME: add check if MPI is really available for this destination.

        if ((use_alternative_parcelports_ ||
             get_config_entry("hpx.parcel.bootstrap", "tcp") == "mpi") &&
             util::mpi_environment::enabled())
        {
            // Find MPI parcelport endpoints ...
            locality dest = find_endpoint(dest_endpoints, connection_mpi);
            if(dest && pports_[connection_mpi])
            {
                return dest;
            }
        }
#endif
#if defined(HPX_PARCELPORT_TCP)
        // FIXME: add check if tcp is really available for this destination.

        std::string enable_tcp =
            get_config_entry("hpx.parcel.tcp.enable", "0");
        if (hpx::util::safe_lexical_cast<int>(enable_tcp, 0))
        {
            // Find ibverbs parcelport endpoints ...
            locality dest = find_endpoint(dest_endpoints, connection_tcp);
            if(dest && pports_[connection_tcp])
            {
                return dest;
            }
        }
#endif

        HPX_THROW_EXCEPTION(network_error, "parcelhandler::find_appropriate_destination",
            "The locality gid cannot be resolved to a valid endpoint. No valid parcelport configured.");
        return locality();
    }

    locality parcelhandler::find_endpoint(endpoints_type const & eps, connection_type type)
    {
        locality res;
        BOOST_FOREACH(locality const & loc, eps)
        {
            if(loc.get_type() == type)
            {
                res = loc;
                break;
            }
        }
        return res;
    }

    // this function  will be called right after pre_main
    void parcelhandler::set_resolved_localities(
        std::map<naming::gid_type, endpoints_type> const& localities)
    {
        mutex_type::scoped_lock l(resolved_endpoints_mtx_);
        if(resolved_endpoints_.empty())
        {
            resolved_endpoints_ = localities;
            return;
        }
        BOOST_FOREACH(resolved_endpoints_type::value_type const & resolved, localities)
        {
            resolved_endpoints_[resolved.first] = resolved.second;
        }
    }

    /// Return the reference to an existing io_service
    util::io_service_pool* parcelhandler::get_thread_pool(char const* name)
    {
        util::io_service_pool* result = 0;
        BOOST_FOREACH(boost::shared_ptr<parcelport> pp, pports_)
        {
            if (pp) {
                result = pp->get_thread_pool(name);
                if (result) return result;
            }
        }
        return result;
    }

    void parcelhandler::rethrow_exception()
    {
        boost::exception_ptr exception;
        {
            // store last error for now only
            mutex_type::scoped_lock l(mtx_);
            boost::swap(exception, exception_);
        }

        if (exception) {
            // report any pending exceptions
            boost::rethrow_exception(exception);
        }
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
        rethrow_exception();

        // properly initialize parcel
        init_parcel(p);

        naming::id_type const* ids = p.get_destinations();
        naming::address* addrs = p.get_destination_addrs();

        bool resolved_locally = true;

#if !defined(HPX_SUPPORT_MULTIPLE_PARCEL_DESTINATIONS)
        if (!addrs[0])
        {
            resolved_locally = resolver_.resolve_local(ids[0], addrs[0]);
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
                resolved_locally = resolver_.resolve_local(ids[0], addrs[0]);
        }
        else {
            boost::dynamic_bitset<> locals;
            resolved_locally = resolver_.resolve_local(ids, addrs, size, locals);
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
            parcelset::locality dest = find_appropriate_destination(addrs[0].locality_);
            policies::message_handler* mh =
                p.get_message_handler(this, dest);

            if (mh) {
                mh->put_parcel(dest, p, wrapped_f);
                return;
            }

            find_parcelport(dest.get_type())->put_parcel(dest, p, wrapped_f);
            return;
        }

        // At least one of the addresses is locally unknown, route the parcel
        // to the AGAS managing the destination.
        ++count_routed_;
        resolver_.route(p, f);
    }

    std::size_t parcelhandler::get_outgoing_queue_length(bool reset) const
    {
        std::size_t parcel_count = 0;
        BOOST_FOREACH(boost::shared_ptr<parcelport> pp, pports_)
        {
            if (pp) parcel_count += pp->get_pending_parcels_count(reset);
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
                    ec == boost::asio::error::broken_pipe ||
                    ec == boost::asio::error::not_connected ||
                    ec == boost::asio::error::eof)
                {
                    return;
                }
            }

            boost::exception_ptr exception =
                hpx::detail::get_exception(hpx::exception(ec),
                    "parcelhandler::default_write_handler", __FILE__,
                    __LINE__, parcelset::dump_parcel(p));

            // store last error for now only
            mutex_type::scoped_lock l(mtx_);
            exception_ = exception;
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
                    action, find_parcelport(loc.get_type()), num_messages, interval, ec));
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
        BOOST_FOREACH(boost::shared_ptr<parcelport> pp, pports_)
        {
            if (pp) return pp->get_locality_name();
        }
        return "<unknown>";
    }

    ///////////////////////////////////////////////////////////////////////////
    bool parcelhandler::enable(bool new_state)
    {
        new_state = enable_parcel_handling_.exchange(new_state, boost::memory_order_acquire);
        BOOST_FOREACH(boost::shared_ptr<parcelport> pp, pports_)
        {
            if (pp) pp->enable(enable_parcel_handling_);
        }

        return new_state;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Performance counter data

    // number of parcels sent
    std::size_t parcelhandler::get_parcel_send_count(
        connection_type pp_type, bool reset) const
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
    std::size_t parcelhandler::get_message_send_count(
        connection_type pp_type, bool reset) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_message_send_count(reset) : 0;
    }

    // number of parcels received
    std::size_t parcelhandler::get_parcel_receive_count(
        connection_type pp_type, bool reset) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_parcel_receive_count(reset) : 0;
    }

    // number of messages received
    std::size_t parcelhandler::get_message_receive_count(
        connection_type pp_type, bool reset) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_message_receive_count(reset) : 0;
    }

    // the total time it took for all sends, from async_write to the
    // completion handler (nanoseconds)
    boost::int64_t parcelhandler::get_sending_time(
        connection_type pp_type, bool reset) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_sending_time(reset) : 0;
    }

    // the total time it took for all receives, from async_read to the
    // completion handler (nanoseconds)
    boost::int64_t parcelhandler::get_receiving_time(
        connection_type pp_type, bool reset) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_receiving_time(reset) : 0;
    }

    // the total time it took for all sender-side serialization operations
    // (nanoseconds)
    boost::int64_t parcelhandler::get_sending_serialization_time(
        connection_type pp_type, bool reset) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_sending_serialization_time(reset) : 0;
    }

    // the total time it took for all receiver-side serialization
    // operations (nanoseconds)
    boost::int64_t parcelhandler::get_receiving_serialization_time(
        connection_type pp_type, bool reset) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_receiving_serialization_time(reset) : 0;
    }

#if defined(HPX_HAVE_SECURITY)
    // the total time it took for all sender-side security operations
    // (nanoseconds)
    boost::int64_t parcelhandler::get_sending_security_time(
        connection_type pp_type, bool reset) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_sending_security_time(reset) : 0;
    }

    // the total time it took for all receiver-side security
    // operations (nanoseconds)
    boost::int64_t parcelhandler::get_receiving_security_time(
        connection_type pp_type, bool reset) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_receiving_security_time(reset) : 0;
    }
#endif

    // total data sent (bytes)
    std::size_t parcelhandler::get_data_sent(connection_type pp_type,
        bool reset) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_data_sent(reset) : 0;
    }

    // total data (uncompressed) sent (bytes)
    std::size_t parcelhandler::get_raw_data_sent(connection_type pp_type,
        bool reset) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_raw_data_sent(reset) : 0;
    }

    // total data received (bytes)
    std::size_t parcelhandler::get_data_received(connection_type pp_type,
        bool reset) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_data_received(reset) : 0;
    }

    // total data (uncompressed) received (bytes)
    std::size_t parcelhandler::get_raw_data_received(connection_type pp_type,
        bool reset) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_raw_data_received(reset) : 0;
    }

    boost::int64_t parcelhandler::get_buffer_allocate_time_sent(
        connection_type pp_type, bool reset) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_buffer_allocate_time_sent(reset) : 0;
    }
    boost::int64_t parcelhandler::get_buffer_allocate_time_received(
        connection_type pp_type, bool reset) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_buffer_allocate_time_received(reset) : 0;
    }

    // connection stack statistics
    boost::int64_t parcelhandler::get_connection_cache_statistics(
        connection_type pp_type,
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
        register_counter_types(connection_tcp);
#if defined(HPX_PARCELPORT_IPC)
        register_counter_types(connection_ipc);
#endif
#if defined(HPX_PARCELPORT_IBVERBS)
        register_counter_types(connection_ibverbs);
#endif
#if defined(HPX_PARCELPORT_MPI)
        register_counter_types(connection_mpi);
#endif

        // register common counters
        HPX_STD_FUNCTION<boost::int64_t(bool)> incoming_queue_length(
            boost::bind(&parcelhandler::get_incoming_queue_length, this, ::_1));
        HPX_STD_FUNCTION<boost::int64_t(bool)> outgoing_queue_length(
            boost::bind(&parcelhandler::get_outgoing_queue_length, this, ::_1));
        HPX_STD_FUNCTION<boost::int64_t(bool)> outgoing_routed_count(
            boost::bind(&parcelhandler::get_parcel_routed_count, this, ::_1));

        performance_counters::generic_counter_type_data const counter_types[] =
        {
            { "/parcelqueue/length/receive",
              performance_counters::counter_raw,
              "returns the number current length of the queue of incoming parcels",
              HPX_PERFORMANCE_COUNTER_V1,
              boost::bind(&performance_counters::locality_raw_counter_creator,
                  _1, incoming_queue_length, _2),
              &performance_counters::locality_counter_discoverer,
              ""
            },
            { "/parcelqueue/length/send",
              performance_counters::counter_raw,
              "returns the number current length of the queue of outgoing parcels",
              HPX_PERFORMANCE_COUNTER_V1,
              boost::bind(&performance_counters::locality_raw_counter_creator,
                  _1, outgoing_queue_length, _2),
              &performance_counters::locality_counter_discoverer,
              ""
            },
            { "/parcels/count/routed",
              performance_counters::counter_raw,
              "returns the number of (outbound) parcel routed through the "
                  "responsible AGAS service",
              HPX_PERFORMANCE_COUNTER_V1,
              boost::bind(&performance_counters::locality_raw_counter_creator,
                  _1, outgoing_routed_count, _2),
              &performance_counters::locality_counter_discoverer,
              ""
            }
        };
        performance_counters::install_counter_types(
            counter_types, sizeof(counter_types)/sizeof(counter_types[0]));
    }

    void parcelhandler::register_counter_types(connection_type pp_type)
    {
        HPX_STD_FUNCTION<boost::int64_t(bool)> num_parcel_sends(
            boost::bind(&parcelhandler::get_parcel_send_count, this, pp_type, ::_1));
        HPX_STD_FUNCTION<boost::int64_t(bool)> num_parcel_receives(
            boost::bind(&parcelhandler::get_parcel_receive_count, this, pp_type, ::_1));

        HPX_STD_FUNCTION<boost::int64_t(bool)> num_message_sends(
            boost::bind(&parcelhandler::get_message_send_count, this, pp_type, ::_1));
        HPX_STD_FUNCTION<boost::int64_t(bool)> num_message_receives(
            boost::bind(&parcelhandler::get_message_receive_count, this, pp_type, ::_1));

        HPX_STD_FUNCTION<boost::int64_t(bool)> sending_time(
            boost::bind(&parcelhandler::get_sending_time, this, pp_type, ::_1));
        HPX_STD_FUNCTION<boost::int64_t(bool)> receiving_time(
            boost::bind(&parcelhandler::get_receiving_time, this, pp_type, ::_1));

        HPX_STD_FUNCTION<boost::int64_t(bool)> sending_serialization_time(
            boost::bind(&parcelhandler::get_sending_serialization_time, this, pp_type, ::_1));
        HPX_STD_FUNCTION<boost::int64_t(bool)> receiving_serialization_time(
            boost::bind(&parcelhandler::get_receiving_serialization_time, this, pp_type, ::_1));

#if defined(HPX_HAVE_SECURITY)
        HPX_STD_FUNCTION<boost::int64_t(bool)> sending_security_time(
            boost::bind(&parcelhandler::get_sending_security_time, this, pp_type, ::_1));
        HPX_STD_FUNCTION<boost::int64_t(bool)> receiving_security_time(
            boost::bind(&parcelhandler::get_receiving_security_time, this, pp_type, ::_1));
#endif
        HPX_STD_FUNCTION<boost::int64_t(bool)> data_sent(
            boost::bind(&parcelhandler::get_data_sent, this, pp_type, ::_1));
        HPX_STD_FUNCTION<boost::int64_t(bool)> data_received(
            boost::bind(&parcelhandler::get_data_received, this, pp_type, ::_1));

        HPX_STD_FUNCTION<boost::int64_t(bool)> data_raw_sent(
            boost::bind(&parcelhandler::get_raw_data_sent, this, pp_type, ::_1));
        HPX_STD_FUNCTION<boost::int64_t(bool)> data_raw_received(
            boost::bind(&parcelhandler::get_raw_data_received, this, pp_type, ::_1));

        HPX_STD_FUNCTION<boost::int64_t(bool)> buffer_allocate_time_sent(
            boost::bind(&parcelhandler::get_buffer_allocate_time_sent, this, pp_type, ::_1));
        HPX_STD_FUNCTION<boost::int64_t(bool)> buffer_allocate_time_received(
            boost::bind(&parcelhandler::get_buffer_allocate_time_received, this, pp_type, ::_1));

        std::string connection_type_name(get_connection_type_name(pp_type));
        performance_counters::generic_counter_type_data const counter_types[] =
        {
            { boost::str(boost::format("/parcels/count/%s/sent") % connection_type_name),
              performance_counters::counter_raw,
              boost::str(boost::format("returns the number of parcels sent using the %s "
                  "connection type for the referenced locality") % connection_type_name),
              HPX_PERFORMANCE_COUNTER_V1,
              boost::bind(&performance_counters::locality_raw_counter_creator,
                  _1, num_parcel_sends, _2),
              &performance_counters::locality_counter_discoverer,
              ""
            },
            { boost::str(boost::format("/parcels/count/%s/received") % connection_type_name),
               performance_counters::counter_raw,
              boost::str(boost::format("returns the number of parcels received using the %s "
                  "connection type for the referenced locality") % connection_type_name),
              HPX_PERFORMANCE_COUNTER_V1,
              boost::bind(&performance_counters::locality_raw_counter_creator,
                  _1, num_parcel_receives, _2),
              &performance_counters::locality_counter_discoverer,
              ""
            },
            { boost::str(boost::format("/messages/count/%s/sent") % connection_type_name),
              performance_counters::counter_raw,
              boost::str(boost::format("returns the number of messages sent using the %s "
                  "connection type for the referenced locality") % connection_type_name),
              HPX_PERFORMANCE_COUNTER_V1,
              boost::bind(&performance_counters::locality_raw_counter_creator,
                  _1, num_message_sends, _2),
              &performance_counters::locality_counter_discoverer,
              ""
            },
            { boost::str(boost::format("/messages/count/%s/received") % connection_type_name),
              performance_counters::counter_raw,
              boost::str(boost::format("returns the number of messages received using the %s "
                  "connection type for the referenced locality") % connection_type_name),
              HPX_PERFORMANCE_COUNTER_V1,
              boost::bind(&performance_counters::locality_raw_counter_creator,
                  _1, num_message_receives, _2),
              &performance_counters::locality_counter_discoverer,
              ""
            },

            { boost::str(boost::format("/data/time/%s/sent") % connection_type_name),
              performance_counters::counter_raw,
              boost::str(boost::format("returns the total time between the start of "
                  "each asynchronous write and the invocation of the write callback "
                  "using the %s connection type for the referenced locality") %
                      connection_type_name),
              HPX_PERFORMANCE_COUNTER_V1,
              boost::bind(&performance_counters::locality_raw_counter_creator,
                  _1, sending_time, _2),
              &performance_counters::locality_counter_discoverer,
              "ns"
            },
            { boost::str(boost::format("/data/time/%s/received") % connection_type_name),
              performance_counters::counter_raw,
              boost::str(boost::format("returns the total time between the start of "
                  "each asynchronous read and the invocation of the read callback "
                  "using the %s connection type for the referenced locality") %
                      connection_type_name),
              HPX_PERFORMANCE_COUNTER_V1,
              boost::bind(&performance_counters::locality_raw_counter_creator,
                  _1, receiving_time, _2),
              &performance_counters::locality_counter_discoverer,
              "ns"
            },
            { boost::str(boost::format("/serialize/time/%s/sent") % connection_type_name),
              performance_counters::counter_raw,
              boost::str(boost::format("returns the total time required to serialize "
                  "all sent parcels using the %s connection type for the referenced "
                  "locality") % connection_type_name),
              HPX_PERFORMANCE_COUNTER_V1,
              boost::bind(&performance_counters::locality_raw_counter_creator,
                  _1, sending_serialization_time, _2),
              &performance_counters::locality_counter_discoverer,
              "ns"
            },
            { boost::str(boost::format("/serialize/time/%s/received") % connection_type_name),
              performance_counters::counter_raw,
              boost::str(boost::format("returns the total time required to de-serialize "
                  "all received parcels using the %s connection type for the referenced "
                  "locality") % connection_type_name),
              HPX_PERFORMANCE_COUNTER_V1,
              boost::bind(&performance_counters::locality_raw_counter_creator,
                  _1, receiving_serialization_time, _2),
              &performance_counters::locality_counter_discoverer,
              "ns"
            },

#if defined(HPX_HAVE_SECURITY)
            { boost::str(boost::format("/security/time/%s/sent") % connection_type_name),
              performance_counters::counter_raw,
              boost::str(boost::format("returns the total time required to perform "
                  "tasks related to security in the parcel layer for all sent parcels "
                  "using the %s connection type for the referenced locality") %
                        connection_type_name),
              HPX_PERFORMANCE_COUNTER_V1,
              boost::bind(&performance_counters::locality_raw_counter_creator,
                  _1, sending_security_time, _2),
              &performance_counters::locality_counter_discoverer,
              "ns"
            },
            { boost::str(boost::format("/security/time/%s/received") % connection_type_name),
              performance_counters::counter_raw,
              boost::str(boost::format("returns the total time required to perform "
                  "tasks related to security in the parcel layer for all received parcels "
                  "using the %s connection type for the referenced locality") %
                        connection_type_name),
              HPX_PERFORMANCE_COUNTER_V1,
              boost::bind(&performance_counters::locality_raw_counter_creator,
                  _1, receiving_security_time, _2),
              &performance_counters::locality_counter_discoverer,
              "ns"
            },
#endif
            { boost::str(boost::format("/data/count/%s/sent") % connection_type_name),
              performance_counters::counter_raw,
              boost::str(boost::format("returns the amount of (uncompressed) parcel "
                  "argument data sent using the %s connection type by the referenced "
                  "locality") % connection_type_name),
              HPX_PERFORMANCE_COUNTER_V1,
              boost::bind(&performance_counters::locality_raw_counter_creator,
                  _1, data_raw_sent, _2),
              &performance_counters::locality_counter_discoverer,
              "bytes"
            },
            { boost::str(boost::format("/data/count/%s/received") % connection_type_name),
              performance_counters::counter_raw,
              boost::str(boost::format("returns the amount of (uncompressed) parcel "
                  "argument data received using the %s connection type by the referenced "
                  "locality") % connection_type_name),
              HPX_PERFORMANCE_COUNTER_V1,
              boost::bind(&performance_counters::locality_raw_counter_creator,
                  _1, data_raw_received, _2),
              &performance_counters::locality_counter_discoverer,
              "bytes"
            },
            { boost::str(boost::format("/serialize/count/%s/sent") % connection_type_name),
              performance_counters::counter_raw,
              boost::str(boost::format("returns the amount of parcel data (including "
                  "headers, possibly compressed) sent using the %s connection type "
                  "by the referenced locality") % connection_type_name),
              HPX_PERFORMANCE_COUNTER_V1,
              boost::bind(&performance_counters::locality_raw_counter_creator,
                  _1, data_sent, _2),
              &performance_counters::locality_counter_discoverer,
              "bytes"
            },
            { boost::str(boost::format("/serialize/count/%s/received") % connection_type_name),
              performance_counters::counter_raw,
              boost::str(boost::format("returns the amount of parcel data (including "
                  "headers, possibly compressed) received using the %s connection type "
                  "by the referenced locality") % connection_type_name),
              HPX_PERFORMANCE_COUNTER_V1,
              boost::bind(&performance_counters::locality_raw_counter_creator,
                  _1, data_received, _2),
              &performance_counters::locality_counter_discoverer,
              "bytes"
            },
            { boost::str(boost::format("/parcels/time/%s/buffer_allocate/received") % connection_type_name),
              performance_counters::counter_raw,
              boost::str(boost::format("returns the time needed to allocate the buffers for serializing using the %s connection type") % connection_type_name),
              HPX_PERFORMANCE_COUNTER_V1,
              boost::bind(&performance_counters::locality_raw_counter_creator,
                  _1, buffer_allocate_time_received, _2),
              &performance_counters::locality_counter_discoverer,
              "ns"
            },
            { boost::str(boost::format("/parcels/time/%s/buffer_allocate/sent") % connection_type_name),
              performance_counters::counter_raw,
              boost::str(boost::format("returns the time needed to allocate the buffers for serializing using the %s connection type") % connection_type_name),
              HPX_PERFORMANCE_COUNTER_V1,
              boost::bind(&performance_counters::locality_raw_counter_creator,
                  _1, buffer_allocate_time_sent, _2),
              &performance_counters::locality_counter_discoverer,
              "ns"
            },
        };
        performance_counters::install_counter_types(
            counter_types, sizeof(counter_types)/sizeof(counter_types[0]));

        // register connection specific performance counters related to connection
        // caches
        HPX_STD_FUNCTION<boost::int64_t(bool)> cache_insertions(
            boost::bind(&parcelhandler::get_connection_cache_statistics,
                this, pp_type, parcelport::connection_cache_insertions, ::_1));
        HPX_STD_FUNCTION<boost::int64_t(bool)> cache_evictions(
            boost::bind(&parcelhandler::get_connection_cache_statistics,
                this, pp_type, parcelport::connection_cache_evictions, ::_1));
        HPX_STD_FUNCTION<boost::int64_t(bool)> cache_hits(
            boost::bind(&parcelhandler::get_connection_cache_statistics,
                this, pp_type, parcelport::connection_cache_hits, ::_1));
        HPX_STD_FUNCTION<boost::int64_t(bool)> cache_misses(
            boost::bind(&parcelhandler::get_connection_cache_statistics,
                this, pp_type, parcelport::connection_cache_misses, ::_1));
        HPX_STD_FUNCTION<boost::int64_t(bool)> cache_reclaims(
            boost::bind(&parcelhandler::get_connection_cache_statistics,
                this, pp_type, parcelport::connection_cache_reclaims, ::_1));

        performance_counters::generic_counter_type_data const connection_cache_types[] =
        {
            { boost::str(boost::format("/parcelport/count/%s/cache-insertions") % connection_type_name),
              performance_counters::counter_raw,
              boost::str(boost::format("returns the number of cache insertions while accessing "
                  "the connection cache for the %s connection type on the referenced "
                  "locality") % connection_type_name),
              HPX_PERFORMANCE_COUNTER_V1,
              boost::bind(&performance_counters::locality_raw_counter_creator,
                  _1, cache_insertions, _2),
              &performance_counters::locality_counter_discoverer,
              ""
            },
            { boost::str(boost::format("/parcelport/count/%s/cache-evictions") % connection_type_name),
              performance_counters::counter_raw,
              boost::str(boost::format("returns the number of cache evictions while accessing "
                  "the connection cache for the %s connection type on the referenced "
                  "locality") % connection_type_name),
              HPX_PERFORMANCE_COUNTER_V1,
              boost::bind(&performance_counters::locality_raw_counter_creator,
                  _1, cache_evictions, _2),
              &performance_counters::locality_counter_discoverer,
              ""
            },
            { boost::str(boost::format("/parcelport/count/%s/cache-hits") % connection_type_name),
              performance_counters::counter_raw,
              boost::str(boost::format("returns the number of cache hits while accessing "
                  "the connection cache for the %s connection type on the referenced "
                  "locality") % connection_type_name),
              HPX_PERFORMANCE_COUNTER_V1,
              boost::bind(&performance_counters::locality_raw_counter_creator,
                  _1, cache_hits, _2),
              &performance_counters::locality_counter_discoverer,
              ""
            },
            { boost::str(boost::format("/parcelport/count/%s/cache-misses") % connection_type_name),
              performance_counters::counter_raw,
              boost::str(boost::format("returns the number of cache misses while accessing "
                  "the connection cache for the %s connection type on the referenced "
                  "locality") % connection_type_name),
              HPX_PERFORMANCE_COUNTER_V1,
              boost::bind(&performance_counters::locality_raw_counter_creator,
                  _1, cache_misses, _2),
              &performance_counters::locality_counter_discoverer,
              ""
            },
            { boost::str(boost::format("/parcelport/count/%s/cache-reclaims") % connection_type_name),
              performance_counters::counter_raw,
              boost::str(boost::format("returns the number of cache reclaims while accessing "
                  "the connection cache for the %s connection type on the referenced "
                  "locality") % connection_type_name),
              HPX_PERFORMANCE_COUNTER_V1,
              boost::bind(&performance_counters::locality_raw_counter_creator,
                  _1, cache_reclaims, _2),
              &performance_counters::locality_counter_discoverer,
              ""
            }
        };
        performance_counters::install_counter_types(connection_cache_types,
            sizeof(connection_cache_types)/sizeof(connection_cache_types[0]));
    }
}}

