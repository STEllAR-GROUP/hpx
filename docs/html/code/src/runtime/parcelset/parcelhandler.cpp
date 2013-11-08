//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c)      2013 Thomas Heller
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
#include <boost/asio/io_service.hpp>
#include <boost/asio/buffer.hpp>
#include <boost/asio/read.hpp>
#include <boost/asio/write.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/ref.hpp>
#include <boost/bind.hpp>
#include <boost/thread.hpp>
#include <boost/thread/condition.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/format.hpp>
#include <boost/foreach.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset
{
    ///////////////////////////////////////////////////////////////////////////
    std::string get_connection_type_name(connection_type t)
    {
        switch(t) {
        case connection_tcpip:
            return "tcpip";

        case connection_shmem:
          return "shmem";

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

    connection_type get_connection_type_from_name(std::string const& t)
    {
        if (!std::strcmp(t.c_str(), "tcpip"))
            return connection_tcpip;

        if (!std::strcmp(t.c_str(), "shmem"))
            return connection_shmem;

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
        std::size_t interval, naming::locality const& loc, connection_type t,
        error_code& ec)
    {
        return ph->get_message_handler(action, type, num, interval, loc, t, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    // A parcel is submitted for transport at the source locality site to
    // the parcel set of the locality with the put-parcel command
    // This function is synchronous.

    struct wait_for_put_parcel
    {
        wait_for_put_parcel() : sema_(new lcos::local::counting_semaphore) {}

        wait_for_put_parcel(wait_for_put_parcel const& other)
            : sema_(other.sema_) {}

        void operator()(boost::system::error_code const&, std::size_t)
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
            threads::threadmanager_base* tm, parcelhandler_queue_base* policy)
      : resolver_(resolver),
        pports_(connection_last),
        tm_(tm),
        parcels_(policy),
        use_alternative_parcelports_(false),
        count_routed_(0)
    {}

    void parcelhandler::initialize(boost::shared_ptr<parcelport> pp)
    {
        BOOST_ASSERT(parcels_);

        // AGAS v2 registers itself in the client before the parcelhandler
        // is booted.
        locality_ = resolver_.get_local_locality();

        parcels_->set_parcelhandler(this);

        attach_parcelport(pp, false);

        util::io_service_pool *pool = 0;
#if defined(HPX_HAVE_PARCELPORT_MPI)
        bool tcpip_bootstrap = (get_config_entry("hpx.parcel.bootstrap", "tcpip") == "tcpip");
        if(tcpip_bootstrap)
        {
            pool = pports_[connection_tcpip]->get_thread_pool("parcel_pool_tcp");
        }
        else
        {
            pool = pports_[connection_mpi]->get_thread_pool("parcel_pool_mpi");
        }
#else
        pool = pports_[connection_tcpip]->get_thread_pool("parcel_pool_tcp");
#endif
        BOOST_ASSERT(0 != pool);


#if defined(HPX_HAVE_PARCELPORT_SHMEM)
        std::string enable_shmem =
            get_config_entry("hpx.parcel.shmem.enable", "0");

        if (boost::lexical_cast<int>(enable_shmem)) {
            attach_parcelport(parcelport::create(
                connection_shmem, hpx::get_config(),
                pool->get_on_start_thread(), pool->get_on_stop_thread()));
        }
#endif
#if defined(HPX_HAVE_PARCELPORT_IBVERBS)
        std::string enable_ibverbs =
            get_config_entry("hpx.parcel.ibverbs.enable", "0");

        if (boost::lexical_cast<int>(enable_ibverbs)) {
            attach_parcelport(parcelport::create(
                connection_ibverbs, hpx::get_config(),
                pool->get_on_start_thread(), pool->get_on_stop_thread()));
        }
#endif
#if defined(HPX_HAVE_PARCELPORT_MPI)
        if(tcpip_bootstrap)
        {
            if (util::mpi_environment::enabled()) {
                attach_parcelport(parcelport::create(
                    connection_mpi, hpx::get_config(),
                    pool->get_on_start_thread(), pool->get_on_stop_thread()));
            }
        }
        else
        {
            std::string enable_tcpip =
                get_config_entry("hpx.parcel.tcpip.enable", "1");
            if (boost::lexical_cast<int>(enable_tcpip)) {
                attach_parcelport(parcelport::create(
                    connection_tcpip, hpx::get_config(),
                    pool->get_on_start_thread(), pool->get_on_stop_thread()));
            }
        }
#endif
    }

    // find and return the specified parcelport
    parcelport* parcelhandler::find_parcelport(connection_type type,
        error_code& ec) const
    {
        if (!pports_[type]) { //-V108
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
    }

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Make sure the specified locality is not held by any
    /// connection caches anymore
    void parcelhandler::remove_from_connection_cache(naming::locality const& loc)
    {
        parcelport* pp = find_parcelport(loc.get_type());
        if (!pp) {
            HPX_THROW_EXCEPTION(network_error,
                "parcelhandler::remove_from_connection_cache",
                "cannot find parcelport for connection type " +
                    get_connection_type_name(loc.get_type()));
            return;
        }
        pp->remove_from_connection_cache(loc);
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

    bool parcelhandler::get_raw_remote_localities(
        std::vector<naming::gid_type>& locality_ids,
        components::component_type type, error_code& ec) const
    {
        std::vector<naming::gid_type> allprefixes;

        bool result = resolver_.get_localities(allprefixes, type, ec);
        if (ec || !result) return false;

        using boost::lambda::_1;
        std::remove_copy_if(allprefixes.begin(), allprefixes.end(),
            std::back_inserter(locality_ids), _1 == locality_);

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

    connection_type parcelhandler::find_appropriate_connection_type(
        naming::locality const& dest)
    {
#if defined(HPX_HAVE_PARCELPORT_SHMEM)
        if (dest.get_type() == connection_tcpip) {
            std::string enable_shmem =
                get_config_entry("hpx.parcel.use_shmem_parcelport", "0");

            // if destination is on the same network node, use shared memory
            // otherwise fall back to tcp
            if (use_alternative_parcelports_ &&
                dest.get_address() == here().get_address() &&
                boost::lexical_cast<int>(enable_shmem))
            {
                if (pports_[connection_shmem])
                    return connection_shmem;
            }
        }
#endif
#if defined(HPX_HAVE_PARCELPORT_IBVERBS)
        // FIXME: add check if ibverbs are really available for this destination.

        if (dest.get_type() == connection_tcpip) {
            std::string enable_ibverbs =
                get_config_entry("hpx.parcel.ibverbs.enable", "0");
            if (use_alternative_parcelports_ &&
                boost::lexical_cast<int>(enable_ibverbs))
            {
                if (pports_[connection_ibverbs])
                    return connection_ibverbs;
            }
        }
#endif
#if defined(HPX_HAVE_PARCELPORT_MPI)
        // FIXME: add check if MPI is really available for this destination.

        if (dest.get_type() == connection_tcpip) {
            if ((use_alternative_parcelports_ ||
                 get_config_entry("hpx.parcel.bootstrap", "tcpip") == "mpi") &&
                 util::mpi_environment::enabled() &&
                 dest.get_rank() != -1)
            {
                if (pports_[connection_mpi])
                    return connection_mpi;
            }
        }
        else if (dest.get_type() == connection_mpi) {
            // fall back to TCP/IP if MPI is disabled
            if (!util::mpi_environment::enabled())
            {
                if (pports_[connection_tcpip])
                    return connection_tcpip;
            }
        }
#endif
        return dest.get_type();
    }

    // this function  will be called right after pre_main
    void parcelhandler::set_resolved_localities(
        std::vector<naming::locality> const& localities)
    {
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
            // report any pending exception
            boost::rethrow_exception(exception);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
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
            resolved_locally = resolver_.resolve(ids[0], addrs[0]);
#else
        std::size_t size = p.size();

        if (0 == size) {
            HPX_THROW_EXCEPTION(network_error, "parcelhandler::put_parcel",
                "no destination address given");
            return;
        }

        if (1 == size) {
            if (!addrs[0])
                resolved_locally = resolver_.resolve(ids[0], addrs[0]);
        }
        else {
            boost::dynamic_bitset<> locals;
            resolved_locally = resolver_.resolve(ids, addrs, size, locals);
        }
#endif

        if (!p.get_parcel_id())
            p.set_parcel_id(parcel::generate_unique_id());

        // If we were able to resolve the address(es) locally we send the
        // parcel directly to the destination.
        if (resolved_locally) {
            // dispatch to the message handler which is associated with the
            // encapsulated action
            connection_type t = find_appropriate_connection_type(addrs[0].locality_);
            policies::message_handler* mh =
                p.get_message_handler(this, addrs[0].locality_, t);

            if (mh) {
                mh->put_parcel(p, f);
                return;
            }

            find_parcelport(t)->put_parcel(p, f);
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
        boost::system::error_code const& ec, std::size_t)
    {
        if (ec) {
            boost::exception_ptr exception =
                hpx::detail::get_exception(hpx::exception(ec),
                    "parcelhandler::default_write_handler", __FILE__, __LINE__);

            // store last error for now only
            mutex_type::scoped_lock l(mtx_);
            exception_ = exception;
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    policies::message_handler* parcelhandler::get_message_handler(
        char const* action, char const* message_handler_type,
        std::size_t num_messages, std::size_t interval,
        naming::locality const& loc, connection_type t, error_code& ec)
    {
        mutex_type::scoped_lock l(handlers_mtx_);
        handler_key_type key(loc, action);
        message_handler_map::iterator it = handlers_.find(key);
        if (it == handlers_.end()) {
            boost::shared_ptr<policies::message_handler> p;

            {
                util::scoped_unlock<mutex_type::scoped_lock> ul(l);
                p.reset(hpx::create_message_handler(message_handler_type,
                    action, find_parcelport(t), num_messages, interval, ec));
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

    bool parcelhandler::supports_connection_cache_statistics(
        connection_type pp_type) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->supports_connection_cache_statistics() : false;
    }

    ///////////////////////////////////////////////////////////////////////////
    void parcelhandler::register_counter_types()
    {
        // register connection specific counters
        register_counter_types(connection_tcpip);
#if defined(HPX_HAVE_PARCELPORT_SHMEM)
        register_counter_types(connection_shmem);
#endif
#if defined(HPX_HAVE_PARCELPORT_IBVERBS)
        register_counter_types(connection_ibverbs);
#endif
#if defined(HPX_HAVE_PARCELPORT_MPI)
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
        // caches, this is not supported by all parcelports
        if (!supports_connection_cache_statistics(pp_type))
            return;

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

