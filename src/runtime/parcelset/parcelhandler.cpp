//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2007      Richard D Guidry Jr
//  Copyright (c) 2011      Bryce Lelbach & Katelyn Kufahl
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/state.hpp>
#include <hpx/exception.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
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

        case connection_portals4:
            return "portals4";

        default:
            break;
        }
        return "<unknown>";
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

    void parcelhandler::sync_put_parcel(parcel& p)
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
            boost::shared_ptr<parcelport> pp, threads::threadmanager_base* tm,
            parcelhandler_queue_base* policy)
      : resolver_(resolver),
        pports_(connection_last),
        tm_(tm),
        parcels_(policy),
        use_alternative_parcelports_(false)
    {
        BOOST_ASSERT(parcels_);

        // AGAS v2 registers itself in the client before the parcelhandler
        // is booted.
        locality_ = resolver_.local_locality();

        parcels_->set_parcelhandler(this);

        attach_parcelport(pp, false);
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

    void parcelhandler::attach_parcelport(boost::shared_ptr<parcelport> pp,
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
    void parcelhandler::stop(bool blocking)
    {
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
        components::component_type type) const
    {
        std::vector<naming::gid_type> allprefixes;
        error_code ec(lightweight);
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
        naming::locality dest)
    {
#if defined(HPX_USE_SHMEM_PARCELPORT)
        if (dest.get_type() == connection_tcpip) {
            // if destination is on the same network node, use shared memory
            // otherwise fall back to tcp
            if (use_alternative_parcelports_ && dest.get_address() == here().get_address())
            {
                if (pports_[connection_shmem])
                    return connection_shmem;
            }
            return connection_tcpip;
        }
#endif
        return dest.get_type();
    }

    // this function  will be called right after pre_main
    void parcelhandler::set_resolved_localities(
        std::vector<naming::locality> const& localities)
    {
#if defined(HPX_USE_SHMEM_PARCELPORT)
        std::string enable_shmem =
            get_config_entry("hpx.parcel.use_shmem_parcelport", "0");

        if (boost::lexical_cast<int>(enable_shmem)) {
            // we use the provided information to decide what types of parcel-ports
            // are needed

            // if there is just one locality, we need no additional network at all
            if (1 == localities.size())
                return;

            // if there are more localities sharing the same network node, we need
            // to instantiate the shmem parcel-port
            std::size_t here_count = 0;
            std::string here_(here().get_address());
            BOOST_FOREACH(naming::locality const& t, localities)
            {
                if (t.get_address() == here_)
                    ++here_count;
            }

            if (here_count > 1) {
                util::io_service_pool* pool =
                    pports_[connection_tcpip]->get_thread_pool("parcel_pool_tcp");
                BOOST_ASSERT(0 != pool);

                attach_parcelport(parcelport::create(
                    connection_shmem, hpx::get_config(),
                    pool->get_on_start_thread(), pool->get_on_stop_thread()));
            }
        }
#endif
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
    void parcelhandler::put_parcel(parcel& p, write_handler_type f)
    {
        rethrow_exception();

        // properly initialize parcel
        init_parcel(p);

        std::vector<naming::gid_type> const& gids = p.get_destinations();
        std::vector<naming::address>& addrs = p.get_destination_addrs();

        if (gids.empty()) {
            HPX_THROW_EXCEPTION(network_error, "parcelhandler::put_parcel",
                "no destination address given");
            return;
        }

        if (gids.size() != addrs.size()) {
            HPX_THROW_EXCEPTION(network_error, "parcelhandler::put_parcel",
                "inconsistent number of destination addresses");
            return;
        }

        if (1 == gids.size()) {
            if (!addrs[0])
                resolver_.resolve(gids[0], addrs[0]);
        }
        else {
            boost::dynamic_bitset<> locals;
            resolver_.resolve(gids, addrs, locals);
        }

        if (!p.get_parcel_id())
            p.set_parcel_id(parcel::generate_unique_id());

        // determine which parcelport to use for sending this parcel
        connection_type t = find_appropriate_connection_type(addrs[0].locality_);

        // send the parcel using the parcelport corresponding to the
        // locality type of the destination
        find_parcelport(t)->put_parcel(p, f);
    }

    std::size_t parcelhandler::get_outgoing_queue_length() const
    {
        std::size_t parcel_count = 0;
        BOOST_FOREACH(boost::shared_ptr<parcelport> pp, pports_)
        {
            if (pp) parcel_count += pp->get_pending_parcels_count();
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
    // Performance counter data

    // number of parcels sent
    std::size_t parcelhandler::get_parcel_send_count(connection_type pp_type) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_parcel_send_count() : 0;
    }

    // number of messages sent
    std::size_t parcelhandler::get_message_send_count(connection_type pp_type) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_message_send_count() : 0;
    }

    // number of parcels received
    std::size_t parcelhandler::get_parcel_receive_count(connection_type pp_type) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_parcel_receive_count() : 0;
    }

    // number of messages received
    std::size_t parcelhandler::get_message_receive_count(connection_type pp_type) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_message_receive_count() : 0;
    }

    // the total time it took for all sends, from async_write to the
    // completion handler (nanoseconds)
    boost::int64_t parcelhandler::get_sending_time(connection_type pp_type) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_sending_time() : 0;
    }

    // the total time it took for all receives, from async_read to the
    // completion handler (nanoseconds)
    boost::int64_t parcelhandler::get_receiving_time(connection_type pp_type) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_receiving_time() : 0;
    }

    // the total time it took for all sender-side serialization operations
    // (nanoseconds)
    boost::int64_t parcelhandler::get_sending_serialization_time(connection_type pp_type) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_sending_serialization_time() : 0;
    }

    // the total time it took for all receiver-side serialization
    // operations (nanoseconds)
    boost::int64_t parcelhandler::get_receiving_serialization_time(connection_type pp_type) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_receiving_serialization_time() : 0;
    }

    // total data sent (bytes)
    std::size_t parcelhandler::get_data_sent(connection_type pp_type) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_data_sent() : 0;
    }

    // total data (uncompressed) sent (bytes)
    std::size_t parcelhandler::get_raw_data_sent(connection_type pp_type) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_raw_data_sent() : 0;
    }

    // total data received (bytes)
    std::size_t parcelhandler::get_data_received(connection_type pp_type) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_data_received() : 0;
    }

    // total data (uncompressed) received (bytes)
    std::size_t parcelhandler::get_raw_data_received(connection_type pp_type) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_raw_data_received() : 0;
    }

    // connection stack statistics
    boost::int64_t parcelhandler::get_connection_cache_statistics(
        connection_type pp_type, parcelport::connection_cache_statistics_type stat_type) const
    {
        error_code ec(lightweight);
        parcelport* pp = find_parcelport(pp_type, ec);
        return pp ? pp->get_connection_cache_statistics(stat_type) : 0;
    }

    ///////////////////////////////////////////////////////////////////////////
    void parcelhandler::register_counter_types()
    {
        // register connection specific counters
        register_counter_types(connection_tcpip);
#if defined(HPX_USE_SHMEM_PARCELPORT)
        register_counter_types(connection_shmem);
#endif

        // register common counters
        HPX_STD_FUNCTION<boost::int64_t()> incoming_queue_length(
            boost::bind(&parcelhandler::get_incoming_queue_length, this));
        HPX_STD_FUNCTION<boost::int64_t()> outgoing_queue_length(
            boost::bind(&parcelhandler::get_outgoing_queue_length, this));

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
            }
        };
        performance_counters::install_counter_types(
            counter_types, sizeof(counter_types)/sizeof(counter_types[0]));
    }

    void parcelhandler::register_counter_types(connection_type pp_type)
    {
        HPX_STD_FUNCTION<boost::int64_t()> num_parcel_sends(
            boost::bind(&parcelhandler::get_parcel_send_count, this, pp_type));
        HPX_STD_FUNCTION<boost::int64_t()> num_parcel_receives(
            boost::bind(&parcelhandler::get_parcel_receive_count, this, pp_type));

        HPX_STD_FUNCTION<boost::int64_t()> num_message_sends(
            boost::bind(&parcelhandler::get_message_send_count, this, pp_type));
        HPX_STD_FUNCTION<boost::int64_t()> num_message_receives(
            boost::bind(&parcelhandler::get_message_receive_count, this, pp_type));

        HPX_STD_FUNCTION<boost::int64_t()> sending_time(
            boost::bind(&parcelhandler::get_sending_time, this, pp_type));
        HPX_STD_FUNCTION<boost::int64_t()> receiving_time(
            boost::bind(&parcelhandler::get_receiving_time, this, pp_type));

        HPX_STD_FUNCTION<boost::int64_t()> sending_serialization_time(
            boost::bind(&parcelhandler::get_sending_serialization_time, this, pp_type));
        HPX_STD_FUNCTION<boost::int64_t()> receiving_serialization_time(
            boost::bind(&parcelhandler::get_receiving_serialization_time, this, pp_type));

        HPX_STD_FUNCTION<boost::int64_t()> data_sent(
            boost::bind(&parcelhandler::get_data_sent, this, pp_type));
        HPX_STD_FUNCTION<boost::int64_t()> data_received(
            boost::bind(&parcelhandler::get_data_received, this, pp_type));

        HPX_STD_FUNCTION<boost::int64_t()> data_raw_sent(
            boost::bind(&parcelhandler::get_raw_data_sent, this, pp_type));
        HPX_STD_FUNCTION<boost::int64_t()> data_raw_received(
            boost::bind(&parcelhandler::get_raw_data_received, this, pp_type));

        HPX_STD_FUNCTION<boost::int64_t()> cache_insertions(
            boost::bind(&parcelhandler::get_connection_cache_statistics,
                this, pp_type, parcelport::connection_cache_insertions));
        HPX_STD_FUNCTION<boost::int64_t()> cache_evictions(
            boost::bind(&parcelhandler::get_connection_cache_statistics,
                this, pp_type, parcelport::connection_cache_evictions));
        HPX_STD_FUNCTION<boost::int64_t()> cache_hits(
            boost::bind(&parcelhandler::get_connection_cache_statistics,
                this, pp_type, parcelport::connection_cache_hits));
        HPX_STD_FUNCTION<boost::int64_t()> cache_misses(
            boost::bind(&parcelhandler::get_connection_cache_statistics,
                this, pp_type, parcelport::connection_cache_misses));

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

            { boost::str(boost::format("/parcelport/count/%s/cache-insertions") % connection_type_name),
              performance_counters::counter_raw,
              boost::str(boost::format("returns the number of cache insertions while accessing "
                  "the connection cache for the %s connection type on the referenced "
                  "locality") % connection_type_name),
              HPX_PERFORMANCE_COUNTER_V1,
              boost::bind(&performance_counters::locality_raw_counter_creator,
                  _1, cache_insertions, _2),
              &performance_counters::locality_counter_discoverer,
              "bytes"
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
              "bytes"
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
              "bytes"
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
              "bytes"
            }
        };
        performance_counters::install_counter_types(
            counter_types, sizeof(counter_types)/sizeof(counter_types[0]));
    }
}}

