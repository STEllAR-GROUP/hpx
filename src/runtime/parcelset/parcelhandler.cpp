//  Copyright (c) 2007-2012 Hartmut Kaiser
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
        if (!pports_[type]) {
            HPX_THROWS_IF(ec, bad_parameter, "parcelhandler::find_parcelport",
                "cannot find parcelport for connection type " +
                    get_connection_type_name(type));
            return 0;
        }

        if (&ec != &throws)
            ec = make_success_code();

        return pports_[type].get();
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
        components::component_type type) const
    {
        error_code ec(lightweight);
        bool result = resolver_.get_localities(locality_ids, type, ec);
        if (ec || !result) return false;

        return !locality_ids.empty();
    }

    connection_type parcelhandler::find_appropriate_connection_type(
        naming::locality dest)
    {
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

        return dest.get_type();
    }

    // this function  will be called right after pre_main
    void parcelhandler::set_resolved_localities(std::vector<naming::locality> const& l)
    {
        // we use the provided information to decide what types of parcel-ports
        // are needed

        // if there is just one locality, we need no additional network at all
        if (1 == l.size())
            return;

        // if there are more localities sharing the same network node, we need
        // to instantiate the shmem parcel-port
        std::size_t here_count = 0;
        std::string here(here().get_address());
        BOOST_FOREACH(naming::locality const& t, l)
        {
            if (t.get_address() == here)
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

    void parcelhandler::put_parcel(parcel& p, write_handler_type f)
    {
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
    void parcelhandler::register_counter_types(connection_type pp_type)
    {
        parcelport* pp = find_parcelport(pp_type);

        HPX_STD_FUNCTION<boost::int64_t()> num_parcel_sends(
            boost::bind(&parcelport::get_parcel_send_count, pp));
        HPX_STD_FUNCTION<boost::int64_t()> num_parcel_receives(
            boost::bind(&parcelport::get_parcel_receive_count, pp));

        HPX_STD_FUNCTION<boost::int64_t()> num_message_sends(
            boost::bind(&parcelport::get_message_send_count, pp));
        HPX_STD_FUNCTION<boost::int64_t()> num_message_receives(
            boost::bind(&parcelport::get_message_receive_count, pp));

        HPX_STD_FUNCTION<boost::int64_t()> sending_time(
            boost::bind(&parcelport::get_sending_time, pp));
        HPX_STD_FUNCTION<boost::int64_t()> receiving_time(
            boost::bind(&parcelport::get_receiving_time, pp));

        HPX_STD_FUNCTION<boost::int64_t()> sending_serialization_time(
            boost::bind(&parcelport::get_sending_serialization_time, pp));
        HPX_STD_FUNCTION<boost::int64_t()> receiving_serialization_time(
            boost::bind(&parcelport::get_receiving_serialization_time, pp));

        HPX_STD_FUNCTION<boost::int64_t()> data_sent(
            boost::bind(&parcelport::get_data_sent, pp));
        HPX_STD_FUNCTION<boost::int64_t()> data_received(
            boost::bind(&parcelport::get_data_received, pp));

        HPX_STD_FUNCTION<boost::int64_t()> data_type_sent(
            boost::bind(&parcelport::get_total_type_sent, pp));
        HPX_STD_FUNCTION<boost::int64_t()> data_type_received(
            boost::bind(&parcelport::get_total_type_received, pp));

        HPX_STD_FUNCTION<boost::int64_t()> incoming_queue_length(
            boost::bind(&parcelhandler::get_incoming_queue_length, this));
        HPX_STD_FUNCTION<boost::int64_t()> outgoing_queue_length(
            boost::bind(&parcelhandler::get_outgoing_queue_length, this));

        performance_counters::generic_counter_type_data const counter_types[] =
        {
            { "/parcels/count/sent", performance_counters::counter_raw,
              "returns the number of sent parcels for the referenced locality",
              HPX_PERFORMANCE_COUNTER_V1,
              boost::bind(&performance_counters::locality_raw_counter_creator,
                  _1, num_parcel_sends, _2),
              &performance_counters::locality_counter_discoverer,
              ""
            },
            { "/parcels/count/received", performance_counters::counter_raw,
              "returns the number of received parcels for the referenced locality",
              HPX_PERFORMANCE_COUNTER_V1,
              boost::bind(&performance_counters::locality_raw_counter_creator,
                  _1, num_parcel_receives, _2),
              &performance_counters::locality_counter_discoverer,
              ""
            },
            { "/messages/count/sent", performance_counters::counter_raw,
              "returns the number of sent messages for the referenced locality",
              HPX_PERFORMANCE_COUNTER_V1,
              boost::bind(&performance_counters::locality_raw_counter_creator,
                  _1, num_message_sends, _2),
              &performance_counters::locality_counter_discoverer,
              ""
            },
            { "/messages/count/received", performance_counters::counter_raw,
              "returns the number of received messages for the referenced locality",
              HPX_PERFORMANCE_COUNTER_V1,
              boost::bind(&performance_counters::locality_raw_counter_creator,
                  _1, num_message_receives, _2),
              &performance_counters::locality_counter_discoverer,
              ""
            },

            { "/data/time/sent", performance_counters::counter_raw,
              "returns the total time between the start of each asynchronous "
              "write and the invocation of the write callback for the referenced "
              "locality",
              HPX_PERFORMANCE_COUNTER_V1,
              boost::bind(&performance_counters::locality_raw_counter_creator,
                  _1, sending_time, _2),
              &performance_counters::locality_counter_discoverer,
              "ns"
            },
            { "/data/time/received", performance_counters::counter_raw,
              "returns the total time between the start of each asynchronous "
              "read and the invocation of the read callback for the referenced "
              "locality",
              HPX_PERFORMANCE_COUNTER_V1,
              boost::bind(&performance_counters::locality_raw_counter_creator,
                  _1, receiving_time, _2),
              &performance_counters::locality_counter_discoverer,
              "ns"
            },
            { "/serialize/time/sent", performance_counters::counter_raw,
              "returns the total time required to serialize all sent parcels "
              "for the referenced locality",
              HPX_PERFORMANCE_COUNTER_V1,
              boost::bind(&performance_counters::locality_raw_counter_creator,
                  _1, sending_serialization_time, _2),
              &performance_counters::locality_counter_discoverer,
              "ns"
            },
            { "/serialize/time/received", performance_counters::counter_raw,
              "returns the total time required to de-serialize all received "
              "parcels for the referenced locality",
              HPX_PERFORMANCE_COUNTER_V1,
              boost::bind(&performance_counters::locality_raw_counter_creator,
                  _1, receiving_serialization_time, _2),
              &performance_counters::locality_counter_discoverer,
              "ns"
            },

            { "/data/count/sent", performance_counters::counter_raw,
              "returns the amount of parcel argument data sent "
              "by the referenced locality",
              HPX_PERFORMANCE_COUNTER_V1,
              boost::bind(&performance_counters::locality_raw_counter_creator,
                  _1, data_type_sent, _2),
              &performance_counters::locality_counter_discoverer,
              "bytes"
            },
            { "/data/count/received", performance_counters::counter_raw,
              "returns the amount of parcel argument data received "
              "by the referenced locality",
              HPX_PERFORMANCE_COUNTER_V1,
              boost::bind(&performance_counters::locality_raw_counter_creator,
                  _1, data_type_received, _2),
              &performance_counters::locality_counter_discoverer,
              "bytes"
            },
            { "/serialize/count/sent", performance_counters::counter_raw,
              "returns the amount of parcel data (including headers) sent "
              "by the referenced locality",
              HPX_PERFORMANCE_COUNTER_V1,
              boost::bind(&performance_counters::locality_raw_counter_creator,
                  _1, data_sent, _2),
              &performance_counters::locality_counter_discoverer,
              "bytes"
            },
            { "/serialize/count/received", performance_counters::counter_raw,
              "returns the amount of parcel data (including headers) received "
              "by the referenced locality",
              HPX_PERFORMANCE_COUNTER_V1,
              boost::bind(&performance_counters::locality_raw_counter_creator,
                  _1, data_received, _2),
              &performance_counters::locality_counter_discoverer,
              "bytes"
            },

            { "/parcelqueue/length/receive", performance_counters::counter_raw,
              "returns the number current length of the queue of incoming parcels",
              HPX_PERFORMANCE_COUNTER_V1,
              boost::bind(&performance_counters::locality_raw_counter_creator,
                  _1, incoming_queue_length, _2),
              &performance_counters::locality_counter_discoverer,
              ""
            },
            { "/parcelqueue/length/send", performance_counters::counter_raw,
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
}}

