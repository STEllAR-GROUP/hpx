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

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset
{
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

    void parcelhandler::parcel_sink(parcelport& pp,
        boost::shared_ptr<std::vector<char> > parcel_data,
        threads::thread_priority priority,
        performance_counters::parcels::data_point const& receive_data)
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
        else
        {
            // create a new thread which decodes and handles the parcel
            threads::thread_init_data data(
                boost::bind(&parcelhandler::decode_parcel, this,
                    boost::ref(pp), parcel_data, receive_data),
                "decode_parcel", 0, priority);
            tm_->register_thread(data);
        }
    }

    threads::thread_state_enum parcelhandler::decode_parcel(
        parcelport& pp, boost::shared_ptr<std::vector<char> > parcel_data,
        performance_counters::parcels::data_point receive_data)
    {
        // protect from un-handled exceptions bubbling up into thread manager
        try {
            try {
                // mark start of serialization
                util::high_resolution_timer timer;
                boost::int64_t overall_add_parcel_time = 0;

                {
                    // De-serialize the parcel data
                    util::portable_binary_iarchive archive(*parcel_data,
                        boost::archive::no_header);

                    std::size_t parcel_count = 0;
                    std::size_t arg_size = 0;

                    archive >> parcel_count;
                    for(std::size_t i = 0; i < parcel_count; ++i)
                    {
                        // de-serialize parcel and add it to incoming parcel queue
                        parcel p;
                        archive >> p;

                        // make sure this parcel ended up on the right locality
                        BOOST_ASSERT(p.get_destination_locality() == here());

                        // incoming argument's size
                        arg_size += p.get_action()->get_argument_size();

                        // be sure not to measure add_parcel as serialization time
                        boost::int64_t add_parcel_time = timer.elapsed_nanoseconds();
                        parcels_->add_parcel(p);
                        overall_add_parcel_time += timer.elapsed_nanoseconds() -
                            add_parcel_time;
                    }

                    // complete received data with parcel count
                    receive_data.num_parcels_ = parcel_count;
                    receive_data.argument_bytes_ = arg_size;
                }

                // store the time required for serialization
                receive_data.serialization_time_ = timer.elapsed_nanoseconds() -
                    overall_add_parcel_time;

                pp.add_received_data(receive_data);
            }
            catch (hpx::exception const& e) {
                LPT_(error)
                    << "decode_parcel: caught hpx::exception: "
                    << e.what();
                hpx::report_error(boost::current_exception());
            }
            catch (boost::system::system_error const& e) {
                LPT_(error)
                    << "decode_parcel: caught boost::system::error: "
                    << e.what();
                hpx::report_error(boost::current_exception());
            }
            catch (boost::exception const&) {
                LPT_(error)
                    << "decode_parcel: caught boost::exception.";
                hpx::report_error(boost::current_exception());
            }
            catch (std::exception const& e) {
                // We have to repackage all exceptions thrown by the
                // serialization library as otherwise we will loose the
                // e.what() description of the problem, due to slicing.
                boost::throw_exception(boost::enable_error_info(
                    hpx::exception(serialization_error, e.what())));
            }
        }
        catch (...) {
            // Prevent exceptions from boiling up into the thread-manager.
            LPT_(error)
                << "decode_parcel: caught unknown exception.";
            hpx::report_error(boost::current_exception());
        }

        return threads::terminated;
    }

    parcelhandler::parcelhandler(naming::resolver_client& resolver,
            parcelport& pp, threads::threadmanager_base* tm,
            parcelhandler_queue_base* policy)
      : resolver_(resolver)
      , tm_(tm)
      , parcels_(policy)
    {
        BOOST_ASSERT(parcels_);

        // AGAS v2 registers itself in the client before the parcelhandler
        // is booted.
        locality_ = resolver_.local_locality();

        parcels_->set_parcelhandler(this);

        attach_parcelport(pp);
    }

    // find and return the specified parcelport
    parcelport* parcelhandler::find_parcelport(connection_type type,
        error_code& ec) const
    {
        std::map<connection_type, parcelport*>::const_iterator it = 
            pports_.find(type);
        if (it == pports_.end()) {
            HPX_THROWS_IF(ec, bad_parameter, "parcelhandler::find_parcelport", 
                "unknown parcel port")
            return 0;
        }

        if (&ec != &throws)
            ec = make_success_code();

        return (*it).second;
    }

    void parcelhandler::attach_parcelport(parcelport& pp)
    {
        // register our callback function with the parcelport
        pp.register_event_handler(
            boost::bind(&parcelhandler::parcel_sink, this, _1, _2, _3, _4)
        );

        // add the new parcelport to the list of parcelports we care about
        pports_.insert(std::make_pair(pp.get_type(), &pp));
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

    void parcelhandler::put_parcel(parcel& p, write_handler_type f)
    {
        // properly initialize parcel
        init_parcel(p);

        std::vector<naming::gid_type> const& gids = p.get_destinations();
        std::vector<naming::address>& addrs = p.get_destination_addrs();

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

        find_parcelport(connection_tcpip)->put_parcel(p, f);
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

        HPX_STD_FUNCTION<boost::int64_t()> data_argument_sent(
            boost::bind(&parcelport::get_total_argument_sent, pp));
        HPX_STD_FUNCTION<boost::int64_t()> data_argument_received(
            boost::bind(&parcelport::get_total_argument_received, pp));

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
                  _1, data_argument_sent, _2),
              &performance_counters::locality_counter_discoverer,
              "bytes"
            },
            { "/data/count/received", performance_counters::counter_raw,
              "returns the amount of parcel argument data received "
              "by the referenced locality",
              HPX_PERFORMANCE_COUNTER_V1,
              boost::bind(&performance_counters::locality_raw_counter_creator,
                  _1, data_argument_received, _2),
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

