//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c) 2013-2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// hpxinspect:nodeprecatedinclude:boost/chrono/chrono.hpp
// hpxinspect:nodeprecatedname:boost::chrono

// This is needed to make everything work with the Intel MPI library header
#include <hpx/config.hpp>
#include <hpx/state.hpp>
#include <hpx/runtime_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/parcelset/parcelport.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/util/io_service_pool.hpp>
#include <hpx/util/runtime_configuration.hpp>
#include <hpx/util/safe_lexical_cast.hpp>
#include <hpx/exception.hpp>

#include <cstdint>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

namespace hpx { namespace parcelset
{
    ///////////////////////////////////////////////////////////////////////////
    parcelport::parcelport(util::runtime_configuration const& ini,
            locality const & here, std::string const& type)
      : applier_(nullptr),
        here_(here),
        max_inbound_message_size_(ini.get_max_inbound_message_size()),
        max_outbound_message_size_(ini.get_max_outbound_message_size()),
        allow_array_optimizations_(true),
        allow_zero_copy_optimizations_(true),
        enable_security_(false),
        async_serialization_(false),
        priority_(hpx::util::get_entry_as<int>(ini,
            "hpx.parcel." + type + ".priority", "0")),
        type_(type)
    {
        std::string key("hpx.parcel.");
        key += type;

        if (hpx::util::get_entry_as<int>(
                ini, key + ".array_optimization", "1") == 0)
        {
            allow_array_optimizations_ = false;
            allow_zero_copy_optimizations_ = false;
        }
        else
        {
            if (hpx::util::get_entry_as<int>(
                    ini, key + ".zero_copy_optimization", "1") == 0)
            {
                allow_zero_copy_optimizations_ = false;
            }
        }

        if (hpx::util::get_entry_as<int>(
                ini, key + ".enable_security", "0") != 0)
        {
            enable_security_ = true;
        }

        if (hpx::util::get_entry_as<int>(
                ini, key + ".async_serialization", "0") != 0)
        {
            async_serialization_ = true;
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // Update performance counter data
    void parcelport::add_received_data(
        performance_counters::parcels::data_point const& data)
    {
        parcels_received_.add_data(data);
    }

    void parcelport::add_sent_data(
        performance_counters::parcels::data_point const& data)
    {
        parcels_sent_.add_data(data);
    }

#if defined(HPX_HAVE_PARCELPORT_ACTION_COUNTERS)
    void parcelport::add_received_data(char const* action,
        performance_counters::parcels::data_point const& data)
    {
        action_parcels_received_.add_data(action, data);
    }

    void parcelport::add_sent_data(char const* action,
        performance_counters::parcels::data_point const& data)
    {
        action_parcels_sent_.add_data(action, data);
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    // number of parcels sent
    std::int64_t parcelport::get_parcel_send_count(bool reset)
    {
        return parcels_sent_.num_parcels(reset);
    }

    // number of messages sent
    std::int64_t parcelport::get_message_send_count(bool reset)
    {
        return parcels_sent_.num_messages(reset);
    }

    // number of parcels received
    std::int64_t parcelport::get_parcel_receive_count(bool reset)
    {
        return parcels_received_.num_parcels(reset);
    }

    // number of messages received
    std::int64_t parcelport::get_message_receive_count(bool reset)
    {
        return parcels_received_.num_messages(reset);
    }

    // the total time it took for all sends, from async_write to the
    // completion handler (nanoseconds)
    std::int64_t parcelport::get_sending_time(bool reset)
    {
        return parcels_sent_.total_time(reset);
    }

    // the total time it took for all receives, from async_read to the
    // completion handler (nanoseconds)
    std::int64_t parcelport::get_receiving_time(bool reset)
    {
        return parcels_received_.total_time(reset);
    }

    // the total time it took for all sender-side serialization operations
    // (nanoseconds)
    std::int64_t parcelport::get_sending_serialization_time(bool reset)
    {
        return parcels_sent_.total_serialization_time(reset);
    }

    // the total time it took for all receiver-side serialization
    // operations (nanoseconds)
    std::int64_t parcelport::get_receiving_serialization_time(bool reset)
    {
        return parcels_received_.total_serialization_time(reset);
    }

#if defined(HPX_HAVE_SECURITY)
    // the total time it took for all sender-side security operations
    // (nanoseconds)
    std::int64_t parcelport::get_sending_security_time(bool reset)
    {
        return parcels_sent_.total_security_time(reset);
    }

    // the total time it took for all receiver-side security
    // operations (nanoseconds)
    std::int64_t parcelport::get_receiving_security_time(bool reset)
    {
        return parcels_received_.total_security_time(reset);
    }
#endif

    // total data sent (bytes)
    std::int64_t parcelport::get_data_sent(bool reset)
    {
        return parcels_sent_.total_bytes(reset);
    }

    // total data (uncompressed) sent (bytes)
    std::int64_t parcelport::get_raw_data_sent(bool reset)
    {
        return parcels_sent_.total_raw_bytes(reset);
    }

    // total data received (bytes)
    std::int64_t parcelport::get_data_received(bool reset)
    {
        return parcels_received_.total_bytes(reset);
    }

    // total data (uncompressed) received (bytes)
    std::int64_t parcelport::get_raw_data_received(bool reset)
    {
        return parcels_received_.total_raw_bytes(reset);
    }

    std::int64_t parcelport::get_buffer_allocate_time_sent(bool reset)
    {
        return parcels_sent_.total_buffer_allocate_time(reset);
    }

    std::int64_t parcelport::get_buffer_allocate_time_received(bool reset)
    {
        return parcels_received_.total_buffer_allocate_time(reset);
    }

    std::int64_t parcelport::get_pending_parcels_count(bool /*reset*/)
    {
        std::lock_guard<lcos::local::spinlock> l(mtx_);
        std::int64_t count = 0;
        for (auto && p : pending_parcels_)
        {
#if defined(HPX_PARCELSET_PENDING_PARCELS_WORKAROUND)
            count += hpx::util::get<0>(p.second)->size();
            HPX_ASSERT(
                hpx::util::get<0>(p.second)->size() ==
                hpx::util::get<1>(p.second).size());
#else
            count += hpx::util::get<0>(p.second).size();
            HPX_ASSERT(
                hpx::util::get<0>(p.second).size() ==
                hpx::util::get<1>(p.second).size());
#endif
        }
        return count;
    }

    ///////////////////////////////////////////////////////////////////////////
#if defined(HPX_HAVE_PARCELPORT_ACTION_COUNTERS)
    // same as above, just separated data for each action
    // number of parcels sent
    std::int64_t parcelport::get_action_parcel_send_count(
        std::string const& action, bool reset)
    {
        if (action.empty())
            return parcels_sent_.num_parcels(reset);
        return action_parcels_sent_.num_parcels(action, reset);
    }

    // number of parcels received
    std::int64_t parcelport::get_action_parcel_receive_count(
        std::string const& action, bool reset)
    {
        if (action.empty())
            return parcels_received_.num_parcels(reset);
        return action_parcels_received_.num_parcels(action, reset);
    }

    // the total time it took for all sender-side serialization operations
    // (nanoseconds)
    std::int64_t parcelport::get_action_sending_serialization_time(
        std::string const& action, bool reset)
    {
        if (action.empty())
            return parcels_sent_.total_serialization_time(reset);
        return action_parcels_sent_.total_serialization_time(action, reset);
    }

    // the total time it took for all receiver-side serialization
    // operations (nanoseconds)
    std::int64_t parcelport::get_action_receiving_serialization_time(
        std::string const& action, bool reset)
    {
        if (action.empty())
            return parcels_received_.total_serialization_time(reset);
        return action_parcels_received_.total_serialization_time(action, reset);
    }

    // total data sent (bytes)
    std::int64_t parcelport::get_action_data_sent(
        std::string const& action, bool reset)
    {
        if (action.empty())
            return parcels_sent_.total_bytes(reset);
        return action_parcels_sent_.total_bytes(action, reset);
    }

    // total data received (bytes)
    std::int64_t parcelport::get_action_data_received(
        std::string const& action, bool reset)
    {
        if (action.empty())
            return parcels_received_.total_bytes(reset);
        return action_parcels_received_.total_bytes(action, reset);
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    std::int64_t HPX_EXPORT get_max_inbound_size(parcelport& pp)
    {
        return pp.get_max_inbound_message_size();
    }
}}

