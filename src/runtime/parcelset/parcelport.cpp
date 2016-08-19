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

#include <boost/chrono/chrono.hpp>

#include <string>
#include <utility>

namespace hpx { namespace parcelset
{
    ///////////////////////////////////////////////////////////////////////////
    parcelport::parcelport(util::runtime_configuration const& ini, locality const & here,
            std::string const& type)
      : applier_(nullptr),
        here_(here),
        max_inbound_message_size_(ini.get_max_inbound_message_size()),
        max_outbound_message_size_(ini.get_max_outbound_message_size()),
        allow_array_optimizations_(true),
        allow_zero_copy_optimizations_(true),
        enable_security_(false),
        async_serialization_(false),
        priority_(hpx::util::get_entry_as<int>(ini, "hpx.parcel." + type + ".priority",
            "0")),
        type_(type)
    {
        std::string key("hpx.parcel.");
        key += type;

        if (hpx::util::get_entry_as<int>(ini, key + ".array_optimization", "1") == 0) {
            allow_array_optimizations_ = false;
            allow_zero_copy_optimizations_ = false;
        }
        else {
            if (hpx::util::get_entry_as<int>(ini, key + ".zero_copy_optimization",
                "1") == 0)
                allow_zero_copy_optimizations_ = false;
        }

        if(hpx::util::get_entry_as<int>(ini, key + ".enable_security", "0") != 0)
        {
            enable_security_ = true;
        }

        if(hpx::util::get_entry_as<int>(ini, key + ".async_serialization", "0") != 0)
        {
            async_serialization_ = true;
        }
    }

    void parcelport::add_received_parcel(parcel p, std::size_t num_thread)
    {
        // do some work (notify event handlers)
        if(applier_)
        {
            while (threads::threadmanager_is(state_starting))
            {
                boost::this_thread::sleep_for(
                    boost::chrono::milliseconds(HPX_NETWORK_RETRIES_SLEEP));
            }

            // Give up if we're shutting down.
            if (threads::threadmanager_is(state_stopping))
            {
    //             LPT_(debug) << "parcelport: add_received_parcel: dropping late "
    //                             "parcel " << p;
                return;
            }

            // write this parcel to the log
    //         LPT_(debug) << "parcelport: add_received_parcel: " << p;

            applier_->schedule_action(std::move(p));
        }
        // If the applier has not been set yet, we are in bootstrapping and
        // need to execute the action directly
        else
        {
            // TODO: Make assertions exceptions
            // decode the action-type in the parcel
            actions::base_action * act = p.get_action();

            // early parcels should only be plain actions
            HPX_ASSERT(actions::base_action::plain_action == act->get_action_type());

            // early parcels can't have continuations
            HPX_ASSERT(!p.get_continuation());

            // We should not allow any exceptions to escape the execution of the
            // action as this would bring down the ASIO thread we execute in.
            try {
                act->get_thread_function(0)(threads::wait_signaled);
            }
            catch (...) {
                hpx::report_error(boost::current_exception());
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    boost::uint64_t HPX_EXPORT get_max_inbound_size(parcelport& pp)
    {
        return pp.get_max_inbound_message_size();
    }
}}

