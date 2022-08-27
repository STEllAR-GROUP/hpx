//  Copyright (c) 2015-2016 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

// config
#include <hpx/config.hpp>
#include <hpx/parcelport_libfabric/config/defines.hpp>
// util
#include <hpx/modules/command_line_handling.hpp>
#include <hpx/modules/format.hpp>
#include <hpx/modules/runtime_configuration.hpp>
#include <hpx/modules/runtime_local.hpp>
#include <hpx/modules/threading_base.hpp>
#include <hpx/modules/timing.hpp>
#include <hpx/plugin/traits/plugin_config_data.hpp>

// The memory pool specialization need to be pulled in before encode_parcels
#include <hpx/parcelset/parcel_buffer.hpp>
#include <hpx/parcelset/parcelport_impl.hpp>
#include <hpx/parcelset_base/parcelport.hpp>
#include <hpx/plugin_factories/parcelport_factory.hpp>
//

// --------------------------------------------------------------------
// Controls whether we are allowed to suspend threads that are sending
// when we have maxed out the number of sends we can handle
#define HPX_PARCELPORT_LIBFABRIC_SUSPEND_WAKE                                  \
    (HPX_PARCELPORT_LIBFABRIC_THROTTLE_SENDS / 2)

// --------------------------------------------------------------------
// Enable the use of hpx small_vector for certain short lived storage
// elements within the parcelport. This can reduce some memory allocations
#define HPX_PARCELPORT_LIBFABRIC_USE_SMALL_VECTOR true

#define HPX_PARCELPORT_LIBFABRIC_IMM_UNSUPPORTED 1

// --------------------------------------------------------------------
#include <hpx/parcelport_libfabric/connection_handler.hpp>
#include <hpx/parcelport_libfabric/header.hpp>
#include <hpx/parcelport_libfabric/libfabric_controller.hpp>
#include <hpx/parcelport_libfabric/libfabric_region_provider.hpp>
#include <hpx/parcelport_libfabric/locality.hpp>
#include <hpx/parcelport_libfabric/parcelport_logging.hpp>
#include <hpx/parcelport_libfabric/performance_counter.hpp>
#include <hpx/parcelport_libfabric/rdma_locks.hpp>
#include <hpx/parcelport_libfabric/rma_memory_pool.hpp>
#include <hpx/parcelport_libfabric/sender.hpp>
#include <hpx/parcelport_libfabric/unordered_map.hpp>

//
#if HPX_PARCELPORT_LIBFABRIC_USE_SMALL_VECTOR
#include <hpx/datastructures/detail/small_vector.hpp>
#endif
//
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace hpx::parcelset::policies;

namespace hpx { namespace parcelset { namespace policies { namespace libfabric {
    // --------------------------------------------------------------------
    // parcelport, the implementation of the parcelport itself
    // --------------------------------------------------------------------
    struct HPX_EXPORT parcelport : public parcelport_impl<parcelport>
    {
    private:
        typedef parcelport_impl<parcelport> base_type;

    public:
        // These are the types used in the parcelport for locking etc
        // Note that spinlock is the only supported mutex that works on HPX+OS threads
        // and condition_variable_any can be used across HPX/OS threads
        typedef hpx::lcos::local::spinlock mutex_type;
        typedef hpx::parcelset::policies::libfabric::scoped_lock<mutex_type>
            scoped_lock;
        typedef hpx::parcelset::policies::libfabric::unique_lock<mutex_type>
            unique_lock;
        typedef rma_memory_region<libfabric_region_provider> region_type;
        typedef memory_region_allocator<libfabric_region_provider>
            allocator_type;

        // --------------------------------------------------------------------
        // main vars used to manage the RDMA controller and interface
        // These are called from a static function, so use static
        // --------------------------------------------------------------------
        libfabric_controller_ptr libfabric_controller_;

        // our local ip address (estimated based on fabric PP address info)
        uint32_t ip_addr_;

        // Not currently working, we support bootstrapping, but when not enabled
        // we should be able to skip it
        bool bootstrap_enabled_;
        bool parcelport_enabled_;

        // @TODO, clean up the allocators, buffers, chunk_pool etc so that there is a
        // more consistent reuse of classes/types.
        // The use of pointer allocators etc is a dreadful hack and needs reworking

        typedef header<HPX_PARCELPORT_LIBFABRIC_MESSAGE_HEADER_SIZE>
            header_type;
        static constexpr unsigned int header_size =
            header_type::header_block_size;
        typedef rma_memory_pool<libfabric_region_provider> memory_pool_type;
        typedef pinned_memory_vector<char, header_size, region_type,
            memory_pool_type>
            snd_data_type;
        typedef parcel_buffer<snd_data_type> snd_buffer_type;
        // when terminating the parcelport, this is used to restrict access
        mutex_type stop_mutex;

        boost::lockfree::stack<sender*,
            boost::lockfree::capacity<HPX_PARCELPORT_LIBFABRIC_THROTTLE_SENDS>,
            boost::lockfree::fixed_sized<true>>
            senders_;

        // Used to help with shutdown
        std::atomic<bool> stopped_;

        memory_pool_type* chunk_pool_;

        // parcelset::gatherer& parcels_sent_;

        // for debugging/performance measurement
        performance_counter<unsigned int> completions_handled_;
        performance_counter<unsigned int> senders_in_use_;

        // --------------------------------------------------------------------
        // Constructor : mostly just initializes the superclass with 'here'
        // --------------------------------------------------------------------
        parcelport(util::runtime_configuration const& ini,
            threads::policies::callback_notifier const& notifier);

        // Start the handling of connections.
        bool do_run();

        // --------------------------------------------------------------------
        // return a sender object back to the parcelport_impl
        // this is used by the send_immediate version of parcelport_impl
        // --------------------------------------------------------------------
        sender* get_connection(
            parcelset::locality const& dest, fi_addr_t& fi_addr);

        void reclaim_connection(sender* s);

        // --------------------------------------------------------------------
        // return a sender object back to the parcelport_impl
        // this is for compatibility with non send_immediate operation
        // --------------------------------------------------------------------
        std::shared_ptr<sender> create_connection(
            parcelset::locality const& dest, error_code& ec);

        ~parcelport();

        /// Should not be used any more as parcelport_impl handles this?
        bool can_bootstrap() const;

        /// Return the name of this locality
        std::string get_locality_name() const;

        parcelset::locality agas_locality(
            util::runtime_configuration const& ini) const;

        parcelset::locality create_locality() const;

        static void suspended_task_debug(const std::string& match);

        void do_stop();

        // --------------------------------------------------------------------
        bool can_send_immediate();

        // --------------------------------------------------------------------
        template <typename Handler>
        bool async_write(Handler&& handler, sender* sender, fi_addr_t addr,
            snd_buffer_type& buffer);

        // --------------------------------------------------------------------
        // This is called to poll for completions and handle all incoming messages
        // as well as complete outgoing messages.
        // --------------------------------------------------------------------
        // Background work
        //
        // This is called whenever the main thread scheduler is idling,
        // is used to poll for events, messages on the libfabric connection
        // --------------------------------------------------------------------
        bool background_work(
            std::size_t num_thread, parcelport_background_mode mode);
        void io_service_work();
        bool background_work_OS_thread();
    };
}}}}    // namespace hpx::parcelset::policies::libfabric

namespace hpx { namespace traits {
    // Inject additional configuration data into the factory registry for this
    // type. This information ends up in the system wide configuration database
    // under the plugin specific section:
    //
    //      [hpx.parcel.libfabric]
    //      ...
    //      priority = 100
    //
    template <>
    struct plugin_config_data<hpx::parcelset::policies::libfabric::parcelport>
    {
        static char const* priority()
        {
            FUNC_START_DEBUG_MSG;
            static int log_init = false;
            if (!log_init)
            {
#if defined(HPX_PARCELPORT_LIBFABRIC_HAVE_LOGGING) ||                          \
    defined(HPX_PARCELPORT_LIBFABRIC_HAVE_DEV_MODE)
                boost::log::add_console_log(std::clog,
                    // This makes the sink to write log records that look like this:
                    // 1: <normal> A normal severity message
                    // 2: <error> An error severity message
                    boost::log::keywords::format =
                        (boost::log::expressions::stream
                            << boost::log::expressions::attr<unsigned int>(
                                   "LineID")
                            << ": <" << boost::log::trivial::severity << "> "
                            << boost::log::expressions::smessage));
                boost::log::add_common_attributes();
#endif
                log_init = true;
            }
            FUNC_END_DEBUG_MSG;
            return "10000";
        }

        // This is used to initialize your parcelport,
        // for example check for availability of devices etc.
        static void init(int*, char***, util::command_line_handling&)
        {
            FUNC_START_DEBUG_MSG;
#ifdef HPX_PARCELPORT_LIBFABRIC_HAVE_PMI
            cfg.ini_config_.push_back("hpx.parcel.bootstrap!=libfabric");
#endif

            FUNC_END_DEBUG_MSG;
        }

        static void destroy() {}

        static char const* call()
        {
            FUNC_START_DEBUG_MSG;
            FUNC_END_DEBUG_MSG;
            // @TODO : check which of these are obsolete after recent changes
            return "provider = "
                   "${HPX_PARCELPORT_LIBFABRIC_"
                   "PROVIDER:" HPX_PARCELPORT_LIBFABRIC_PROVIDER "}\n"
                   "domain = "
                   "${HPX_PARCELPORT_LIBFABRIC_"
                   "DOMAIN:" HPX_PARCELPORT_LIBFABRIC_DOMAIN "}\n"
                   "endpoint = "
                   "${HPX_PARCELPORT_LIBFABRIC_"
                   "ENDPOINT:" HPX_PARCELPORT_LIBFABRIC_ENDPOINT "}\n";
        }
    };
}}    // namespace hpx::traits
