//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_THREADS_RESOURCE_MANAGER_JAN_16_2013_0830AM)
#define HPX_RUNTIME_THREADS_RESOURCE_MANAGER_JAN_16_2013_0830AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/move.hpp>
#include <hpx/runtime/threads/topology.hpp>
#include <hpx/runtime/threads/thread_executor.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <boost/detail/scoped_enum_emulation.hpp>

#include <boost/atomic.hpp>
#include <boost/shared_ptr.hpp>

#include <vector>

namespace hpx { namespace  threads
{
    ///////////////////////////////////////////////////////////////////////////
    /// Status of a given processing unit
    BOOST_SCOPED_ENUM_START(punit_status)
    {
        unassigned = 0,
        reserved = 1,
        assigned = 2
    };
    BOOST_SCOPED_ENUM_END

    /// In short, there are two main responsibilities of the Resource Manager:
    ///
    /// * Initial Allocation: Allocating resources to executors when executors
    ///   are created.
    /// * Dynamic Migration: Constantly monitoring utilization of resources
    ///   by executors, and dynamically migrating resources between them
    ///   (not implemented yet).
    ///
    class resource_manager
    {
        typedef lcos::local::spinlock mutex_type;
        struct tag {};

        // mapping of physical core to virtual core
        typedef std::pair<std::size_t, std::size_t>  coreids_type;

    public:
        resource_manager();

        // Request an initial resource allocation
        std::size_t initial_allocation(detail::manage_executor* proxy,
            error_code& ec = throws);

        // Stop the executor identified by the given cookie
        void stop_executor(std::size_t cookie, error_code& ec = throws);

        // Detach the executor identified by the given cookie
        void detach(std::size_t cookie, error_code& ec = throws);

        // Return the singleton resource manager instance
        static resource_manager& get();

    protected:
        std::vector<coreids_type> allocate_virt_cores(
            detail::manage_executor* proxy, std::size_t min_punits,
            std::size_t max_punits, error_code& ec);

        std::size_t reserve_processing_units(
            std::size_t use_count, std::size_t desired,
            std::vector<BOOST_SCOPED_ENUM(punit_status)>& available_punits);

    private:
        mutable mutex_type mtx_;
        boost::atomic<std::size_t> next_cookie_;

        ///////////////////////////////////////////////////////////////////////
        // Store information about the physical processing units available to
        // this resource manager.
        struct punit_data
        {
            punit_data() : use_count_(0) {}

            std::size_t use_count_;   // number of schedulers using this core
        };

        typedef std::vector<punit_data> punit_array_type;
        punit_array_type punits_;

        ///////////////////////////////////////////////////////////////////////
        // Store information about the virtual processing unit allocation for
        // each of the scheduler proxies attached to this resource manager.
        struct proxy_data
        {
        public:
            proxy_data(detail::manage_executor* proxy,
                    std::vector<coreids_type> && core_ids)
              : proxy_(proxy), core_ids_(std::move(core_ids))
            {}

            proxy_data(proxy_data const& rhs)
              : proxy_(rhs.proxy_),
                core_ids_(rhs.core_ids_)
            {}

            proxy_data(proxy_data && rhs)
              : proxy_(std::move(rhs.proxy_)),
                core_ids_(std::move(rhs.core_ids_))
            {}

            proxy_data& operator=(proxy_data const& rhs)
            {
                if (this != &rhs) {
                    proxy_ = rhs.proxy_;
                    core_ids_ = rhs.core_ids_;
                }
                return *this;
            }

            proxy_data& operator=(proxy_data && rhs)
            {
                if (this != &rhs) {
                    proxy_ = std::move(rhs.proxy_);
                    core_ids_ = std::move(rhs.core_ids_);
                }
                return *this;
            }

            boost::shared_ptr<detail::manage_executor> proxy_;  // hold on to proxy
            std::vector<coreids_type> core_ids_;
            // map physical to logical puinit ids
        };

        typedef std::map<std::size_t, proxy_data> proxies_map_type;
        proxies_map_type proxies_;

        threads::topology const& topology_;
    };
}}

#endif
