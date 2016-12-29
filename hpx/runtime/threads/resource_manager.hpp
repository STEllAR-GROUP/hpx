//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c) 2015 Nidhi Makhijani
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_THREADS_RESOURCE_MANAGER_JAN_16_2013_0830AM)
#define HPX_RUNTIME_THREADS_RESOURCE_MANAGER_JAN_16_2013_0830AM

#include <hpx/config.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/runtime/threads/thread_executor.hpp>
#include <hpx/runtime/threads/topology.hpp>

#include <boost/atomic.hpp>

#include <cstddef>
#include <map>
#include <memory>
#include <utility>
#include <vector>

namespace hpx { namespace  threads
{
    ///////////////////////////////////////////////////////////////////////////
    /// Status of a given processing unit
    enum class punit_status
    {
        unassigned = 0,
        reserved = 1,
        assigned = 2
    };

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
        // allocate virtual cores
        // called by initial_allocation
        std::vector<coreids_type> allocate_virt_cores(
                detail::manage_executor* proxy, std::size_t min_punits,
                std::size_t max_punits, error_code& ec);

        // reserve virtual cores for scheduler
        std::size_t reserve_processing_units(
                std::size_t use_count, std::size_t desired,
                std::vector<punit_status>& available_punits);

        // reserve virtual cores for scheduler at higher use count
        std::size_t reserve_at_higher_use_count(
                std::size_t desired,
                std::vector<punit_status>& available_punits);

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

        threads::topology const& topology_;

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

            // hold on to proxy
            std::shared_ptr<detail::manage_executor> proxy_;

            // map physical to logical puinit ids
            std::vector<coreids_type> core_ids_;
        };

        typedef std::map<std::size_t, proxy_data> proxies_map_type;
        proxies_map_type proxies_;

        ///////////////////////////////////////////////////////////////////////
        // Used to store information during static and dynamic allocation.
        struct allocation_data
        {
            allocation_data()
              : allocation_(0),
                scaled_allocation_(0.0),
                num_borrowed_cores_(0),
                num_owned_cores_(0),
                min_proxy_cores_(0),
                max_proxy_cores_(0)
            {}

            // The scheduler proxy this allocation data is for.
            std::shared_ptr<detail::manage_executor> proxy_;  // hold on to proxy

            // Additional allocation to give to a scheduler after proportional
            // allocation decisions are made.
            std::size_t allocation_;

            // Scaled allocation value during proportional allocation.
            double scaled_allocation_;

            std::size_t num_borrowed_cores_; // borrowed cores held by scheduler
            std::size_t num_owned_cores_;    // owned cores held by scheduler
            std::size_t min_proxy_cores_;    // min cores required by scheduler
            std::size_t max_proxy_cores_;    // max cores required by scheduler
        };

        struct static_allocation_data : public allocation_data
        {
            static_allocation_data()
              : adjusted_desired_(0),
                num_cores_stolen_(0)
            {}

            // A field used during static allocation to decide on an allocation
            // proportional to each scheduler's desired value.
            double adjusted_desired_;

            // Keeps track of stolen cores during static allocation.
            std::size_t num_cores_stolen_;
        };

        typedef std::map<std::size_t, static_allocation_data>
            allocation_data_map_type;
        allocation_data_map_type proxies_static_allocation_data;

        // stores static allocation data for all schedulers
        std::size_t preprocess_static_allocation(std::size_t min_punits,
            std::size_t max_punits);

        // constants used for parameters to the release_scheduler API
        static std::size_t const release_borrowed_cores = std::size_t(-1);
        static std::size_t const release_cores_to_min = std::size_t(-2);

        // release cores from scheduler
        bool release_scheduler_resources(
            allocation_data_map_type::iterator it,
            std::size_t number_to_free,
            std::vector<punit_status>& available_punits);

        // release cores from all schedulers
        // calls release_scheduler_resources
        std::size_t release_cores_on_existing_schedulers(
            std::size_t number_to_free,
            std::vector<punit_status>& available_punits,
            std::size_t new_allocation);

        // distribute cores to schedulers proportional to max_punits of
        // the schedulers
        std::size_t redistribute_cores_among_all(std::size_t reserved,
            std::size_t min_punits, std::size_t max_punits,
            std::vector<punit_status>& available_punits,
            std::size_t new_allocation);

        void roundup_scaled_allocations(
            allocation_data_map_type &scaled_static_allocation_data,
            std::size_t total_allocated);
    };
}}

#endif
