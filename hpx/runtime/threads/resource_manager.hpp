//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_THREADS_RESOURCE_MANAGER_JAN_16_2013_0830AM)
#define HPX_RUNTIME_THREADS_RESOURCE_MANAGER_JAN_16_2013_0830AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/move.hpp>
#include <hpx/runtime/threads/thread_executor.hpp>
#include <hpx/lcos/local/spinlock.hpp>

#include <boost/atomic.hpp>
#include <boost/shared_ptr.hpp>

#include <vector>

namespace hpx { namespace  threads
{
    /// In short, there are two main responsibilities of the Resource Manager:
    ///
    /// * Initial Allocation: Allocating resources to executors when executors 
    ///   are created.
    /// * Dynamic Migration: Constantly monitoring utilization of resources 
    ///   by executors, and dynamically migrating resources between them.
    ///
    class resource_manager
    {
        typedef lcos::local::spinlock mutex_type;
        struct tag {};

    public:
        resource_manager();

        // Request an initial resource allocation
        std::size_t initial_allocation(detail::manage_executor* proxy,
            error_code& ec = throws);

        // Detach the executor identified with the given cookie
        void detach(std::size_t cookie, error_code& ec = throws);

        // Return the singleton resource manager instance
        static resource_manager& get();

    private:
        mutable mutex_type mtx_;
        boost::atomic<std::size_t> next_cookie_;

        struct proxy_data
        {
        private:
            BOOST_MOVABLE_BUT_NOT_COPYABLE(proxy_data);

        public:
            proxy_data(detail::manage_executor* proxy, 
                    BOOST_RV_REF(std::vector<std::size_t>) virt_core)
              : proxy_(proxy), virt_cores_(boost::move(virt_core))
            {}

            proxy_data(BOOST_RV_REF(proxy_data) rhs)
              : proxy_(boost::move(rhs.proxy_)),
                virt_cores_(boost::move(rhs.virt_cores_))
            {}

            proxy_data& operator=(BOOST_RV_REF(proxy_data) rhs)
            {
                if (this != &rhs) {
                    proxy_ = boost::move(rhs.proxy_);
                    virt_cores_ = boost::move(rhs.virt_cores_);
                }
                return *this;
            }

            boost::shared_ptr<detail::manage_executor> proxy_;
            std::vector<std::size_t> virt_cores_;
        };

        typedef std::map<std::size_t, proxy_data> proxies_map_type;
        proxies_map_type proxies_;
    };
}}

#endif
