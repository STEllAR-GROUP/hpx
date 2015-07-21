//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/threads/thread_executor.hpp>
#include <hpx/runtime/threads/resource_manager.hpp>
#include <hpx/lcos/local/once.hpp>
#include <hpx/util/reinitializable_static.hpp>

#include <boost/thread/locks.hpp>

namespace hpx { namespace threads
{
    ///////////////////////////////////////////////////////////////////////////
    resource_manager& resource_manager::get()
    {
        typedef util::reinitializable_static<resource_manager, tag> static_type;

        static_type instance;
        return instance.get();
    }

    ///////////////////////////////////////////////////////////////////////////
    resource_manager::resource_manager()
      : next_cookie_(0),
        punits_(get_os_thread_count()),
        topology_(get_topology())
    {}

    // Request an initial resource allocation
    std::size_t resource_manager::initial_allocation(
        detail::manage_executor* proxy, error_code& ec)
    {
        if (0 == proxy) {
            HPX_THROWS_IF(ec, bad_parameter,
                "resource_manager::init_allocation",
                "manage_executor pointer is a nullptr");
            return std::size_t(-1);
        }

        // ask executor for its policies
        error_code ec1(lightweight);
        std::size_t min_punits = proxy->get_policy_element(detail::min_concurrency, ec1);
        if (ec1) min_punits = 1;
        std::size_t max_punits = proxy->get_policy_element(detail::max_concurrency, ec1);
        if (ec1) max_punits = get_os_thread_count();

        // lock the resource manager from this point on
        boost::lock_guard<mutex_type> l(mtx_);

        // allocate initial resources for the given executor
        std::vector<std::pair<std::size_t, std::size_t> > cores =
            allocate_virt_cores(proxy, min_punits, max_punits, ec);
        if (ec) return std::size_t(-1);

        // attach the given proxy to this resource manager
        std::size_t cookie = ++next_cookie_;
        proxies_.insert(proxies_map_type::value_type(
            cookie, proxy_data(proxy, std::move(cores))));

        if (&ec != &throws)
            ec = make_success_code();
        return cookie;
    }

    // Find 'desired' amount of processing units which have the given use count
    // (use count is the number of schedulers associated with a given processing
    // unit).
    //
    // the resource manager is locked while executing this function
    std::size_t resource_manager::reserve_processing_units(
        std::size_t use_count, std::size_t desired,
        std::vector<BOOST_SCOPED_ENUM(punit_status)>& available_punits)
    {
        std::size_t available = 0;
        for (std::size_t i = 0; i != punits_.size(); ++i)
        {
            if (use_count == punits_[i].use_count_)
            {
                available_punits[i] = punit_status::reserved;
                if (++available == desired)
                    break;
            }
        }
        return available;
    }

    // the resource manager is locked while executing this function
    std::vector<std::pair<std::size_t, std::size_t> >
    resource_manager::allocate_virt_cores(
        detail::manage_executor* proxy, std::size_t min_punits,
        std::size_t max_punits, error_code& ec)
    {
        std::vector<coreids_type> core_ids;

        // array of available processing units
        std::vector<BOOST_SCOPED_ENUM(punit_status)> available_punits(
            get_os_thread_count(), punit_status::unassigned);

        // find all available processing units with zero use count
        std::size_t reserved = reserve_processing_units(0, max_punits,
            available_punits);
        if (reserved < max_punits)
        {
            // insufficient available cores found, try to redistribute
            // processing units
        }

        // processing units found, inform scheduler
        std::size_t punit = 0;
        for (std::size_t i = 0; i != available_punits.size(); ++i)
        {
            if (available_punits[i] == punit_status::reserved) //-V104
            {
                proxy->add_processing_unit(punit, i, ec);
                if (ec) break;

                core_ids.push_back(std::make_pair(i, punit));
                ++punit;

                // update use count for reserved processing units
                ++punits_[i].use_count_;
            }
        }
        HPX_ASSERT(punit <= max_punits);

        if (ec) {
            // on error, remove the already assigned virtual cores
            for (std::size_t j = 0; j != punit; ++j)
            {
                proxy->remove_processing_unit(j, ec);
                --punits_[j].use_count_;
            }
            return std::vector<coreids_type>();
        }

        if (&ec != &throws)
            ec = make_success_code();
        return core_ids;
    }

    // Stop the executor identified with the given cookie
    void resource_manager::stop_executor(std::size_t cookie, error_code& ec)
    {
        boost::lock_guard<mutex_type> l(mtx_);
        proxies_map_type::iterator it = proxies_.find(cookie);
        if (it == proxies_.end()) {
            HPX_THROWS_IF(ec, bad_parameter, "resource_manager::detach",
                "the given cookie is not known to the resource manager");
            return;
        }

        // inform executor to give up virtual cores
        proxy_data& p = (*it).second;
        for (coreids_type coreids : p.core_ids_)
        {
            p.proxy_->remove_processing_unit(coreids.second, ec);
        }
    }

    // Detach the executor identified with the given cookie
    void resource_manager::detach(std::size_t cookie, error_code& ec)
    {
        boost::lock_guard<mutex_type> l(mtx_);
        proxies_map_type::iterator it = proxies_.find(cookie);
        if (it == proxies_.end()) {
            HPX_THROWS_IF(ec, bad_parameter, "resource_manager::detach",
                "the given cookie is not known to the resource manager");
            return;
        }

        // adjust resource usage count
        proxy_data& p = (*it).second;
        for (coreids_type coreids : p.core_ids_)
        {
            HPX_ASSERT(punits_[coreids.first].use_count_ != 0);
            --punits_[coreids.first].use_count_;
        }

        proxies_.erase(cookie);
    }
}}

