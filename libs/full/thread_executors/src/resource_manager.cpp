//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c) 2015 Nidhi Makhijani
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_THREAD_EXECUTORS_COMPATIBILITY)
#include <hpx/assert.hpp>
#include <hpx/execution/detail/execution_parameter_callbacks.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/topology.hpp>
#include <hpx/static_reinit/reinitializable_static.hpp>
#include <hpx/thread_executors/resource_manager.hpp>
#include <hpx/thread_executors/thread_executor.hpp>
#include <hpx/threading_base/thread_helpers.hpp>
#include <hpx/type_support/unused.hpp>

#include <algorithm>
#include <cstddef>
#include <mutex>
#include <utility>
#include <vector>

namespace hpx { namespace threads {
    ///////////////////////////////////////////////////////////////////////////
    resource_manager& resource_manager::get()
    {
        typedef util::reinitializable_static<resource_manager, tag> static_type;

        static_type instance;
        return instance.get();
    }

    ///////////////////////////////////////////////////////////////////////////
    resource_manager::resource_manager()
      : next_cookie_(0)
      , punits_(parallel::execution::detail::get_os_thread_count())
      , topology_(create_topology())
    {
    }

    // Request an initial resource allocation
    std::size_t resource_manager::initial_allocation(
        detail::manage_executor* proxy, error_code& ec)
    {
        if (nullptr == proxy)
        {
            HPX_THROWS_IF(ec, bad_parameter,
                "resource_manager::init_allocation",
                "manage_executor pointer is a nullptr");
            return std::size_t(-1);
        }

        // ask executor for its policies
        error_code ec1(lightweight);
        std::size_t min_punits =
            proxy->get_policy_element(detail::min_concurrency, ec1);
        if (ec1)
            min_punits = 1;
        std::size_t max_punits =
            proxy->get_policy_element(detail::max_concurrency, ec1);
        if (ec1)
            max_punits = parallel::execution::detail::get_os_thread_count();

        // lock the resource manager from this point on
        std::lock_guard<mutex_type> l(mtx_);

        // allocate initial resources for the given executor
        std::vector<std::pair<std::size_t, std::size_t>> cores =
            allocate_virt_cores(proxy, min_punits, max_punits, ec);
        if (ec)
            return std::size_t(-1);

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
        std::vector<punit_status>& available_punits)
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

    // reserve cores at higher use counts (use count > 0)
    // used when cores cannot be allocated by stealing
    // calls reserve_processing_units
    std::size_t resource_manager::reserve_at_higher_use_count(
        std::size_t desired, std::vector<punit_status>& available_punits)
    {
        std::size_t use_count = 1;
        std::size_t available = 0;

        while (available < desired)
        {
            available += reserve_processing_units(
                use_count++, desired - available, available_punits);
        }

        return available;
    }

    // Instructs a scheduler proxy to free up a fixed number of resources
    // This is only a temporary release of resources.
    // The use count on the global core is decremented and the scheduler
    // proxy remembers the core as temporarily released
    // release_cores_to_min - scheduler should release all cores above its minimum
    // release_borrowed_cores - scheduler should release all its borrowed cores
    bool resource_manager::release_scheduler_resources(
        allocation_data_map_type::iterator it, std::size_t number_to_free,
        std::vector<punit_status>& /* available_punits */)
    {
        static_allocation_data& st = (*it).second;
        std::size_t borrowed_cores;
        std::size_t owned_cores;

        proxies_map_type::iterator iter;
        iter = proxies_.find((*it).first);
        // FIXME: handle iter == proxies.end()

        proxy_data& p = (*iter).second;

        if (number_to_free == release_borrowed_cores)
        {
            // We should only get one request to release borrowed cores - there
            // should be no cores already stolen at this time.
            HPX_ASSERT(st.num_cores_stolen_ == 0);

            number_to_free = borrowed_cores = st.num_borrowed_cores_;
        }
        else if (number_to_free == release_cores_to_min)
        {
            HPX_ASSERT(st.num_borrowed_cores_ == 0 ||
                st.num_cores_stolen_ >= st.num_borrowed_cores_);
            HPX_ASSERT(st.num_owned_cores_ >= st.min_proxy_cores_);

            // Number to stolen includes all borrowed cores, if any, and
            // possibly some owned cores.
            number_to_free = st.num_owned_cores_ - st.min_proxy_cores_ -
                (st.num_cores_stolen_ - st.num_borrowed_cores_);
            borrowed_cores = 0;
        }
        else
        {
            // If we're asked to release a specific number of cores, borrowed
            // cores should already have been released, and we should
            // not encounter any borrowed cores during our search.
            HPX_ASSERT(st.num_cores_stolen_ == st.num_borrowed_cores_);
            HPX_ASSERT(st.num_owned_cores_ >= st.min_proxy_cores_);
            HPX_ASSERT(number_to_free > 0);

            if (number_to_free > st.num_owned_cores_ - st.min_proxy_cores_)
                return false;

            borrowed_cores = 0;
        }

        // owned_cores number of cores can be released only
        HPX_ASSERT(number_to_free >= borrowed_cores &&
            number_to_free <= release_cores_to_min);
        owned_cores = number_to_free - borrowed_cores;

        if (number_to_free > 0)
        {
            for (coreids_type& coreids : p.core_ids_)
            {
                // continue if this core is not used by the current executor
                if (coreids.second == std::size_t(-1))
                    continue;

                // check all cores
                if (punits_[coreids.first].use_count_ > 0 || owned_cores > 0)
                {
                    // The proxy remembers this processor as gone ..
                    // TODO

                    // remove this core form the current scheduler
                    error_code ec(lightweight);
                    p.proxy_->remove_processing_unit(coreids.second, ec);

                    // this virtual core is not running on the current
                    // scheduler anymore
                    coreids.second = std::size_t(-1);

                    // increase number of stolen cores
                    ++(*it).second.num_cores_stolen_;

                    // reduce use_count as core is released
                    --punits_[coreids.first].use_count_;

                    if (punits_[coreids.first].use_count_ > 0)
                    {
                        --owned_cores;
                    }

                    if (--number_to_free == 0)
                    {
                        return true;
                    }
                }
            }
        }

        // The scheduler proxy does not have any cores available to free.
        return false;
    }

    // Instructs existing schedulers to release cores. Then tries to reserve
    // available cores for the new scheduler
    //
    // The parameter number_to_free can be one of the two special values:
    //    release_cores_to_min: used to release cores until they are at min, or
    //    release_borrowed_cores: only release borrowed cores.
    //
    std::size_t resource_manager::release_cores_on_existing_schedulers(
        std::size_t request, std::size_t number_to_free,
        std::vector<punit_status>& available_punits, std::size_t new_allocation)
    {
        HPX_ASSERT(!proxies_static_allocation_data_.empty());

        // Ask previously allocated schedulers to release surplus cores,
        // until either the request is satisfied, or we're out of
        // schedulers.
        bool released_cores = false;

        for (allocation_data_map_type::iterator it =
                 proxies_static_allocation_data_.begin();
             it != proxies_static_allocation_data_.end(); ++it)
        {
            // skip new allocation
            if ((*it).first == new_allocation)
                continue;

            // check each scheduler
            if (release_scheduler_resources(
                    it, number_to_free, available_punits))
            {
                released_cores = true;
            }
        }

        std::size_t available = 0;
        if (released_cores)
        {
            // reserve resources if available
            available = reserve_processing_units(0, request, available_punits);
        }

        return available;
    }

    // Tries to redistribute cores allocated to all schedulers proportional
    // to each schedulers maximum processing units
    // and reserve any freed cores for the new scheduler.
    std::size_t resource_manager::redistribute_cores_among_all(
        std::size_t reserved, std::size_t min_punits,
        std::size_t /* max_punits */,
        std::vector<punit_status>& available_punits, std::size_t new_allocation)
    {
        std::size_t available = 0;
        HPX_ASSERT(!proxies_static_allocation_data_.empty() &&
            proxies_static_allocation_data_.find(new_allocation) !=
                proxies_static_allocation_data_.end());

        // Try to proportionally allocate cores to all schedulers w/o
        // over-subscription. The proportions used will be max_punits for each
        // scheduler, except that no existing scheduler will be forced to
        // increase the current allocation.
        if (proxies_static_allocation_data_.size() > 1)
        {
            std::size_t total_minimum = min_punits;

            // sum of cores that have been previously reserved and cores that
            // were reserved during this allocation attempt.
            std::size_t num_schedulers = 1;    // includes the current scheduler

            // total_allocated is the number of cores allocated to new
            // scheduler so far plus the number of 'owned' cores allocated to
            // all existing schedulers.
            std::size_t total_allocated = reserved;

            // Let total_allocated be the number of cores we have allocated to
            // the new scheduler so far, plus the number of 'owned' cores
            // allocated to all existing schedulers.
            // Let s0,...sn-1 be the currently allocated schedulers with 'desired'
            // des[0],...,des[n-1] and 'minimum' min[0],...,min[n-1].
            //
            // The new scheduler requesting an allocation is sn with desired
            // des[n] and minimum min[n].
            for (auto& data : proxies_static_allocation_data_)
            {
                // skip new allocation
                if (data.first == new_allocation)
                    continue;

                static_allocation_data st = data.second;

                // Only take into account existing schedulers that have > Min.
                // We work with the number of 'owned' cores here instead of the
                // number of 'allocated' cores (which includes borrowed cores).
                // The borrowed cores should already have been released, but
                // they are accounted for in the total allocated count, until
                // the release is confirmed.
                if (st.num_owned_cores_ > st.min_proxy_cores_)
                {
                    ++num_schedulers;
                    total_minimum += st.min_proxy_cores_;
                    total_allocated += st.num_owned_cores_;
                }
            }

            if (num_schedulers > 1 && total_minimum <= total_allocated)
            {
                // We have found schedulers with cores greater than min.
                // Moreover, the sum of all cores already allocated to
                // existing schedulers can at least satisfy all minimums
                // (including the min requirement of the current scheduler).
                double total_desired = 0.0;

#if defined(HPX_DEBUG)
                // epsilon allows forgiveness of reasonable round-off errors
                constexpr double epsilon = 1e-7;
#endif
                allocation_data_map_type scaled_static_allocation_data;
                scaled_static_allocation_data[new_allocation] =
                    proxies_static_allocation_data_[new_allocation];

                for (auto const& data : proxies_static_allocation_data_)
                {
                    // skip new allocation
                    if (data.first == new_allocation)
                        continue;

                    static_allocation_data st = data.second;
                    if (st.num_owned_cores_ > st.min_proxy_cores_)
                    {
                        HPX_ASSERT(std::size_t(st.adjusted_desired_) ==
                            st.max_proxy_cores_);

                        scaled_static_allocation_data.insert(
                            allocation_data_map_type::value_type(
                                data.first, st));

                        total_desired += st.adjusted_desired_;
                    }
                }

                double scaling = 0.0;
                while (true)
                {
                    // We're trying to pick a scaling factor r such that
                    // r * (Sum { desired[j] | j = 0,...,n }) == totalAllocated.
                    scaling = double(total_allocated) / total_desired;

                    // multiply the scaling factor by each schedulers 'desired'.
                    for (auto& data : proxies_static_allocation_data_)
                    {
                        static_allocation_data& st = data.second;
                        st.scaled_allocation_ = st.adjusted_desired_ * scaling;
                    }

                    // Convert the floating point scaled allocations into
                    // integer allocations, using the algorithm below.
                    roundup_scaled_allocations(
                        scaled_static_allocation_data, total_allocated);

                    bool re_calculate = false;
                    for (auto& data : scaled_static_allocation_data)
                    {
                        // skip new allocation
                        if (data.first == new_allocation)
                            continue;

                        // Keep recursing until previous allocations do not
                        // increase (excluding the current scheduler).
                        static_allocation_data& st = data.second;
                        if (st.allocation_ > st.num_owned_cores_)
                        {
                            double modifier =
                                static_cast<double>(st.num_owned_cores_) /
                                static_cast<double>(st.allocation_);

                            // Reduce adjusted_desired by multiplying it with
                            // 'modifier', to try to bias allocation to the
                            // original size or less.
                            total_desired -=
                                st.adjusted_desired_ * (1.0 - modifier);
                            st.adjusted_desired_ *= modifier;

                            re_calculate = true;
                        }
                    }

                    if (re_calculate)
                    {
#if defined(HPX_DEBUG)
                        double sum_desired = 0.0;
                        for (auto const& data : scaled_static_allocation_data)
                        {
                            sum_desired += data.second.adjusted_desired_;
                        }
                        HPX_ASSERT(total_desired >= sum_desired - epsilon &&
                            total_desired <= sum_desired + epsilon);
#endif
                        continue;
                    }

                    for (auto& data : proxies_static_allocation_data_)
                    {
                        // Keep recursing until all allocations are no greater
                        // than desired (including the current scheduler).
                        static_allocation_data& st = data.second;
                        if (st.allocation_ > st.max_proxy_cores_)
                        {
                            double modifier =
                                static_cast<double>(st.max_proxy_cores_) /
                                static_cast<double>(st.allocation_);

                            // Reduce adjusted_desired by multiplying with it
                            // 'modifier', to try to bias allocation to desired
                            // or less.
                            total_desired -=
                                st.adjusted_desired_ * (1.0 - modifier);
                            st.adjusted_desired_ *= modifier;

                            re_calculate = true;
                        }
                    }

                    if (re_calculate)
                    {
#if defined(HPX_DEBUG)
                        double sum_desired = 0.0;
                        for (auto const& data : scaled_static_allocation_data)
                        {
                            sum_desired += data.second.adjusted_desired_;
                        }
                        HPX_ASSERT(total_desired >= sum_desired - epsilon &&
                            total_desired <= sum_desired + epsilon);
#endif
                        continue;
                    }

                    for (auto& data : proxies_static_allocation_data_)
                    {
                        // Keep recursing until all allocations are at least
                        // minimum (including the current scheduler).
                        static_allocation_data& st = data.second;
                        if (st.min_proxy_cores_ > st.allocation_)
                        {
                            double new_desired =
                                static_cast<double>(st.min_proxy_cores_) /
                                scaling;

                            // Bias desired to get allocation closer to min.
                            total_desired += new_desired - st.adjusted_desired_;
                            st.adjusted_desired_ = new_desired;

                            re_calculate = true;
                        }
                    }

                    if (re_calculate)
                    {
#if defined(HPX_DEBUG)
                        double sum_desired = 0.0;
                        for (auto const& data : scaled_static_allocation_data)
                        {
                            sum_desired += data.second.adjusted_desired_;
                        }
                        HPX_ASSERT(total_desired >= sum_desired - epsilon &&
                            total_desired <= sum_desired + epsilon);
#endif
                        continue;
                    }

#if defined(HPX_DEBUG)
                    hpx::error_code ec(lightweight);
                    for (auto const& data : scaled_static_allocation_data)
                    {
                        // skip new allocation
                        if (data.first == new_allocation)
                            continue;

                        HPX_ASSERT(data.second.proxy_->get_policy_element(
                                       hpx::threads::detail::min_concurrency,
                                       ec) <= data.second.allocation_);
                    }

                    static_allocation_data const& st =
                        scaled_static_allocation_data[new_allocation];
                    HPX_ASSERT(st.min_proxy_cores_ <= st.allocation_);
#endif
                    break;
                }    // end of while(true)

                //
                static_allocation_data const& st =
                    scaled_static_allocation_data[new_allocation];
                if (st.allocation_ > total_allocated)
                {
                    allocation_data_map_type::iterator end =
                        scaled_static_allocation_data.end();

                    for (allocation_data_map_type::iterator it =
                             scaled_static_allocation_data.begin();
                         it != end; ++it)
                    {
                        // skip new allocation
                        if ((*it).first == new_allocation)
                            continue;

                        static_allocation_data const& st = (*it).second;
                        std::size_t reduce_by =
                            st.num_owned_cores_ - st.allocation_;
                        if (reduce_by > 0)
                        {
                            release_scheduler_resources(
                                it, reduce_by, available_punits);
                        }
                    }

                    // Reserve out of the cores we just freed.
                    available = reserve_processing_units(
                        0, st.allocation_ - reserved, available_punits);
                }

                scaled_static_allocation_data.clear();
            }
        }
        return available;
    }

    /// Denote the n+1 scaled allocations by: r[1],..., r[n].
    ///
    /// Split r[j] into b[j] and fract[j] where b[j] is the integral floor of
    /// r[j] and fract[j] is the fraction truncated.
    ///
    /// Sort the set { r[j] | j = 1,...,n } from largest fract[j] to smallest.
    ///
    /// For each j = 0, 1, 2,...  if fract[j] > 0, then set b[j] += 1 and pay
    /// for the cost of 1-fract[j] by rounding fract[j0] -> 0 from the end
    /// (j0=n, n-1, n-2,...) -- stop before j > j0.
    ///
    /// total_allocated is the sum of all scaled_allocation_
    /// upon entry, which after the function call is over will
    /// necessarily be equal to the sum of all allocation_.
    ///
    void resource_manager::roundup_scaled_allocations(
        allocation_data_map_type& scaled_static_allocation_data,
        std::size_t total_allocated)
    {
        HPX_ASSERT(!scaled_static_allocation_data.empty());

        // epsilon allows forgiveness of reasonable round-off errors
        double epsilon = 1e-07;
        double fraction = 0.0;

        for (auto& data : scaled_static_allocation_data)
        {
            static_allocation_data& st = data.second;
            st.allocation_ = static_cast<std::size_t>(st.scaled_allocation_);
            st.scaled_allocation_ -= static_cast<double>(st.allocation_);
        }

        // Sort by scaled_allocation
        using item = std::pair<double, allocation_data_map_type::iterator>;
        using helper_type = std::vector<item>;

        helper_type d;
        d.reserve(scaled_static_allocation_data.size());

        for (allocation_data_map_type::iterator it =
                 scaled_static_allocation_data.begin();
             it != scaled_static_allocation_data.end(); ++it)
        {
            d.emplace_back(it->second.scaled_allocation_, it);
        }

        std::sort(d.begin(), d.end(), [](item const& rhs, item const& lhs) {
            return rhs.first < lhs.first;
        });

        // Round up those with the largest truncation, stealing the fraction
        // from those with the least.
        helper_type::reverse_iterator rit = d.rbegin();
        for (helper_type::iterator it = d.begin(); it != d.end(); ++it)
        {
            while (fraction > epsilon && rit != d.rend())
            {
                if (rit->second->second.scaled_allocation_ > epsilon)
                {
                    do
                    {
                        fraction -= rit->second->second.scaled_allocation_;
                        rit->second->second.scaled_allocation_ = 0.0;
                        ++rit;
                    } while (fraction > epsilon && rit != d.rend());
                    HPX_ASSERT(it <= rit.base());
                }
                else
                {
                    ++rit;
                    HPX_ASSERT(it <= rit.base());
                }
            }

            if (it <= rit.base())
            {
                auto& alloc_data = it->second->second;
                if (alloc_data.scaled_allocation_ > epsilon)
                {
                    fraction += (1.0 - alloc_data.scaled_allocation_);
                    alloc_data.scaled_allocation_ = 0.0;
                    alloc_data.allocation_ += 1;
                }
            }
            else
            {
                break;
            }
        }

        HPX_ASSERT(fraction <= epsilon && fraction >= -epsilon);

#if defined(HPX_DEBUG)
        std::size_t sum_allocation = 0;
        for (auto& data : d)
        {
            sum_allocation += data.second->second.allocation_;
        }
        HPX_ASSERT(sum_allocation == total_allocated);
#else
        HPX_UNUSED(total_allocated);
#endif
    }

    // store all information required for static allocation in
    // proxies_static_allocation_data_
    // also store new proxy scheduler
    std::size_t resource_manager::preprocess_static_allocation(
        std::size_t min_punits, std::size_t max_punits)
    {
        proxies_map_type::iterator it;
        proxies_static_allocation_data_.clear();

        for (auto& proxy : proxies_)
        {
            proxy_data& p = proxy.second;
            static_allocation_data st;
            st.proxy_ = p.proxy_;

            // ask executor for its policies
            error_code ec1(lightweight);
            st.min_proxy_cores_ =
                p.proxy_->get_policy_element(detail::min_concurrency, ec1);
            if (ec1)
                st.min_proxy_cores_ = 1;

            st.max_proxy_cores_ =
                p.proxy_->get_policy_element(detail::max_concurrency, ec1);
            if (ec1)
                st.max_proxy_cores_ =
                    parallel::execution::detail::get_os_thread_count();

            st.num_borrowed_cores_ = 0;
            st.num_owned_cores_ = 0;
            st.adjusted_desired_ = static_cast<double>(st.max_proxy_cores_);
            st.num_cores_stolen_ = 0;

            for (coreids_type coreids : p.core_ids_)
            {
                if (punits_[coreids.first].use_count_ > 1)
                    st.num_borrowed_cores_++;
                if (punits_[coreids.first].use_count_ == 1)
                    st.num_owned_cores_++;
            }

            proxies_static_allocation_data_.insert(
                std::make_pair(proxy.first, st));
        }

        std::size_t cookie = next_cookie_ + 1;

        static_allocation_data st;
        st.min_proxy_cores_ = min_punits;
        st.max_proxy_cores_ = max_punits;
        st.adjusted_desired_ = static_cast<double>(max_punits);
        st.num_cores_stolen_ = 0;
        proxies_static_allocation_data_.insert(
            allocation_data_map_type::value_type(cookie, st));

        return cookie;
    }

    // the resource manager is locked while executing this function
    std::vector<std::pair<std::size_t, std::size_t>>
    resource_manager::allocate_virt_cores(detail::manage_executor* proxy,
        std::size_t min_punits, std::size_t max_punits, error_code& ec)
    {
        std::vector<coreids_type> core_ids;

        // array of available processing units
        std::vector<punit_status> available_punits(
            punits_.size(), punit_status::unassigned);

        // find all available processing units with zero use count
        std::size_t reserved =
            reserve_processing_units(0, max_punits, available_punits);
        if (reserved < max_punits)
        {
            // insufficient available cores found, try to share
            // processing units
            std::size_t cookie =
                preprocess_static_allocation(min_punits, max_punits);

            std::size_t num_to_release = max_punits - reserved;
            reserved += release_cores_on_existing_schedulers(
                num_to_release, num_to_release, available_punits, cookie);

            if (reserved < max_punits)
            {
                reserved += redistribute_cores_among_all(
                    reserved, min_punits, max_punits, available_punits, cookie);

                if (reserved < min_punits)
                {
                    num_to_release = min_punits - reserved;
                    reserved +=
                        release_cores_on_existing_schedulers(num_to_release,
                            num_to_release, available_punits, cookie);

                    if (reserved < min_punits)
                    {
                        reserved += reserve_at_higher_use_count(
                            min_punits - reserved, available_punits);
                    }
                }
            }

            HPX_ASSERT(reserved >= min_punits);
        }

        // processing units found, inform scheduler
        std::size_t punit = 0;
        for (std::size_t i = 0; i != available_punits.size(); ++i)
        {
            // allocate only as many processing units as necessary
            if (available_punits[i] == punit_status::reserved)
            {
                proxy->add_processing_unit(punit, i, ec);
                if (ec)
                    break;

                core_ids.push_back(std::make_pair(i, punit));

                // update use count for reserved processing units
                ++punits_[i].use_count_;

                // no need to reserve more cores than requested
                if (++punit == max_punits)
                    break;
            }
        }
        HPX_ASSERT(punit <= max_punits);

        if (ec)
        {
            // on error, remove the already assigned virtual cores
            for (std::size_t j = 0; j != available_punits.size(); ++j)
            {
                if (available_punits[j] == punit_status::reserved &&
                    j != max_punits)
                {
                    proxy->remove_processing_unit(j, ec);
                    --punits_[j].use_count_;
                }
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
        std::lock_guard<mutex_type> l(mtx_);
        proxies_map_type::iterator it = proxies_.find(cookie);
        if (it == proxies_.end())
        {
            HPX_THROWS_IF(ec, bad_parameter, "resource_manager::stop_executor",
                "the given cookie is not known to the resource manager");
            return;
        }

        // inform executor to give up virtual cores
        proxy_data& p = (*it).second;
        for (coreids_type coreids : p.core_ids_)
        {
            if (coreids.second != std::size_t(-1))
            {
                p.proxy_->remove_processing_unit(coreids.second, ec);
            }
        }
    }

    // Detach the executor identified with the given cookie
    void resource_manager::detach(std::size_t cookie, error_code& ec)
    {
        std::lock_guard<mutex_type> l(mtx_);
        proxies_map_type::iterator it = proxies_.find(cookie);
        if (it == proxies_.end())
        {
            HPX_THROWS_IF(ec, bad_parameter, "resource_manager::detach",
                "the given cookie is not known to the resource manager");
            return;
        }

        // adjust resource usage count
        proxy_data& p = (*it).second;
        for (coreids_type& coreids : p.core_ids_)
        {
            if (coreids.second != std::size_t(-1))
            {
                HPX_ASSERT(punits_[coreids.first].use_count_ != 0);
                --punits_[coreids.first].use_count_;

                coreids.second = std::size_t(-1);
            }
        }

        proxies_.erase(cookie);
    }

    // Return the current schedulers and the corresponding allocation data
    std::vector<resource_allocation> resource_manager::get_resource_allocation()
        const
    {
        std::vector<resource_allocation> result;
        proxies_map_type::const_iterator end = proxies_.end();
        for (proxies_map_type::const_iterator it = proxies_.begin(); it != end;
             ++it)
        {
            result.emplace_back(
                (*it).second.proxy_->get_description(), (*it).second.core_ids_);
        }
        return result;
    }

    std::vector<resource_allocation> get_resource_allocation()
    {
        resource_manager& rm = resource_manager::get();
        return rm.get_resource_allocation();
    }
}}    // namespace hpx::threads
#endif
