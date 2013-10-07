//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/threads/policies/affinity_data.hpp>
#include <hpx/runtime/threads/topology.hpp>

#include <algorithm>

#include <boost/format.hpp>
#include <boost/foreach.hpp>

namespace hpx { namespace threads { namespace policies { namespace detail
{
    inline std::size_t count_initialized(std::vector<mask_type> const& masks)
    {
        std::size_t count = 0;
        BOOST_FOREACH(mask_cref_type m, masks)
        {
            if(any(m))
                ++count;
        }
        return count;
    }

    void affinity_data::init_cached_pu_nums(std::size_t hardware_concurrency)
    {
        pu_nums_.resize(num_threads_);
        for (std::size_t i = 0; i != num_threads_; ++i)
            pu_nums_[i] = get_pu_num(i, hardware_concurrency);
    }

    affinity_data::affinity_data(std::size_t num_threads)
      : num_threads_(num_threads), pu_offset_(0), pu_step_(1),
        affinity_domain_("pu"), affinity_masks_(), pu_nums_()
    {}

    void affinity_data::init(init_affinity_data const& data)
    {
        // initialize from command line
        pu_offset_ = data.pu_offset_;
        pu_step_ = data.pu_step_;
        affinity_domain_ = data.affinity_domain_;

        std::size_t num_system_pus = hardware_concurrency();
#if defined(HPX_HAVE_HWLOC)
        if (!data.affinity_desc_.empty()) {
            affinity_masks_.resize(num_threads_);
            for (std::size_t i = 0; i != num_threads_; ++i)
                threads::resize(affinity_masks_[i], num_system_pus);

            parse_affinity_options(data.affinity_desc_, affinity_masks_);

            std::size_t num_initialized = count_initialized(affinity_masks_);
            if (num_initialized != num_threads_) {
                HPX_THROW_EXCEPTION(bad_parameter,
                    "affinity_data::affinity_data",
                    boost::str(
                        boost::format("The number of OS threads requested "
                            "(%1%) does not match the number of threads to "
                            "bind (%2%)") % num_threads_ % num_initialized));
            }
        }
#endif
        init_cached_pu_nums(num_system_pus);
    }

    // means of adding a processing unit after initialization
    void affinity_data::add_punit(std::size_t virt_core, std::size_t thread_num)
    {
        std::size_t num_system_pus = hardware_concurrency();

        // initialize affinity_masks and set the mask for the given virt_core
        if (affinity_masks_.empty())
        {
            affinity_masks_.resize(num_threads_);
            for (std::size_t i = 0; i != num_threads_; ++i)
                threads::resize(affinity_masks_[i], num_system_pus);
        }
        threads::set(affinity_masks_[virt_core], thread_num);

        // find first used pu, which is then stored as the pu_offset
        std::size_t first_pu = std::size_t(-1);
        for (std::size_t i = 0; i != num_threads_; ++i)
        {
            std::size_t first = threads::find_first(affinity_masks_[i]);
            first_pu = (std::min)(first_pu, first);
        }
        if (first_pu != std::size_t(-1))
            pu_offset_ = first_pu;

        init_cached_pu_nums(num_system_pus);
    }

    mask_cref_type affinity_data::get_pu_mask(topology const& topology,
        std::size_t num_thread, bool numa_sensitive) const
    {
        // if we have individual, predefined affinity masks, return those
        if (!affinity_masks_.empty())
            return affinity_masks_[num_thread];

        // otherwise return mask based on affinity domain
        std::size_t pu_num = get_pu_num(num_thread);
        if (0 == std::string("pu").find(affinity_domain_)) {
            // The affinity domain is 'processing unit', just convert the
            // pu-number into a bit-mask.
            return topology.get_thread_affinity_mask(pu_num, numa_sensitive);
        }
        if (0 == std::string("core").find(affinity_domain_)) {
            // The affinity domain is 'core', return a bit mask corresponding
            // to all processing units of the core containing the given
            // pu_num.
            return topology.get_core_affinity_mask(pu_num, numa_sensitive);
        }
        if (0 == std::string("numa").find(affinity_domain_)) {
            // The affinity domain is 'numa', return a bit mask corresponding
            // to all processing units of the NUMA domain containing the
            // given pu_num.
            return topology.get_numa_node_affinity_mask(pu_num, numa_sensitive);
        }

        // The affinity domain is 'machine', return a bit mask corresponding
        // to all processing units of the machine.
        BOOST_ASSERT(0 == std::string("machine").find(affinity_domain_));
        return topology.get_machine_affinity_mask();
    }

    std::size_t affinity_data::get_pu_num(std::size_t num_thread,
        std::size_t hardware_concurrency) const
    {
        // The offset shouldn't be larger than the number of available
        // processing units.
        BOOST_ASSERT(pu_offset_ < hardware_concurrency);

        // The distance between assigned processing units shouldn't be zero
        BOOST_ASSERT(pu_step_ > 0 && pu_step_ < hardware_concurrency);

        // We 'scale' the thread number to compute the corresponding
        // processing unit number.
        //
        // The base line processing unit number is computed from the given
        // pu-offset and pu-step.
        std::size_t num_pu = pu_offset_ + pu_step_ * num_thread;

        // We add an additional offset, which allows to 'roll over' if the
        // pu number would get larger than the number of available
        // processing units. Note that it does not make sense to 'roll over'
        // farther than the given pu-step.
        std::size_t offset = (num_pu / hardware_concurrency) % pu_step_;

        // The resulting pu number has to be smaller than the available
        // number of processing units.
        return (num_pu + offset) % hardware_concurrency;
    }
}}}}
