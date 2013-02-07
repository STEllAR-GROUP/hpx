//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/threads/policies/affinity_data.hpp>

#include <algorithm>

#include <boost/format.hpp>

namespace hpx { namespace threads { namespace policies { namespace detail
{
    inline std::size_t count_initialized(std::vector<mask_type> const& masks)
    {
        return masks.size() - std::count(masks.begin(), masks.end(), 0);
    }

    affinity_data::affinity_data(std::size_t num_threads, 
            std::size_t pu_offset, std::size_t pu_step,
            std::string const& affinity_domain, std::string const& affinity_desc)
      : pu_offset_(pu_offset), pu_step_(pu_step),
        affinity_domain_(affinity_domain), affinity_masks_()
    {
        if (!affinity_desc.empty()) {
            affinity_masks_.resize(num_threads, 0);
            parse_affinity_options(affinity_desc, affinity_masks_);

            std::size_t num_initialized = count_initialized(affinity_masks_);
            if (num_initialized != num_threads) {
                HPX_THROW_EXCEPTION(bad_parameter,
                    "affinity_data::affinity_data",
                    boost::str(
                        boost::format("The number of OS threads requested "
                            "(%1%) does not match the number of threads to "
                            "bind (%2%)") % num_threads % num_initialized));
            }
        }
    }

    mask_type affinity_data::get_pu_mask(topology const& topology, 
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

    std::size_t affinity_data::get_pu_num(std::size_t num_thread) const
    {
        // The offset shouldn't be larger than the number of available
        // processing units.
        BOOST_ASSERT(pu_offset_ < hardware_concurrency());

        // The distance between assigned processing units shouldn't be zero
        BOOST_ASSERT(pu_step_ > 0 && pu_step_ < hardware_concurrency());

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
        std::size_t offset = (num_pu / hardware_concurrency()) % pu_step_;

        // The resulting pu number has to be smaller than the available
        // number of processing units.
        return (num_pu + offset) % hardware_concurrency();
    }
}}}}
