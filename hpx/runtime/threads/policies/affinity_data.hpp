//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_THREADMANAGER_POLICIES_AFFINITY_DATA_JAN_11_2013_0922PM)
#define HPX_THREADMANAGER_POLICIES_AFFINITY_DATA_JAN_11_2013_0922PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/threads/topology.hpp>

#include <string>

namespace hpx { namespace threads { namespace policies { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    // Structure holding the information related to thread affinity selection
    // for the shepherd threads of this instance
    struct affinity_data
    {
//         affinity_data()
//           : pu_offset_(0), pu_step_(1),
//             affinity_domain_("pu"), affinity_masks_(),
//             pu_nums_()
//         {}

        affinity_data(std::size_t num_threads, std::size_t pu_offset, 
            std::size_t pu_step, std::string const& affinity_domain, 
            std::string const& affinity_desc);

        mask_cref_type get_pu_mask(topology const& topology,
            std::size_t num_thread, bool numa_sensitive) const;

        std::size_t get_pu_num(std::size_t num_thread) const
        {
            BOOST_ASSERT(num_thread < pu_nums_.size());
            return pu_nums_[num_thread];
        }

    private:
        std::size_t init_pu_num(std::size_t num_thread,
            std::size_t hardware_concurrency) const;

        std::size_t pu_offset_; ///< offset of the first processing unit to use
        std::size_t pu_step_;   ///< step between used processing units
        std::string affinity_domain_;
        std::vector<mask_type> affinity_masks_;
        std::vector<std::size_t> pu_nums_;
    };
}}}}

#endif


