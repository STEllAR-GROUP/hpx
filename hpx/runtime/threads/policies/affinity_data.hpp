//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_THREADMANAGER_POLICIES_AFFINITY_DATA_JAN_11_2013_0922PM)
#define HPX_THREADMANAGER_POLICIES_AFFINITY_DATA_JAN_11_2013_0922PM

#include <hpx/config.hpp>
#include <hpx/runtime/threads/topology.hpp>

#include <string>
#include <vector>

namespace hpx { namespace threads { namespace policies
{
    ///////////////////////////////////////////////////////////////////////////
    struct init_affinity_data
    {
        init_affinity_data(std::size_t pu_offset = std::size_t(-1),
                std::size_t pu_step = 1, std::string const& affinity = "pu",
                std::string const& affinity_desc = "")
          : used_cores_(0),
            pu_offset_(pu_offset),
            pu_step_(pu_step),
            affinity_domain_(affinity),
            affinity_desc_(affinity_desc)
        {}

        init_affinity_data(std::string const& affinity,
                std::string const& affinity_desc)
          : used_cores_(0),
            pu_offset_(std::size_t(-1)),
            pu_step_(1),
            affinity_domain_(affinity),
            affinity_desc_(affinity_desc)
        {}

        std::size_t used_cores_;
        std::size_t pu_offset_;
        std::size_t pu_step_;
        std::string affinity_domain_;
        std::string affinity_desc_;
    };
}}}

namespace hpx { namespace threads { namespace policies { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    // Structure holding the information related to thread affinity selection
    // for the shepherd threads of this instance
    struct affinity_data
    {
        affinity_data(std::size_t num_threads);

        std::size_t init(init_affinity_data const& data,
                topology const & toplogy);

        mask_cref_type get_pu_mask(topology const& topology,
            std::size_t num_thread, bool numa_sensitive) const;

        std::size_t get_pu_num(std::size_t num_thread) const
        {
            HPX_ASSERT(num_thread < pu_nums_.size());
            return pu_nums_[num_thread];
        }

        void add_punit(std::size_t virt_core, std::size_t thread_num,
            topology const& topology);

    protected:
        void init_cached_pu_nums(std::size_t hardware_concurrency,
                topology const & topology);
        std::size_t get_pu_num(std::size_t num_thread,
            std::size_t hardware_concurrency) const;

    private:
        std::size_t num_threads_;   ///< number of processing units managed
        std::size_t pu_offset_;     ///< offset of the first processing unit to use
        std::size_t pu_step_;       ///< step between used processing units
        std::string affinity_domain_;
        std::vector<mask_type> affinity_masks_;
        std::vector<std::size_t> pu_nums_;
        mask_type no_affinity_;     ///< mask of processing units which have no affinity
    };
}}}}

#endif


