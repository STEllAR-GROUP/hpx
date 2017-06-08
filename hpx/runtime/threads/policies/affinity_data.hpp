//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_RUNTIME_THREADS_POLICIES_AFFINITY_DATA_HPP
#define HPX_RUNTIME_THREADS_POLICIES_AFFINITY_DATA_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/threads/topology.hpp>

#include <boost/atomic.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace hpx { namespace threads { namespace policies { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    // Structure holding the information related to thread affinity selection
    // for the shepherd threads of this instance
    struct HPX_EXPORT affinity_data
    {

        affinity_data();

        std::size_t init(util::command_line_handling const& cfg_);

        void set_num_threads(size_t num_threads){
            num_threads_ = num_threads;
        }

        std::size_t get_num_threads() const{
            return num_threads_;
        }

        mask_cref_type get_pu_mask(std::size_t num_thread, bool numa_sensitive) const;

        mask_cref_type get_used_pus_mask() const;
        std::size_t get_thread_occupancy(std::size_t pid) const;

        std::size_t get_pu_num(std::size_t num_thread) const
        {
            HPX_ASSERT(num_thread < pu_nums_.size());
            return pu_nums_[num_thread];
        }

        void add_punit(std::size_t virt_core, std::size_t thread_num);

    protected:
        void init_cached_pu_nums(std::size_t hardware_concurrency);
        std::size_t get_pu_num(std::size_t num_thread,
            std::size_t hardware_concurrency) const;

    private:
        std::size_t num_threads_;                           ///< number of processing units managed
        std::size_t pu_offset_;                             ///< offset of the first processing unit to use
        std::size_t pu_step_;                               ///< step between used processing units
        std::size_t used_cores_;
        std::string affinity_domain_;
        std::vector<mask_type> affinity_masks_;
        std::vector<std::size_t> pu_nums_;
        mask_type no_affinity_;                             ///< mask of processing units which have no affinity
        static boost::atomic<int> instance_number_counter_; ///< counter for instance numbers

    };
}}}}

#endif /*HPX_RUNTIME_THREADS_POLICIES_AFFINITY_DATA_HPP*/
