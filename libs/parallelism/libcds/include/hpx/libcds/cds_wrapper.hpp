//  Copyright (c) 2020 Weile Wei
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/functional.hpp>

#include <cstddef>

/// \cond NODETAIL
namespace hpx { namespace cds {

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        /// \cond NOINTERNAL
        using get_num_concurrent_hazard_pointer_threads_type =
            hpx::util::function_nonser<std::size_t()>;

        HPX_PARALLELISM_EXPORT void
        set_get_num_concurrent_hazard_pointer_threads(
            get_num_concurrent_hazard_pointer_threads_type f);

        HPX_PARALLELISM_EXPORT std::size_t
        get_num_concurrent_hazard_pointer_threads();
        /// \endcond
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    enum class smr_t
    {
        hazard_pointer_hpxthread,
        hazard_pointer_stdthread,
        rcu
    };

    // this wrapper will initialize libCDS
    struct HPX_PARALLELISM_EXPORT libcds_wrapper
    {
        // hazard_pointer_count, max_thread_count, max_retired_pointer_count
        // are only used in hazard pointer
        //
        // hazard_pointer_count is corresponding var nHazardPtrCount
        // in libcds that defines Hazard pointer count per thread;
        //
        // max_concurrent_attach_thread_ is corresponding var nMaxThreadCount
        // in libcds that defines Max count of simultaneous working thread
        // in your application
        //
        // max_retired_pointer_count is corresponding var
        // nMaxRetiredPtrCount in libcds that defines Capacity
        // of the array of retired objects for the thread
        libcds_wrapper(smr_t smr_type = smr_t::hazard_pointer_hpxthread,
            std::size_t hazard_pointer_count = 1,
            std::size_t max_thread_count =
                detail::get_num_concurrent_hazard_pointer_threads(),
            std::size_t max_retired_pointer_count = 16);

        ~libcds_wrapper();

    private:
        friend struct hpxthread_manager_wrapper;
        friend struct stdthread_manager_wrapper;

        smr_t smr_;
    };
}}    // namespace hpx::cds
