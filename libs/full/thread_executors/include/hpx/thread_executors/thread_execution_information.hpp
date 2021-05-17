//  Copyright (c) 2017-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#if defined(HPX_HAVE_THREAD_EXECUTORS_COMPATIBILITY)
#include <hpx/threading_base/scheduler_mode.hpp>
#include <hpx/topology/topology.hpp>

#include <hpx/execution/executors/execution_information.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

// NOTE: Thread executors are deprecated and will be removed. Until then this
// forward declaration serves to make sure the thread_executors module does not
// depend on the runtime_local module (although it really does).
namespace hpx {
    namespace threads {
        class executor;
    }

    HPX_EXPORT std::size_t get_os_thread_count(threads::executor const& exec);
}    // namespace hpx

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads {

    template <typename Executor, typename Parameters>
    std::enable_if_t<hpx::traits::is_threads_executor<Executor>::value,
        std::size_t>
    tag_invoke(hpx::parallel::execution::processing_units_count_t, Parameters&&,
        Executor&& exec)
    {
        return get_os_thread_count(exec);
    }

    template <typename Executor>
    std::enable_if_t<hpx::traits::is_threads_executor<Executor>::value, bool>
    tag_invoke(
        hpx::parallel::execution::has_pending_closures_t, Executor&& exec)
    {
        return exec.num_pending_closures() ? true : false;
    }

    template <typename Executor>
    std::enable_if_t<hpx::traits::is_threads_executor<Executor>::value,
        threads::mask_cref_type>
    tag_invoke(hpx::parallel::execution::get_pu_mask_t, Executor&& exec,
        threads::topology& topo, std::size_t thread_num)
    {
        return exec.get_pu_mask(topo, thread_num);
    }

    template <typename Executor, typename Mode>
    std::enable_if_t<hpx::traits::is_threads_executor<Executor>::value>
    tag_invoke(hpx::parallel::execution::set_scheduler_mode_t, Executor&& exec,
        Mode mode)
    {
        exec.set_scheduler_mode(
            static_cast<threads::policies::scheduler_mode>(mode));
    }
}}    // namespace hpx::threads
#endif
