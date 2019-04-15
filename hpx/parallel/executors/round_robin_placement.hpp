//  Copyright (c) 2019 Mikael Simberg
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_EXECUTORS_ROUND_ROBIN_PLACEMENT)
#define HPX_PARALLEL_EXECUTORS_ROUND_ROBIN_PLACEMENT

#include <hpx/config.hpp>
#include <hpx/parallel/executors/execution_parameters_fwd.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime/threads/thread_enums.hpp>
#include <hpx/traits/is_executor_parameters.hpp>

#include <cstddef>
#include <type_traits>

namespace hpx { namespace parallel { namespace execution {
    ///////////////////////////////////////////////////////////////////////////
    /// Loop tasks are given thread hints in a round-robin fashion. For example,
    /// with 10 tasks and 3 threads the tasks would be placed as follows:
    ///
    /// - tasks 1, 4, 7, 10: thread 1
    /// - tasks 2, 5, 8: thread 2
    /// - tasks 3, 6, 9: thread 3
    struct round_robin_placement
    {
        /// Construct a \a round_robin_placement executor parameters object
        HPX_CONSTEXPR round_robin_placement() {}

        /// \cond NOINTERNAL
        template <typename Executor>
        hpx::threads::thread_schedule_hint get_schedule_hint(Executor&,
            std::size_t task_idx, std::size_t /* num_tasks */,
            std::size_t cores)
        {
            return hpx::threads::thread_schedule_hint(
                hpx::threads::thread_schedule_hint_mode_thread,
                task_idx % cores);
        }
        /// \endcond

    private:
        /// \cond NOINTERNAL
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive& ar, const unsigned int version)
        {
        }
        /// \endcond
    };
}}}

namespace hpx { namespace parallel { namespace execution {
    /// \cond NOINTERNAL
    template <>
    struct is_executor_parameters<parallel::execution::round_robin_placement>
      : std::true_type
    {
    };
    /// \endcond
}}}

#endif
