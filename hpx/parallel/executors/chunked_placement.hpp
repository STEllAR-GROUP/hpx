//  Copyright (c) 2019 Mikael Simberg
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_EXECUTORS_CHUNKED_PLACEMENT)
#define HPX_PARALLEL_EXECUTORS_CHUNKED_PLACEMENT

#include <hpx/config.hpp>
#include <hpx/parallel/executors/execution_parameters_fwd.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime/threads/thread_enums.hpp>
#include <hpx/traits/is_executor_parameters.hpp>

#include <cstddef>
#include <type_traits>

namespace hpx { namespace parallel { namespace execution {
    ///////////////////////////////////////////////////////////////////////////
    /// Loop tasks are given thread hints in a chunked fashion. For example,
    /// with 10 tasks and 3 threads the tasks would be placed as follows:
    ///
    /// - tasks 1, 2, 3: thread 1
    /// - tasks 4, 5, 6: thread 2
    /// - tasks 7, 8, 9, 10: thread 3
    struct chunked_placement
    {
        /// Construct a \a chunked_placement executor parameters object
        HPX_CONSTEXPR chunked_placement() {}

        /// \cond NOINTERNAL
        template <typename Executor>
        hpx::threads::thread_schedule_hint get_schedule_hint(Executor&,
            std::size_t task_idx, std::size_t num_tasks, std::size_t cores)
        {
            return hpx::threads::thread_schedule_hint(
                hpx::threads::thread_schedule_hint_mode_thread,
                task_idx * cores / num_tasks);
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
    struct is_executor_parameters<parallel::execution::chunked_placement>
      : std::true_type
    {
    };
    /// \endcond
}}}

#endif
