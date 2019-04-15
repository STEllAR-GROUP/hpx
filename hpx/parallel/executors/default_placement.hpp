//  Copyright (c) 2019 Mikael Simberg
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_EXECUTORS_SCHEDULER_PLACEMENT)
#define HPX_PARALLEL_EXECUTORS_SCHEDULER_PLACEMENT

#include <hpx/config.hpp>
#include <hpx/parallel/executors/execution_parameters_fwd.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime/threads/thread_enums.hpp>
#include <hpx/traits/is_executor_parameters.hpp>

#include <cstddef>
#include <type_traits>

namespace hpx { namespace parallel { namespace execution {
    ///////////////////////////////////////////////////////////////////////////
    /// Lets the scheduler decide the task placement during bulk execution.
    struct scheduler_placement
    {
        /// Construct a \a scheduler_placement executor parameters object
        HPX_CONSTEXPR scheduler_placement() {}

        /// \cond NOINTERNAL
        template <typename Executor>
        hpx::threads::thread_schedule_hint get_schedule_hint(Executor&,
            std::size_t task_idx, std::size_t num_tasks, std::size_t cores)
        {
            return hpx::threads::thread_schedule_hint();
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
    struct is_executor_parameters<parallel::execution::scheduler_placement>
      : std::true_type
    {
    };
    /// \endcond
}}}

#endif
