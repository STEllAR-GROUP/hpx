//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/parallel_fork_executor.hpp

#if !defined(HPX_PARALLEL_EXECUTORS_PARALLEL_FORK_EXECUTOR_MAY_15_2015_0402PM)
#define HPX_PARALLEL_EXECUTORS_PARALLEL_FORK_EXECUTOR_MAY_15_2015_0402PM

#include <hpx/config.hpp>
#include <hpx/traits/is_executor.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/executors/executor_traits.hpp>
#include <hpx/runtime/threads/thread_executor.hpp>
#include <hpx/util/decay.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v3)
{
    ///////////////////////////////////////////////////////////////////////////
    /// A \a parallel_fork_executor creates groups of parallel execution agents
    /// which execute in threads implicitly created by the executor. This
    /// executor prefers executing newly created threads first before continuing
    /// with the creating thread.
    ///
    struct parallel_fork_executor : executor_tag
    {
#if defined(DOXYGEN)
        /// Create a new parallel fork executor
        parallel_fork_executor() {}
#endif

        /// \cond NOINTERNAL
        template <typename F>
        static void apply_execute(F && f)
        {
            return hpx::apply(std::forward<F>(f));
        }

        template <typename F>
        static hpx::future<typename hpx::util::result_of<
            typename hpx::util::decay<F>::type()
        >::type>
        async_execute(F && f)
        {
            return hpx::async(launch::fork, std::forward<F>(f));
        }
        /// \endcond
    };
}}}

#endif
