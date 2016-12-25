//  Copyright (c) 2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_EXECUTORS_EXECUTION_FWD_DEC_23_0712PM)
#define HPX_PARALLEL_EXECUTORS_EXECUTION_FWD_DEC_23_0712PM

#include <hpx/config.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(concurrency_v2) {
    namespace execution
{
    ///////////////////////////////////////////////////////////////////////////
    /// Function invocations executed by a group of sequential execution agents
    /// execute in sequential order.
    struct sequenced_execution_tag {};

    /// Function invocations executed by a group of parallel execution agents
    /// execute in unordered fashion. Any such invocations executing in the
    /// same thread are indeterminately sequenced with respect to each other.
    ///
    /// \note \a parallel_execution_tag is weaker than
    ///       \a sequenced_execution_tag.
    struct parallel_execution_tag {};

    /// Function invocations executed by a group of vector execution agents are
    /// permitted to execute in unordered fashion when executed in different
    /// threads, and un-sequenced with respect to one another when executed in
    /// the same thread.
    ///
    /// \note \a unsequenced_execution_tag is weaker than
    ///       \a parallel_execution_tag.
    struct unsequenced_execution_tag {};

    /// \cond NOINTERNAL
    struct task_policy_tag
    {
        HPX_CONSTEXPR task_policy_tag() {}
    };
    /// \endcond
}}}}

#endif

