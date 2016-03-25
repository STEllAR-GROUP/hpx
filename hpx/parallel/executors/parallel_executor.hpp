//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/parallel_executor.hpp

#if !defined(HPX_PARALLEL_EXECUTORS_PARALLEL_EXECUTOR_MAY_13_2015_1057AM)
#define HPX_PARALLEL_EXECUTORS_PARALLEL_EXECUTOR_MAY_13_2015_1057AM

#include <hpx/config.hpp>
#include <hpx/traits/is_executor.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/executors/executor_traits.hpp>
#include <hpx/parallel/executors/auto_chunk_size.hpp>
#include <hpx/runtime/threads/thread_executor.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v3)
{
    ///////////////////////////////////////////////////////////////////////////
    /// A \a parallel_executor creates groups of parallel execution agents
    /// which execute in threads implicitly created by the executor. This
    /// executor prefers continuing with the creating thread first before
    /// executing newly created threads.
    struct parallel_executor : executor_tag
    {
        /// Associate the auto_chunk_size executor parameters type as a default
        /// with this executor.
        typedef auto_chunk_size executor_parameters_type;

        /// Create a new parallel executor
        explicit parallel_executor(launch l = launch::async)
          : l_(l)
        {}

        /// \cond NOINTERNAL
        template <typename F>
        static void apply_execute(F && f)
        {
            hpx::apply(std::forward<F>(f));
        }

        template <typename F>
        hpx::future<typename hpx::util::result_of<F()>::type>
        async_execute(F && f)
        {
            return hpx::async(l_, std::forward<F>(f));
        }
        /// \endcond

    private:
        /// \cond NOINTERNAL
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            ar & l_;
        }
        /// \endcond

    private:
        /// \cond NOINTERNAL
        launch l_;
        /// \endcond
    };
}}}

#endif
