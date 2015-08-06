//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/thread_executor.hpp

#if !defined(HPX_PARALLEL_EXECUTORS_THREAD_EXECUTOR_MAY_15_2015_0546PM)
#define HPX_PARALLEL_EXECUTORS_THREAD_EXECUTOR_MAY_15_2015_0546PM

#include <hpx/config.hpp>
#include <hpx/traits/is_executor.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/executors/executor_traits.hpp>
#include <hpx/parallel/executors/auto_chunk_size.hpp>
#include <hpx/runtime/threads/thread_executor.hpp>
#include <hpx/util/decay.hpp>

#include <type_traits>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v3)
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        /// \cond NOINTERNAL
        struct threads_executor : executor_tag
        {
            // Associate the auto_chunk_size executor parameters type as a
            // default with all executors derived from this.
            typedef auto_chunk_size executor_parameters_type;

            threads_executor(threads::executor exec)
              : exec_(exec)
            {}

            template <typename F>
            hpx::future<typename hpx::util::result_of<
                typename hpx::util::decay<F>::type()
            >::type>
            async_execute(F && f)
            {
                return hpx::async(exec_, std::forward<F>(f));
            }

            std::size_t os_thread_count()
            {
                return hpx::get_os_thread_count(exec_);
            }

            bool has_pending_closures() const
            {
                return exec_.num_pending_closures() != 0;
            }

        private:
            threads::executor exec_;
        };
        /// \endcond
    }
}}}

#endif
