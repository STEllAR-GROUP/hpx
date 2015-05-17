//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/thread_executor.hpp

#if !defined(HPX_PARALLEL_EXECUTORS_THREAD_EXECUTOR_MAY_15_2015_0546PM)
#define HPX_PARALLEL_EXECUTORS_THREAD_EXECUTOR_MAY_15_2015_0546PM

#include <hpx/config.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/executors/executor_traits.hpp>
#include <hpx/runtime/threads/thread_executor.hpp>
#include <hpx/util/decay.hpp>

#include <type_traits>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v3)
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        /// \cond NOINTERNAL
        struct threads_executor
        {
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

        private:
            threads::executor exec_;
        };

        template <>
        struct is_executor<threads_executor>
          : std::true_type
        {};
        /// \endcond
    }
}}}

#endif
