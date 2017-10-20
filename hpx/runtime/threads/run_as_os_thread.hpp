//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_THREADS_RUN_AS_OS_THREAD_MAR_12_2016_0220PM)
#define HPX_THREADS_RUN_AS_OS_THREAD_MAR_12_2016_0220PM

#include <hpx/config.hpp>
#include <hpx/parallel/executors/service_executors.hpp>
#include <hpx/parallel/executors/execution.hpp>
#include <hpx/runtime/threads_fwd.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/deferred_call.hpp>
#include <hpx/util/result_of.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace threads
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename ... Ts>
    hpx::future<typename util::invoke_result<F, Ts...>::type>
    run_as_os_thread(F && f, Ts &&... vs)
    {
        HPX_ASSERT(get_self_ptr() != nullptr);

        parallel::execution::io_pool_executor scheduler;
        return parallel::execution::async_execute(scheduler,
            std::forward<F>(f), std::forward<Ts>(vs)...);
    }
}}

#endif
