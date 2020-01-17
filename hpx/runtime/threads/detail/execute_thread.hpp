//  Copyright (c) 2019-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_THREADS_DETAIL_EXECUTE_THREAD_DEC_01_2019_0126PM)
#define HPX_RUNTIME_THREADS_DETAIL_EXECUTE_THREAD_DEC_01_2019_0126PM

#include <hpx/config.hpp>
#include <hpx/runtime/threads/thread_data.hpp>

namespace hpx { namespace threads { namespace detail {

    HPX_API_EXPORT bool execute_thread(thread_data* thrd);

}}}    // namespace hpx::threads::detail

#endif
