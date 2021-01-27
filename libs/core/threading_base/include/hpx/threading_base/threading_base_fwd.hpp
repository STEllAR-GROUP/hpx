//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/runtime/threads/thread_data_fwd.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/coroutines/coroutine_fwd.hpp>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/coroutines/thread_id_type.hpp>
#include <hpx/functional/function.hpp>
#include <hpx/functional/unique_function.hpp>
#include <hpx/modules/errors.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>

#if defined(HPX_HAVE_APEX)
namespace hpx { namespace util { namespace external_timer {
    struct task_wrapper;
}}}    // namespace hpx::util::external_timer
#endif

namespace hpx { namespace threads {

    class thread_data;
    class thread_data_stackful;
    class thread_data_stackless;

    namespace policies {
        struct scheduler_base;
    }
    class HPX_CORE_EXPORT thread_pool_base;

    /// \cond NOINTERNAL
    using thread_id_type = thread_id;

    using coroutine_type = coroutines::coroutine;
    using stackless_coroutine_type = coroutines::stackless_coroutine;

    using thread_result_type = std::pair<thread_schedule_state, thread_id_type>;
    using thread_arg_type = thread_restart_state;

    using thread_function_sig = thread_result_type(thread_arg_type);
    using thread_function_type =
        util::unique_function_nonser<thread_function_sig>;

    using thread_self = coroutines::detail::coroutine_self;
    using thread_self_impl_type = coroutines::detail::coroutine_impl;

    using thread_result_type = std::pair<thread_schedule_state, thread_id_type>;
    using thread_arg_type = thread_restart_state;

    using thread_function_sig = thread_result_type(thread_arg_type);
    using thread_function_type =
        util::unique_function_nonser<thread_function_sig>;

#if defined(HPX_HAVE_APEX)
    HPX_CORE_EXPORT std::shared_ptr<hpx::util::external_timer::task_wrapper>
    get_self_timer_data(void);
    HPX_CORE_EXPORT void set_self_timer_data(
        std::shared_ptr<hpx::util::external_timer::task_wrapper> data);
#endif
    /// \endcond
}}    // namespace hpx::threads

namespace std {
    template <>
    struct hash<::hpx::threads::thread_id>
    {
        std::size_t operator()(::hpx::threads::thread_id const& v) const
            noexcept
        {
            std::hash<::hpx::threads::thread_data const*> hasher_;
            return hasher_(static_cast<::hpx::threads::thread_data*>(v.get()));
        }
    };
}    // namespace std
