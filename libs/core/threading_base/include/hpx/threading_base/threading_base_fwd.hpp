//  Copyright (c) 2007-2024 Hartmut Kaiser
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
#include <hpx/errors/exception_fwd.hpp>
#include <hpx/functional/move_only_function.hpp>

#include <cstddef>
#include <cstdint>
#include <utility>

#if defined(HPX_HAVE_APEX)
#include <memory>

namespace hpx::util::external_timer {

    struct task_wrapper;
}    // namespace hpx::util::external_timer
#endif

namespace hpx::threads {

    class HPX_CORE_EXPORT thread_data;    // forward declaration only
    class thread_data_stackful;
    class thread_data_stackless;

    class thread_init_data;
    struct thread_description;

    namespace policies {

        struct HPX_CORE_EXPORT scheduler_base;
    }
    class HPX_CORE_EXPORT thread_pool_base;

    /// \cond NOINTERNAL
    using thread_id_ref_type = thread_id_ref;
    using thread_id_type = thread_id;

    using coroutine_type = coroutines::coroutine;
    using stackless_coroutine_type = coroutines::stackless_coroutine;

    using thread_result_type = std::pair<thread_schedule_state, thread_id_type>;
    using thread_arg_type = thread_restart_state;

    using thread_function_sig = thread_result_type(thread_arg_type);
    using thread_function_type = hpx::move_only_function<thread_function_sig>;

    using thread_self = coroutines::detail::coroutine_self;
    using thread_self_impl_type = coroutines::detail::coroutine_impl;

#if defined(HPX_HAVE_APEX)
    HPX_CORE_EXPORT std::shared_ptr<hpx::util::external_timer::task_wrapper>
    get_self_timer_data();
    HPX_CORE_EXPORT void set_self_timer_data(
        std::shared_ptr<hpx::util::external_timer::task_wrapper> data);
#endif
    /// \endcond

    ////////////////////////////////////////////////////////////////////////////
    /// The function \a get_self_id_data returns the data of the HPX thread id
    /// associated with the current thread (or nullptr if the current thread is
    /// not a HPX thread).
    HPX_CORE_EXPORT thread_data* get_self_id_data() noexcept;

    namespace detail {

        HPX_CORE_EXPORT void set_self_ptr(thread_self*) noexcept;
    }

    ///////////////////////////////////////////////////////////////////////
    /// The function \a get_self returns a reference to the (OS thread
    /// specific) self reference to the current HPX thread.
    HPX_CORE_EXPORT thread_self& get_self();

    /// The function \a get_self_ptr returns a pointer to the (OS thread
    /// specific) self reference to the current HPX thread.
    HPX_CORE_EXPORT thread_self* get_self_ptr() noexcept;

    /// The function \a get_ctx_ptr returns a pointer to the internal data
    /// associated with each coroutine.
    HPX_CORE_EXPORT thread_self_impl_type* get_ctx_ptr();

    /// The function \a get_self_ptr_checked returns a pointer to the (OS
    /// thread specific) self reference to the current HPX thread.
    HPX_CORE_EXPORT thread_self* get_self_ptr_checked(error_code& ec = throws);

    /// The function \a get_self_id returns the HPX thread id of the current
    /// thread (or zero if the current thread is not a HPX thread).
    HPX_CORE_EXPORT thread_id_type get_self_id() noexcept;

    /// The function \a get_outer_self_id returns the HPX thread id of
    /// the current outer thread (or zero if the current thread is not a HPX
    /// thread). This now always returns the same as \a get_self_id, even for
    /// directly executed threads.
    HPX_DEPRECATED_V(1, 10,
        "hpx::threads::get_outer_self_id is deprecated, use "
        "hpx::threads::get_self_id instead")
    inline thread_id_type get_outer_self_id() noexcept
    {
        return get_self_id();
    }

    /// The function \a get_parent_id returns the HPX thread id of the
    /// current thread's parent (or zero if the current thread is not a
    /// HPX thread).
    ///
    /// \note This function will return a meaningful value only if the
    ///       code was compiled with HPX_HAVE_THREAD_PARENT_REFERENCE
    ///       being defined.
    HPX_CORE_EXPORT thread_id_type get_parent_id() noexcept;

    /// The function \a get_parent_phase returns the HPX phase of the
    /// current thread's parent (or zero if the current thread is not a
    /// HPX thread).
    ///
    /// \note This function will return a meaningful value only if the
    ///       code was compiled with HPX_HAVE_THREAD_PARENT_REFERENCE
    ///       being defined.
    HPX_CORE_EXPORT std::size_t get_parent_phase() noexcept;

    /// The function \a get_self_stacksize returns the stack size of the
    /// current thread (or zero if the current thread is not a HPX thread).
    HPX_CORE_EXPORT std::ptrdiff_t get_self_stacksize() noexcept;

    /// The function \a get_self_stacksize_enum returns the stack size of the /
    //current thread (or thread_stacksize::default if the current thread is not
    //a HPX thread).
    HPX_CORE_EXPORT thread_stacksize get_self_stacksize_enum() noexcept;

    /// The function \a get_parent_locality_id returns the id of the locality of
    /// the current thread's parent (or zero if the current thread is not a
    /// HPX thread).
    ///
    /// \note This function will return a meaningful value only if the
    ///       code was compiled with HPX_HAVE_THREAD_PARENT_REFERENCE
    ///       being defined.
    HPX_CORE_EXPORT std::uint32_t get_parent_locality_id() noexcept;

    /// The function \a get_self_component_id returns the lva of the component
    /// the current thread is acting on
    ///
    /// \note This function will return a meaningful value only if the code was
    ///       compiled with HPX_HAVE_THREAD_TARGET_ADDRESS being defined.
    HPX_CORE_EXPORT std::uint64_t get_self_component_id() noexcept;
}    // namespace hpx::threads
