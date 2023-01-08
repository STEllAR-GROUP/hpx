//  Copyright (c) 2017-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file scoped_annotation.hpp

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_THREAD_DESCRIPTION)
#include <hpx/threading_base/thread_description.hpp>
#include <hpx/threading_base/thread_helpers.hpp>

#if HPX_HAVE_ITTNOTIFY != 0
#include <hpx/modules/itt_notify.hpp>
#elif defined(HPX_HAVE_APEX)
#include <hpx/threading_base/external_timer.hpp>
#endif
#endif

#include <string>
#include <type_traits>

namespace hpx {

    namespace detail {

        HPX_CORE_EXPORT char const* store_function_annotation(std::string name);
    }    // namespace detail

#if defined(HPX_HAVE_THREAD_DESCRIPTION)
    ///////////////////////////////////////////////////////////////////////////
#if defined(HPX_COMPUTE_DEVICE_CODE)
    struct [[nodiscard]] scoped_annotation
    {
        HPX_NON_COPYABLE(scoped_annotation);

        explicit constexpr scoped_annotation(char const*) noexcept {}

        template <typename F>
        explicit HPX_HOST_DEVICE constexpr scoped_annotation(F&&) noexcept
        {
        }

        // add empty (but non-trivial) destructor to silence warnings
        HPX_HOST_DEVICE ~scoped_annotation() {}
    };
#elif HPX_HAVE_ITTNOTIFY != 0
    struct [[nodiscard]] scoped_annotation
    {
        HPX_NON_COPYABLE(scoped_annotation);

        explicit scoped_annotation(char const* name)
          : task_(thread_domain_, hpx::util::itt::string_handle(name))
        {
            auto* self = hpx::threads::get_self_ptr();
            if (self != nullptr)
            {
                desc_ = threads::get_thread_id_data(self->get_thread_id())
                            ->set_description(name);
            }
        }

        explicit scoped_annotation(std::string name)
          : task_(thread_domain_,
                hpx::util::itt::string_handle(
                    detail::store_function_annotation(name)))
        {
            auto* self = hpx::threads::get_self_ptr();
            if (self != nullptr)
            {
                char const* name_c_str =
                    detail::store_function_annotation(HPX_MOVE(name));
                desc_ = threads::get_thread_id_data(self->get_thread_id())
                            ->set_description(name_c_str);
            }
        }

        template <typename F,
            typename =
                std::enable_if_t<!std::is_same_v<std::decay_t<F>, std::string>>>
        explicit scoped_annotation(F&& f)
          : task_(thread_domain_,
                hpx::traits::get_function_annotation_itt<std::decay_t<F>>::call(
                    f))
        {
            auto* self = hpx::threads::get_self_ptr();
            if (self != nullptr)
            {
                desc_ =
                    threads::get_thread_id_data(self->get_thread_id())
                        ->set_description(hpx::threads::thread_description(f));
            }
        }

        ~scoped_annotation()
        {
            auto* self = hpx::threads::get_self_ptr();
            if (self != nullptr)
            {
                threads::get_thread_id_data(self->get_thread_id())
                    ->set_description(desc_);
            }
        }

    private:
        hpx::util::itt::thread_domain thread_domain_;
        hpx::util::itt::task task_;
        hpx::threads::thread_description desc_;
    };
#else
    struct [[nodiscard]] scoped_annotation
    {
        HPX_NON_COPYABLE(scoped_annotation);

        explicit scoped_annotation(char const* name)
        {
            auto* self = hpx::threads::get_self_ptr();
            if (self != nullptr)
            {
                desc_ = threads::get_thread_id_data(self->get_thread_id())
                            ->set_description(name);
            }

#if defined(HPX_HAVE_APEX)
            /* update the task wrapper in APEX to use the specified name */
            threads::set_self_timer_data(hpx::util::external_timer::update_task(
                threads::get_self_timer_data(), std::string(name)));
#endif
        }

        explicit scoped_annotation(std::string name)
        {
            auto* self = hpx::threads::get_self_ptr();
            if (self != nullptr)
            {
                char const* name_c_str =
#if defined(HPX_HAVE_APEX)
                    detail::store_function_annotation(name);
#else
                    detail::store_function_annotation(HPX_MOVE(name));
#endif
                desc_ = threads::get_thread_id_data(self->get_thread_id())
                            ->set_description(name_c_str);
            }

#if defined(HPX_HAVE_APEX)
            /* update the task wrapper in APEX to use the specified name */
            threads::set_self_timer_data(hpx::util::external_timer::update_task(
                threads::get_self_timer_data(), HPX_MOVE(name)));
#endif
        }

        template <typename F,
            typename =
                std::enable_if_t<!std::is_same_v<std::decay_t<F>, std::string>>>
        explicit scoped_annotation(F&& f)
        {
            auto* self = hpx::threads::get_self_ptr();
            if (self != nullptr)
            {
                desc_ =
                    threads::get_thread_id_data(self->get_thread_id())
                        ->set_description(hpx::threads::thread_description(f));
            }

#if defined(HPX_HAVE_APEX)
            /* no need to update the task description in APEX, because
             * this same description was used when the task was created. */
#endif
        }

        ~scoped_annotation()
        {
            auto* self = hpx::threads::get_self_ptr();
            if (self != nullptr)
            {
                threads::get_thread_id_data(self->get_thread_id())
                    ->set_description(desc_);
            }
        }

        hpx::threads::thread_description desc_;
    };
#endif

#else
    /// \brief scoped_annotation associates a \c name with a section of code
    ///        (scope). It can be used to visualize code execution in profiling
    ///        tools like \a Intel \a VTune, \a Apex \a Profiler, etc. That
    ///        allows analysing performance to figure out which part(s) of code
    ///        is (are) responsible for performance degradation, etc.
    struct [[nodiscard]] scoped_annotation
    {
        HPX_NON_COPYABLE(scoped_annotation);

        explicit constexpr scoped_annotation(char const* /*name*/) noexcept {}

        template <typename F>
        explicit HPX_HOST_DEVICE constexpr scoped_annotation(F&& /*f*/) noexcept
        {
        }

        // add empty (but non-trivial) destructor to silence warnings
        HPX_HOST_DEVICE ~scoped_annotation() {}
    };
#endif
}    // namespace hpx

namespace hpx::util {

    using annotate_function HPX_DEPRECATED_V(1, 8,
        "hpx::util::scoped_annotation has been deprecated, please use "
        "hpx::scoped_annotation instead.") = hpx::scoped_annotation;
}    // namespace hpx::util
