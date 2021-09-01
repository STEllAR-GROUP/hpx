//  Copyright (c) 2017-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_THREAD_DESCRIPTION)
#include <hpx/functional/detail/invoke.hpp>
#include <hpx/functional/traits/get_function_address.hpp>
#include <hpx/functional/traits/get_function_annotation.hpp>
#include <hpx/threading_base/thread_description.hpp>
#include <hpx/threading_base/thread_helpers.hpp>
#include <hpx/type_support/decay.hpp>

#if HPX_HAVE_ITTNOTIFY != 0
#include <hpx/modules/itt_notify.hpp>
#elif defined(HPX_HAVE_APEX)
#include <hpx/threading_base/external_timer.hpp>
#endif
#endif

#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>

namespace hpx { namespace util {
    namespace detail {
        HPX_CORE_EXPORT char const* store_function_annotation(std::string name);
    }    // namespace detail

#if defined(HPX_HAVE_THREAD_DESCRIPTION)
    ///////////////////////////////////////////////////////////////////////////
#if defined(HPX_COMPUTE_DEVICE_CODE)
    struct HPX_NODISCARD annotate_function
    {
        HPX_NON_COPYABLE(annotate_function);

        explicit constexpr annotate_function(char const*) noexcept {}

        template <typename F>
        explicit HPX_HOST_DEVICE constexpr annotate_function(F&&) noexcept
        {
        }

        // add empty (but non-trivial) destructor to silence warnings
        HPX_HOST_DEVICE ~annotate_function() {}
    };
#elif HPX_HAVE_ITTNOTIFY != 0
    struct HPX_NODISCARD annotate_function
    {
        HPX_NON_COPYABLE(annotate_function);

        explicit annotate_function(char const* name)
          : task_(thread_domain_, hpx::util::itt::string_handle(name))
        {
        }
        template <typename F>
        explicit annotate_function(F&& f)
          : task_(thread_domain_,
                hpx::traits::get_function_annotation_itt<std::decay_t<F>>::call(
                    f))
        {
        }

    private:
        hpx::util::itt::thread_domain thread_domain_;
        hpx::util::itt::task task_;
    };
#else
    struct HPX_NODISCARD annotate_function
    {
        HPX_NON_COPYABLE(annotate_function);

        explicit annotate_function(char const* name)
        {
            auto* self = hpx::threads::get_self_ptr();
            if (self != nullptr)
            {
                desc_ = threads::get_thread_id_data(self->get_thread_id())
                            ->set_description(name);
            }

#if defined(HPX_HAVE_APEX)
            /* update the task wrapper in APEX to use the specified name */
            threads::set_self_timer_data(external_timer::update_task(
                threads::get_self_timer_data(), std::string(name)));
#endif
        }

        explicit annotate_function(std::string name)
        {
            auto* self = hpx::threads::get_self_ptr();
            if (self != nullptr)
            {
                char const* name_c_str =
#if defined(HPX_HAVE_APEX)
                    detail::store_function_annotation(name);
#else
                    detail::store_function_annotation(std::move(name));
#endif
                desc_ = threads::get_thread_id_data(self->get_thread_id())
                            ->set_description(name_c_str);
            }

#if defined(HPX_HAVE_APEX)
            /* update the task wrapper in APEX to use the specified name */
            threads::set_self_timer_data(external_timer::update_task(
                threads::get_self_timer_data(), std::move(name)));
#endif
        }

        template <typename F,
            typename =
                std::enable_if_t<!std::is_same_v<std::decay_t<F>, std::string>>>
        explicit annotate_function(F&& f)
        {
            auto* self = hpx::threads::get_self_ptr();
            if (self != nullptr)
            {
                desc_ = threads::get_thread_id_data(self->get_thread_id())
                            ->set_description(hpx::util::thread_description(f));
            }

#if defined(HPX_HAVE_APEX)
            /* no need to update the task description in APEX, because
             * this same description was used when the task was created. */
#endif
        }

        ~annotate_function()
        {
            auto* self = hpx::threads::get_self_ptr();
            if (self != nullptr)
            {
                threads::get_thread_id_data(self->get_thread_id())
                    ->set_description(desc_);
            }
        }

        hpx::util::thread_description desc_;
    };
#endif

    namespace detail {
        template <typename F>
        struct annotated_function
        {
            using fun_type = util::decay_unwrap_t<F>;

            annotated_function() noexcept
              : name_(nullptr)
            {
            }

            annotated_function(F const& f, char const* name)
              : f_(f)
              , name_(name)
            {
            }

            annotated_function(F&& f, char const* name)
              : f_(std::move(f))
              , name_(name)
            {
            }

            template <typename... Ts>
            invoke_result_t<fun_type, Ts...> operator()(Ts&&... ts)
            {
                annotate_function annotate(get_function_annotation());
                return HPX_INVOKE(f_, std::forward<Ts>(ts)...);
            }

            template <typename Archive>
            void serialize(Archive& ar, unsigned int const /*version*/)
            {
                // clang-format off
                ar & f_;
                // clang-format on
            }

            ///////////////////////////////////////////////////////////////////
            /// \brief Returns the function address
            ///
            /// This function returns the passed function address.
            /// \param none
            constexpr std::size_t get_function_address() const
            {
                return traits::get_function_address<fun_type>::call(f_);
            }

            ///////////////////////////////////////////////////////////////////
            /// \brief Returns the function annotation
            ///
            /// This function returns the function annotation, if it has a name
            /// name is returned, name is returned; if name is empty the typeid
            /// is returned
            ///
            /// \param none
            constexpr char const* get_function_annotation() const noexcept
            {
                return name_ ? name_ : typeid(f_).name();
            }

            constexpr fun_type const& get_bound_function() const noexcept
            {
                return f_;
            }

        private:
            fun_type f_;
            char const* name_;
        };
    }    // namespace detail

    template <typename F>
    detail::annotated_function<std::decay_t<F>> annotated_function(
        F&& f, char const* name = nullptr)
    {
        typedef detail::annotated_function<std::decay_t<F>> result_type;

        return result_type(std::forward<F>(f), name);
    }

    template <typename F>
    detail::annotated_function<std::decay_t<F>> annotated_function(
        F&& f, std::string name)
    {
        typedef detail::annotated_function<std::decay_t<F>> result_type;

        // Store string in a set to ensure it lives for the entire duration of
        // the task.
        char const* name_c_str =
            detail::store_function_annotation(std::move(name));
        return result_type(std::forward<F>(f), name_c_str);
    }

#else
    ///////////////////////////////////////////////////////////////////////////
    struct HPX_NODISCARD annotate_function
    {
        HPX_NON_COPYABLE(annotate_function);

        explicit constexpr annotate_function(char const* /*name*/) noexcept {}

        template <typename F>
        explicit HPX_HOST_DEVICE constexpr annotate_function(F&& /*f*/) noexcept
        {
        }

        // add empty (but non-trivial) destructor to silence warnings
        HPX_HOST_DEVICE ~annotate_function() {}
    };

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Given a function as an argument, the user can annotate_function
    /// as well.
    /// Annotating includes setting the thread description per thread id.
    ///
    /// \param function
    template <typename F>
    constexpr F&& annotated_function(F&& f, char const* = nullptr) noexcept
    {
        return std::forward<F>(f);
    }

    template <typename F>
    constexpr F&& annotated_function(F&& f, std::string const&) noexcept
    {
        return std::forward<F>(f);
    }
#endif
}}    // namespace hpx::util

namespace hpx { namespace traits {

#if defined(HPX_HAVE_THREAD_DESCRIPTION)
    ///////////////////////////////////////////////////////////////////////////
    template <typename F>
    struct get_function_address<util::detail::annotated_function<F>>
    {
        static constexpr std::size_t call(
            util::detail::annotated_function<F> const& f) noexcept
        {
            return f.get_function_address();
        }
    };

    template <typename F>
    struct get_function_annotation<util::detail::annotated_function<F>>
    {
        static constexpr char const* call(
            util::detail::annotated_function<F> const& f) noexcept
        {
            return f.get_function_annotation();
        }
    };
#endif
}}    // namespace hpx::traits
