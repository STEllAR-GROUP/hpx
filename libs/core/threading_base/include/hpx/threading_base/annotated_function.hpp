//  Copyright (c) 2017 Hartmut Kaiser
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
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
    ///////////////////////////////////////////////////////////////////////////
#if defined(HPX_COMPUTE_DEVICE_CODE)
    struct annotate_function
    {
        HPX_NON_COPYABLE(annotate_function);

        explicit annotate_function(std::string) {}
        template <typename F>
        explicit HPX_HOST_DEVICE annotate_function(F&&)
        {
        }

        // add empty (but non-trivial) destructor to silence warnings
        HPX_HOST_DEVICE ~annotate_function() {}
    };
#elif HPX_HAVE_ITTNOTIFY != 0
    struct annotate_function
    {
        HPX_NON_COPYABLE(annotate_function);

        explicit annotate_function(std::string name)
          : name_(std::move(name))
          , task_(thread_domain_, hpx::util::itt::string_handle(name.c_str()))
        {
        }
        template <typename F>
        explicit annotate_function(F&& f)
          : task_(thread_domain_,
                hpx::traits::get_function_annotation_itt<
                    typename std::decay<F>::type>::call(f))
        {
        }

    private:
        std::string name_;
        hpx::util::itt::thread_domain thread_domain_;
        hpx::util::itt::task task_;
    };
#else
    struct annotate_function
    {
        HPX_NON_COPYABLE(annotate_function);

        explicit annotate_function(std::string name)
          : name_(std::move(name))
          , desc_(hpx::threads::get_self_ptr() ?
                    hpx::threads::set_thread_description(
                        hpx::threads::get_self_id(), name_.c_str()) :
                    nullptr)
        {
#if defined(HPX_HAVE_APEX)
            /* update the task wrapper in APEX to use the specified name */
            threads::set_self_timer_data(external_timer::update_task(
                threads::get_self_timer_data(), name_));
#endif
        }

        template <typename F>
        explicit annotate_function(F&& f)
          : desc_(hpx::threads::get_self_ptr() ?
                    hpx::threads::set_thread_description(
                        hpx::threads::get_self_id(),
                        hpx::util::thread_description(f)) :
                    nullptr)
        {
#if defined(HPX_HAVE_APEX)
            /* no need to update the task description in APEX, because
             * this same description was used when the task was created. */
#endif
        }

        ~annotate_function()
        {
            if (hpx::threads::get_self_ptr())
            {
                hpx::threads::set_thread_description(
                    hpx::threads::get_self_id(), desc_);
            }
        }

        std::string name_;
        hpx::util::thread_description desc_;
    };
#endif

    namespace detail {
        template <typename F>
        struct annotated_function
        {
            annotated_function() noexcept
              : name_("")
            {
            }

            annotated_function(F const& f, std::string name)
              : f_(f)
              , name_(std::move(name))
            {
            }

            annotated_function(F&& f, std::string name)
              : f_(std::move(f))
              , name_(std::move(name))
            {
            }

        public:
            template <typename... Ts>
            typename invoke_result<typename util::decay_unwrap<F>::type,
                Ts...>::type
            operator()(Ts&&... ts)
            {
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

            std::size_t get_function_address() const
            {
                return traits::get_function_address<
                    typename util::decay_unwrap<F>::type>::call(f_);
            }

            ///////////////////////////////////////////////////////////////////
            /// \brief Returns the function annotation
            ///
            /// This function returns the function annotation, if it has a name
            /// name is returned, name is returned; if name is empty the typeid
            /// is returned
            ///
            /// \param none
            char const* get_function_annotation() const noexcept
            {
                return !name_.empty() ? name_.c_str() : typeid(f_).name();
            }

        private:
            typename util::decay_unwrap<F>::type f_;
            std::string name_;
        };
    }    // namespace detail

    template <typename F>
    detail::annotated_function<typename std::decay<F>::type> annotated_function(
        F&& f, std::string name = "")
    {
        typedef detail::annotated_function<typename std::decay<F>::type>
            result_type;

        return result_type(std::forward<F>(f), std::move(name));
    }

#else
    ///////////////////////////////////////////////////////////////////////////
    struct annotate_function
    {
        HPX_NON_COPYABLE(annotate_function);

        explicit annotate_function(std::string /*name*/) {}
        template <typename F>
        explicit HPX_HOST_DEVICE annotate_function(F&& /*f*/)
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
    F&& annotated_function(F&& f, std::string = "")
    {
        return std::forward<F>(f);
    }
#endif
}}    // namespace hpx::util

#if defined(HPX_HAVE_THREAD_DESCRIPTION)
namespace hpx { namespace traits {
    template <typename F>
    struct get_function_address<util::detail::annotated_function<F>>
    {
        static std::size_t call(
            util::detail::annotated_function<F> const& f) noexcept
        {
            return f.get_function_address();
        }
    };

    template <typename F>
    struct get_function_annotation<util::detail::annotated_function<F>>
    {
        static char const* call(
            util::detail::annotated_function<F> const& f) noexcept
        {
            return f.get_function_annotation();
        }
    };
}}    // namespace hpx::traits
#endif
