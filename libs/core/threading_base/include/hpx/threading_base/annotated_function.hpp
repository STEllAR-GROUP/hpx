//  Copyright (c) 2017-2022 Hartmut Kaiser
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
#include <hpx/threading_base/scoped_annotation.hpp>
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

namespace hpx {

#if defined(HPX_HAVE_THREAD_DESCRIPTION)
    ///////////////////////////////////////////////////////////////////////////
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
              : f_(HPX_MOVE(f))
              , name_(name)
            {
            }

            template <typename... Ts>
            hpx::util::invoke_result_t<fun_type, Ts...> operator()(Ts&&... ts)
            {
                scoped_annotation annotate(get_function_annotation());
                return HPX_INVOKE(f_, HPX_FORWARD(Ts, ts)...);
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
            constexpr std::size_t get_function_address() const noexcept
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
        using result_type = detail::annotated_function<std::decay_t<F>>;

        return result_type(HPX_FORWARD(F, f), name);
    }

    template <typename F>
    detail::annotated_function<std::decay_t<F>> annotated_function(
        F&& f, std::string name)
    {
        using result_type = detail::annotated_function<std::decay_t<F>>;

        // Store string in a set to ensure it lives for the entire duration of
        // the task.
        char const* name_c_str =
            hpx::detail::store_function_annotation(HPX_MOVE(name));
        return result_type(HPX_FORWARD(F, f), name_c_str);
    }

#else
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Returns a function annotated with the given annotation.
    ///
    /// Annotating includes setting the thread description per thread id.
    ///
    /// \param function
    template <typename F>
    constexpr F&& annotated_function(F&& f, char const* = nullptr) noexcept
    {
        return HPX_FORWARD(F, f);
    }

    template <typename F>
    constexpr F&& annotated_function(F&& f, std::string const&) noexcept
    {
        return HPX_FORWARD(F, f);
    }
#endif
}    // namespace hpx

namespace hpx::traits {

#if defined(HPX_HAVE_THREAD_DESCRIPTION)
    ///////////////////////////////////////////////////////////////////////////
    template <typename F>
    struct get_function_address<hpx::detail::annotated_function<F>>
    {
        static constexpr std::size_t call(
            hpx::detail::annotated_function<F> const& f) noexcept
        {
            return f.get_function_address();
        }
    };

    template <typename F>
    struct get_function_annotation<hpx::detail::annotated_function<F>>
    {
        static constexpr char const* call(
            hpx::detail::annotated_function<F> const& f) noexcept
        {
            return f.get_function_annotation();
        }
    };
#endif
}    // namespace hpx::traits

namespace hpx::util {

    template <typename F>
    HPX_DEPRECATED_V(1, 8, "Please use hpx::annotated_function instead.")
    constexpr decltype(auto)
        annotated_function(F&& f, char const* name = nullptr) noexcept
    {
        return hpx::annotated_function(HPX_FORWARD(F, f), name);
    }

    template <typename F>
    HPX_DEPRECATED_V(1, 8, "Please use hpx::annotated_function instead.")
    constexpr decltype(auto)
        annotated_function(F&& f, std::string const& name) noexcept
    {
        return hpx::annotated_function(HPX_FORWARD(F, f), name);
    }
}    // namespace hpx::util
