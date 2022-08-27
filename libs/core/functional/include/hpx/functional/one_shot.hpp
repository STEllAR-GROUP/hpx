//  Copyright (c) 2011-2012 Thomas Heller
//  Copyright (c) 2013-2016 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/functional/invoke_result.hpp>
#include <hpx/functional/traits/get_function_address.hpp>
#include <hpx/functional/traits/get_function_annotation.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace hpx { namespace util {
    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        template <typename F>
        class one_shot_wrapper    //-V690
        {
        public:
            // default constructor is needed for serialization
            constexpr one_shot_wrapper()
#if defined(HPX_DEBUG)
              : _called(false)
#endif
            {
            }

            template <typename F_,
                typename = typename std::enable_if<
                    std::is_constructible<F, F_>::value>::type>
            constexpr explicit one_shot_wrapper(F_&& f)
              : _f(HPX_FORWARD(F_, f))
#if defined(HPX_DEBUG)
              , _called(false)
#endif
            {
            }

            constexpr one_shot_wrapper(one_shot_wrapper&& other)
              : _f(HPX_MOVE(other._f))
#if defined(HPX_DEBUG)
              , _called(other._called)
#endif
            {
#if defined(HPX_DEBUG)
                other._called = true;
#endif
            }

            void check_call()
            {
#if defined(HPX_DEBUG)
                HPX_ASSERT(!_called);
                _called = true;
#endif
            }

            template <typename... Ts>
            constexpr HPX_HOST_DEVICE
                typename util::invoke_result<F, Ts...>::type
                operator()(Ts&&... vs)
            {
                check_call();

                return HPX_INVOKE(HPX_MOVE(_f), HPX_FORWARD(Ts, vs)...);
            }

            template <typename Archive>
            void serialize(Archive& ar, unsigned int const /*version*/)
            {
                ar& _f;
            }

            constexpr std::size_t get_function_address() const
            {
                return traits::get_function_address<F>::call(_f);
            }

            constexpr char const* get_function_annotation() const
            {
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
                return traits::get_function_annotation<F>::call(_f);
#else
                return nullptr;
#endif
            }

#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
            util::itt::string_handle get_function_annotation_itt() const
            {
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
                return traits::get_function_annotation_itt<F>::call(_f);
#else
                static util::itt::string_handle sh("one_shot_wrapper");
                return sh;
#endif
            }
#endif

        public:    // exposition-only
            F _f;
#if defined(HPX_DEBUG)
            bool _called;
#endif
        };
    }    // namespace detail

    template <typename F>
    constexpr detail::one_shot_wrapper<typename std::decay<F>::type> one_shot(
        F&& f)
    {
        typedef detail::one_shot_wrapper<typename std::decay<F>::type>
            result_type;

        return result_type(HPX_FORWARD(F, f));
    }
}}    // namespace hpx::util

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace traits {
    ///////////////////////////////////////////////////////////////////////////
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
    template <typename F>
    struct get_function_address<util::detail::one_shot_wrapper<F>>
    {
        static constexpr std::size_t call(
            util::detail::one_shot_wrapper<F> const& f) noexcept
        {
            return f.get_function_address();
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename F>
    struct get_function_annotation<util::detail::one_shot_wrapper<F>>
    {
        static constexpr char const* call(
            util::detail::one_shot_wrapper<F> const& f) noexcept
        {
            return f.get_function_annotation();
        }
    };

#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
    template <typename F>
    struct get_function_annotation_itt<util::detail::one_shot_wrapper<F>>
    {
        static util::itt::string_handle call(
            util::detail::one_shot_wrapper<F> const& f) noexcept
        {
            return f.get_function_annotation_itt();
        }
    };
#endif
#endif
}}    // namespace hpx::traits

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace serialization {
    template <typename Archive, typename F>
    void serialize(Archive& ar,
        ::hpx::util::detail::one_shot_wrapper<F>& one_shot_wrapper,
        unsigned int const version = 0)
    {
        one_shot_wrapper.serialize(ar, version);
    }
}}    // namespace hpx::serialization
