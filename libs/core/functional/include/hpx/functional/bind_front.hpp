//  Copyright (c) 2011-2012 Thomas Heller
//  Copyright (c) 2013-2019 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file bind_front.hpp
/// \page hpx::bind_front
/// \headerfile hpx/functional.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/datastructures/member_pack.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/functional/invoke_result.hpp>
#include <hpx/functional/one_shot.hpp>
#include <hpx/functional/traits/get_function_address.hpp>
#include <hpx/functional/traits/get_function_annotation.hpp>
#include <hpx/type_support/decay.hpp>
#include <hpx/type_support/pack.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace hpx::detail {

    template <typename F, typename Ts, typename... Us>
    struct invoke_bound_front_result;

    template <typename F, typename... Ts, typename... Us>
    struct invoke_bound_front_result<F, util::pack<Ts...>, Us...>
      : util::invoke_result<F, Ts..., Us...>
    {
    };

    template <typename F, typename Ts, typename... Us>
    using invoke_bound_front_result_t =
        typename invoke_bound_front_result<F, Ts, Us...>::type;

    ///////////////////////////////////////////////////////////////////////
    template <typename F, typename Is, typename... Ts>
    class bound_front;

    template <typename F, std::size_t... Is, typename... Ts>
    class bound_front<F, util::index_pack<Is...>, Ts...>
    {
    public:
        bound_front() = default;    // needed for serialization

        template <typename F_, typename... Ts_,
            typename = std::enable_if_t<std::is_constructible_v<F, F_>>>
        constexpr explicit bound_front(F_&& f, Ts_&&... vs)
          : _f(HPX_FORWARD(F_, f))
          , _args(std::piecewise_construct, HPX_FORWARD(Ts_, vs)...)
        {
        }

#if !defined(__NVCC__) && !defined(__CUDACC__)
        bound_front(bound_front const&) = default;
        bound_front(bound_front&&) = default;
#else
        constexpr HPX_HOST_DEVICE bound_front(bound_front const& other)
          : _f(other._f)
          , _args(other._args)
        {
        }

        constexpr HPX_HOST_DEVICE bound_front(bound_front&& other)
          : _f(HPX_MOVE(other._f))
          , _args(HPX_MOVE(other._args))
        {
        }
#endif

        bound_front& operator=(bound_front const&) = delete;
        bound_front& operator=(bound_front&&) = delete;

        ~bound_front() = default;

        template <typename... Us>
        constexpr HPX_HOST_DEVICE
            invoke_bound_front_result_t<F&, util::pack<Ts&...>, Us&&...>
            operator()(Us&&... vs) &
        {
            return HPX_INVOKE(
                _f, _args.template get<Is>()..., HPX_FORWARD(Us, vs)...);
        }

        template <typename... Us>
        constexpr HPX_HOST_DEVICE invoke_bound_front_result_t<F const&,
            util::pack<Ts const&...>, Us&&...>
        operator()(Us&&... vs) const&
        {
            return HPX_INVOKE(
                _f, _args.template get<Is>()..., HPX_FORWARD(Us, vs)...);
        }

        template <typename... Us>
        constexpr HPX_HOST_DEVICE
            invoke_bound_front_result_t<F&&, util::pack<Ts&&...>, Us&&...>
            operator()(Us&&... vs) &&
        {
            return HPX_INVOKE(HPX_MOVE(_f),
                HPX_MOVE(_args).template get<Is>()..., HPX_FORWARD(Us, vs)...);
        }

        template <typename... Us>
        constexpr HPX_HOST_DEVICE invoke_bound_front_result_t<F const&&,
            util::pack<Ts const&&...>, Us&&...>
        operator()(Us&&... vs) const&&
        {
            return HPX_INVOKE(HPX_MOVE(_f),
                HPX_MOVE(_args).template get<Is>()..., HPX_FORWARD(Us, vs)...);
        }

        template <typename Archive>
        void serialize(Archive& ar, unsigned int const /*version*/)
        {
            // clang-format off
                ar & _f;
                ar & _args;
            // clang-format on
        }

        [[nodiscard]] constexpr std::size_t get_function_address()
            const noexcept
        {
            return traits::get_function_address<F>::call(_f);
        }

        [[nodiscard]] constexpr char const* get_function_annotation()
            const noexcept
        {
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
            return traits::get_function_annotation<F>::call(_f);
#else
            return nullptr;
#endif
        }

#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
        [[nodiscard]] util::itt::string_handle get_function_annotation_itt()
            const
        {
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
            return traits::get_function_annotation_itt<F>::call(_f);
#else
            static util::itt::string_handle sh("bound_front");
            return sh;
#endif
        }
#endif

    private:
        F _f;
        util::member_pack_for<Ts...> _args;
    };
}    // namespace hpx::detail

namespace hpx {

    /// \brief Function template \c bind_front generates a forwarding call wrapper
    ///        for \c f. Calling this wrapper is equivalent to invoking \c f with its
    ///        first \c sizeof...(Ts) parameters bound to \c vs.
    ///
    /// \param f    Callable object (function object, pointer to function,
    ///             reference to function, pointer to member function, or pointer
    ///             to data member) that will be bound to some arguments
    /// \param vs   list of the arguments to bind to the first or \c sizeof...(Ts)
    ///             parameters of \c f
    /// \returns    A function object of type \c T that is unspecified, except that
    ///             the types of objects returned by two calls to \c hpx::bind_front
    ///             with the same arguments are the same.
    template <typename F, typename... Ts>
    constexpr detail::bound_front<std::decay_t<F>,
        util::make_index_pack_t<sizeof...(Ts)>, util::decay_unwrap_t<Ts>...>
    bind_front(F&& f, Ts&&... vs)
    {
        using result_type = detail::bound_front<std::decay_t<F>,
            util::make_index_pack_t<sizeof...(Ts)>,
            util::decay_unwrap_t<Ts>...>;

        return result_type(HPX_FORWARD(F, f), HPX_FORWARD(Ts, vs)...);
    }

    // nullary functions do not need to be bound again
    template <typename F>
    constexpr std::decay_t<F> bind_front(F&& f)    //-V524
    {
        return HPX_FORWARD(F, f);
    }
}    // namespace hpx

namespace hpx::util {

    template <typename F, typename... Ts>
    HPX_DEPRECATED_V(1, 8,
        "hpx::util::bind_front is deprecated, use hpx::bind_front instead")
    constexpr decltype(auto) bind_front(F&& f, Ts&&... ts)
    {
        return hpx::bind_front(HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
    }
}    // namespace hpx::util

///////////////////////////////////////////////////////////////////////////////
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
namespace hpx::traits {

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename... Ts>
    struct get_function_address<hpx::detail::bound_front<F, Ts...>>
    {
        [[nodiscard]] static constexpr std::size_t call(
            hpx::detail::bound_front<F, Ts...> const& f) noexcept
        {
            return f.get_function_address();
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename... Ts>
    struct get_function_annotation<hpx::detail::bound_front<F, Ts...>>
    {
        [[nodiscard]] static constexpr char const* call(
            hpx::detail::bound_front<F, Ts...> const& f) noexcept
        {
            return f.get_function_annotation();
        }
    };

#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
    template <typename F, typename... Ts>
    struct get_function_annotation_itt<hpx::detail::bound_front<F, Ts...>>
    {
        [[nodiscard]] static util::itt::string_handle call(
            hpx::detail::bound_front<F, Ts...> const& f) noexcept
        {
            return f.get_function_annotation_itt();
        }
    };
#endif
}    // namespace hpx::traits
#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx::serialization {

    // serialization of the bound_front object
    template <typename Archive, typename F, typename... Ts>
    void serialize(Archive& ar, ::hpx::detail::bound_front<F, Ts...>& bound,
        unsigned int const version = 0)
    {
        bound.serialize(ar, version);
    }
}    // namespace hpx::serialization
