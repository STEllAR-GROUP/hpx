//  Copyright (c) 2013-2015 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/functional/detail/invoke.hpp>
#include <hpx/functional/invoke_result.hpp>

#include <utility>

namespace hpx { namespace util {
    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        template <typename MemberPointer>
        struct mem_fn
        {
            constexpr explicit mem_fn(MemberPointer pm)
              : _pm(pm)
            {
            }

            constexpr mem_fn(mem_fn const& other)
              : _pm(other._pm)
            {
            }

            template <typename... Ts>
            constexpr typename util::invoke_result<MemberPointer, Ts...>::type
            operator()(Ts&&... vs) const
            {
                return HPX_INVOKE(_pm, HPX_FORWARD(Ts, vs)...);
            }

            MemberPointer _pm;
        };
    }    // namespace detail

    template <typename M, typename C>
    constexpr detail::mem_fn<M C::*> mem_fn(M C::*pm)
    {
        return detail::mem_fn<M C::*>(pm);
    }

    template <typename R, typename C, typename... Ps>
    constexpr detail::mem_fn<R (C::*)(Ps...)> mem_fn(R (C::*pm)(Ps...))
    {
        return detail::mem_fn<R (C::*)(Ps...)>(pm);
    }

    template <typename R, typename C, typename... Ps>
    constexpr detail::mem_fn<R (C::*)(Ps...) const> mem_fn(
        R (C::*pm)(Ps...) const)
    {
        return detail::mem_fn<R (C::*)(Ps...) const>(pm);
    }
}}    // namespace hpx::util
