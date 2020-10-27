//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/naming_base/address.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/traits/is_continuation.hpp>

#include <type_traits>

namespace hpx { namespace applier { namespace detail {
    // forward declaration only
    template <typename Action, typename Continuation, typename... Ts>
    inline bool apply_l_p(Continuation&& c, naming::id_type const& target,
        naming::address&& addr, threads::thread_priority priority, Ts&&... vs);

    template <typename Action, typename... Ts>
    inline bool apply_l_p(naming::id_type const& target, naming::address&& addr,
        threads::thread_priority priority, Ts&&... vs);

    template <typename Action, typename Continuation, typename... Ts>
    inline bool apply_r_p(naming::address&& addr, Continuation&& c,
        naming::id_type const& id, threads::thread_priority priority,
        Ts&&... vs);

    template <typename Action, typename... Ts>
    inline bool apply_r_p(naming::address&& addr, naming::id_type const& id,
        threads::thread_priority priority, Ts&&... vs);

    template <typename Action, typename Continuation, typename Callback,
        typename... Ts>
    inline bool apply_r_p_cb(naming::address&& addr, Continuation&& c,
        naming::id_type const& id, threads::thread_priority priority,
        Callback&& cb, Ts&&... vs);

    template <typename Action, typename Callback, typename... Ts>
    inline bool apply_r_p_cb(naming::address&& addr, naming::id_type const& id,
        threads::thread_priority priority, Callback&& cb, Ts&&... vs);
}}}    // namespace hpx::applier::detail

namespace hpx { namespace detail {
    template <typename Action, typename Continuation, typename... Ts>
    typename std::enable_if<traits::is_continuation<Continuation>::value,
        bool>::type
    apply_impl(Continuation&& c, hpx::id_type const& id,
        threads::thread_priority priority, Ts&&... vs);

    template <typename Action, typename Continuation, typename... Ts>
    typename std::enable_if<traits::is_continuation<Continuation>::value,
        bool>::type
    apply_impl(Continuation&& c, hpx::id_type const& id, naming::address&& addr,
        threads::thread_priority priority, Ts&&... vs);

    template <typename Action, typename Continuation, typename Callback,
        typename... Ts>
    typename std::enable_if<traits::is_continuation<Continuation>::value,
        bool>::type
    apply_cb_impl(Continuation&& c, hpx::id_type const& id,
        threads::thread_priority priority, Callback&& cb, Ts&&... vs);

    template <typename Action, typename... Ts>
    bool apply_impl(
        hpx::id_type const& id, threads::thread_priority priority, Ts&&... vs);

    template <typename Action, typename... Ts>
    bool apply_impl(hpx::id_type const& id, naming::address&&,
        threads::thread_priority priority, Ts&&... vs);

    template <typename Action, typename Callback, typename... Ts>
    bool apply_cb_impl(hpx::id_type const& id,
        threads::thread_priority priority, Callback&& cb, Ts&&... vs);
}}    // namespace hpx::detail
