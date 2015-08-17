//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_APPLY_IMPLEMENTATIONS_FWD_APR_13_2015_0945AM)
#define HPX_APPLY_IMPLEMENTATIONS_FWD_APR_13_2015_0945AM

#include <hpx/config.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/traits/is_continuation.hpp>

namespace hpx { namespace applier { namespace detail
{
    // forward declaration only
    template <typename Action, typename Continuation, typename ...Ts>
    inline bool apply_l_p(Continuation && c,
        naming::id_type const& target, naming::address&& addr,
        threads::thread_priority priority, Ts&&... vs);

    template <typename Action, typename ...Ts>
    inline bool apply_l_p(
        naming::id_type const& target, naming::address&& addr,
        threads::thread_priority priority, Ts&&... vs);

    template <typename Action, typename Continuation, typename ...Ts>
    inline bool apply_r_p(naming::address&& addr, Continuation && c,
        naming::id_type const& id, threads::thread_priority priority,
        Ts&&... vs);

    template <typename Action, typename ...Ts>
    inline bool apply_r_p(naming::address&& addr,
        naming::id_type const& id, threads::thread_priority priority,
        Ts&&... vs);

    template <typename Action, typename Continuation, typename Callback, typename ...Ts>
    inline bool apply_r_p_cb(naming::address&& addr,
        Continuation && c, naming::id_type const& id,
        threads::thread_priority priority, Callback && cb, Ts&&... vs);

    template <typename Action, typename Callback, typename ...Ts>
    inline bool apply_r_p_cb(naming::address&& addr, naming::id_type const& id,
        threads::thread_priority priority, Callback && cb, Ts&&... vs);
}}}

namespace hpx { namespace detail
{
    template <typename Action, typename Continuation, typename ...Ts>
    typename boost::enable_if_c<
        traits::is_continuation<Continuation>::value, bool
    >::type
    apply_impl(Continuation && c,
        hpx::id_type const& id, threads::thread_priority priority, Ts&&... vs);

    template <typename Action, typename Continuation, typename Callback, typename ...Ts>
    typename boost::enable_if_c<
        traits::is_continuation<Continuation>::value, bool
    >::type
    apply_cb_impl(Continuation && c,
        hpx::id_type const& id, threads::thread_priority priority,
        Callback&& cb, Ts&&... vs);

    template <typename Action, typename ...Ts>
    bool apply_impl(
        hpx::id_type const& id, threads::thread_priority priority, Ts&&... vs);

    template <typename Action, typename Callback, typename ...Ts>
    bool apply_cb_impl(
        hpx::id_type const& id, threads::thread_priority priority,
        Callback&& cb, Ts&&... vs);
}}

#endif
