//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

///////////////////////////////////////////////////////////////////////////////
#if !defined(HPX_RUNTIME_APPLIER_APPLY_COLOCATED_CALLBACK_FWD_APR_15_2015_0831AM)
#define HPX_RUNTIME_APPLIER_APPLY_COLOCATED_CALLBACK_FWD_APR_15_2015_0831AM

#include <hpx/config.hpp>
#include <hpx/runtime/actions/action_support.hpp>
#include <hpx/runtime/naming/id_type.hpp>

namespace hpx { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Callback, typename ...Ts>
    bool apply_colocated_cb(naming::id_type const& gid, Callback&& cb,
        Ts&&... vs);

    template <typename Component, typename Signature, typename Derived,
        typename Callback, typename ...Ts>
    bool apply_colocated_cb(
        hpx::actions::basic_action<Component, Signature, Derived> /*act*/,
        naming::id_type const& gid, Callback&& cb, Ts&&... vs);

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Continuation, typename Callback, typename ...Ts>
    bool apply_colocated_cb(Continuation && cont,
        naming::id_type const& gid, Callback&& cb, Ts&&... vs);

    template <typename Continuation, typename Component, typename Signature,
        typename Derived, typename Callback, typename ...Ts>
    bool apply_colocated_cb(
        Continuation && cont,
        hpx::actions::basic_action<Component, Signature, Derived> /*act*/,
        naming::id_type const& gid, Callback&& cb, Ts&&... vs);
}}

#if defined(HPX_HAVE_COLOCATED_BACKWARDS_COMPATIBILITY)
namespace hpx
{
    using hpx::detail::apply_colocated_cb;
}
#endif

#endif
