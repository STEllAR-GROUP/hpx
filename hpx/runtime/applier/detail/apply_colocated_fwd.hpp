//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_APPLIER_APPLY_COLOCATED_FWD_APR_15_2015_0830AM)
#define HPX_RUNTIME_APPLIER_APPLY_COLOCATED_FWD_APR_15_2015_0830AM

#include <hpx/config.hpp>
#include <hpx/runtime/actions/action_support.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/traits/is_continuation.hpp>

namespace hpx { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename ...Ts>
    bool apply_colocated(naming::id_type const& gid, Ts&&... vs);

    template <typename Component, typename Signature, typename Derived,
        typename ...Ts>
    bool apply_colocated(
        hpx::actions::basic_action<Component, Signature, Derived> /*act*/,
        naming::id_type const& gid, Ts&&... vs);

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Continuation, typename ...Ts>
    typename std::enable_if<
        traits::is_continuation<Continuation>::value
      , bool
    >::type
    apply_colocated(Continuation && cont,
        naming::id_type const& gid, Ts&&... vs);

    template <typename Continuation, typename Component, typename Signature,
        typename Derived, typename ...Ts>
    bool apply_colocated(
        Continuation && cont,
        hpx::actions::basic_action<Component, Signature, Derived> /*act*/,
        naming::id_type const& gid, Ts&&... vs);
}}

#if defined(HPX_HAVE_COLOCATED_BACKWARDS_COMPATIBILITY)
namespace hpx
{
    using hpx::detail::apply_colocated;
}
#endif

#endif
