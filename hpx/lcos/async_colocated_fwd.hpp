//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

///////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_LCOS_ASYNC_COLOCATED_FWD_FEB_01_2014_0107PM)
#define HPX_LCOS_ASYNC_COLOCATED_FWD_FEB_01_2014_0107PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/traits.hpp>
#include <hpx/util/move.hpp>

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename ...Ts>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::remote_result_type
        >::type>
    async_colocated(naming::id_type const& gid, Ts&&... vs);

    ///////////////////////////////////////////////////////////////////////////
    template <
        typename Component, typename Result, typename Arguments,
        typename Derived, typename ...Ts>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Derived>::remote_result_type
        >::type>
    async_colocated(
        hpx::actions::action<Component, Result, Arguments, Derived> /*act*/
      , naming::id_type const& gid, Ts&&... vs);
}

#endif
