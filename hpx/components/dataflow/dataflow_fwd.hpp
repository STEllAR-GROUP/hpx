//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_LCOS_FUTURE_DATAFLOW_FWD_HPP
#define HPX_LCOS_FUTURE_DATAFLOW_FWD_HPP

#include <hpx/traits/promise_local_result.hpp>
#include <hpx/components/dataflow/is_dataflow.hpp>

namespace hpx { namespace lcos 
{
    template <
        typename Action
      , typename Result =
            typename traits::promise_local_result<
                typename Action::result_type
            >::type
      , typename DirectExecute = typename Action::direct_execution
    >
    struct dataflow;
}}

namespace hpx
{
    namespace traits
    {
        template <typename Action, typename Result, typename DirectExecute>
        struct is_dataflow<hpx::lcos::dataflow<Action, Result, DirectExecute> >
            : boost::mpl::true_
        {};
    }
}

#endif
