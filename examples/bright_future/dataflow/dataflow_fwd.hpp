
//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef EXAMPLES_BRIGHT_FUTURE_DATAFLOW_FWD_HPP
#define EXAMPLES_BRIGHT_FUTURE_DATAFLOW_FWD_HPP

#include <hpx/traits/promise_local_result.hpp>
#include <examples/bright_future/dataflow/is_dataflow.hpp>

namespace hpx { namespace lcos {
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

        template <typename Action, typename Result, typename DirectExecute, typename F>
        struct handle_gid<hpx::lcos::dataflow<Action, Result, DirectExecute>, F>
        {
            static bool call(
                hpx::lcos::dataflow<Action, Result, DirectExecute> const &df
              , F const& f
            )
            {
                f(df.get_gid());
                return true;
            }
        };
    }
}

#endif
