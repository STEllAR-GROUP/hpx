//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_LCOS_DATAFLOW_BASE_FWD_HPP
#define HPX_LCOS_DATAFLOW_BASE_FWD_HPP

#include <hpx/traits/promise_remote_result.hpp>
#include <hpx/components/dataflow/is_dataflow.hpp>

namespace hpx { namespace lcos
{
    template <
        typename Result
      , typename RemoteResult
            = typename traits::promise_remote_result<Result>::type
    >
    struct dataflow_base;
}}

namespace hpx
{
    namespace traits
    {
        template <typename Result, typename RemoteResult>
        struct is_dataflow<hpx::lcos::dataflow_base<Result, RemoteResult> >
            : boost::mpl::true_
        {};
    }
}

#endif
