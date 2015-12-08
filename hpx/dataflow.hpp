//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_DATAFLOW_DEC_07_2015_1130AM)
#define HPX_DATAFLOW_DEC_07_2015_1130AM

#include <hpx/lcos/local/dataflow.hpp>

// distributed dataflow: invokes given function (or executor) when ready
namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename ...Ts>
    BOOST_FORCEINLINE
    auto dataflow(F && f, Ts &&... ts)
    ->  decltype(lcos::detail::dataflow_dispatch<
            typename util::decay<F>::type>::call(
                std::forward<F>(f), std::forward<Ts>(ts)...
            ))
    {
        return lcos::detail::dataflow_dispatch<
                typename util::decay<F>::type
            >::call(std::forward<F>(f), std::forward<Ts>(ts)...);
    }
}

#endif
