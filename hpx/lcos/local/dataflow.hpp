//  Copyright (c)      2013 Thomas Heller
//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_LCOS_LOCAL_DATAFLOW_HPP
#define HPX_LCOS_LOCAL_DATAFLOW_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_LOCAL_DATAFLOW_COMPATIBILITY)
#include <hpx/lcos/dataflow.hpp>

#include <utility>

namespace hpx { namespace lcos { namespace local
{
    template <typename F, typename ...Ts>
    HPX_DEPRECATED(HPX_DEPRECATED_MSG) HPX_FORCEINLINE
    auto dataflow(F && f, Ts &&... ts)
    ->  decltype(lcos::detail::dataflow_dispatch<F>::call(
            std::forward<F>(f), std::forward<Ts>(ts)...))
    {
        return lcos::detail::dataflow_dispatch<F>::call(
            std::forward<F>(f), std::forward<Ts>(ts)...);
    }
}}}
#endif

#endif /*HPX_LCOS_LOCAL_DATAFLOW_HPP*/
