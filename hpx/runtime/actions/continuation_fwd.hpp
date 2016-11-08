//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_ACTIONS_CONTINUATION_HPP)
#define HPX_RUNTIME_ACTIONS_CONTINUATION_HPP

#include <hpx/config.hpp>

namespace hpx { namespace actions
{
    class HPX_EXPORT continuation;

    template <typename Result, typename RemoteResult = Result>
    struct typed_continuation;

    template <
        typename Result, typename RemoteResult, typename F, typename ...Ts>
    void trigger(typed_continuation<Result, RemoteResult>&& cont, F&& f, Ts&&... vs);
}}

#endif

