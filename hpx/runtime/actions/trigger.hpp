//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_RUNTIME_ACTIONS_TRIGGER_HPP
#define HPX_RUNTIME_ACTIONS_TRIGGER_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/actions/continuation_fwd.hpp>
#include <hpx/util/invoke.hpp>
#include <hpx/type_support/unused.hpp>

#include <exception>
#include <utility>

namespace hpx { namespace actions
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Result, typename RemoteResult, typename F, typename... Ts>
    void trigger(typed_continuation<Result, RemoteResult>&& cont,
        F&& f, Ts&&... vs)
    {
        try
        {
            cont.trigger_value(util::invoke(
                std::forward<F>(f), std::forward<Ts>(vs)...));
        } catch (...) {
            // make sure hpx::exceptions are propagated back to the client
            cont.trigger_error(std::current_exception());
        }
    }

    // Overload when return type is "void" aka util::unused_type
    template <typename Result, typename F, typename... Ts>
    void trigger(typed_continuation<Result, util::unused_type>&& cont,
        F&& f, Ts&&... vs)
    {
        try
        {
            util::invoke(std::forward<F>(f), std::forward<Ts>(vs)...);
            cont.trigger();
        } catch (...) {
            // make sure hpx::exceptions are propagated back to the client
            cont.trigger_error(std::current_exception());
        }
    }
}}

#endif
