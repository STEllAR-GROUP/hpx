//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_ACTIONS_TRIGGER_HPP
#define HPX_ACTIONS_TRIGGER_HPP

#include <hpx/exception.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/actions/continuation_fwd.hpp>
#include <hpx/traits/is_future.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/invoke.hpp>
#include <hpx/util/result_of.hpp>
#include <hpx/util/unused.hpp>

#include <utility>
#include <type_traits>

namespace hpx { namespace actions
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename Result, typename RemoteResult, typename F,
            typename... Ts>
        void trigger_impl(std::false_type,
            typed_continuation<Result, RemoteResult>&& cont, F&& f, Ts&&... vs)
        {
            try {
                cont.trigger_value(
                    util::invoke(std::forward<F>(f), std::forward<Ts>(vs)...));
            }
            catch (...) {
                // make sure hpx::exceptions are propagated back to the client
                cont.trigger_error(std::current_exception());
            }
        }

        // Overload when return type is "void" aka util::unused_type
        template <typename Result, typename RemoteResult, typename F,
            typename... Ts>
        void trigger_impl(std::true_type,
            typed_continuation<Result, RemoteResult>&& cont, F&& f, Ts&&... vs)
        {
            try {
                util::invoke(std::forward<F>(f), std::forward<Ts>(vs)...);
                cont.trigger();
            }
            catch (...) {
                // make sure hpx::exceptions are propagated back to the client
                cont.trigger_error(std::current_exception());
            }
        }
    }

    template <typename Result, typename RemoteResult, typename F,
        typename... Ts>
    void trigger(
        typed_continuation<Result, RemoteResult>&& cont, F&& f, Ts&&... vs)
    {
        typename std::is_same<RemoteResult, util::unused_type>::type is_void;

        detail::trigger_impl(is_void, std::move(cont), std::forward<F>(f),
            std::forward<Ts>(vs)...);
    }
}}

#endif
