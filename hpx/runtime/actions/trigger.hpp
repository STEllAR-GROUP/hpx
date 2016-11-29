//  Copyright (c) 2007-2016 Hartmut Kaiser
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

namespace hpx { namespace actions {
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {

        struct deferred_trigger
        {
            // special handling of actions returning a future
            template <typename Result, typename RemoteResult, typename Future>
            void operator()(std::false_type,
                typed_continuation<Result, RemoteResult>&& cont, Future&& result)
            {
                try {
                    HPX_ASSERT(result.is_ready());
                    cont.trigger_value(hpx::util::detail::decay_copy(result.get()));
                }
                catch (...) {
                    // make sure hpx::exceptions are propagated back to the client
                    cont.trigger_error(boost::current_exception());
                }
            }

            template <typename Result, typename RemoteResult, typename Future>
            void operator()(std::true_type,
                typed_continuation<Result, RemoteResult>&& cont, Future&& result)
            {
                try {
                    HPX_ASSERT(result.is_ready());
                    result.get();                   // rethrow exceptions
                    cont.trigger();
                }
                catch (...) {
                    // make sure hpx::exceptions are propagated back to the client
                    cont.trigger_error(boost::current_exception());
                }
            }
        };

        template <typename Result, typename RemoteResult, typename F, typename ...Ts>
        void trigger_impl_future(std::true_type,
            typed_continuation<Result, RemoteResult>&& cont, F&& f, Ts&&... vs)
        {
            typedef
                typename std::is_same<RemoteResult, util::unused_type>::type
                is_void;

            auto result = util::invoke(std::forward<F>(f),
                std::forward<Ts>(vs)...);

            typedef typename hpx::util::decay<decltype(result)>::type future_type;

            deferred_trigger trigger;

            if(result.is_ready())
            {
                trigger(
                    is_void(), std::move(cont), std::move(result));
                return;
            }

            result.then(
                hpx::util::bind(
                    hpx::util::one_shot(trigger)
                  , is_void()
                  , std::move(cont) //-V575
                  , util::placeholders::_1
                )
            );
        }

        template <typename Result, typename RemoteResult, typename F, typename ...Ts>
        void trigger_impl(std::false_type,
            typed_continuation<Result, RemoteResult>&& cont, F&& f, Ts&&... vs)
        {
            try {
                cont.trigger_value(
                        util::invoke(std::forward<F>(f), std::forward<Ts>(vs)...));
            }
            catch (...) {
                // make sure hpx::exceptions are propagated back to the client
                cont.trigger_error(boost::current_exception());
            }
        }

        // Overload when return type is "void" aka util::unused_type
        template <typename Result, typename RemoteResult, typename F, typename ...Ts>
        void trigger_impl(std::true_type, typed_continuation<Result, RemoteResult>&& cont,
            F&& f, Ts&&... vs)
        {
            try {
                util::invoke(std::forward<F>(f), std::forward<Ts>(vs)...);
                cont.trigger();
            }
            catch (...) {
                // make sure hpx::exceptions are propagated back to the client
                cont.trigger_error(boost::current_exception());
            }
        }

        template <typename Result, typename RemoteResult, typename F, typename ...Ts>
        void trigger_impl_future(std::false_type,
            typed_continuation<Result, RemoteResult>&& cont, F&& f, Ts&&... vs)
        {
            typename std::is_same<RemoteResult, util::unused_type>::type is_void;

            trigger_impl(is_void, std::move(cont), std::forward<F>(f),
                std::forward<Ts>(vs)...);
        }
    }

    template <typename Result, typename RemoteResult, typename F, typename ...Ts>
    void trigger(typed_continuation<Result, RemoteResult>&& cont,
        F&& f, Ts&&... vs)
    {
        typedef typename util::result_of<F(Ts...)>::type result_type;
        traits::is_future<result_type> is_future;

        detail::trigger_impl_future(is_future, std::move(cont),
            std::forward<F>(f), std::forward<Ts>(vs)...);
    }
}}

#endif
