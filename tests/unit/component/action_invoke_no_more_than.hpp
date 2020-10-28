//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/actions_base/traits/action_decorate_continuation.hpp>
#include <hpx/assert.hpp>
#include <hpx/components_base/traits/action_decorate_function.hpp>
#include <hpx/functional/bind_back.hpp>
#include <hpx/futures/traits/is_future.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/synchronization/counting_semaphore.hpp>
#include <hpx/synchronization/spinlock.hpp>
#include <hpx/type_support/static.hpp>

#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

namespace hpx { namespace actions { namespace detail {
    ///////////////////////////////////////////////////////////////////////////
    template <typename Semaphore>
    struct signal_on_exit
    {
        signal_on_exit(Semaphore& sem)
          : sem_(sem)
        {
            sem_.wait();
        }

        ~signal_on_exit()
        {
            sem_.signal();
        }

    private:
        Semaphore& sem_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, int N>
    struct action_decorate_function_semaphore
    {
        typedef lcos::local::counting_semaphore_var<hpx::lcos::local::spinlock,
            N>
            semaphore_type;

        struct tag
        {
        };

        static semaphore_type& get_sem()
        {
            util::static_<semaphore_type, tag> sem;
            return sem.get();
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, int N>
    struct action_decorate_function
    {
        static constexpr bool value = true;
        // This wrapper is needed to stop infinite recursion when
        // trying to get the possible additional function decoration
        // from the component
        struct action_wrapper
        {
            typedef typename Action::component_type component_type;
        };

        static_assert(!Action::direct_execution::value,
            "explicit decoration of direct actions is not supported");

        typedef action_decorate_function_semaphore<Action, N>
            construct_semaphore_type;

        // If the action returns something which is not a future, we inject
        // a semaphore into the call graph.
        static threads::thread_result_type thread_function(
            threads::thread_restart_state state,
            threads::thread_function_type f)
        {
            typedef typename construct_semaphore_type::semaphore_type
                semaphore_type;

            signal_on_exit<semaphore_type> on_exit(
                construct_semaphore_type::get_sem());
            return f(state);
        }

        template <typename F>
        static threads::thread_function_type call(
            naming::address::address_type lva, F&& f, std::false_type)
        {
            return util::one_shot(
                util::bind_back(&action_decorate_function::thread_function,
                    traits::action_decorate_function<action_wrapper>::call(
                        lva, std::forward<F>(f))));
        }

        // If the action returns a future we wait on the semaphore as well,
        // however it will be signaled once the future gets ready only.
        static threads::thread_result_type thread_function_future(
            threads::thread_restart_state state,
            threads::thread_function_type f)
        {
            construct_semaphore_type::get_sem().wait();
            return f(state);
        }

        template <typename F>
        static threads::thread_function_type call(
            naming::address::address_type lva, F&& f, std::true_type)
        {
            return util::one_shot(util::bind_back(
                &action_decorate_function::thread_function_future,
                traits::action_decorate_function<action_wrapper>::call(
                    lva, std::forward<F>(f))));
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename F>
        static threads::thread_function_type call(
            naming::address::address_type lva, F&& f)
        {
            typedef typename Action::result_type result_type;
            typedef traits::is_future<result_type> is_future;
            return call(lva, std::forward<F>(f), is_future());
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, int N>
    struct wrapped_continuation
    {
        typedef action_decorate_function_semaphore<Action, N>
            construct_semaphore_type;

        naming::address addr_;

        template <typename T>
        void operator()(naming::id_type const& id, T&& t)
        {
            if (id)
            {
                hpx::set_lco_value(id, std::move(addr_), std::forward<T>(t));
            }
            construct_semaphore_type::get_sem().signal();
        }

        void operator()(naming::id_type const& id)
        {
            if (id)
            {
                trigger_lco_event(id, std::move(addr_));
            }
            construct_semaphore_type::get_sem().signal();
        }

        template <typename Archive>
        void serialize(Archive& ar, unsigned int)
        {
            ar& addr_;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, int N>
    struct action_decorate_continuation
    {
        static_assert(!Action::direct_execution::value,
            "explicit decoration of direct actions is not supported");

        typedef action_decorate_function_semaphore<Action, N>
            construct_semaphore_type;

        typedef typename traits::action_continuation<Action>::type
            continuation_type;

        ///////////////////////////////////////////////////////////////////////
        // If the action returns something which is not a future, we do nothing
        // special.
        static bool call(continuation_type&, std::false_type)
        {
            return false;
        }

        // If the action returns a future we wrap the given continuation to
        // be able to signal the semaphore after the wrapped action has
        // returned.
        static bool call(continuation_type& c, std::true_type)
        {
            c = continuation_type(
                c.get_id(), wrapped_continuation<Action, N>{c.get_addr()});
            return true;
        }

        ///////////////////////////////////////////////////////////////////////
        static bool call(continuation_type& cont)
        {
            typedef typename Action::result_type result_type;
            typedef traits::is_future<result_type> is_future;
            return call(cont, is_future());
        }
    };
}}}    // namespace hpx::actions::detail

///////////////////////////////////////////////////////////////////////////////
#define HPX_ACTION_INVOKE_NO_MORE_THAN(action, maxnum)                         \
    namespace hpx { namespace traits {                                         \
            template <>                                                        \
            struct action_decorate_function<action>                            \
              : hpx::actions::detail::action_decorate_function<action, maxnum> \
            {                                                                  \
            };                                                                 \
                                                                               \
            template <>                                                        \
            struct action_decorate_continuation<action>                        \
              : hpx::actions::detail::action_decorate_continuation<action,     \
                    maxnum>                                                    \
            {                                                                  \
            };                                                                 \
        }                                                                      \
    }                                                                          \
/**/

#endif
