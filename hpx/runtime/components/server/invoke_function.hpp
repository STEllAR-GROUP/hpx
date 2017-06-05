//  Copyright (c) 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_INVOKE_FUNCTION_JUL_21_2015_0521PM)
#define HPX_COMPONENTS_INVOKE_FUNCTION_JUL_21_2015_0521PM

#include <hpx/config.hpp>
#include <hpx/runtime/actions/basic_action.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/detail/pack.hpp>
#include <hpx/util/result_of.hpp>

#include <utility>

namespace hpx { namespace components { namespace server
{
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        // simple utility action which invoke an arbitrary function
        template <typename F, typename ...Ts>
        struct invoke_function
        {
            static typename util::invoke_result<F, Ts...>::type
            call (F f, Ts... ts)
            {
                return f(std::move(ts)...);
            }
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    // action definition exposing invoke_function<>
    template <typename F, typename ...Ts>
    struct invoke_function_action
      : ::hpx::actions::action<
            typename util::invoke_result<F, Ts...>::type(*)(F, Ts...),
            &detail::invoke_function<F, Ts...>::call,
            invoke_function_action<F, Ts...> >
    {};
}}}

#endif

