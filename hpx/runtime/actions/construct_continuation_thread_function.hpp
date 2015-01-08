//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_ACTIONS_CONSTRUCT_CONTINUATION_FUNCTION_FEB_22_2012_1143AM)
#define HPX_RUNTIME_ACTIONS_CONSTRUCT_CONTINUATION_FUNCTION_FEB_22_2012_1143AM

#include <hpx/config/forceinline.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/result_of.hpp>

#include <boost/type_traits/is_void.hpp>
#include <boost/utility/enable_if.hpp>

namespace detail
{
    ///////////////////////////////////////////////////////////////////////
    struct continuation_thread_function
    {
        typedef threads::thread_state_enum result_type;

        template <typename F>
        BOOST_FORCEINLINE
        typename boost::disable_if_c<
            boost::is_void<typename util::result_of<F()>::type>::value,
            result_type
        >::type operator()(continuation_type cont, F&& f) const
        {
            try {
                LTM_(debug) << " with continuation(" << cont->get_gid() << ")";

                cont->trigger(f());
            }
            catch (...) {
                // make sure hpx::exceptions are propagated back to the client
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }

        template <typename F>
        BOOST_FORCEINLINE
        typename boost::enable_if_c<
            boost::is_void<typename util::result_of<F()>::type>::value,
            result_type
        >::type operator()(continuation_type cont, F&& f) const
        {
            try {
                LTM_(debug) << " with continuation(" << cont->get_gid() << ")";

                f();
                cont->trigger();
            }
            catch (...) {
                // make sure hpx::exceptions are propagated back to the client
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };

    ///////////////////////////////////////////////////////////////////////
    template <typename F>
    threads::thread_function_type
    construct_continuation_thread_function(continuation_type cont, F&& f)
    {
        return util::bind(
            util::one_shot(continuation_thread_function()),
            std::move(cont), std::forward<F>(f));
    }
}

#endif
