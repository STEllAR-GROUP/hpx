//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_UTIL_REMOTE_LOOP_OCT_15_2014_0938PM)
#define HPX_PARALLEL_UTIL_REMOTE_LOOP_OCT_15_2014_0938PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/runtime/actions/plain_action.hpp>

namespace hpx { namespace parallel { namespace util { namespace remote
{
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        // Helper class to repeatedly call a function starting from a given
        // iterator position.
        template <typename Iter, typename F>
        struct loop_invoker
        {
            ///////////////////////////////////////////////////////////////////
            static Iter call(Iter it, Iter end, F const& f)
            {
                for (/**/; it != end; ++it)
                    f(*it);
                return it;
            }
        };

        template <typename Iter, typename F>
        struct loop_invoker_action
            : hpx::actions::make_action<
                Iter (*)(Iter, Iter, F const&),
                &loop_invoker<Iter, F>::call,
                loop_invoker_action<Iter, F>
            >
        {};
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename LocalIter, typename F>
    BOOST_FORCEINLINE hpx::future<LocalIter>
    loop_async(id_type id, LocalIter begin, LocalIter end, F && f)
    {
        typename detail::loop_invoker_action<
                LocalIter, hpx::util::decay<F>::type
            > act;
        return hpx::async(act, id, begin, end, std::forward<F>(f));
    }

    template <typename LocalIter, typename F>
    BOOST_FORCEINLINE LocalIter
    loop(id_type id, LocalIter begin, LocalIter end, F && f)
    {
        return loop_async(id, first, last, std::forwrad<F>(f)).get();
    }
}}}}

HPX_REGISTER_PLAIN_ACTION_TEMPLATE(
    (template <typename Iter, typename F>),
    (hpx::parallel::util::remote::detail::loop_invoker_action<Iter, F>)
)

#endif
