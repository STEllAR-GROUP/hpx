//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_UTIL_REMOTE_LOOP_OCT_15_2014_0938PM)
#define HPX_PARALLEL_UTIL_REMOTE_LOOP_OCT_15_2014_0938PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/runtime/actions/plain_action.hpp>

#include <hpx/parallel/algorithms/for_each.hpp>

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
            static Iter sequential(Iter begin, Iter end, F f)
            {
                typedef typename Iter::base_iterator_type iterator;

                iterator last = end.base_iterator();
                for (iterator it = begin.base_iterator(); it != last;
                     (void) ++it, ++begin)
                {
                    f(it);
                }

                return begin;
            }

            static Iter parallel(Iter begin, Iter end, F f)
            {
                typedef typename Iter::base_iterator_type iterator;

                return parallel::for_each(parallel::par, begin.base_iterator(),
                    end.base_iterator(), f);
            }
        };

        template <typename Iter, typename F>
        struct sequential_loop_invoker_action
            : hpx::actions::make_action<
                Iter (*)(Iter, Iter, F),
                &loop_invoker<Iter, F>::sequential,
                sequential_loop_invoker_action<Iter, F>
            >
        {};

        template <typename Iter, typename F>
        struct parallel_loop_invoker_action
            : hpx::actions::make_action<
                Iter (*)(Iter, Iter, F),
                &loop_invoker<Iter, F>::parallel,
                parallel_loop_invoker_action<Iter, F>
            >
        {};
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename SegIter, typename LocalIter, typename F>
    BOOST_FORCEINLINE hpx::future<LocalIter>
    segmented_sequential_loop_async(SegIter sit, LocalIter begin, LocalIter end,
        F && f)
    {
        if (begin == end)
            return hpx::make_ready_future(begin);

        typename detail::sequential_loop_invoker_action<
                LocalIter, typename hpx::util::decay<F>::type
            > act;
        return hpx::async_colocated(act, sit->get_id(), begin, end,
            std::forward<F>(f));
    }

    template <typename SegIter, typename LocalIter, typename F>
    BOOST_FORCEINLINE LocalIter
    segmented_sequential_loop(SegIter sit, LocalIter begin, LocalIter end,
        F && f)
    {
        if (begin == end)
            return begin;

        return segmented_sequential_loop_async(sit, begin, end,
            std::forward<F>(f)).get();
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename LocalIter, typename F>
    BOOST_FORCEINLINE hpx::future<LocalIter>
    parallel_loop_async(id_type id, LocalIter begin, LocalIter end, F && f)
    {
        typename detail::parallel_loop_invoker_action<
                LocalIter, typename hpx::util::decay<F>::type
            > act;
        return hpx::async_colocated(act, id, begin, end, std::forward<F>(f));
    }

    template <typename LocalIter, typename F>
    BOOST_FORCEINLINE LocalIter
    parallel_loop(id_type id, LocalIter begin, LocalIter end, F && f)
    {
        return parallel_loop_async(id, begin, end, std::forward<F>(f)).get();
    }
}}}}

HPX_REGISTER_PLAIN_ACTION_TEMPLATE(
    (template <typename Iter, typename F>),
    (hpx::parallel::util::remote::detail::sequential_loop_invoker_action<Iter, F>))

HPX_REGISTER_PLAIN_ACTION_TEMPLATE(
    (template <typename Iter, typename F>),
    (hpx::parallel::util::remote::detail::parallel_loop_invoker_action<Iter, F>))

#endif
