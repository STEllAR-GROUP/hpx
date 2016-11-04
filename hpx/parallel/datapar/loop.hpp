//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_DATAPAR_UTIL_LOOP_SEP_07_2016_1217PM)
#define HPX_PARALLEL_DATAPAR_UTIL_LOOP_SEP_07_2016_1217PM

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR)
#include <hpx/parallel/datapar/detail/loop_vc.hpp>
#include <hpx/parallel/datapar/detail/loop_boost_simd.hpp>
#include <hpx/parallel/datapar/execution_policy_fwd.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename Iter1, typename Iter2>
    HPX_HOST_DEVICE HPX_FORCEINLINE
    auto loop_step(parallel::v1::datapar_execution_policy, std::false_type,
            F && f, Iter1& it1, Iter2& it2)
    ->  decltype(
            detail::datapar_loop_step2<Iter1, Iter2>::call1(
                std::forward<F>(f), it1, it2)
            )
    {
        return detail::datapar_loop_step2<Iter1, Iter2>::call1(
            std::forward<F>(f), it1, it2);
    }

    template <typename F, typename Iter1, typename Iter2>
    HPX_HOST_DEVICE HPX_FORCEINLINE
    auto loop_step(parallel::v1::datapar_execution_policy, std::true_type,
            F && f, Iter1& it1, Iter2& it2)
    ->  decltype(
            detail::datapar_loop_step2<Iter1, Iter2>::callv(
                std::forward<F>(f), it1, it2)
            )
    {
        return detail::datapar_loop_step2<Iter1, Iter2>::callv(
            std::forward<F>(f), it1, it2);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename Iter1, typename Iter2>
    HPX_HOST_DEVICE HPX_FORCEINLINE
    auto loop_step(parallel::v1::datapar_task_execution_policy, std::false_type,
            F && f, Iter1& it1, Iter2& it2)
    ->  decltype(
            detail::datapar_loop_step2<Iter1, Iter2>::call1(
                std::forward<F>(f), it1, it2)
            )
    {
        return detail::datapar_loop_step2<Iter1, Iter2>::call1(
            std::forward<F>(f), it1, it2);
    }

    template <typename F, typename Iter1, typename Iter2>
    HPX_HOST_DEVICE HPX_FORCEINLINE
    auto loop_step(parallel::v1::datapar_task_execution_policy, std::true_type,
            F && f, Iter1& it1, Iter2& it2)
    ->  decltype(
            detail::datapar_loop_step2<Iter1, Iter2>::callv(
                std::forward<F>(f), it1, it2)
            )
    {
        return detail::datapar_loop_step2<Iter1, Iter2>::callv(
            std::forward<F>(f), it1, it2);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iter>
    HPX_HOST_DEVICE HPX_FORCEINLINE
    bool loop_optimization(parallel::v1::datapar_execution_policy,
        Iter const& first1, Iter const& last1)
    {
        return detail::loop_optimization<Iter>::call(first1, last1);
    }

    template <typename Iter>
    HPX_HOST_DEVICE HPX_FORCEINLINE
    bool loop_optimization(parallel::v1::datapar_task_execution_policy,
        Iter const& first1, Iter const& last1)
    {
        return detail::loop_optimization<Iter>::call(first1, last1);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Begin, typename End, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE Begin
    loop(parallel::v1::datapar_execution_policy, Begin begin, End end, F && f)
    {
        return detail::datapar_loop<Begin>::call(begin, end, std::forward<F>(f));
    }

    template <typename Begin, typename End, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE Begin
    loop(parallel::v1::datapar_task_execution_policy, Begin begin, End end, F && f)
    {
        return detail::datapar_loop<Begin>::call(begin, end, std::forward<F>(f));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename VecOnly, typename Iter1, typename Iter2, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE std::pair<Iter1, Iter2>
    loop2(parallel::v1::datapar_execution_policy, VecOnly,
        Iter1 first1, Iter1 last1, Iter2 first2, F && f)
    {
        return detail::datapar_loop2<VecOnly, Iter1, Iter2>::call(
            first1, last1, first2, std::forward<F>(f));
    }

    template <typename VecOnly, typename Iter1, typename Iter2, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE std::pair<Iter1, Iter2>
    loop2(parallel::v1::datapar_task_execution_policy, VecOnly,
        Iter1 first1, Iter1 last1, Iter2 first2, F && f)
    {
        return detail::datapar_loop2<VecOnly, Iter1, Iter2>::call(
            first1, last1, first2, std::forward<F>(f));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE Iter
    loop_n(parallel::v1::datapar_execution_policy, Iter it,
        std::size_t count, F && f)
    {
        return detail::datapar_loop_n<Iter>::call(it, count, std::forward<F>(f));
    }

    template <typename Iter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE Iter
    loop_n(parallel::v1::datapar_task_execution_policy, Iter it,
        std::size_t count, F && f)
    {
        return detail::datapar_loop_n<Iter>::call(it, count, std::forward<F>(f));
    }
}}}

#endif
#endif

