//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_DATAPAR_TRANSFORM_LOOP_SEP_08_2016_0657PM)
#define HPX_PARALLEL_DATAPAR_TRANSFORM_LOOP_SEP_08_2016_0657PM

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR)
#include <hpx/parallel/datapar/detail/vc/transform_loop.hpp>
#include <hpx/parallel/datapar/detail/boost_simd/transform_loop.hpp>
#include <hpx/parallel/datapar/execution_policy_fwd.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Iter, typename OutIter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE
    std::pair<Iter, OutIter>
    transform_loop_n(parallel::v1::datapar_execution_policy, Iter it,
        std::size_t count, OutIter dest, F && f)
    {
        return detail::datapar_transform_loop_n<Iter>::call(it, count, dest,
            std::forward<F>(f));
    }

    template <typename Iter, typename OutIter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE
    std::pair<Iter, OutIter>
    transform_loop_n(parallel::v1::datapar_task_execution_policy, Iter it,
        std::size_t count, OutIter dest, F && f)
    {
        return detail::datapar_transform_loop_n<Iter>::call(it, count, dest,
            std::forward<F>(f));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iter, typename OutIter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE
    std::pair<Iter, OutIter>
    transform_loop(parallel::v1::datapar_execution_policy, Iter it, Iter end,
        OutIter dest, F && f)
    {
        return detail::datapar_transform_loop<Iter>::call(it, end, dest,
            std::forward<F>(f));
    }

    template <typename Iter, typename OutIter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE
    std::pair<Iter, OutIter>
    transform_loop(parallel::v1::datapar_task_execution_policy, Iter it,
        Iter end, OutIter dest, F && f)
    {
        return detail::datapar_transform_loop<Iter>::call(it, end, dest,
            std::forward<F>(f));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename InIter1, typename InIter2, typename OutIter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE
    hpx::util::tuple<InIter1, InIter2, OutIter>
    transform_binary_loop_n(parallel::v1::datapar_execution_policy,
        InIter1 first1, std::size_t count, InIter2 first2, OutIter dest, F && f)
    {
        return detail::datapar_transform_binary_loop_n<InIter1, InIter2>::call(
            first1, count, first2, dest, std::forward<F>(f));
    }

    template <typename InIter1, typename InIter2, typename OutIter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE
    hpx::util::tuple<InIter1, InIter2, OutIter>
    transform_binary_loop_n(parallel::v1::datapar_task_execution_policy,
        InIter1 first1, std::size_t count, InIter2 first2, OutIter dest, F && f)
    {
        return detail::datapar_transform_binary_loop_n<InIter1, InIter2>::call(
            first1, count, first2, dest, std::forward<F>(f));
    }

    ///////////////////////////////////////////////////////////////////////////

    template <typename InIter1, typename InIter2, typename OutIter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE
    hpx::util::tuple<InIter1, InIter2, OutIter>
    transform_binary_loop(parallel::v1::datapar_execution_policy,
        InIter1 first1, InIter1 last1, InIter2 first2, OutIter dest, F && f)
    {
        return detail::datapar_transform_binary_loop<InIter1, InIter2>::call(
            first1, last1, first2, dest, std::forward<F>(f));
    }

    template <typename InIter1, typename InIter2, typename OutIter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE
    hpx::util::tuple<InIter1, InIter2, OutIter>
    transform_binary_loop(parallel::v1::datapar_task_execution_policy,
        InIter1 first1, InIter1 last1, InIter2 first2, OutIter dest, F && f)
    {
        return detail::datapar_transform_binary_loop<InIter1, InIter2>::call(
            first1, last1, first2, dest, std::forward<F>(f));
    }

    template <typename InIter1, typename InIter2, typename OutIter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE
    hpx::util::tuple<InIter1, InIter2, OutIter>
    transform_binary_loop(parallel::v1::datapar_execution_policy,
        InIter1 first1, InIter1 last1, InIter2 first2, InIter2 last2,
        OutIter dest, F && f)
    {
        return detail::datapar_transform_binary_loop<InIter1, InIter2>::call(
            first1, last1, first2, last2, dest, std::forward<F>(f));
    }

    template <typename InIter1, typename InIter2,
        typename OutIter, typename F, typename Proj1, typename Proj2>
    HPX_HOST_DEVICE HPX_FORCEINLINE
    hpx::util::tuple<InIter1, InIter2, OutIter>
    transform_binary_loop(parallel::v1::datapar_task_execution_policy,
        InIter1 first1, InIter1 last1, InIter2 first2, InIter2 last2,
        OutIter dest, F && f)
    {
        return detail::datapar_transform_binary_loop<InIter1, InIter2>::call(
            first1, last1, first2, last2, dest, std::forward<F>(f));
    }
}}}

#endif
#endif

