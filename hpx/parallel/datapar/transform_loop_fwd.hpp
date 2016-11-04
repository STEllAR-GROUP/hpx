//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_DATAPAR_TRANSFORM_LOOP_FWD_SEP_22_2016_0811PM)
#define HPX_PARALLEL_DATAPAR_TRANSFORM_LOOP_FWD_SEP_22_2016_0811PM

#include <hpx/config.hpp>
#include <hpx/util/tuple.hpp>

#include <cstddef>
#include <utility>

namespace hpx { namespace parallel { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy, typename Iter, typename OutIter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE
    std::pair<Iter, OutIter>
    transform_loop_n(ExPolicy &&, Iter it, std::size_t count, OutIter dest,
        F && f);

    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy, typename Iter, typename OutIter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE
    std::pair<Iter, OutIter>
    transform_loop(ExPolicy &&, Iter it, Iter end, OutIter dest, F && f);

    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy, typename InIter1, typename InIter2,
        typename OutIter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE
    hpx::util::tuple<InIter1, InIter2, OutIter>
    transform_binary_loop_n(ExPolicy &&, InIter1 first1, std::size_t count,
        InIter2 first2, OutIter dest, F && f);

    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy, typename InIter1, typename InIter2,
        typename OutIter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE
    hpx::util::tuple<InIter1, InIter2, OutIter>
    transform_binary_loop(ExPolicy &&, InIter1 first1, InIter1 last1,
        InIter2 first2, OutIter dest, F && f);

    template <typename ExPolicy, typename InIter1, typename InIter2,
        typename OutIter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE
    hpx::util::tuple<InIter1, InIter2, OutIter>
    transform_binary_loop(ExPolicy &&, InIter1 first1, InIter1 last1,
        InIter2 first2, InIter2 last2, OutIter dest, F && f);
}}}

#endif

