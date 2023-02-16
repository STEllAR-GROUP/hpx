//  Copyright (c) 2007-2023 Hartmut Kaiser
//  Copyright (c) 2014 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/iterator_support/zip_iterator.hpp>
#include <hpx/modules/execution.hpp>
#include <hpx/parallel/util/result_types.hpp>

#include <utility>

namespace hpx::parallel::detail {

    ///////////////////////////////////////////////////////////////////////////
    template <int N, typename R, typename ZipIter>
    constexpr R get_iter(ZipIter&& zipiter)
    {
        return hpx::get<N>(zipiter.get_iterator_tuple());
    }

    template <int N, typename R, typename ZipIter>
    R get_iter(hpx::future<ZipIter>&& zipiter)
    {
        using result_type = typename hpx::tuple_element<N,
            typename ZipIter::iterator_tuple_type>::type;

        return hpx::make_future<result_type>(
            HPX_MOVE(zipiter), [](ZipIter zipiter) {
                return get_iter<N, result_type>(HPX_MOVE(zipiter));
            });
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename ZipIter>
    constexpr typename ZipIter::iterator_tuple_type get_iter_tuple(
        ZipIter&& zipiter)
    {
        return zipiter.get_iterator_tuple();
    }

    template <typename ZipIter>
    hpx::future<typename ZipIter::iterator_tuple_type> get_iter_tuple(
        hpx::future<ZipIter>&& zipiter)
    {
        using result_type = typename ZipIter::iterator_tuple_type;
        return hpx::make_future<result_type>(HPX_MOVE(zipiter),
            [](ZipIter zipiter) { return get_iter_tuple(HPX_MOVE(zipiter)); });
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename ZipIter>
    constexpr std::pair<typename hpx::tuple_element<0,
                            typename ZipIter::iterator_tuple_type>::type,
        typename hpx::tuple_element<1,
            typename ZipIter::iterator_tuple_type>::type>
    get_iter_pair(ZipIter&& zipiter)
    {
        using iterator_tuple_type = typename ZipIter::iterator_tuple_type;

        iterator_tuple_type t = zipiter.get_iterator_tuple();
        return std::make_pair(hpx::get<0>(t), hpx::get<1>(t));
    }

    template <typename ZipIter>
    hpx::future<std::pair<typename hpx::tuple_element<0,
                              typename ZipIter::iterator_tuple_type>::type,
        typename hpx::tuple_element<1,
            typename ZipIter::iterator_tuple_type>::type>>
    get_iter_pair(hpx::future<ZipIter>&& zipiter)
    {
        using iterator_tuple_type = typename ZipIter::iterator_tuple_type;

        using result_type =
            std::pair<typename hpx::tuple_element<0, iterator_tuple_type>::type,
                typename hpx::tuple_element<1, iterator_tuple_type>::type>;

        return hpx::make_future<result_type>(HPX_MOVE(zipiter),
            [](ZipIter zipiter) { return get_iter_pair(HPX_MOVE(zipiter)); });
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename ZipIter>
    constexpr util::in_in_result<
        typename hpx::tuple_element<0,
            typename ZipIter::iterator_tuple_type>::type,
        typename hpx::tuple_element<1,
            typename ZipIter::iterator_tuple_type>::type>
    get_iter_in_in_result(ZipIter&& zipiter)
    {
        using iterator_tuple_type = typename ZipIter::iterator_tuple_type;

        using result_type = util::in_in_result<
            typename hpx::tuple_element<0, iterator_tuple_type>::type,
            typename hpx::tuple_element<1, iterator_tuple_type>::type>;

        iterator_tuple_type t = zipiter.get_iterator_tuple();
        return result_type{hpx::get<0>(t), hpx::get<1>(t)};
    }

    template <typename ZipIter>
    hpx::future<
        util::in_in_result<typename hpx::tuple_element<0,
                               typename ZipIter::iterator_tuple_type>::type,
            typename hpx::tuple_element<1,
                typename ZipIter::iterator_tuple_type>::type>>
    get_iter_in_in_result(hpx::future<ZipIter>&& zipiter)
    {
        using iterator_tuple_type = typename ZipIter::iterator_tuple_type;

        using result_type = util::in_in_result<
            typename hpx::tuple_element<0, iterator_tuple_type>::type,
            typename hpx::tuple_element<1, iterator_tuple_type>::type>;

        return hpx::make_future<result_type>(HPX_MOVE(zipiter),
            [](ZipIter iter) { return get_iter_in_in_result(HPX_MOVE(iter)); });
    }
}    // namespace hpx::parallel::detail
