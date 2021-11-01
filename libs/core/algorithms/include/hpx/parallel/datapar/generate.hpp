//  Copyright (c) 2021 Srinivas Yadav
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR)
#include <hpx/execution/traits/is_execution_policy.hpp>
#include <hpx/execution/traits/vector_pack_alignment_size.hpp>
#include <hpx/functional/tag_dispatch.hpp>
#include <hpx/parallel/algorithms/detail/generate.hpp>
#include <hpx/parallel/datapar/loop.hpp>
#include <hpx/parallel/util/result_types.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { inline namespace v1 { namespace detail {

    template <typename Iterator>
    struct datapar_generate_helper
    {
        typedef typename std::decay<Iterator>::type iterator_type;
        typedef
            typename std::iterator_traits<iterator_type>::value_type value_type;

        template <typename Iter, typename F>
        static void call(Iter first, std::size_t count, F&& f)
        {
            using V = std::decay_t<decltype(f())>;

            static constexpr std::size_t size =
                traits::vector_pack_size<V>::value;
            std::size_t len = count;

            for (std::int64_t len_v = std::int64_t(len - (size + 1)); len_v > 0;
                 len_v -= size, len -= size)
            {
                auto tmp = f();

                if (util::detail::is_data_aligned(first))
                    traits::vector_pack_store<V, value_type>::aligned(
                        tmp, first);
                else
                    traits::vector_pack_store<V, value_type>::unaligned(
                        tmp, first);
                std::advance(first, size);
            }
            auto tmp = f();
            std::size_t i = 0;
            for (/* */; len != 0; --len)
            {
                *first++ = tmp[i++];
            }
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    struct datapar_generate
    {
        template <typename ExPolicy, typename Iter, typename Sent, typename F>
        HPX_HOST_DEVICE HPX_FORCEINLINE static Iter call(
            ExPolicy&&, Iter first, Sent last, F&& f)
        {
            using result_type = std::decay_t<decltype(f())>;
            static_assert(traits::is_vector_pack<result_type>::value ||
                    traits::is_scalar_vector_pack<result_type>::value,
                "Function object must return a vector_pack");

            std::size_t count = std::distance(first, last);
            datapar_generate_helper<Iter>::call(
                first, count, std::forward<F>(f));
            return first;
        }

    };

    template <typename ExPolicy, typename Iter, typename Sent, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE typename std::enable_if<
        hpx::is_vectorpack_execution_policy<ExPolicy>::value, Iter>::type
    tag_dispatch(
        sequential_generate_t, ExPolicy&& policy, Iter first, Sent last, F&& f)
    {
        return datapar_generate::call(
            std::forward<ExPolicy>(policy), first, last, std::forward<F>(f));
    }

    ///////////////////////////////////////////////////////////////////////////
    struct datapar_generate_n
    {
        template <typename ExPolicy, typename Iter, typename F>
        HPX_HOST_DEVICE HPX_FORCEINLINE static Iter call(
            ExPolicy&&, Iter first, std::size_t count, F&& f)
        {
            using result_type = std::decay_t<decltype(f())>;
            static_assert(traits::is_vector_pack<result_type>::value ||
                    traits::is_scalar_vector_pack<result_type>::value,
                "Function object must return a vector_pack");

            datapar_generate_helper<Iter>::call(
                first, count, std::forward<F>(f));
            return first;
        }

    };

    template <typename ExPolicy, typename Iter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE typename std::enable_if<
        hpx::is_vectorpack_execution_policy<ExPolicy>::value, Iter>::type
    tag_dispatch(sequential_generate_n_t, ExPolicy&& policy, Iter first,
        std::size_t count, F&& f)
    {
        return datapar_generate_n::call(
            std::forward<ExPolicy>(policy), first, count, std::forward<F>(f));
    }
}}}}    // namespace hpx::parallel::v1::detail
#endif
