//  Copyright (c) 2021 Srinivas Yadav
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/local/config.hpp>

#if defined(HPX_HAVE_DATAPAR)
#include <hpx/execution/traits/is_execution_policy.hpp>
#include <hpx/functional/tag_dispatch.hpp>
#include <hpx/parallel/datapar/transform_loop.hpp>
#include <hpx/parallel/util/result_types.hpp>
#include <hpx/parallel/util/transfer.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { namespace util {

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        template <typename Iterator>
        struct datapar_copy_n
        {
            template <typename InIter, typename OutIter>
            HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
                iterators_datapar_compatible<InIter, OutIter>::value &&
                    iterator_datapar_compatible<InIter>::value &&
                    iterator_datapar_compatible<OutIter>::value,
                in_out_result<InIter, OutIter>>::type
            call(InIter first, std::size_t count, OutIter dest)
            {
                auto ret =
                    util::transform_loop_n_ind<hpx::execution::simd_policy>(
                        first, count, dest, [](auto& v) { return v; });

                return util::in_out_result<InIter, OutIter>{
                    std::move(ret.first), std::move(ret.second)};
            }

            template <typename InIter, typename OutIter>
            HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
                !iterators_datapar_compatible<InIter, OutIter>::value ||
                    !iterator_datapar_compatible<InIter>::value ||
                    !iterator_datapar_compatible<OutIter>::value,
                in_out_result<InIter, OutIter>>::type
            call(InIter first, std::size_t count, OutIter dest)
            {
                return util::copy_n<hpx::execution::sequenced_policy>(
                    first, count, dest);
            }
        };
    }    // namespace detail

    template <typename ExPolicy, typename InIter, typename OutIter>
    HPX_HOST_DEVICE HPX_FORCEINLINE typename std::enable_if<
        hpx::is_vectorpack_execution_policy<ExPolicy>::value,
        in_out_result<InIter, OutIter>>::type
    tag_dispatch(hpx::parallel::util::copy_n_t<ExPolicy>, InIter first,
        std::size_t count, OutIter dest)
    {
        return detail::datapar_copy_n<InIter>::call(first, count, dest);
    }
}}}    // namespace hpx::parallel::util
#endif
