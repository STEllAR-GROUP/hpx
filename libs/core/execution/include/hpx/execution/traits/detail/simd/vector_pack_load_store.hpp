//  Copyright (c) 2021 Srinivas Yadav
//  Copyright (c) 2016-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR_EXPERIMENTAL_SIMD)

#include <hpx/execution/traits/detail/simd/vector_pack_simd.hpp>

#include <cstddef>
#include <iterator>
#include <memory>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::parallel::traits {

    ///////////////////////////////////////////////////////////////////////////
    template <typename V, typename ValueType, typename Enable>
    struct vector_pack_load
    {
        template <typename Iter>
        HPX_HOST_DEVICE HPX_FORCEINLINE static V aligned(Iter const& iter)
        {
            return V(
                std::addressof(*iter), datapar::experimental::vector_aligned);
        }

        template <typename Iter>
        HPX_HOST_DEVICE HPX_FORCEINLINE static V unaligned(Iter const& iter)
        {
            return *iter;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename V, typename ValueType, typename Enable>
    struct vector_pack_store
    {
        template <typename Iter>
        HPX_HOST_DEVICE HPX_FORCEINLINE static void aligned(
            V& value, Iter const& iter)
        {
            value.copy_to(
                std::addressof(*iter), datapar::experimental::vector_aligned);
        }

        template <typename Iter>
        HPX_HOST_DEVICE HPX_FORCEINLINE static void unaligned(
            V& value, Iter const& iter)
        {
            *iter = value;
        }
    };
}    // namespace hpx::parallel::traits

#endif
