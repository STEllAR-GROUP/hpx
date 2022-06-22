//  Copyright (c) 2022 Srinivas Yadav
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR_EVE)
#include <eve/eve.hpp>
#include <eve/function/store.hpp>
#include <eve/memory/aligned_ptr.hpp>

#include <memory>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace traits {

    ///////////////////////////////////////////////////////////////////////////
    template <typename V, typename ValueType, typename Enable>
    struct vector_pack_load
    {
        template <typename Iter>
        HPX_HOST_DEVICE HPX_FORCEINLINE static V aligned(Iter const& iter)
        {
            return V(
                eve::as_aligned(std::addressof(*iter), eve::cardinal_t<V>{}));
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
            eve::store(value,
                eve::as_aligned(std::addressof(*iter), eve::cardinal_t<V>{}));
        }

        template <typename Iter>
        HPX_HOST_DEVICE HPX_FORCEINLINE static void unaligned(
            V& value, Iter const& iter)
        {
            *iter = value;
            return;
        }
    };
}}}    // namespace hpx::parallel::traits

#endif
