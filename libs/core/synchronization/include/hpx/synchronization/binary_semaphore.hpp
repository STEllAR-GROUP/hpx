//  Copyright (c) 2007-2022 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/synchronization/counting_semaphore.hpp>
#include <hpx/synchronization/spinlock.hpp>

#include <cstddef>

#if defined(HPX_MSVC_WARNING_PRAGMA)
#pragma warning(push)
#pragma warning(disable : 4251)
#endif

////////////////////////////////////////////////////////////////////////////////
namespace hpx {

    ///////////////////////////////////////////////////////////////////////////
    // A binary semaphore should be more efficient than the default
    // implementation of a counting semaphore with a unit resource count.
    namespace detail {

        template <typename Mutex = hpx::spinlock>
        class binary_semaphore : public counting_semaphore<1, Mutex>
        {
        public:
            binary_semaphore(binary_semaphore const&) = delete;
            binary_semaphore& operator=(binary_semaphore const&) = delete;
            binary_semaphore(binary_semaphore&&) = delete;
            binary_semaphore& operator=(binary_semaphore&&) = delete;

        public:
            explicit binary_semaphore(std::ptrdiff_t value = 1)
              : counting_semaphore<1, Mutex>(value)
            {
            }

            ~binary_semaphore() = default;
        };
    }    // namespace detail

    using binary_semaphore = detail::binary_semaphore<>;
}    // namespace hpx

/// \cond NOINTERN
namespace hpx::lcos::local {

    template <typename Mutex = hpx::spinlock>
    using cpp20_binary_semaphore HPX_DEPRECATED_V(1, 8,
        "hpx::lcos::local::cpp20_binary_semaphore is deprecated, use "
        "hpx::binary_semaphore instead") = hpx::detail::binary_semaphore<Mutex>;
}
/// \endcond

#if defined(HPX_MSVC_WARNING_PRAGMA)
#pragma warning(pop)
#endif
