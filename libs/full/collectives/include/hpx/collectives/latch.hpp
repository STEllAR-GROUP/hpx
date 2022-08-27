//  Copyright (c) 2007-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/collectives/detail/latch.hpp>
#include <hpx/components/client_base.hpp>
#include <hpx/futures/future.hpp>

#include <cstddef>
#include <exception>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::distributed {

    class HPX_EXPORT latch
      : public components::client_base<latch, hpx::lcos::server::latch>
    {
        typedef components::client_base<latch, hpx::lcos::server::latch>
            base_type;

    public:
        latch() = default;

        /// Initialize the latch
        ///
        /// Requires: count >= 0.
        /// Synchronization: None
        /// Postconditions: counter_ == count.
        ///
        explicit latch(std::ptrdiff_t count);

        /// Extension: Create a client side representation for the existing
        /// \a server#latch instance with the given global id \a id.
        latch(hpx::id_type const& id)
          : base_type(id)
        {
        }

        /// Extension: Create a client side representation for the existing
        /// \a server#latch instance with the given global id \a id.
        latch(hpx::future<hpx::id_type>&& f)
          : base_type(HPX_MOVE(f))
        {
        }

        /// Extension: Create a client side representation for the existing
        /// \a server#latch instance with the given global id \a id.
        latch(hpx::shared_future<hpx::id_type> const& id)
          : base_type(id)
        {
        }
        latch(hpx::shared_future<hpx::id_type>&& id)
          : base_type(HPX_MOVE(id))
        {
        }

        ///////////////////////////////////////////////////////////////////////
        /// Decrements counter_ by 1 . Blocks at the synchronization point
        /// until counter_ reaches 0.
        ///
        /// Requires: counter_ > 0.
        ///
        /// Synchronization: Synchronizes with all calls that block on this
        /// latch and with all is_ready calls on this latch that return true.
        ///
        /// \throws Nothing.
        ///
        void count_down_and_wait()
        {
            count_down_and_wait_async().get();
        }

        /// Decrements counter_ by update . Blocks at the synchronization point
        /// until counter_ reaches 0.
        ///
        /// Requires: counter_ > 0.
        ///
        /// Synchronization: Synchronizes with all calls that block on this
        /// latch and with all is_ready calls on this latch that return true.
        ///
        /// \throws Nothing.
        ///
        void arrive_and_wait()
        {
            count_down_and_wait_async().get();
        }

        /// Decrements counter_ by n. Does not block.
        ///
        /// Requires: counter_ >= n and n >= 0.
        ///
        /// Synchronization: Synchronizes with all calls that block on this
        /// latch and with all is_ready calls on this latch that return true .
        ///
        /// \throws Nothing.
        ///
        void count_down(std::ptrdiff_t n)
        {
            count_down_async(n).get();
        }

        /// Returns: counter_ == 0. Does not block.
        ///
        /// \throws Nothing.
        ///
        bool is_ready() const noexcept
        {
            return is_ready_async().get();
        }

        /// Returns: counter_ == 0. Does not block.
        ///
        /// \throws Nothing.
        ///
        bool try_wait() const noexcept
        {
            return is_ready_async().get();
        }

        /// If counter_ is 0, returns immediately. Otherwise, blocks the
        /// calling thread at the synchronization point until counter_
        /// reaches 0.
        ///
        /// \throws Nothing.
        ///
        void wait() const
        {
            return wait_async().get();
        }

        /// \cond NOINTERNAL
        // extended API
        hpx::future<void> count_down_and_wait_async();

        hpx::future<void> count_down_async(std::ptrdiff_t n);

        hpx::future<bool> is_ready_async() const;

        hpx::future<void> wait_async() const;

        hpx::future<void> set_exception_async(std::exception_ptr const& e);

        void set_exception(std::exception_ptr const& e)
        {
            set_exception_async(e).get();
        }
        /// \endcond
    };
}    // namespace hpx::distributed

namespace hpx::lcos {

    /// \cond NOINTERNAL
    using latch HPX_DEPRECATED_V(1, 8,
        "hpx::lcos::latch is deprecated, use hpx::distributed::latch instead") =
        hpx::distributed::latch;
    /// \endcond
}    // namespace hpx::lcos
