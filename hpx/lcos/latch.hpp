//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_LATCH_APR_19_2015_1002AM)
#define HPX_LCOS_LATCH_APR_19_2015_1002AM

#include <hpx/exception.hpp>
#include <hpx/runtime/components/client_base.hpp>
#include <hpx/runtime/components/new.hpp>
#include <hpx/lcos/server/latch.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos
{
    class latch : public components::client_base<latch, lcos::server::latch>
    {
        typedef components::client_base<latch, lcos::server::latch> base_type;

    public:
        latch()
        {}

        /// Initialize the latch
        ///
        /// Requires: count >= 0.
        /// Synchronization: None
        /// Postconditions: counter_ == count.
        ///
        explicit latch(std::ptrdiff_t count)
          : base_type(hpx::new_<lcos::server::latch>(hpx::find_here(), count))
        {
        }

        /// Extension: Create a client side representation for the existing
        /// \a server#latch instance with the given global id \a id.
        latch(naming::id_type id)
          : base_type(id)
        {}

        /// Extension: Create a client side representation for the existing
        /// \a server#latch instance with the given global id \a id.
        latch(hpx::future<naming::id_type> && id)
          : base_type(id.share())
        {}

        /// Extension: Create a client side representation for the existing
        /// \a server#latch instance with the given global id \a id.
        latch(hpx::shared_future<naming::id_type> const& id)
          : base_type(id)
        {}

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
        bool is_ready() const HPX_NOEXCEPT
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
        hpx::future<void> count_down_and_wait_async()
        {
            lcos::server::latch::set_event_action act;
            return hpx::async(act, get_id());
        }

        hpx::future<void> count_down_async(std::ptrdiff_t n)
        {
            lcos::server::latch::set_value_action act;
            return hpx::async(act, get_id(), std::move(n));
        }

        hpx::future<bool> is_ready_async() const
        {
            lcos::server::latch::get_value_action act;
            return hpx::async(act, get_id());
        }

        hpx::future<void> wait_async() const
        {
            lcos::server::latch::wait_action act;
            return hpx::async(act, get_id());
        }

        hpx::future<void> set_exception_async(boost::exception_ptr const& e)
        {
            lcos::server::latch::set_exception_action act;
             return hpx::async(act, get_id(), e);
        }
        void set_exception(boost::exception_ptr const& e)
        {
            set_exception_async(e).get();
        }
        /// \endcond
    };
}}

#endif

