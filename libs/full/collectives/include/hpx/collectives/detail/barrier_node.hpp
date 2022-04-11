//  Copyright (c) 2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/actions/transfer_action.hpp>
#include <hpx/actions_base/component_action.hpp>
#include <hpx/assert.hpp>
#include <hpx/async_distributed/base_lco.hpp>
#include <hpx/async_distributed/transfer_continuation_action.hpp>
#include <hpx/components_base/server/managed_component_base.hpp>
#include <hpx/components_base/traits/managed_component_policies.hpp>
#include <hpx/modules/futures.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/synchronization/barrier.hpp>
#include <hpx/synchronization/spinlock.hpp>
#include <hpx/thread_support/atomic_count.hpp>
#include <hpx/type_support/unused.hpp>

#include <cstddef>
#include <string>
#include <vector>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace distributed { namespace detail {

    struct HPX_EXPORT barrier_node;
}}}    // namespace hpx::distributed::detail

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace traits {
    template <>
    struct managed_component_dtor_policy<hpx::distributed::detail::barrier_node>
    {
        typedef managed_object_is_lifetime_controlled type;
    };
}}    // namespace hpx::traits

namespace hpx { namespace distributed { namespace detail {

    struct barrier_node : hpx::lcos::base_lco
    {
        typedef components::managed_component<barrier_node> wrapping_type;
        typedef hpx::spinlock mutex_type;

        barrier_node();
        barrier_node(std::string base_name, std::size_t num, std::size_t rank);

        void set_event();
        hpx::future<void> gather();

        hpx::future<void> wait(bool async);

        HPX_DEFINE_COMPONENT_DIRECT_ACTION(barrier_node, gather)

    private:
        hpx::util::atomic_count count_;

        std::vector<hpx::id_type> children_;

    public:
        std::string base_name_;
        std::size_t rank_;
        std::size_t num_;
        std::size_t arity_;
        std::size_t cut_off_;

    private:
        hpx::promise<void> gather_promise_;
        hpx::promise<void> broadcast_promise_;
        hpx::barrier<> local_barrier_;

        hpx::future<void> do_wait(hpx::future<void> future);

        template <typename>
        friend struct components::detail_adl_barrier::init;

        void set_back_ptr(components::managed_component<barrier_node>* bp)
        {
            HPX_ASSERT(bp);
            HPX_UNUSED(bp);
        }

        // intrusive reference counting
        friend void intrusive_ptr_add_ref(barrier_node* p) noexcept
        {
            ++p->count_;
        }

        // intrusive reference counting
        friend void intrusive_ptr_release(barrier_node* p) noexcept
        {
            if (p && --p->count_ == 0)
            {
                delete p;
            }
        }
    };
}}}    // namespace hpx::distributed::detail

HPX_REGISTER_ACTION_DECLARATION(
    hpx::distributed::detail::barrier_node::gather_action,
    barrier_node_gather_action)

#include <hpx/config/warnings_suffix.hpp>
