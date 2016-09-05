//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_LCOS_DETAIL_BARRIER_NODE_HPP
#define HPX_LCOS_DETAIL_BARRIER_NODE_HPP

#include <hpx/config.hpp>
#include <hpx/lcos/base_lco.hpp>
#include <hpx/lcos/local/promise.hpp>
#include <hpx/lcos/local/barrier.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/util/atomic_count.hpp>
#include <hpx/traits/managed_component_policies.hpp>

#include <cstddef>
#include <string>
#include <vector>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace lcos { namespace detail {
    struct HPX_EXPORT barrier_node;
}}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx {
namespace traits {
    template <>
    struct managed_component_dtor_policy<
        lcos::detail::barrier_node>
    {
        typedef managed_object_is_lifetime_controlled type;
    };
}
}

namespace hpx { namespace lcos { namespace detail {
    struct barrier_node : base_lco
    {
        typedef components::managed_component<barrier_node> wrapping_type;
        typedef hpx::lcos::local::spinlock mutex_type;

        barrier_node();
        barrier_node(std::string base_name, std::size_t num, std::size_t rank);
        void set_event();
        hpx::future<void> gather();

        hpx::future<void> wait(bool async);

        HPX_DEFINE_COMPONENT_DIRECT_ACTION(barrier_node, gather);

    private:
        hpx::util::atomic_count count_;

        std::vector<naming::id_type> children_;

    public:
        std::string base_name_;
        std::size_t rank_;
        std::size_t num_;
        std::size_t arity_;
        std::size_t cut_off_;
    private:
        hpx::lcos::local::promise<void> gather_promise_;
        hpx::lcos::local::promise<void> broadcast_promise_;
        hpx::lcos::local::barrier local_barrier_;

        template <typename This>
        hpx::future<void> do_wait(This this_, hpx::future<void> future);

        template <typename>
        friend struct components::detail_adl_barrier::init;

        void set_back_ptr(components::managed_component<barrier_node>* bp)
        {
            HPX_ASSERT(bp);
        }

        // intrusive reference counting
        friend void intrusive_ptr_add_ref(barrier_node* p)
        {
            ++p->count_;
        }

        // intrusive reference counting
        friend void intrusive_ptr_release(barrier_node* p)
        {
            if (--p->count_ == 0)
            {
                delete p;
            }
        }
    };
}}}

HPX_REGISTER_ACTION_DECLARATION(hpx::lcos::detail::barrier_node::gather_action,
    barrier_node_gather_action);

#include <hpx/config/warnings_suffix.hpp>

#endif
