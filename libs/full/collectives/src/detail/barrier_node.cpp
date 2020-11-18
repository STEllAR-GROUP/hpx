//  Copyright (c) 2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/assert.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/async_combinators/when_all.hpp>
#include <hpx/collectives/detail/barrier_node.hpp>
#include <hpx/components_base/component_type.hpp>
#include <hpx/modules/async_distributed.hpp>
#include <hpx/modules/execution.hpp>
#include <hpx/modules/futures.hpp>
#include <hpx/modules/memory.hpp>
#include <hpx/pack_traversal/unwrap.hpp>
#include <hpx/util/from_string.hpp>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

typedef hpx::components::managed_component<hpx::lcos::detail::barrier_node>
    barrier_type;

HPX_REGISTER_COMPONENT_HEAP(barrier_type)
HPX_DEFINE_COMPONENT_NAME(
    hpx::lcos::detail::barrier_node, hpx_lcos_barrier_node);

HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    hpx::lcos::detail::barrier_node, hpx::components::component_barrier)

HPX_REGISTER_ACTION(
    hpx::lcos::detail::barrier_node::gather_action, barrier_node_gather_action);

namespace hpx { namespace lcos { namespace detail {
    barrier_node::barrier_node()
      : count_(0)
      , rank_(0)
      , num_(0)
      , arity_(0)
      , cut_off_(0)
      , local_barrier_(0)
    {
        HPX_ASSERT(false);
    }

    barrier_node::barrier_node(
        std::string base_name, std::size_t num, std::size_t rank)
      : count_(0)
      , base_name_(base_name)
      , rank_(rank)
      , num_(num)
      , arity_(hpx::util::from_string<std::size_t>(
            get_config_entry("hpx.lcos.collectives.arity", 32)))
      , cut_off_(hpx::util::from_string<std::size_t>(
            get_config_entry("hpx.lcos.collectives.cut_off", -1)))
      , local_barrier_(num)
    {
        if (num_ >= cut_off_)
        {
            std::vector<std::size_t> ids;
            ids.reserve(children_.size());

            for (std::size_t i = 1; i <= arity_; ++i)
            {
                std::size_t id = (arity_ * rank_) + i;
                if (id >= num)
                    break;
                ids.push_back(id);
            }

            children_ =
                hpx::util::unwrap(hpx::find_from_basename(base_name_, ids));

            return;
        }

        if (rank_ != 0)
        {
            HPX_ASSERT(num_ < cut_off_);
            children_.push_back(hpx::find_from_basename(base_name_, 0).get());
        }
    }

    hpx::future<void> barrier_node::wait(bool async)
    {
        hpx::intrusive_ptr<barrier_node> this_(this);
        future<void> result;

        if (num_ < cut_off_)
        {
            if (rank_ != 0)
            {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
                HPX_ASSERT(children_.size() == 1);
                hpx::lcos::base_lco::set_event_action action;
                result = hpx::async(action, children_[0]);
#else
                HPX_ASSERT(false);
#endif
            }
            else
            {
                if (async)
                {
                    result = hpx::async(&barrier_node::set_event, this_);
                }
                else
                {
                    set_event();
                    result = hpx::make_ready_future();
                }
            }

            // keep everything alive until future has become ready
            traits::detail::get_shared_state(result)->set_on_completed(
                [this_ = std::move(this_)] {});

            return result;
        }

        if (rank_ == 0)
        {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
            // The root process calls the gather action on its children
            // once all those return, we know that everyone entered the
            // barrier
            std::vector<hpx::future<void>> futures;
            futures.reserve(children_.size());
            for (hpx::id_type& id : children_)
            {
                barrier_node::gather_action action;
                futures.push_back(hpx::async(action, id));
            }
            result = hpx::when_all(futures);
#else
            HPX_ASSERT(false);
#endif
        }
        else
        {
            // Non-root flags that it entered the barrier...
            gather_promise_.set_value();

            // The broadcast_promise is set once the root knows that everyone
            // entered the barrier
            result = broadcast_promise_.get_future();
        }

        // keep everything alive until future has become ready
        result = do_wait(this_, std::move(result));

        traits::detail::get_shared_state(result)->set_on_completed(
            [this_ = std::move(this_)] {});

        return result;
    }

    template <typename This>
    hpx::future<void> barrier_node::do_wait(
        This this_, hpx::future<void> future)
    {
        if (rank_ == 0)
        {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
            return future.then(hpx::launch::sync,
                [this_ = std::move(this_)](hpx::future<void>&& f) {
                    // Trigger possible errors...
                    f.get();

                    std::vector<hpx::future<void>> futures;
                    futures.reserve(this_->children_.size());

                    // After the root process received the notification that
                    // everyone entered the barrier, it will broadcast to
                    // everyone that they can leave the barrier
                    for (hpx::id_type& id : this_->children_)
                    {
                        base_lco::set_event_action action;
                        futures.push_back(hpx::async(action, id));
                    }

                    return hpx::when_all(futures);
                });
#else
            HPX_ASSERT(false);
            return hpx::make_ready_future();
#endif
        }

        return future.then(hpx::launch::sync,
            [this_ = std::move(this_)](hpx::future<void>&& f) {
                // Trigger possible errors...
                f.get();

                // Once the non-roots are ready to leave the barrier, we
                // need to reset our promises such that the barrier can be
                // reused.
                this_->broadcast_promise_ = hpx::lcos::local::promise<void>();
                this_->gather_promise_ = hpx::lcos::local::promise<void>();
            });
    }

    template hpx::future<void> barrier_node::do_wait(
        hpx::intrusive_ptr<barrier_node>, hpx::future<void>);
    template hpx::future<void> barrier_node::do_wait(
        barrier_node*, hpx::future<void>);

    hpx::future<void> barrier_node::gather()
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        // We recursively gather the information that everyone entered the
        // barrier. The recursion is started from the root node.
        HPX_ASSERT(rank_ != 0);
        std::vector<hpx::future<void>> futures;
        futures.reserve(children_.size());
        for (hpx::id_type& id : children_)
        {
            barrier_node::gather_action action;
            futures.push_back(hpx::async(action, id));
        }

        hpx::intrusive_ptr<barrier_node> this_(this);

        // Once we know that all our children entered the barrier, we flag ourself
        auto result = hpx::when_all(futures).then(
            hpx::launch::sync, [this_](hpx::future<void> f) {
                // Trigger possible errors...
                f.get();
                return this_->gather_promise_.get_future();
            });

        // keep everything alive until future has become ready
        traits::detail::get_shared_state(result)->set_on_completed(
            [this_ = std::move(this_)] {});

        return result;
#else
        HPX_ASSERT(false);
        return hpx::make_ready_future();
#endif
    }

    void barrier_node::set_event()
    {
        if (num_ < cut_off_)
        {
            local_barrier_.wait();
            return;
        }

#if !defined(HPX_COMPUTE_DEVICE_CODE)
        // We recursively broadcast the information that everyone entered the
        // barrier. The recursion is started from the root node.
        HPX_ASSERT(rank_ != 0);
        std::vector<hpx::future<void>> futures;
        futures.reserve(children_.size());
        for (hpx::id_type& id : children_)
        {
            base_lco::set_event_action action;
            futures.push_back(hpx::async(action, id));
        }

        // Once we notified our children, we mark ourself ready.
        hpx::intrusive_ptr<barrier_node> this_(this);
        hpx::when_all(futures)
            .then(hpx::launch::sync,
                [this_ = std::move(this_)](future<void> f) {
                    // Trigger possible errors...
                    f.get();
                    this_->broadcast_promise_.set_value();
                })
            .get();
#else
        HPX_ASSERT(false);
#endif
    }
}}}    // namespace hpx::lcos::detail
