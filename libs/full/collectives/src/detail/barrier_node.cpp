//  Copyright (c) 2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/assert.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/async_combinators/when_all.hpp>
#include <hpx/collectives/detail/barrier_node.hpp>
#include <hpx/components/basename_registration_fwd.hpp>
#include <hpx/components_base/component_type.hpp>
#include <hpx/modules/async_distributed.hpp>
#include <hpx/modules/execution.hpp>
#include <hpx/modules/futures.hpp>
#include <hpx/modules/logging.hpp>
#include <hpx/modules/memory.hpp>
#include <hpx/pack_traversal/unwrap.hpp>
#include <hpx/util/from_string.hpp>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

using barrier_type =
    hpx::components::managed_component<hpx::distributed::detail::barrier_node>;

HPX_REGISTER_COMPONENT_HEAP(barrier_type)
HPX_DEFINE_COMPONENT_NAME(
    hpx::distributed::detail::barrier_node, hpx_lcos_barrier_node)

HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(hpx::distributed::detail::barrier_node,
    to_int(hpx::components::component_enum_type::barrier))

HPX_REGISTER_ACTION(hpx::distributed::detail::barrier_node::gather_action,
    barrier_node_gather_action)

namespace hpx::distributed::detail {

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
        std::string base_name, std::size_t const num, std::size_t const rank)
      : count_(0)
      , base_name_(HPX_MOVE(base_name))
      , rank_(rank)
      , num_(num)
      , arity_(hpx::util::from_string<std::size_t>(
            get_config_entry("hpx.lcos.collectives.arity", 32)))    //-V112
      , cut_off_(hpx::util::from_string<std::size_t>(get_config_entry(
            "hpx.lcos.collectives.cut_off", static_cast<std::size_t>(-1))))
      , local_barrier_(static_cast<std::ptrdiff_t>(num))
    {
        LRT_(info).format("creating barrier_node: base_name({}), num({}), "
                          "rank({}), cutoff({}), arity({})",
            base_name_, num, rank, cut_off_, arity_);

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

            children_ = hpx::unwrap(hpx::find_from_basename(base_name_, ids));

            return;
        }

        if (rank_ != 0)
        {
            children_.push_back(hpx::find_from_basename(base_name_, 0).get());
        }
    }

    hpx::future<void> barrier_node::wait(bool const async)
    {
        LRT_(info).format(
            "barrier_node::wait: async({}), rank_({})", async, rank_);

        hpx::intrusive_ptr<barrier_node> this_(this);
        hpx::future<void> result;

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
                [this_ = HPX_MOVE(this_)] {
                    LRT_(info).format(
                        "barrier_node::wait: rank_({}): waiting done",
                        this_->rank_);
                });

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
        result = do_wait(HPX_MOVE(result));

        traits::detail::get_shared_state(result)->set_on_completed(
            [this_ = HPX_MOVE(this_)] {
                LRT_(info).format("barrier_node::wait: rank_({}): waiting done",
                    this_->rank_);
            });

        return result;
    }

    hpx::future<void> barrier_node::do_wait(hpx::future<void> future)
    {
        hpx::intrusive_ptr<barrier_node> this_(this);

        LRT_(info).format("barrier_node::do_wait: rank_({})", rank_);

        if (rank_ == 0)
        {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
            hpx::future<void> result =
                future.then(hpx::launch::sync, [this_](hpx::future<void>&& f) {
                    LRT_(info).format("barrier_node::do_wait: rank_({}): "
                                      "entering barrier done",
                        this_->rank_);

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

            traits::detail::get_shared_state(result)->set_on_completed(
                [this_ = HPX_MOVE(this_)] {
                    LRT_(info).format(
                        "barrier_node::do_wait: rank_({}): all done",
                        this_->rank_);
                });

            return result;
#else
            HPX_ASSERT(false);
            return hpx::make_ready_future();
#endif
        }

        hpx::future<void> result =
            future.then(hpx::launch::sync, [this_](hpx::future<void>&& f) {
                LRT_(info).format(
                    "barrier_node::do_wait: rank_({}): entering barrier done",
                    this_->rank_);

                // Trigger possible errors...
                f.get();

                // Once the non-roots are ready to leave the barrier, we
                // need to reset our promises such that the barrier can be
                // reused.
                this_->broadcast_promise_ = hpx::promise<void>();
                this_->gather_promise_ = hpx::promise<void>();
            });

        // keep everything alive until future has become ready
        traits::detail::get_shared_state(result)->set_on_completed(
            [this_ = HPX_MOVE(this_)] {
                LRT_(info).format(
                    "barrier_node::do_wait: rank_({}): all done", this_->rank_);
            });

        return result;
    }

    hpx::future<void> barrier_node::gather()
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        LRT_(info).format(
            "barrier_node::gather: rank_({}): recursively gather", rank_);

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

        // Once we know that all our children entered the barrier, we
        // flag ourselves
        hpx::future<void> result = hpx::when_all(futures).then(
            hpx::launch::sync, [this_](hpx::future<void>&& f) {
                LRT_(info).format(
                    "barrier_node::gather: rank_({}): recursive gather done",
                    this_->rank_);

                // Trigger possible errors...
                f.get();
                return this_->gather_promise_.get_future();
            });

        // keep everything alive until future has become ready
        traits::detail::get_shared_state(result)->set_on_completed(
            [this_ = HPX_MOVE(this_)] {
                LRT_(info).format(
                    "barrier_node::gather: rank_({}): all done", this_->rank_);
            });

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
            LRT_(info).format(
                "barrier_node::set_event: rank_({}): entering local barrier",
                rank_);

            local_barrier_.arrive_and_wait();

            LRT_(info).format(
                "barrier_node::set_event: rank_({}): exiting local barrier",
                rank_);
            return;
        }

#if !defined(HPX_COMPUTE_DEVICE_CODE)
        LRT_(info).format(
            "barrier_node::set_event: rank_({}): recursively broadcast", rank_);

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

        // Once we notified our children, we mark ourselves ready.
        hpx::intrusive_ptr<barrier_node> this_(this);
        hpx::future<void> result = hpx::when_all(futures).then(
            hpx::launch::sync, [this_](hpx::future<void>&& f) {
                LRT_(info).format(
                    "barrier_node::set_event: rank_({}): mark ourselves ready",
                    this_->rank_);

                // Trigger possible errors...
                f.get();
                this_->broadcast_promise_.set_value();
            });

        // keep everything alive until future has become ready
        traits::detail::get_shared_state(result)->set_on_completed(
            [this_ = HPX_MOVE(this_)] {
                LRT_(info).format(
                    "barrier_node::set_event: rank_({}): all done",
                    this_->rank_);
            });

        (void) result;    // don't wait for the future
#else
        HPX_ASSERT(false);
#endif
    }
}    // namespace hpx::distributed::detail
