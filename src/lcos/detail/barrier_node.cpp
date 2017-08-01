//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/async.hpp>
#include <hpx/lcos/detail/barrier_node.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/when_all.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/unwrap.hpp>

#include <boost/intrusive_ptr.hpp>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

typedef hpx::components::managed_component<hpx::lcos::detail::barrier_node> barrier_type;

HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    hpx::lcos::detail::barrier_node, hpx::components::component_barrier)

HPX_REGISTER_ACTION(hpx::lcos::detail::barrier_node::gather_action,
    barrier_node_gather_action);

namespace hpx { namespace lcos { namespace detail {

    barrier_node::barrier_node()
      : count_(0),
        local_barrier_(0)
    {
        HPX_ASSERT(false);
    }

    barrier_node::barrier_node(std::string base_name, std::size_t num, std::size_t rank)
      : count_(0),
        base_name_(base_name),
        rank_(rank),
        num_(num),
        arity_(std::stol(get_config_entry("hpx.lcos.collectives.arity", 32))),
        cut_off_(std::stol(get_config_entry("hpx.lcos.collectives.cut_off", -1))),
        local_barrier_(num)
    {
        if (num_ >= cut_off_)
        {
            std::vector<std::size_t> ids;
            ids.reserve(children_.size());

            for (std::size_t i = 1; i <= arity_; ++i)
            {
                std::size_t id = (arity_ * rank_) + i;
                if (id >= num) break;
                ids.push_back(id);
            }

            children_ = hpx::util::unwrap(
                hpx::find_from_basename(base_name_, ids));

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
        if (num_ < cut_off_)
        {
            if (rank_ != 0)
            {
                HPX_ASSERT(children_.size() == 1);
                hpx::lcos::base_lco::set_event_action action;
                return hpx::async(action, children_[0]);
            }
            else
            {
                if (async)
                {
                    boost::intrusive_ptr<barrier_node> this_(this);
                    return hpx::async(&barrier_node::set_event, this_);
                }
                set_event();
                return hpx::make_ready_future();
            }
        }

        future<void> result;

        if (rank_ == 0)
        {
            // The root process calls the gather action on its children
            // once all those return, we know that everyone entered the
            // barrier
            std::vector<hpx::future<void> > futures;
            futures.reserve(children_.size());
            for(hpx::id_type& id : children_)
            {
                barrier_node::gather_action action;
                futures.push_back(hpx::async(action, id));
            }
            result = hpx::when_all(futures);
        }
        else
        {
            // Non-root flags that it entered the barrier...
            gather_promise_.set_value();

            // The broadcast_promise is set once the root knows that everyone
            // entered the barrier
            result = broadcast_promise_.get_future();
        }

        if (async)
        {
            boost::intrusive_ptr<barrier_node> this_(this);
            return do_wait(this_, std::move(result));
        }
        return do_wait(this, std::move(result));
    }

    template <typename This>
    hpx::future<void> barrier_node::do_wait(This this_,
        hpx::future<void> future)
    {
        if (rank_ == 0)
        {
            return future.then(hpx::launch::sync,
                [this_](hpx::future<void>&& f)
                {
                    // Trigger possible errors...
                    f.get();

                    std::vector<hpx::future<void> > futures;
                    futures.reserve(this_->children_.size());

                    // After the root process received the notification that
                    // everyone entered the barrier, it will broadcast to
                    // everyone that they can leave the barrier
                    for(hpx::id_type& id : this_->children_)
                    {
                        base_lco::set_event_action action;
                        futures.push_back(hpx::async(action, id));
                    }

                    return hpx::when_all(futures);
                });
        }

        return future.then(hpx::launch::sync,
            [this_](hpx::future<void>&& f)
            {
                // Trigger possible errors...
                f.get();

                // Once the non-roots are ready to leave the barrier, we
                // need to reset our promises such that the barrier can be
                // reused.
                this_->broadcast_promise_ = hpx::lcos::local::promise<void>();
                this_->gather_promise_ = hpx::lcos::local::promise<void>();
            });
    }

    template hpx::future<void>
        barrier_node::do_wait(
            boost::intrusive_ptr<barrier_node>, hpx::future<void>);
    template hpx::future<void>
        barrier_node::do_wait(
            barrier_node*, hpx::future<void>);

    hpx::future<void> barrier_node::gather()
    {
        // We recursively gather the information that everyone entered the
        // barrier. The recursion is started from the root node.
        HPX_ASSERT(rank_ != 0);
        std::vector<hpx::future<void> > futures;
        futures.reserve(children_.size());
        for(hpx::id_type& id : children_)
        {
            barrier_node::gather_action action;
            futures.push_back(hpx::async(action, id));
        }

        // Once we know that all our children entered the barrier, we flag ourself
        return hpx::when_all(futures).then(hpx::launch::sync,
            [this](hpx::future<void> f)
            {
                // Trigger possible errors...
                f.get();
                return gather_promise_.get_future();
            });
    }

    void barrier_node::set_event()
    {
        if (num_ < cut_off_)
        {
            local_barrier_.wait();
            return;
        }

        // We recursively broadcast the information that everyone entered the
        // barrier. The recursion is started from the root node.
        HPX_ASSERT(rank_ != 0);
        std::vector<hpx::future<void> > futures;
        futures.reserve(children_.size());
        for(hpx::id_type& id : children_)
        {
            base_lco::set_event_action action;
            futures.push_back(hpx::async(action, id));
        }

        // Once we notified our children, we mark ourself ready.
        hpx::when_all(futures).then(hpx::launch::sync,
            [this](future<void> f)
            {
                // Trigger possible errors...
                f.get();
                broadcast_promise_.set_value();
            });
    }
}}}
