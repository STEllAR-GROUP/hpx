//  Copyright (c) 2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/assert.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/async_combinators/when_all.hpp>
#include <hpx/collectives/barrier.hpp>
#include <hpx/modules/execution.hpp>
#include <hpx/modules/memory.hpp>
#include <hpx/runtime/basename_registration.hpp>
#include <hpx/runtime/components/server/component_heap.hpp>
#include <hpx/runtime_configuration/runtime_configuration.hpp>
#include <hpx/runtime_local/run_as_hpx_thread.hpp>
#include <hpx/runtime_local/runtime_local.hpp>
#include <hpx/state.hpp>
#include <hpx/type_support/unused.hpp>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx {
    bool is_stopped_or_shutting_down();
}

namespace hpx { namespace lcos {
    barrier::barrier(std::string const& base_name)
      : node_(new (hpx::components::component_heap<wrapping_type>().alloc())
                wrapping_type(new wrapped_type(base_name,
                    static_cast<std::size_t>(
                        hpx::get_num_localities(hpx::launch::sync)),
                    static_cast<std::size_t>(hpx::get_locality_id()))))
    {
        if ((*node_)->num_ >= (*node_)->cut_off_ || (*node_)->rank_ == 0)
        {
            register_with_basename(
                base_name, node_->get_unmanaged_id(), (*node_)->rank_)
                .get();
        }
    }

    barrier::barrier(std::string const& base_name, std::size_t num)
      : node_(new (hpx::components::component_heap<wrapping_type>().alloc())
                wrapping_type(new wrapped_type(base_name, num,
                    static_cast<std::size_t>(hpx::get_locality_id()))))
    {
        if ((*node_)->num_ >= (*node_)->cut_off_ || (*node_)->rank_ == 0)
        {
            register_with_basename(
                base_name, node_->get_unmanaged_id(), (*node_)->rank_)
                .get();
        }
    }

    barrier::barrier(
        std::string const& base_name, std::size_t num, std::size_t rank)
      : node_(new (hpx::components::component_heap<wrapping_type>().alloc())
                wrapping_type(new wrapped_type(base_name, num, rank)))
    {
        if ((*node_)->num_ >= (*node_)->cut_off_ || (*node_)->rank_ == 0)
        {
            register_with_basename(
                base_name, node_->get_unmanaged_id(), (*node_)->rank_)
                .get();
        }
    }

    barrier::barrier(std::string const& base_name,
        std::vector<std::size_t> const& ranks, std::size_t rank)
    {
        auto rank_it = std::find(ranks.begin(), ranks.end(), rank);
        HPX_ASSERT(rank_it != ranks.end());

        std::size_t barrier_rank = std::distance(ranks.begin(), rank_it);
        node_.reset(
            new (hpx::components::component_heap<wrapping_type>().alloc())
                wrapping_type(
                    new wrapped_type(base_name, ranks.size(), barrier_rank)));

        if ((*node_)->num_ >= (*node_)->cut_off_ || (*node_)->rank_ == 0)
        {
            register_with_basename(
                base_name, node_->get_unmanaged_id(), (*node_)->rank_)
                .get();
        }
    }

    barrier::barrier() = default;

    barrier::barrier(barrier&& other)
      : node_(std::move(other.node_))
    {
        other.node_.reset();
    }

    barrier& barrier::operator=(barrier&& other)
    {
        release();
        node_ = std::move(other.node_);
        other.node_.reset();

        return *this;
    }

    barrier::~barrier()
    {
        release();
    }

    void barrier::wait()
    {
        (*node_)->wait(false).get();
    }

    future<void> barrier::wait(hpx::launch::async_policy)
    {
        return (*node_)->wait(true);
    }

    void barrier::release()
    {
        if (node_)
        {
            if (hpx::get_runtime_ptr() != nullptr &&
                hpx::threads::threadmanager_is(state_running) &&
                !hpx::is_stopped_or_shutting_down())
            {
                // make sure this runs as an HPX thread
                if (hpx::threads::get_self_ptr() == nullptr)
                {
                    hpx::threads::run_as_hpx_thread(&barrier::release, this);
                }

                hpx::future<void> f;
                if ((*node_)->num_ >= (*node_)->cut_off_ ||
                    (*node_)->rank_ == 0)
                {
                    f = hpx::unregister_with_basename(
                        (*node_)->base_name_, (*node_)->rank_);
                }

                // we need to wait on everyone to have its name unregistered,
                // and hold on to our node long enough...
                hpx::intrusive_ptr<wrapping_type> node = node_;
                hpx::when_all(f, wait(hpx::launch::async))
                    .then(hpx::launch::sync,
                        [node = std::move(node)](hpx::future<void> f) {
                            HPX_UNUSED(node);
                            f.get();
                        })
                    .get();
            }
            intrusive_ptr_release(node_->get());
            node_.reset();
        }
    }

    void barrier::detach()
    {
        if (node_)
        {
            if (hpx::get_runtime_ptr() != nullptr &&
                hpx::threads::threadmanager_is(state_running) &&
                !hpx::is_stopped_or_shutting_down())
            {
                if ((*node_)->num_ >= (*node_)->cut_off_ ||
                    (*node_)->rank_ == 0)
                {
                    hpx::unregister_with_basename(
                        (*node_)->base_name_, (*node_)->rank_);
                }
            }
            intrusive_ptr_release(node_->get());
            node_.reset();
        }
    }

    barrier barrier::create_global_barrier()
    {
        runtime& rt = get_runtime();
        util::runtime_configuration const& cfg = rt.get_config();
        return barrier("/0/hpx/global_barrier",
            static_cast<std::size_t>(cfg.get_num_localities()));
    }

    barrier& barrier::get_global_barrier()
    {
        static barrier b;
        return b;
    }

    void barrier::synchronize()
    {
        static barrier& b = get_global_barrier();
        HPX_ASSERT(b.node_);
        b.wait();
    }
}}    // namespace hpx::lcos
