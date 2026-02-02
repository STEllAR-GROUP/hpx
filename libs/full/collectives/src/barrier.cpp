//  Copyright (c) 2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/assert.hpp>
#include <hpx/collectives/barrier.hpp>
#include <hpx/components/basename_registration.hpp>
#include <hpx/components_base/server/component_heap.hpp>
#include <hpx/modules/async_base.hpp>
#include <hpx/modules/async_combinators.hpp>
#include <hpx/modules/execution.hpp>
#include <hpx/modules/memory.hpp>
#include <hpx/modules/runtime_configuration.hpp>
#include <hpx/modules/runtime_local.hpp>
#include <hpx/modules/type_support.hpp>

#include <array>
#include <atomic>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace distributed {

    barrier::barrier(std::string const& base_name)
      : node_(hpx::construct_at(
            static_cast<wrapping_type*>(
                hpx::components::component_heap<wrapping_type>().alloc()),
            new wrapped_type(base_name,
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
      : node_(hpx::construct_at(
            static_cast<wrapping_type*>(
                hpx::components::component_heap<wrapping_type>().alloc()),
            new wrapped_type(base_name, num,
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
      : node_(hpx::construct_at(
            static_cast<wrapping_type*>(
                hpx::components::component_heap<wrapping_type>().alloc()),
            new wrapped_type(base_name, num, rank)))
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
        auto const rank_it = std::find(ranks.begin(), ranks.end(), rank);
        HPX_ASSERT(rank_it != ranks.end());

        std::size_t const barrier_rank = std::distance(ranks.begin(), rank_it);
        node_.reset(hpx::construct_at(
            static_cast<wrapping_type*>(
                hpx::components::component_heap<wrapping_type>().alloc()),
            new wrapped_type(base_name, ranks.size(), barrier_rank)));

        if ((*node_)->num_ >= (*node_)->cut_off_ || (*node_)->rank_ == 0)
        {
            register_with_basename(
                base_name, node_->get_unmanaged_id(), (*node_)->rank_)
                .get();
        }
    }

    barrier::barrier() = default;

    barrier::barrier(barrier&& other) noexcept
      : node_(HPX_MOVE(other.node_))
    {
        other.node_.reset();
    }

    barrier& barrier::operator=(barrier&& other) noexcept
    {
        release();
        node_ = HPX_MOVE(other.node_);
        other.node_.reset();

        return *this;
    }

    barrier::~barrier()
    {
        release();
    }

    void barrier::wait() const
    {
        (*node_)->wait(false).get();
    }

    hpx::future<void> barrier::wait(hpx::launch::async_policy) const
    {
        return (*node_)->wait(true);
    }

    void barrier::release()
    {
        if (node_)
        {
            if (hpx::get_runtime_ptr() != nullptr &&
                hpx::threads::threadmanager_is(hpx::state::running) &&
                !hpx::is_stopped_or_shutting_down())
            {
                // make sure this runs as an HPX thread
                if (hpx::threads::get_self_ptr() == nullptr)
                {
                    hpx::run_as_hpx_thread(&barrier::release, this);
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
                        [node = HPX_MOVE(node)](hpx::future<void> f) {
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
                hpx::threads::threadmanager_is(hpx::state::running) &&
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

    std::array<barrier, 2> barrier::create_global_barrier()
    {
        runtime& rt = get_runtime();
        util::runtime_configuration const& cfg = rt.get_config();
        barrier b1("/0/hpx/global_barrier0",
            static_cast<std::size_t>(cfg.get_num_localities()));
        barrier b2("/0/hpx/global_barrier1",
            static_cast<std::size_t>(cfg.get_num_localities()));
        return {{HPX_MOVE(b1), HPX_MOVE(b2)}};
    }

    std::array<barrier, 2>& barrier::get_global_barrier()
    {
        static std::array<barrier, 2> bs = {};
        return bs;
    }

    void barrier::synchronize()
    {
        static std::atomic<std::size_t> gen = 0;
        std::array<barrier, 2>& b = get_global_barrier();

        if (!b[0].node_ || !b[1].node_)
        {
            return;
        }

        b[++gen % 2].wait();
    }
}}    // namespace hpx::distributed
