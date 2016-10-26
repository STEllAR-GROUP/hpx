//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/lcos/barrier.hpp>
#include <hpx/lcos/detail/barrier_node.hpp>
#include <hpx/runtime.hpp>
#include <hpx/runtime/basename_registration.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/util/runtime_configuration.hpp>

#include <cstddef>
#include <string>

namespace hpx { namespace lcos {
    barrier::barrier(std::string const& base_name)
      : node_(new wrapping_type(new wrapped_type(
            base_name, hpx::get_num_localities(hpx::launch::sync),
            hpx::get_locality_id()
        )))
    {
        if ((*node_)->num_ >= (*node_)->cut_off_ || (*node_)->rank_ == 0)
            register_with_basename(
                base_name, node_->get_unmanaged_id(), (*node_)->rank_).get();
    }

    barrier::barrier(std::string const& base_name, std::size_t num)
      : node_(new wrapping_type(new wrapped_type(
            base_name, num, hpx::get_locality_id()
        )))
    {
        if ((*node_)->num_ >= (*node_)->cut_off_ || (*node_)->rank_ == 0)
            register_with_basename(
                base_name, node_->get_unmanaged_id(), (*node_)->rank_).get();
    }

    barrier::barrier(std::string const& base_name, std::size_t num, std::size_t rank)
      : node_(new wrapping_type(new wrapped_type(base_name, num, rank)))
    {
        if ((*node_)->num_ >= (*node_)->cut_off_ || (*node_)->rank_ == 0)
            register_with_basename(
                base_name, node_->get_unmanaged_id(), (*node_)->rank_).get();
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
            if ((*node_)->num_ >= (*node_)->cut_off_ || (*node_)->rank_ == 0)
                hpx::unregister_with_basename(
                    (*node_)->base_name_, (*node_)->rank_).get();

            // we need to wait on everyone to have its name unregistered...
            wait();
            node_.reset();
        }
    }

    barrier& barrier::get_global_barrier()
    {
        runtime& rt = get_runtime();
        util::runtime_configuration const& cfg = rt.get_config();
        static barrier b("/hpx/global_barrier", cfg.get_num_localities());
        return b;
    }

    void barrier::synchronize()
    {
        static barrier& b = get_global_barrier();
        b.wait();
    }
}}
