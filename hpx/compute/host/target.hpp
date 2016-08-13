///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#ifndef HPX_COMPUTE_HOST_TARGET_HPP
#define HPX_COMPUTE_HOST_TARGET_HPP

#include <hpx/config.hpp>

#include <hpx/lcos/future.hpp>
#include <hpx/runtime/threads/topology.hpp>

#include <cstddef>
#include <utility>

namespace hpx { namespace compute { namespace host
{
    struct HPX_EXPORT target
    {
        typedef hpx::threads::mask_type native_handle_type;

        // Constructs default target
        target()
          : mask_(hpx::threads::get_topology().get_machine_affinity_mask())
        {
        }

        // Constructs target from a given device ID
        explicit target(hpx::threads::mask_type mask)
          : mask_(mask)
        {
        }

        native_handle_type const& native_handle() const
        {
            return mask_;
        }

        native_handle_type & native_handle()
        {
            return mask_;
        }

        std::pair<std::size_t, std::size_t> num_pus() const;

        void synchronize() const
        {
            // nothing to do here...
        }

        hpx::future<void> get_future() const
        {
            return hpx::make_ready_future();
        }

    private:
        native_handle_type mask_;
    };
}}}

#endif
