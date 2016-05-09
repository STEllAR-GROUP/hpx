///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#ifndef HPX_COMPUTE_HOST_TARGET_HPP
#define HPX_COMPUTE_HOST_TARGET_HPP

#include <hpx/config.hpp>

#include <hpx/runtime/threads/topology.hpp>

namespace hpx { namespace compute { namespace host
{
    struct target
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

    private:
        native_handle_type mask_;
    };
}}}

#endif
