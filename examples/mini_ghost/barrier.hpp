//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_EXAMPLES_MINI_GHOST_BARRIER_HPP
#define HPX_EXAMPLES_MINI_GHOST_BARRIER_HPP

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/naming.hpp>
#include <hpx/include/lcos.hpp>

#include <boost/cstdint.hpp>

#define HPX_MINI_GHOST_BARRIER "/mini_ghost/barrier"

namespace mini_ghost
{
    namespace detail
    {
        hpx::lcos::barrier& get_barrier()
        {
            hpx::util::static_<hpx::lcos::barrier> b;
            return b.get();
        }
    }

    void barrier_wait()
    {
        // Wait for the barrier to release all localities
        HPX_ASSERT(detail::get_barrier().get_gid());
        detail::get_barrier().wait();
    }

    void free_barrier()
    {
        detail::get_barrier() = hpx::lcos::barrier();
    }

    void create_barrier()
    {
        boost::uint32_t rank = hpx::get_locality_id();
        if (0 == rank) {
            // create a barrier we will use at the start and end of each run to
            // synchronize
            boost::uint64_t nranks = hpx::get_num_localities_sync();

            hpx::lcos::barrier& b = detail::get_barrier();
            b = hpx::lcos::barrier::create(hpx::find_here(), nranks);

            // Register the global id of the barrier with AGAS so that other
            // localities will be able to retrieve it.
            hpx::id_type id = hpx::unmanaged(b.get_gid());
            hpx::agas::register_name_sync(HPX_MINI_GHOST_BARRIER, id.get_gid());
        }

        // make sure the barrier is released before exiting
        hpx::register_pre_shutdown_function(&mini_ghost::free_barrier);
    }

    void find_barrier()
    {
        boost::uint32_t rank = hpx::get_locality_id();
        if (rank != 0) {
            // Wait for the barrier to be registered with AGAS before continuing
            hpx::id_type id = hpx::agas::on_symbol_namespace_event(
                HPX_MINI_GHOST_BARRIER, hpx::agas::symbol_ns_bind, true).get();

            detail::get_barrier() = hpx::lcos::barrier(id);
        }
    }
}

#endif
