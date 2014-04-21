//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_EXAMPLES_MINI_GHOST_BARRIER_HPP
#define HPX_EXAMPLES_MINI_GHOST_BARRIER_HPP

#include <hpx/lcos/barrier.hpp>

#define HPX_EXAMPLES_MINI_GHOST_BARRIER "/mini_ghost/barrier"

namespace mini_ghost {
    hpx::lcos::barrier & get_barrier()
    {
        hpx::util::static_<hpx::lcos::barrier> b;
        return b.get();
    }

    void barrier_wait()
    {
        HPX_ASSERT(get_barrier().get_gid());
        get_barrier().wait();
    }

    void create_barrier()
    {
        hpx::id_type here = hpx::find_here();
        uint64_t rank = hpx::naming::get_locality_id_from_id(here);

        // create a barrier we will use at the start and end of each run to
        // synchronize
        if(0 == rank) {
            uint64_t nranks = hpx::get_num_localities().get();
            hpx::lcos::barrier & b = get_barrier();
            b.create(hpx::find_here(), nranks);
            hpx::naming::id_type id = b.get_gid();
            id.make_unmanaged();
            hpx::agas::register_name_sync(HPX_EXAMPLES_MINI_GHOST_BARRIER, hpx::naming::detail::strip_credits_from_gid(id.get_gid()));
        }
    }

    //----------------------------------------------------------------------------
    void find_barrier()
    {
        hpx::id_type here = hpx::find_here();
        uint64_t rank = hpx::naming::get_locality_id_from_id(here);

        if (rank != 0) {
            hpx::id_type id;
            while(!id)
            {
                id = hpx::agas::resolve_name_sync(HPX_EXAMPLES_MINI_GHOST_BARRIER);
            }
            get_barrier() = hpx::lcos::barrier(id);
        }
    }
}

#endif
