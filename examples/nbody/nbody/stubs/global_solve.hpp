//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Adelstein-Lelbach
//  Copyright (c)      2012 Daniel Kogler
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_NBODY_GLOBAL_SOLVE_STUB)
#define HPX_NBODY_GLOBAL_SOLVE_STUB

#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/runtime/applier/apply.hpp>
#include <hpx/lcos/async.hpp>

#include "../server/global_solve.hpp"

namespace examples { namespace stubs
{
    ///////////////////////////////////////////////////////////////////////////
    //[global_solve_stubs_inherit
    struct global_solve
      : hpx::components::stub_base<server::global_solve>
    //]
    {
        static int solve_init(hpx::naming::id_type const& gid,
        std::string filename){
            return async<
                server::global_solve::init_action>(gid, filename, gid).get();
        }
        static void solve_run(hpx::naming::id_type const& gid, 
        int it, int bch, int tch){
            async<server::global_solve::run_action>(gid, it, bch, tch).get();
        }
        static void solve_report(hpx::naming::id_type const& gid,
        std::string directory){
            async<server::global_solve::report_action>(gid, directory).get();
        }
        static void solve_calculate(hpx::naming::id_type const& gid,
        bool cont, bool odd, vector<int> const& cargs){
            async<
                server::global_solve::calc_action>(
                gid, cont, odd, cargs).get();
        }
    };
}}

#endif

