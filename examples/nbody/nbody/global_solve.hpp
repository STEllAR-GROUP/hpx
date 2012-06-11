//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Adelstein-Lelbach
//  Copyright (c)      2012 Daniel Kogler
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_NBODY_GLOBAL_SOLVE_HPP)
#define HPX_NBODY_GLOBAL_SOLVE_HPP

#include <hpx/runtime/components/client_base.hpp>

#include "stubs/global_solve.hpp"

namespace examples{
    ///////////////////////////////////////////////////////////////////////////
    class global_solve
      : public hpx::components::client_base<
            global_solve, stubs::global_solve>{
        //[global_solve_base_type
        typedef hpx::components::client_base<
            global_solve, stubs::global_solve> base_type;
        //]

    public:
        //constructor without gid
        global_solve()
        {}

        //server::global_solve instance with the given GID.
        global_solve(hpx::naming::id_type const& gid)
          : base_type(gid)
        {}

        //initialize the particle history space
        int init(std::string filename){
            BOOST_ASSERT(this->gid_);
            return this->base_type::solve_init(this->gid_, filename);
        }

        //run the entire series of simulations
        void run(int iters, int bch, int tch){
            BOOST_ASSERT(this->gid_);
            return this->base_type::solve_run(this->gid_, iters, bch, tch);
        }

        //output the results to a series of files with a specified directory
        void report(std::string directory){
            BOOST_ASSERT(this->gid_);
            return this->base_type::solve_report(this->gid_, directory);
        }

        //output the results to a series of files with a specified directory
        void calculate(bool cont, bool odd, vector<int> const& cargs){
            BOOST_ASSERT(this->gid_);
            return this->base_type::solve_calculate(
                this->gid_, cont, odd, cargs);
        }
    };
}

#endif

