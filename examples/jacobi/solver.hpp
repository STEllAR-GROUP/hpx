
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef JACOBI_SOLVER_HPP
#define JACOBI_SOLVER_HPP

#include "server/solver.hpp"
#include "grid.hpp"

#include <hpx/include/naming.hpp>

namespace jacobi
{
    struct solver
    {
        hpx::naming::id_type id;

        solver(grid const & g, std::size_t nx, std::size_t line_block)
        {
            // make get the type of the solver component
            hpx::components::component_type
                solver_type = hpx::components::get_component_type<
                    server::solver
                >();
            
            // get list of locality prefixes
            std::vector<hpx::naming::id_type> localities =
                hpx::find_all_localities(solver_type);

            BOOST_ASSERT(localities.size() > 0);

            typedef
                hpx::components::server::create_one_component_action3<
                    hpx::components::managed_component<server::solver>
                  , grid
                  , std::size_t
                  , std::size_t
                >::type
                create_component_action;

            id
                = hpx::async<create_component_action>(
                    localities[0]
                  , solver_type
                  , g
                  , nx
                  , line_block
                ).get();
        }

        void run(std::size_t max_iterations)
        {
            hpx::async<server::solver::run_action>(id, max_iterations).get();
        }

    };
}

#endif
