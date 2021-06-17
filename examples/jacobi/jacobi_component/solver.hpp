
//  Copyright (c) 2012 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include "grid.hpp"
#include "server/solver.hpp"

#include <hpx/assert.hpp>
#include <hpx/include/naming.hpp>
#include <hpx/include/runtime.hpp>

#include <cstddef>
#include <utility>
#include <vector>

namespace jacobi {
    struct solver : hpx::components::client_base<solver, server::solver>
    {
        typedef hpx::components::client_base<solver, server::solver> base_type;

        solver(hpx::future<hpx::id_type>&& id)
          : base_type(std::move(id))
        {
        }

        solver(grid const& g, std::size_t nx, std::size_t line_block)
        {
            // make get the type of the solver component
            hpx::components::component_type solver_type =
                hpx::components::get_component_type<server::solver>();

            // get list of locality prefixes
            std::vector<hpx::naming::id_type> localities =
                hpx::find_all_localities(solver_type);

            HPX_ASSERT(localities.size() > 0);

            this->base_type::operator=(
                hpx::components::create_async<server::solver>(
                    localities[0], g, nx, line_block));
        }

        void run(std::size_t max_iterations)
        {
            hpx::async<server::solver::run_action>(
                this->get_id(), max_iterations)
                .get();
        }
    };
}    // namespace jacobi
