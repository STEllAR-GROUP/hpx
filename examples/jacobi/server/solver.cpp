
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "solver.hpp"

typedef hpx::components::managed_component<
    jacobi::server::solver
> solver_type;

HPX_REGISTER_MINIMAL_GENERIC_COMPONENT_FACTORY(solver_type, solver);


HPX_REGISTER_ACTION_EX(
    jacobi::server::solver::run_action
  , jacobi_server_solver_run_action
)
