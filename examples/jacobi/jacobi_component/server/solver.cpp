
//  Copyright (c) 2012 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx.hpp>
#include "solver.hpp"

typedef hpx::components::component<
    jacobi::server::solver
> solver_type;

HPX_REGISTER_COMPONENT(solver_type, solver);

HPX_REGISTER_ACTION(
    jacobi::server::solver::run_action
  , jacobi_server_solver_run_action
)
#endif
