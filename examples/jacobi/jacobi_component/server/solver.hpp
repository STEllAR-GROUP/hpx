
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef JACOBI_SERVER_SOLVER_HPP
#define JACOBI_SERVER_SOLVER_HPP

#include "../grid.hpp"
#include "../stencil_iterator.hpp"
#include "stencil_iterator.hpp"

#include <hpx/include/components.hpp>
#include <hpx/lcos/future_wait.hpp>
#include <hpx/util/high_resolution_timer.hpp>

#include <cstddef>
#include <vector>

namespace jacobi
{
    namespace server
    {
        struct HPX_COMPONENT_EXPORT solver
            : hpx::components::component_base<solver>
        {
            typedef
                hpx::components::component_base<solver>
                base_type;
            typedef hpx::components::component<solver> component_type;

            solver()
            {
                HPX_ASSERT(false);
            }

            ~solver()
            {
                HPX_ASSERT(stencil_iterators.size() == ny);
                for(std::size_t y = 0; y < ny; ++y)
                {
                    HPX_ASSERT(stencil_iterators[y].id);
                }
            }

            solver(grid const & g, std::size_t nx, std::size_t line_block)
                : ny(g.rows.size())
                , nx(nx)
                //, stencil_iterators(g.rows.size())
            {
                stencil_iterators.reserve(ny);

                std::vector<hpx::id_type> ids =
                    hpx::new_<server::stencil_iterator[]>(
                        hpx::default_layout(hpx::find_all_localities()), ny).get();

                std::vector<hpx::lcos::shared_future<void> > init_futures;
                init_futures.reserve(ny);
                std::size_t y = 0;
                for (hpx::naming::id_type const& id : ids)
                {
                    //std::cout << y << " " << id << "\n";
                    jacobi::stencil_iterator r; r.id = id;
                    init_futures.push_back(r.init(g.rows[y], y, nx, ny, line_block));
                    stencil_iterators.push_back(r);
                    ++y;
                }
                HPX_ASSERT(y == ny);

                std::vector<hpx::lcos::shared_future<void> > boundary_futures;
                hpx::lcos::wait(
                    init_futures
                  , [&](std::size_t y)
                    {
                        if(y > 0 && y < ny-1)
                        {
                            HPX_ASSERT(stencil_iterators[y-1].id);
                            HPX_ASSERT(stencil_iterators[y].id);
                            HPX_ASSERT(stencil_iterators[y+1].id);
                            hpx::wait_all(init_futures[y-1]);
                            hpx::wait_all(init_futures[y+1]);
                            boundary_futures.push_back(
                                stencil_iterators[y].setup_boundary(
                                    stencil_iterators[y-1]
                                  , stencil_iterators[y+1]
                                )
                            );
                        }
                    }
                );
                HPX_ASSERT(stencil_iterators[0].id);
                hpx::wait_all(boundary_futures);
                HPX_ASSERT(stencil_iterators[0].id);
            }

            void run(std::size_t max_iterations)
            {
                HPX_ASSERT(stencil_iterators[0].id);

                hpx::util::high_resolution_timer t;

                t.restart();
                /*
                for(std::size_t y = 1; y < ny-1; ++y)
                {
                    run_futures.push_back(stencil_iterators[y].run(max_iterations));
                }
                HPX_ASSERT(stencil_iterators[0].id);
                hpx::wait_all(run_futures);
                HPX_ASSERT(stencil_iterators[0].id);
                */

                for(std::size_t iter = 0; iter < max_iterations; ++iter)
                {
                    std::vector<hpx::lcos::shared_future<void> > run_futures;
                    run_futures.reserve(ny-2);
                    for(std::size_t y = 1; y < ny-1; ++y)
                    {
                        run_futures.push_back(
                            stencil_iterators[y].step()
                        );
                    }
                    hpx::wait_all(run_futures);
                }

                double time_elapsed = t.elapsed();
                hpx::cout << nx << "x" << ny << " "
                     << ((double((nx-2)*(ny-2) * max_iterations)/1e6)/time_elapsed)
                     << " MLUPS\n" << hpx::flush;
            }

            HPX_DEFINE_COMPONENT_ACTION(solver, run, run_action);

            std::size_t ny;
            std::size_t nx;
            std::vector<jacobi::stencil_iterator> stencil_iterators;
        };
    }
}

HPX_REGISTER_ACTION_DECLARATION(
    jacobi::server::solver::run_action
  , jacobi_server_solver_run_action
)

#endif
