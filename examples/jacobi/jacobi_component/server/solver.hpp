
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
#include <hpx/components/distributing_factory/distributing_factory.hpp>
#include <hpx/util/high_resolution_timer.hpp>

#include <vector>

namespace jacobi
{
    namespace server
    {
        struct HPX_COMPONENT_EXPORT solver
            : hpx::components::managed_component_base<
                solver
              , hpx::components::detail::this_type
              , hpx::traits::construct_with_back_ptr
            >
        {
            typedef
                hpx::components::managed_component_base<
                    solver
                  , hpx::components::detail::this_type
                  , hpx::traits::construct_with_back_ptr
                >
                base_type;
            typedef hpx::components::managed_component<solver> component_type;

            solver(component_type * back_ptr)
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

            solver(component_type * back_ptr, grid const & g, std::size_t nx, std::size_t line_block)
                : base_type(back_ptr)
                , ny(g.rows.size())
                , nx(nx)
                //, stencil_iterators(g.rows.size())
            {
                stencil_iterators.reserve(ny);
                hpx::components::distributing_factory factory;
                factory.create(hpx::find_here());

                // make get the type of the solver component
                hpx::components::component_type
                    type = hpx::components::get_component_type<
                        server::stencil_iterator
                    >();

                hpx::components::distributing_factory::result_type si_allocated =
                    factory.create_components(type, ny);

                std::vector<hpx::lcos::future<void> > init_futures;
                init_futures.reserve(ny);
                std::size_t y = 0;
                BOOST_FOREACH(hpx::naming::id_type id, hpx::util::locality_results(si_allocated))
                {
                    //std::cout << y << " " << id << "\n";
                    jacobi::stencil_iterator r; r.id = id;
                    init_futures.push_back(r.init(g.rows[y], y, nx, ny, line_block));
                    stencil_iterators.push_back(r);
                    ++y;
                }
                HPX_ASSERT(y == ny);

                std::vector<hpx::lcos::future<void> > boundary_futures;
                hpx::lcos::wait(
                    init_futures
                  , [&](std::size_t y)
                    {
                        if(y > 0 && y < ny-1)
                        {
                            HPX_ASSERT(stencil_iterators[y-1].id);
                            HPX_ASSERT(stencil_iterators[y].id);
                            HPX_ASSERT(stencil_iterators[y+1].id);
                            hpx::lcos::wait(init_futures[y-1]);
                            hpx::lcos::wait(init_futures[y+1]);
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
                hpx::lcos::wait(boundary_futures);
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
                hpx::lcos::wait(run_futures);
                HPX_ASSERT(stencil_iterators[0].id);
                */

                for(std::size_t iter = 0; iter < max_iterations; ++iter)
                {
                    std::vector<hpx::lcos::future<void> > run_futures;
                    run_futures.reserve(ny-2);
                    for(std::size_t y = 1; y < ny-1; ++y)
                    {
                        run_futures.push_back(
                            stencil_iterators[y].step()
                        );
                    }
                    hpx::lcos::wait(run_futures);
                }

                double time_elapsed = t.elapsed();
                hpx::cout << nx << "x" << ny << " "
                     << ((double((nx-2)*(ny-2) * max_iterations)/1e6)/time_elapsed) << " MLUPS\n" << hpx::flush;
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
