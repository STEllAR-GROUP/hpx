//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include "grid.hpp"
#include <cmath>

#include <hpx/hpx.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/components/plain_component_factory.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/components/distributing_factory/distributing_factory.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/lcos/async.hpp>
#include <hpx/include/iostreams.hpp>

#include "server/remote_lse.hpp"

using bright_future::update;
using bright_future::update_residuum;

typedef bright_future::grid<double> grid_type;
typedef grid_type::size_type size_type;

using hpx::util::high_resolution_timer;

using hpx::cout;
using hpx::flush;

struct fun_base
{
    virtual ~fun_base() {}

    template <typename Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
    }
};

typedef bright_future::server::remote_lse<double> remote_lse_type;
typedef bright_future::lse_config<double> lse_config;
typedef bright_future::server::remote_lse<double>::range_type range_type;

/*
#include <boost/phoenix.hpp>
#include <boost/phoenix/stl/cmath.hpp>
using boost::phoenix::placeholders::arg1;
using boost::phoenix::placeholders::arg2;
using boost::phoenix::local_names::_a;
using boost::phoenix::local_names::_x;
using boost::phoenix::local_names::_y;
using boost::phoenix::sin;
*/

struct init_rhs_fun
    : fun_base
{
    double operator()(size_type x, size_type y, lse_config const & c)
    {
        return
            39.478 * sin((x * c.hx) * 6.283) * sinh((y * c.hy) * 6.283);
            /*
            let(
                _x = (arg1) * c.hx  // actual x coordinate
              , _y = (arg2) * c.hy  // actual y coordinate
            )
            [
                39.478 * sin(_x * 6.283) * sinh(_y * 6.283)
            ]
            (x, y);
            */
    }
};

struct init_u_fun
    : fun_base
{
    double operator()(size_type x, size_type y, lse_config const & c)
    {
        //cout << "hx " << c.hx << "hy " << c.hy << "\n" << flush;
        //return
        double value =
            y == (c.n_y - 1) ? sin((x * c.hx) * 6.283) * sinh(6.283) : 0.0;
            /*
            let(
                _x = arg1 * c.hx  // actual x coordinate
            )
            [
                if_else(
                    arg2 == 0u
                  , sin(_x * 6.283) * sinh(6.283)
                  , 0.0
                )
            ]
            (x, y);
        */

        return value;
    }
};

struct update_fun
    : fun_base
{
    double operator()(grid_type const & u, grid_type const & rhs, size_type x, size_type y, lse_config const & c)
    {
        return update(u, rhs, x, y, c.hx_sq, c.hy_sq, c.div, c.relaxation);
    }
};

struct identity_fun
    : fun_base
{
    double operator()(grid_type const & u, grid_type const & rhs, size_type x, size_type y, lse_config const & c)
    {
        return u(x, y);
    }
};

struct output_fun
    : fun_base
{
    boost::shared_ptr<std::ofstream> file;

    output_fun() {}
    output_fun(std::string const & output) : file(new std::ofstream(output.c_str())) {}

    double operator()(grid_type const & u, grid_type const & rhs, size_type x, size_type y, lse_config const & c)
    {
        (*file) << x * c.hx << " " << y * c.hy << " " << u(x, y) << "\n";
        return u(x, y);
    }
};

struct update_residuum_fun
    : fun_base
{
    void operator()(grid_type & u, size_type x, size_type y)
    {
        cout << "init!\n" << flush;
    }
};


void gs(
    size_type n_x
  , size_type n_y
  , double hx
  , double hy
  , double k
  , double relaxation
  , unsigned max_iterations
  , unsigned iteration_block
  , unsigned block_size
  , std::string const & output
)
{
    {
        hpx::components::component_type type
            = hpx::components::get_component_type<remote_lse_type>();

        typedef
            hpx::components::distributing_factory
            distributing_factory;

        distributing_factory
            factory(
                distributing_factory::create_sync(hpx::find_here())
            );

        distributing_factory::async_create_result_type
            result = factory.create_components_async(type, 1);

        distributing_factory::result_type results = result.get();
        distributing_factory::iterator_range_type
            parts = hpx::components::server::locality_results(results);

        hpx::naming::id_type remote_id = *parts.first;

        typedef hpx::lcos::future<void> promise_type;

        async<remote_lse_type::init_action>(remote_id, n_x, n_y, hx, hy).get();

        typedef bright_future::grid<promise_type> promise_grid_type;

        // set our initial values, setting the top boundary to be a dirichlet
        // boundary condition
        {
            promise_grid_type init_rhs_promises(n_x, n_y);
            for(size_type y = 0; y < n_y; ++y)
            {
                for(size_type x = 0; x < n_x; ++x)
                {
                    init_rhs_promises(x, y) =
                        hpx::lcos::async<remote_lse_type::init_rhs_action>(
                            remote_id, init_rhs_fun(), x, y);
                }
            }
            BOOST_FOREACH(promise_type & promise, init_rhs_promises)
            {
                promise.get();
            }
        }

        typedef std::vector<promise_grid_type> iteration_dependencies_type;

        iteration_dependencies_type iteration_dependencies(max_iterations+1, promise_grid_type(n_x, n_y));

        for(size_type y = 0; y < n_y; ++y)
        {
            for(size_type x = 0; x < n_x; ++x)
            {
                iteration_dependencies[0](x, y) =
                    hpx::lcos::async<remote_lse_type::init_u_action>(
                        remote_id, init_u_fun(), x, y);
                iteration_dependencies[0](x, y).get();
            }
        }

        high_resolution_timer t;
        for(unsigned iter = 0; iter < max_iterations; iter++)// += iteration_block)
        {
            unsigned const jter = 0;
            // split up iterations so we don't need to check the residual every iteration
            //for(unsigned jter = 0; jter < iteration_block; ++jter)
            {
                // update the "red" points
                size_type y, x/*, block_size*/;
                promise_grid_type & current = iteration_dependencies[iter + jter + 1];
                promise_grid_type & prev = iteration_dependencies[iter + jter];

                // Boundary regions
                for(y = 0; y < n_y; ++y)
                {
                    {
                        std::vector<promise_type> dependencies;
                        dependencies.push_back(prev(0,y));
                        current(0, y) =
                            hpx::lcos::async<remote_lse_type::apply_action>(
                                remote_id, identity_fun(), 0, y, dependencies);
                    }
                    {
                        std::vector<promise_type> dependencies;
                        dependencies.push_back(prev(n_x-1,y));
                        current(n_x-1, y) =
                            hpx::lcos::async<remote_lse_type::apply_action>(
                                remote_id, identity_fun(), n_x-1, y, dependencies);
                    }
                }
                for(x = 1; x < n_x-1; ++x)
                {
                    {
                        std::vector<promise_type> dependencies;
                        dependencies.push_back(prev(x,0));
                        current(x, 0) =
                            hpx::lcos::async<remote_lse_type::apply_action>(
                                remote_id, identity_fun(), x, 0, dependencies);
                    }
                    {
                        std::vector<promise_type> dependencies;
                        dependencies.push_back(prev(x,n_y-1));
                        current(x, n_y-1) =
                            hpx::lcos::async<remote_lse_type::apply_action>(
                                remote_id, identity_fun(), x, n_y-1, dependencies);
                    }
                }

                // Inner region
                for(y = 1; y < n_y-1; ++y)
                {
                    for(x = 1; x < n_x-1; ++x)
                    {
                        // wait on all the previous results for now to continue the update.
                        std::vector<promise_type> dependencies;
                        dependencies.reserve(5);
                        dependencies.push_back(prev(x,y));
                        dependencies.push_back(prev(x+1,y));
                        dependencies.push_back(prev(x,y+1));
                        dependencies.push_back(current(x-1,y));
                        dependencies.push_back(current(x,y-1));

                        current(x, y) =
                            hpx::lcos::async<remote_lse_type::apply_action>(
                                remote_id, update_fun(), x, y, dependencies);

                        //u(x, y) = update(u, rhs, x, y, hx_sq, hy_sq, div, relaxation);
                    }
                }
                /*
                // update the "black" points
                for(y = 1; y < n_y - 1; ++y)
                {
                    for(x = ((y+1)%2) + 1; x < n_x - 1; ++x)
                    {
                        //u(x, y) = update(u, rhs, x, y, hx_sq, hy_sq, div, relaxation);
                    }
                }
                */
            }
            /*
            // check if we converged yet
            grid_type residuum(u.x(), u.y());
            size_type y, x;
#pragma omp parallel for private(x, y)
            for(y = 1; y < n_y - 1; ++y)
            {
                for(x = 1; x < n_x - 1; ++x)
                {
                    residuum(x, y) = update_residuum(u, rhs, x, y, hx_sq, hy_sq, k);
                }
            }
            double r = 0.0;
#pragma omp parallel for reduction(+:r)
            for(unsigned i = 0; i < residuum.size(); ++i)
            {
                r = r + residuum[i] * residuum[i];
            }

            if(std::sqrt(r) <= 1e-10)
            {
                break;
            }
            */
        }

        BOOST_FOREACH(promise_type & promise, iteration_dependencies[max_iterations])
        {
            promise.get();
        }

        double time_elapsed = t.elapsed();
        cout << (n_x*n_y) << " " << time_elapsed << "\n" << flush;

        if(!output.empty())
        {
            output_fun f(output);
            remote_lse_type::apply_func_type out = f;

            for(size_type x = 0; x < n_x; ++x)
            {
                for(size_type y = 0; y < n_y; ++y)
                {
                    hpx::lcos::async<remote_lse_type::apply_action>(
                        remote_id, out, x, y, std::vector<promise_type>()).get();
                }
                (*f.file) << "\n";
            }
        }
    }
}
