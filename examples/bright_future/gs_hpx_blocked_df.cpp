
//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>

#include "grid.hpp"
#include <cmath>

#include <hpx/hpx.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/components/plain_component_factory.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/components/distributing_factory/distributing_factory.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/iostreams.hpp>

#include <hpx/components/dataflow/dataflow.hpp>
#include <hpx/components/dataflow/dataflow_trigger.hpp>

#include "server/remote_lse.hpp"

using bright_future::update;
using bright_future::update_residuum;

typedef bright_future::grid<double> grid_type;

using hpx::util::high_resolution_timer;

using hpx::cout;
using hpx::flush;
using hpx::lcos::dataflow;
using hpx::lcos::dataflow_base;
using hpx::lcos::dataflow_trigger;
using hpx::find_here;

// this is just a simple base class for my functors to be serializable
// this is required by my function implementation, because arguments to hpx
// actions need to be serializable
struct fun_base
{
    virtual ~fun_base() {}

    template <typename Archive>
    void serialize(Archive &, const unsigned int)
    {
        BOOST_ASSERT(false);
    }
};

// various typedefs:
// remote_lse_type is a hpx component to hold the grid, rhs and the other
// parameters of our PDE, for example the step size.
typedef bright_future::server::remote_lse<double> remote_lse_type;
typedef bright_future::lse_config<double> lse_config;
typedef bright_future::server::remote_lse<double>::range_type range_type;

// this is the functor to initialize the RHS of my partial differential equation
struct init_rhs_fun
    : fun_base
{
    double operator()(std::size_t x, std::size_t y, lse_config const & c)
    {
        return
            39.478 * sin((double(x) * c.hx) * 6.283) * sinh((double(y) * c.hy) * 6.283);
    }
};

// functor to initialize the grid with boundary conditions
struct init_u_fun
    : fun_base
{
    double operator()(std::size_t x, std::size_t y, lse_config const & c)
    {
        double value =
            y == (c.n_y - 1) ? sin((double(x) * c.hx) * 6.283) * sinh(6.283) : 0.0;

        return value;
    }
};

// the stencil code
struct update_fun
    : fun_base
{
    double operator()(grid_type const & u, grid_type const & rhs, std::size_t x, std::size_t y, lse_config const & c)
    {
        return
            u(x, y)
            + (
                (
                    (
                        (u(x - 1, y) + u(x + 1, y)) / c.hx_sq
                      + (u(x, y - 1) + u(x, y + 1)) / c.hy_sq
                      + rhs(x, y)
                    )
                    / c.div
                )
                - u(x, y)
            )
            * c.relaxation
            ;
    }
};

struct identity_fun
    : fun_base
{
    double operator()(grid_type const & u, grid_type const &, std::size_t x, std::size_t y, lse_config const &)
    {
        return u(x, y);
    }
};

// apply function to output our result
struct output_fun
    : fun_base
{
    boost::shared_ptr<std::ofstream> file;

    output_fun() {}
    output_fun(std::string const & output) : file(new std::ofstream(output.c_str())) {}

    double operator()(grid_type const & u, grid_type const &, std::size_t x, std::size_t y, lse_config const & c)
    {
        (*file) << double(x) * c.hx << " " << double(y) * c.hy << " " << u(x, y) << "\n";
        return u(x, y);
    }
};

// not used currently
struct update_residuum_fun
    : fun_base
{
    void operator()(grid_type &, std::size_t, std::size_t)
    {
    }
};

void gs(
    std::size_t n_x
  , std::size_t n_y
  , double hx
  , double hy
  , double //k
  , double //relaxation
  , unsigned max_iterations
  , unsigned //iteration_block
  , unsigned block_size
  , std::size_t //cache_block
  , std::string const & output
)
{
    {
        // initialization of hox component. this can be hidden nicely
        // this is not something to worry about right now.
        // it creates the remote_lse_type and registers it with the hpx agas
        // server. a remote_id is created, this is a "pointer" to our (possibly)
        // remotely created LSE.
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

        hpx::cout << typeid(hpx::traits::is_dataflow<dataflow_trigger>::type).name() << "\n" << hpx::endl;

        // initialization of hpx component ends here.

        typedef hpx::lcos::dataflow_base<void> dataflow_type;

        // The init future:
        // a hpx future takes an action. this action is a remotely callable
        // member function on an instance of remote_lse_type
        // the init function has the signature
        // void(unsigned n_x, unsigned n_y, double hx, double hy)

        // we create a temporary init_future object. the first parameter is the
        // id on which object we want to call the action. the remaining
        // parameters are the parameters to be passed to the action, see comment
        // above
        hpx::async<remote_lse_type::init_action>(
            remote_id, n_x, n_y, n_x, n_y, hx, hy).get();

        // this type represents our grid, instead of doubles, we just use
        // promises as value types.
        typedef bright_future::grid<dataflow_type> promise_grid_type;
        // this type is used to hold the promise grids of the different iterations
        typedef std::vector<promise_grid_type> iteration_dependencies_type;

        std::size_t n_x_block = n_x/block_size+1;
        std::size_t n_y_block = n_y/block_size+1;

        // initialize the iteration dependencies:
        // we need max_iterations+1 elements in that vector. The 0th entry
        // is used for the initialization of the grid.
        iteration_dependencies_type
            iteration_dependencies(
                2//max_iterations+1
              , promise_grid_type(n_x_block, n_y_block)
            );
        promise_grid_type init_rhs_promises(n_x_block, n_y_block);

        // set our initial values, setting the top boundary to be a dirichlet
        // boundary condition
        {
            typedef
                hpx::lcos::dataflow<remote_lse_type::init_rhs_blocked_action>
                init_rhs_dataflow;

            cout << "initializing rhs\n" << flush;
            // we opened another scope to just have this vector temporarily, we
            // won't need it after the initializiation
            for(std::size_t y = 0, y_block = 0; y < n_y; y += block_size, ++y_block)
            {
                for(std::size_t x = 0, x_block = 0; x < n_x; x += block_size, ++x_block)
                {
                    init_rhs_promises(x_block, y_block) =
                        init_rhs_dataflow(
                            remote_id
                          , init_rhs_fun()
                          , range_type(x, (std::min)(n_x, x + block_size))
                          , range_type(y, (std::min)(n_y, y + block_size))
                        );
                }
            }

            typedef
                hpx::lcos::dataflow<remote_lse_type::init_u_blocked_action>
                init_u_dataflow;

            cout << "initializing u\n" << flush;
            // initialize our grid. This will serve as the dependency of the
            // first loop below.
            for(std::size_t y = 0, y_block = 0; y < n_y; y += block_size, ++y_block)
            {
                for(std::size_t x = 0, x_block = 0; x < n_x; x += block_size, ++x_block)
                {
                    // set the promise of the initialization.
                    iteration_dependencies[0](x_block, y_block) =
                        init_u_dataflow(   // invoke the init future.
                            remote_id
                          , init_u_fun() // pass the initialization function
                          , range_type(x, (std::min)(n_x, x + block_size))
                          , range_type(y, (std::min)(n_y, y + block_size))
                        );
                }
            }

            // wait for the rhs initialization to finish.
            //BOOST_FOREACH(dataflow_type & promise, init_rhs_promises)
            //{
            //    promise.get();
            //}
        }
        typedef
            hpx::lcos::dataflow<remote_lse_type::apply_action>
            apply_future;

        typedef
            hpx::lcos::dataflow<remote_lse_type::apply_region_df_action>
            apply_region_dataflow;

        cout << "finished initializing ...\n" << flush;

        hpx::async<remote_lse_type::clear_timestamps_action>(remote_id).get();

        high_resolution_timer t;

        // our real work starts here.
        for(unsigned iter = 0; iter < max_iterations; ++iter)
        {
            {
                promise_grid_type & prev = iteration_dependencies[iter%2];
                promise_grid_type & current = iteration_dependencies[(iter + 1)%2];

                // in every iteration we want to compute this:
                for(std::size_t y_block = 0, y = 0; y_block < n_y; y_block += block_size, ++y)
                {
                    for(std::size_t x_block = 0, x = 0; x_block < n_x; x_block += block_size, ++x)
                    {
                        // set up the x and y ranges, this takes care of the
                        // boundaries, so we don't access invalid rgid points.
                        range_type
                            x_range(
                                x == 0             ? 1     : x_block
                              , x + 1 == n_x_block ? n_x-1 : x_block + block_size
                              );
                        range_type
                            y_range(
                                y == 0             ? 1     : y_block
                              , y + 1 == n_y_block ? n_y-1 : y_block + block_size
                              );

                        std::vector<dataflow_base<void> > deps;
                        // we need to be sure to wait for the previous iteration.
                        deps.push_back(prev(x, y));

                        if(iter==0)
                        {
                            deps.push_back(init_rhs_promises(x, y));
                        }
                        // these are our dependencies to update this specific
                        // block

                        // keep in mind, our loop goes from top-left to bottom
                        // right.

                        if(x + 1 < n_x_block) // are we on the boundary?
                        {
                            // add the right block of the previous iteration
                            // to our list of dependencies
                            deps.push_back(prev(x+1, y));
                        }

                        if(y + 1 < n_y_block) // are we on the boundary?
                        {
                            // add the upper block of the previous iteration
                            // to our list of dependencies
                            deps.push_back(prev(x, y+1));
                        }

                        if(x > 0) // are we on the boundary?
                        {
                            // add the upper block of the current iteration
                            // to our list of dependencies
                            deps.push_back(current(x-1, y));
                        }

                        if(y > 0) // are we on the boundary?
                        {
                            // add the upper block of the current iteration
                            // to our list of dependencies
                            deps.push_back(current(x, y-1));
                        }

                        //deps.get();
                        current(x, y) =
                            // call the update action
                            // this will exectue a loop like this:
                            // for(std::size_t y = x_range.first; y < y_range.second; ++y)
                            //     for(std::size_t x = x_range.first; x < x_range.second; ++x)
                            //         u(x, y) = update_fun()(x, y, u, rhs, ...)
                            // the loop will be executed after all the dependencies
                            // are finished.
                            apply_region_dataflow(
                                remote_id
                              //, deps
                              , update_fun()
                              , x_range
                              , y_range
                              , dataflow_trigger(find_here(), deps)
                            );
                        //cout << "." << flush;
                    }
                }
            }
        }

        cout << "finished building dataflow tree\n" << flush;

        // wait for the last iteration to finish.
        BOOST_FOREACH(dataflow_type & promise, iteration_dependencies[max_iterations%2])
        {
            promise.get_future().get();
            std::cout << "." << flush;
        }
        std::cout << "\n" << flush;

        hpx::async<remote_lse_type::print_timestamps_action>(remote_id).get();

        double time_elapsed = t.elapsed();
        cout << time_elapsed << "\n" << flush;

        if(!output.empty())
        {
            output_fun f(output);
            remote_lse_type::apply_func_type out = f;

            for(std::size_t x = 0; x < n_x; ++x)
            {
                for(std::size_t y = 0; y < n_y; ++y)
                {
                    apply_future(
                        remote_id, out, x, y
                      , std::vector<hpx::lcos::future<void> >()
                    ).get_future().get();
                }
                (*f.file) << "\n";
            }
        }
    }
}
