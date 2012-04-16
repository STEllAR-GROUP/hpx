
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
#include <algorithm>

#include <hpx/components/dataflow/dataflow.hpp>
#include <hpx/components/dataflow/dataflow_trigger.hpp>
#include <hpx/components/dataflow/async_dataflow_wait.hpp>

/*
#include <google/profiler.h>
#include <google/heap-profiler.h>
*/

#include "server/remote_lse.hpp"

using bright_future::grid;
using bright_future::update;
using bright_future::update_residuum;

typedef bright_future::grid<double> grid_type;
typedef grid_type::size_type size_type;

using hpx::util::high_resolution_timer;

using hpx::cout;
using hpx::flush;
/*
#include <iostream>
using std::cout;
using std::flush;
*/
using hpx::lcos::dataflow;
using hpx::lcos::dataflow_trigger;
using hpx::lcos::dataflow_base;
using hpx::find_here;
using hpx::find_all_localities;
using hpx::find_here;
using hpx::lcos::future;
using hpx::lcos::wait;
using hpx::naming::id_type;
using hpx::naming::get_locality_from_id;

// this is just a simple base class for my functors to be serializable
// this is required by my function implementation, because arguments to hpx
// actions need to be serializable
struct fun_base
{
    virtual ~fun_base() {}

    template <typename Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
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
{
    size_type x_start;
    size_type y_start;

    init_rhs_fun() {}

    init_rhs_fun(size_type x, size_type y)
        : x_start(x)
        , y_start(y)
    {}

    double operator()(size_type x_local, size_type y_local, lse_config const & c)
    {
        size_type x = x_start + x_local - 1;
        size_type y = y_start + y_local - 1;
        return
            39.478 * sin((x * c.hx) * 6.283) * sinh((y * c.hy) * 6.283);
    }

    template <typename Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & x_start;
        ar & y_start;
    }
};

// functor to initialize the grid with boundary conditions
struct init_u_fun
{
    size_type x_start;
    size_type y_start;

    init_u_fun() {}

    init_u_fun(size_type x, size_type y)
        : x_start(x)
        , y_start(y)
    {}

    double operator()(size_type x_local, size_type y_local, lse_config const & c)
    {
        size_type x = x_start + x_local - 1;
        size_type y = y_start + y_local - 1;
        double value =
            y == (c.n_y - 1) ? sin((x * c.hx) * 6.283) * sinh(6.283) : 0.0;

        return value;
    }

    template <typename Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & x_start;
        ar & y_start;
    }
};

// the stencil code
struct update_fun
    : fun_base
{
    double operator()(grid_type const & u, grid_type const & rhs, size_type x, size_type y, lse_config const & c)
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
    double operator()(grid_type const & u, grid_type const & rhs, size_type x, size_type y, lse_config const & c)
    {
        return u(x, y);
    }
};

// apply function to output our result
struct output_fun
{
    std::string file_name;
    size_type x_start;
    size_type y_start;

    output_fun() {}
    output_fun(std::string const & output, size_type x, size_type y)
        : file_name(output)
        , x_start(x)
        , y_start(y)
    {}

    double operator()(grid_type const & u, grid_type const & rhs, size_type x_local, size_type y_local, lse_config const & c)
    {
        std::ofstream file(file_name.c_str(), std::ios_base::app | std::ios_base::out);
        size_type x = x_start + x_local - 1;
        size_type y = y_start + y_local - 1;
        file << x * c.hx << " " << y * c.hy << " " << u(x_local, y_local) << "\n";
        return u(x_local, y_local);
    }

    template <typename Archive>
    void load(Archive & ar, const unsigned int version)
    {
        ar & file_name;
        ar & x_start;
        ar & y_start;
    }

    template <typename Archive>
    void save(Archive & ar, const unsigned int version) const
    {
        ar & file_name;
        ar & x_start;
        ar & y_start;
    }

    BOOST_SERIALIZATION_SPLIT_MEMBER();
};

// not used currently
struct update_residuum_fun
    : fun_base
{
    void operator()(grid_type & u, size_type x, size_type y)
    {
    }
};

void dependency()
{
    //cout << "dependency complete\n" << flush;
}
typedef hpx::actions::plain_action0<&dependency> dependency_action;
HPX_REGISTER_PLAIN_ACTION(dependency_action);

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
  , std::size_t cache_block
  , std::string const & output
)
{
    {
        bool debug = false;
        // initialization of hpx component. this can be hidden nicely
        // this is not something to worry about right now.
        // it creates the remote_lse_type and registers it with the hpx agas
        // server. a remote_id is created, this is a "pointer" to our (possibly)
        // remotely created LSE.
        hpx::components::component_type type
            = hpx::components::get_component_type<remote_lse_type>();

        typedef
            hpx::components::distributing_factory
            distributing_factory;

        std::vector<id_type> prefixes = find_all_localities(type);

        distributing_factory
            factory(
                distributing_factory::create_sync(hpx::find_here())
            );

        double num_blocks_sqrt = std::floor(std::sqrt(static_cast<double>(prefixes.size())));

        double num_blocks = 0;
        if(std::abs(num_blocks_sqrt - std::sqrt(static_cast<double>(prefixes.size()))) > 1e-9)
        {
            num_blocks_sqrt = num_blocks_sqrt + 1;
        }
        num_blocks = num_blocks_sqrt * num_blocks_sqrt;

        distributing_factory::async_create_result_type
            result = factory.create_components_async(type, num_blocks);

        if(debug)
        {
            BOOST_FOREACH(id_type const & prefix, prefixes)
            {
                cout << prefix << "\n" << flush;
            }
        }

        distributing_factory::result_type results = result.get();
        distributing_factory::iterator_range_type
            parts = hpx::components::server::locality_results(results);


        size_type n_x_block = num_blocks_sqrt;
        size_type n_y_block = num_blocks_sqrt;

        size_type n_x_local = n_x / n_x_block + 1;
        size_type n_y_local = n_y / n_y_block + 1;

        size_type n_x_local_block = n_x_local/block_size+1;
        size_type n_y_local_block = n_y_local/block_size+1;

        if(debug)
        {
            cout
                << "N_x       " << n_x << "\n"
                << "N_y       " << n_y << "\n"
                << "N_x_block " << n_x_block << "\n"
                << "N_y_block " << n_y_block << "\n"
                << "N_x_local " << n_x_local << "\n"
                << "N_y_local " << n_y_local << "\n"
                << flush;
        }

        typedef grid<hpx::naming::id_type> id_grid_type;
        id_grid_type grid_ids(n_x_block, n_y_block);
        {
            size_type x = 0;
            size_type y = 0;

            BOOST_FOREACH(hpx::naming::id_type const & id, parts)
            {
                using hpx::naming::id_type;
                using hpx::naming::strip_credit_from_gid;

                grid_ids(x, y) = id_type(
                    strip_credit_from_gid(id.get_gid()),
                    id_type::unmanaged);

                if(++x > n_x_block - 1)
                {
                    x = 0;
                    ++y;
                }
            }
        }

        // initalization of hpx component ends here.

        typedef hpx::lcos::dataflow_base<void> dataflow_type;

        // The init future:
        // a hpx future takes an action. this action is a remotely callable
        // member function on an instance of remote_lse_type
        // the init function has the signature
        // void(unsigned n_x, unsigned n_y, double hx, double hy)
        for(size_type y_block = 0; y_block < n_y_block; ++y_block)
        {
            for(size_type x_block = 0; x_block < n_x_block; ++x_block)
            {
                hpx::async<remote_lse_type::init_action>(
                    grid_ids(x_block, y_block)
                  , n_x_local + 2
                  , n_y_local + 2
                  , n_x
                  , n_y
                  , hx
                  , hy
                ).get();
            }
        }

        if(debug) cout << "init1\n" << flush;

        typedef grid<dataflow_type> promise_grid_type;
        typedef grid<promise_grid_type> promise_grid_block_type;
        typedef std::vector<promise_grid_block_type> iteration_dependencies_type;

        iteration_dependencies_type
            iteration_dependencies(
                2
              , promise_grid_block_type(
                    n_x_block
                  , n_y_block
                  , cache_block
                  , promise_grid_type(
                        n_x_local_block
                      , n_y_local_block
                    )
                )
            );

        promise_grid_block_type
            rhs_promise(
                n_x_block
              , n_y_block
              , cache_block
              , promise_grid_type(
                    n_x_local_block
                  , n_y_local_block
                )
            );

        {
            typedef
                hpx::lcos::dataflow<remote_lse_type::init_rhs_blocked_action>
                init_rhs_dataflow;

            typedef
                hpx::lcos::dataflow<remote_lse_type::init_u_blocked_action>
                init_u_dataflow;

            for(size_type y_block = 0; y_block < n_y_block; ++y_block)
            {
                for(size_type x_block = 0; x_block < n_x_block; ++x_block)
                {
                    for(size_type y = 0, yy = 0; y < n_y_local + 2; y += block_size, ++yy)
                    {
                        for(size_type x = 0, xx = 0; x < n_x_local + 2; x += block_size, ++xx)
                        {
                            range_type
                                x_range(
                                    x
                                  , (std::min)(n_x_local + 2, x + block_size)
                                );

                            range_type
                                y_range(
                                    y
                                  , (std::min)(n_y_local + 2, y + block_size)
                                );

                            rhs_promise(x_block, y_block)(xx, yy) =
                                init_rhs_dataflow(
                                    grid_ids(x_block, y_block)
                                  , init_rhs_fun(
                                        x_block * n_x_local
                                      , y_block * n_y_local
                                    )
                                  , x_range
                                  , y_range
                                );

                            iteration_dependencies[0](x_block, y_block)(xx, yy) =
                                init_u_dataflow(
                                    grid_ids(x_block, y_block)
                                  , init_u_fun(
                                        x_block * n_x_local
                                      , y_block * n_y_local
                                    )
                                  , x_range
                                  , y_range
                                );
                        }
                    }
                }
            }
        }
        if(debug) cout << "init2\n" << flush;

        typedef
            hpx::lcos::dataflow<remote_lse_type::apply_region_df_action>
            apply_region_dataflow;

        //ProfilerStart("gs_hpx_distributed-cpu.prof");
        //HeapProfilerStart("gs_hpx_distributed-heap.heap");
        high_resolution_timer t;

        for(unsigned iter = 0; iter < max_iterations; ++iter)
        {
            promise_grid_block_type & prev_block = iteration_dependencies.at(iter%2);
            promise_grid_block_type & current_block = iteration_dependencies.at((iter + 1)%2);
            for(size_type y_block = 0; y_block < n_y_block; ++y_block)
            {
                for(size_type x_block = 0; x_block < n_x_block; ++x_block)
                {
                    typedef
                        hpx::lcos::dataflow<remote_lse_type::get_col_action>
                        get_column_dataflow;
                    typedef
                        hpx::lcos::dataflow<remote_lse_type::get_row_action>
                        get_row_dataflow;
                    typedef
                        hpx::lcos::dataflow<remote_lse_type::update_top_boundary_action>
                        update_top_boundary_dataflow;
                    typedef
                        hpx::lcos::dataflow<remote_lse_type::update_bottom_boundary_action>
                        update_bottom_boundary_dataflow;
                    typedef
                        hpx::lcos::dataflow<remote_lse_type::update_left_boundary_action>
                        update_left_boundary_dataflow;
                    typedef
                        hpx::lcos::dataflow<remote_lse_type::update_right_boundary_action>
                        update_right_boundary_dataflow;

                    promise_grid_type & prev = prev_block(x_block, y_block);
                    promise_grid_type & current = current_block(x_block, y_block);

                    for(size_type y = 1, yy = 0; y < n_y_local + 1; y += block_size, ++yy)
                    {
                        for(size_type x = 1, xx = 0; x < n_x_local + 1; x += block_size, ++xx)
                        {
                            range_type
                                x_range(
                                    xx == 0 && x_block == 0 ? 2 : x
                                  , xx + 1 == n_x_local_block
                                        ? x_block + 1 == n_x_block
                                            ? n_x - (x_block * n_x_local)
                                            : (std::min)(n_x_local + 1, x + block_size)
                                        : (std::min)(n_x_local + 1, x + block_size)
                                );

                            range_type
                                y_range(
                                    yy == 0 && y_block == 0 ? 2 : y
                                  , yy + 1 == n_y_local_block
                                        ? y_block + 1 == n_y_block
                                            ? n_y - (y_block * n_y_local)
                                            : (std::min)(n_y_local + 1, y + block_size)
                                        : (std::min)(n_y_local + 1, y + block_size)
                                );

                            std::vector<dataflow_base<void> > deps;
                            deps.reserve(9);

                            deps.push_back(prev(xx, yy));
                            if(iter == 0)
                            {
                                deps.push_back(rhs_promise(x_block, y_block)(xx, yy));
                            }

                            if(xx + 1 < n_x_local_block)
                            {
                                deps.push_back(prev(xx + 1, yy));
                            }

                            if(yy + 1 < n_y_local_block)
                            {
                                deps.push_back(prev(xx, yy + 1));
                            }

                            if(xx > 0)
                            {
                                deps.push_back(current(xx - 1, yy));
                            }

                            if(yy > 0)
                            {
                                deps.push_back(current(xx, yy - 1));
                            }

                            if(xx == 0 && x_block > 0)
                            {
                                deps.push_back(
                                    update_left_boundary_dataflow(
                                        grid_ids(x_block, y_block)
                                      , get_column_dataflow(
                                            grid_ids(x_block - 1, y_block)
                                          , n_x_local
                                          , y_range
                                        )
                                      , y_range
                                      , prev_block(x_block - 1, y_block)(
                                            n_x_local_block -1
                                          , yy
                                        )
                                    )
                                );
                            }

                            if(
                                xx + 1 == n_x_local_block
                             && x_block + 1 < n_x_block
                            )
                            {
                                deps.push_back(
                                    update_right_boundary_dataflow(
                                        grid_ids(x_block, y_block)
                                      , get_column_dataflow(
                                            grid_ids(x_block + 1, y_block)
                                          , 1
                                          , y_range
                                        )
                                      , y_range
                                      , prev_block(x_block + 1, y_block)(
                                            0
                                          , yy
                                        )
                                    )
                                );
                                //cout << "r" << flush;
                            }

                            if(yy == 0 && y_block > 0)
                            {
                                deps.push_back(
                                    update_top_boundary_dataflow(
                                        grid_ids(x_block, y_block)
                                      , get_row_dataflow(
                                            grid_ids(x_block, y_block - 1)
                                          , n_y_local
                                          , x_range
                                        )
                                      , x_range
                                      , prev_block(x_block, y_block - 1)(
                                            xx
                                          , n_y_local_block -1
                                        )
                                    )
                                );
                                //cout << "t" << flush;
                            }

                            if(yy + 1 == n_y_local_block && y_block + 1 < n_y_block)
                            {
                                deps.push_back(
                                    update_bottom_boundary_dataflow(
                                        grid_ids(x_block, y_block)
                                      , get_row_dataflow(
                                            grid_ids(x_block, y_block + 1)
                                          , 1
                                          , x_range
                                        )
                                      , x_range
                                      , prev_block(x_block, y_block + 1)(
                                            xx
                                          , 0
                                        )
                                    )
                                );
                                //cout << "b" << flush;
                            }

                            /*
                            cout << "x " << x_range.first << " " << x_range.second << "\n" << flush;
                            cout << "y " << y_range.first << " " << y_range.second << "\n" << flush;
                            */
                            current(xx, yy) =
                                apply_region_dataflow(
                                    grid_ids(x_block, y_block)
                                  , update_fun()
                                  , x_range
                                  , y_range
                                  , dataflow_trigger(grid_ids(x_block, y_block), deps)
                                );
                            //cout << ".! " << flush;
                        }
                    }
                }
            }
            //cout << "done\n" << flush;
        }

        // wait for the last iteration to finish.
        if(debug) cout << "dataflow tree construction completed " << flush;
        BOOST_FOREACH(promise_grid_type & block, iteration_dependencies[max_iterations%2])
        {
//             wait(block.data_handle(), [&](size_t){if(debug) cout << "." << flush; });
            BOOST_FOREACH(dataflow_base<void> & promise, block)
            {
                cout << "." << flush;
                promise.get_future().get();
            }
        }
        if(debug) cout << "\n";

        double time_elapsed = t.elapsed();
        cout << ((((n_x-2)*(n_y-2) * max_iterations)/1e6)/time_elapsed) << " MLUP/S\n" << flush;
        //ProfilerStop();
        //HeapProfilerDump("computation finished");
        //HeapProfilerStop();

        if(!output.empty())
        {
            // make sure to have an empty file
            {
                std::ofstream(output.c_str());
            }
            for(size_type x_block = 0; x_block < n_x_block; ++x_block)
            {
                for(size_type y_block = 0; y_block < n_y_block; ++y_block)
                {
                    range_type
                        x_range(
                            1
                          , x_block + 1 == n_x_block
                          ? n_x - x_block * n_x_local + 1
                          : n_x_local+1
                        );

                    range_type
                        y_range(
                            1
                          , y_block + 1 == n_y_block
                          ? n_y - y_block * n_y_local + 1
                          : n_y_local+1
                        );

                    cout << "." << flush;
                    apply_region_dataflow(
                        grid_ids(x_block, y_block)
                      , output_fun(
                            output
                          , x_block * n_x_local
                          , y_block * n_y_local
                        )
                      , x_range
                      , y_range
                    ).get_future().get();
                }
            }
            cout << "\n" << flush;
        }
    }
}

