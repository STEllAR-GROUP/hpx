
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
#include <hpx/lcos/eager_future.hpp>
#include <hpx/include/iostreams.hpp>

#include "server/remote_lse.hpp"
#include "dataflow/dataflow.hpp"

using bright_future::grid;
using bright_future::update;
using bright_future::update_residuum;

typedef bright_future::grid<double> grid_type;
typedef grid_type::size_type size_type;

using hpx::util::high_resolution_timer;

using hpx::cout;
using hpx::flush;
using hpx::lcos::dataflow;
using hpx::lcos::dataflow_base;
using hpx::find_here;
using hpx::find_all_localities;
using hpx::find_here;
using hpx::lcos::promise;
using hpx::lcos::wait;
using hpx::naming::id_type;

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
  , std::string const & output
)
{
    {
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
        
        std::vector<id_type> prefixes = find_all_localities();

        distributing_factory
            factory(
                distributing_factory::create_sync(hpx::find_here())
            );

        distributing_factory::async_create_result_type
            result = factory.create_components_async(type, prefixes.size());

        distributing_factory::result_type results = result.get();
        distributing_factory::iterator_range_type
            parts = hpx::components::server::locality_results(results);

        size_type n_x_block = std::sqrt(prefixes.size());
        size_type n_y_block = std::sqrt(prefixes.size());

        size_type n_x_local = n_x / n_x_block + 1;
        size_type n_y_local = n_y / n_y_block + 1;

        cout
            << "N_x       " << n_x << "\n"
            << "N_y       " << n_y << "\n"
            << "N_x_block " << n_x_block << "\n"
            << "N_y_block " << n_y_block << "\n"
            << "N_x_local " << n_x_local << "\n"
            << "N_y_local " << n_y_local << "\n"
            << flush;

        typedef grid<hpx::naming::id_type> id_grid_type;
        id_grid_type grid_ids(n_x_block, n_y_block);
        {
            size_type x = 0;
            size_type y = 0;

            BOOST_FOREACH(hpx::naming::id_type const & id, parts)
            {
                grid_ids(x, y) = id;
                
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
        typedef
            hpx::lcos::eager_future<remote_lse_type::init_action>
            init_dataflow;

        for(size_type y_block = 0; y_block < n_y_block; ++y_block)
        {
            for(size_type x_block = 0; x_block < n_x_block; ++x_block)
            {
                /*
                size_type x_end
                    = x_block == n_x_block - 1
                    ? n_x - x_block* n_x_local + 1
                    : n_x_local + 1;
                size_type y_end
                    = y_block == n_y_block - 1
                    ? n_y - y_block * n_y_local + 1
                    : n_y_local + 1;

                cout << "x range: " << x_block * n_x_local << " " << (x_block + 1) * n_x_local << "\n" << flush;
                cout << "y range: " << y_block * n_y_local << " " << (y_block + 1) * n_y_local << "\n" << flush;
                */
                init_dataflow(
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

        cout << "init1\n" << flush;

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
                    for(size_type y = 0; y < n_y_local + 2; y += block_size)
                    {
                        for(size_type x = 0; x < n_x_local + 2; x += block_size)
                        {
                            range_type
                                x_range(
                                    x
                                  , x + 1 == n_x_local + 2
                                  ? n_x - (x_block + x * n_x_local + 2)
                                  : std::min(n_x_local + 2, x + block_size)
                                );
                            
                            range_type
                                y_range(
                                    y
                                  , y + 1 == n_y_local + 2
                                  ? n_y - (y_block + y * n_y_local + 2)
                                  : std::min(n_y_local + 2, y + block_size)
                                );

                            init_rhs_dataflow(
                                grid_ids(x_block, y_block)
                              , init_rhs_fun(
                                    x_block * n_x_local
                                  , y_block * n_y_local
                                )
                              , x_range
                              , y_range
                            ).get();
                            
                            init_u_dataflow(
                                grid_ids(x_block, y_block)
                              , init_u_fun(
                                    x_block * n_x_local
                                  , y_block * n_y_local
                                )
                              , x_range
                              , y_range
                            ).get();
                        }
                    }
                }
            }
        }
        cout << "init2\n" << flush;

        typedef
            hpx::lcos::dataflow<remote_lse_type::apply_region_df_action>
            apply_region_dataflow;

        high_resolution_timer t;
        
        for(unsigned iter = 0; iter < max_iterations; ++iter)
        {
            for(size_type y_block = 0; y_block < n_y_block; ++y_block)
            {
                for(size_type x_block = 0; x_block < n_x_block; ++x_block)
                {
                    // boundary updates ....
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

                    for(size_type y = 1; y < n_y_local + 1; y += block_size)
                    {
                        for(size_type x = 1; x < n_x_local + 1; x += block_size)
                        {
                            range_type
                                x_range(
                                    x_block == 0 && x == 1
                                  ? 2
                                  : x
                                  , x + block_size >= n_x_local + 1
                                  ? x_block + 1 == n_x_block
                                    ? n_x - (x_block * n_x_local)
                                    : n_x_local + 1
                                  : n_x_local + 1
                                );
                            
                            range_type
                                y_range(
                                    y_block == 0 && y == 1
                                  ? 2
                                  : y
                                  , y + block_size >= n_y_local + 1
                                  ? y_block + 1 == n_y_block
                                    ? n_y - (y_block * n_y_local)
                                    : n_y_local + 1
                                  : n_y_local + 1
                                );

                            if(x == 1 && x_block > 0)
                            {
                                update_left_boundary_dataflow(
                                    grid_ids(x_block, y_block)
                                  , get_column_dataflow(
                                        grid_ids(x_block - 1, y_block)
                                      , n_y_local
                                      , y_range
                                    )
                                  , y_range
                                ).get();
                            }

                            if(x == n_x_local && x_block < n_x_local)
                            {
                                update_right_boundary_dataflow(
                                    grid_ids(x_block, y_block)
                                  , get_column_dataflow(
                                        grid_ids(x_block + 1, y_block)
                                      , 1
                                      , y_range
                                    )
                                  , y_range
                                ).get();
                            }
                            
                            if(y == 1 && y_block > 0)
                            {
                                update_top_boundary_dataflow(
                                    grid_ids(x_block, y_block)
                                  , get_row_dataflow(
                                        grid_ids(x_block, y_block - 1)
                                      , n_x_local
                                      , x_range
                                    )
                                  , x_range
                                ).get();
                            }
                            
                            if(y == n_y_local && y_block < n_y_local)
                            {
                                update_bottom_boundary_dataflow(
                                    grid_ids(x_block, y_block)
                                  , get_row_dataflow(
                                        grid_ids(x_block, y_block + 1)
                                      , 1
                                      , x_range
                                    )
                                  , x_range
                                ).get();
                            }

                            /*
                            cout << "x " << x_range.first << " " << x_range.second << "\n" << flush;
                            cout << "y " << y_range.first << " " << y_range.second << "\n" << flush;
                            */

                            apply_region_dataflow(
                                grid_ids(x_block, y_block)
                              , update_fun()
                              , x_range
                              , y_range
                            ).get();
                        }
                    }
                }
            }
        }

        double time_elapsed = t.elapsed();
        cout << time_elapsed << "\n" << flush;
        
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
                    ).get();
                }
            }
            cout << "\n" << flush;
        }
    }
}

