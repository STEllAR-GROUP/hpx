//  Copyright (c) 2012 Thomas Heller
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
#include <hpx/components/remote_object/new.hpp>
#include "dataflow_object.hpp"
#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/lcos/async.hpp>
#include <hpx/include/iostreams.hpp>
#include <algorithm>

#include <hpx/components/dataflow/dataflow.hpp>
#include <hpx/components/dataflow/dataflow_trigger.hpp>
#include <hpx/components/dataflow/async_dataflow_wait.hpp>

#undef min

using bright_future::grid;
using bright_future::range_type;
using bright_future::jacobi_kernel_simple;

typedef bright_future::grid<double> grid_type;
typedef grid_type::size_type size_type;

using hpx::util::high_resolution_timer;

using hpx::cout;
using hpx::flush;
using hpx::lcos::dataflow_trigger;
using hpx::lcos::dataflow_base;
using hpx::find_here;
using hpx::find_all_localities;
using hpx::find_here;
using hpx::lcos::wait;
using hpx::naming::id_type;
using hpx::naming::get_locality_from_id;

using hpx::components::new_;

using hpx::components::dataflow_object;

struct update_fun
{
    typedef void result_type;
    
    range_type x_range;
    range_type y_range;
    size_type old;
    size_type n;
    size_type cache_block;

    update_fun() {}

    update_fun(range_type x, range_type y, size_type old, size_type n, size_type c)
        : x_range(x)
        , y_range(y)
        , old(old)
        , n(n)
        , cache_block(c)
    {}

    void operator()(std::vector<grid_type> & u) const
    {
        jacobi_kernel_simple(
            u
          , x_range
          , y_range
          , old, n
          , cache_block
        );
    }
    
    template <typename Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & x_range;
        ar & y_range;
        ar & old;
        ar & n;
        ar & cache_block;
    }
};

struct output_fun
{
    std::string file_name;
    range_type x_range;
    range_type y_range;
    size_type cur;

    output_fun() {}

    output_fun(std::string const & output, range_type x, range_type y, size_type cur)
        : file_name(output)
        , x_range(x)
        , y_range(y)
        , cur(cur)
    {}

    typedef void result_type;

    void operator()(std::vector<grid_type> & u) const
    {
        std::ofstream file(file_name.c_str(), std::ios_base::app | std::ios_base::out);
        for(size_type x = x_range.first; x < x_range.second; ++x)
        {
            for(size_type y = y_range.first; y < y_range.second; ++y)
            {
                file << x << " " << y << " " << u[cur](x, y) << "\n";
            }
            file << "\n";
        }
    }
    
    template <typename Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & file_name;
        ar & x_range;
        ar & y_range;
    }
};

void gs(
    size_type n_x
  , size_type n_y
  , double hx_
  , double hy_
  , double k_
  , double relaxation_
  , unsigned max_iterations
  , unsigned iteration_block
  , unsigned block_size
  , std::size_t cache_block
  , std::string const & output
)
{
    typedef dataflow_object<std::vector<grid_type> > object_type;

    object_type u(new_<std::vector<grid_type> >(find_here(), 2, grid_type(n_x, n_y, block_size, 1)).get());

    n_x = n_x -1;
    n_y = n_y -1;

    high_resolution_timer t;
    size_type old = 0;
    size_type n = 1;
    size_type n_x_block = (n_x - 2)/block_size + 1;
    size_type n_y_block = (n_y - 2)/block_size + 1;
    typedef dataflow_base<void> promise;
    typedef grid<promise> promise_grid_type;
    std::vector<promise_grid_type> deps(2, promise_grid_type(n_x_block, n_y_block));

    t.restart();
    for(std::size_t iter = 0; iter < max_iterations; ++iter)//iter += iteration_block)
    {
        for(size_type y = 1, yy = 0; y < n_y - 1; y += block_size, ++yy)
        {
            size_type y_end = (std::min)(y + block_size, n_y-1);
            for(size_type x = 1, xx = 0; x < n_x - 1; x += block_size, ++xx)
            {
                size_type x_end = (std::min)(x + block_size, n_x-1);
                if(iter > 0)
                {
                    std::vector<promise > trigger;
                    trigger.reserve(5);
                    trigger.push_back(deps[old](xx,yy));
                    if(xx + 1 < n_x_block)
                        trigger.push_back(deps[old](xx+1, yy));
                    if(xx > 0)
                        trigger.push_back(deps[old](xx-1, yy));
                    if(yy + 1 < n_y_block)
                        trigger.push_back(deps[old](xx, yy+1));
                    if(yy > 0)
                        trigger.push_back(deps[old](xx, yy-1));

                    deps[n](xx, yy)
                        = u.apply(
                            update_fun(
                                range_type(x, x_end)
                              , range_type(y, y_end)
                              , old
                              , n
                              , cache_block
                            )
                          , dataflow_trigger(u.gid_, trigger)
                        );
                }
                else
                {
                    deps[n](xx, yy)
                        = u.apply(
                            update_fun(
                                range_type(x, x_end)
                              , range_type(y, y_end)
                              , old
                              , n
                              , cache_block
                            )
                        );
                }
            }
        }
        std::swap(old, n);
    }
    for(size_type y = 1, yy = 0; y < n_y - 1; y += block_size, ++yy)
    {
        for(size_type x = 1, xx = 0; x < n_x - 1; x += block_size, ++xx)
        {
            deps[old](xx, yy).get();
        }
    }

    double time_elapsed = t.elapsed();
    cout << n_x-1 << "x" << n_y-1 << " "
         << ((((n_x-2)*(n_y-2) * max_iterations)/1e6)/time_elapsed) << " MLUP/S\n" << flush;

    if(!output.empty())
    {
        // make sure to have an empty file
        {
            std::ofstream(output.c_str());
        }
        u.apply(output_fun(output, range_type(0, n_x), range_type(0, n_y), old)).get();
    }
}
