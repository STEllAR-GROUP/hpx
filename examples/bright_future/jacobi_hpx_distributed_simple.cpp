
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
#include "distributed_new.hpp"
#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/lcos/eager_future.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/components/dataflow/dataflow.hpp>
#include <hpx/components/dataflow/dataflow_trigger.hpp>
#include <hpx/components/dataflow/async_dataflow_wait.hpp>

#include "create_grid_dim.hpp"

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

using hpx::components::distributed_new;

using hpx::components::object;
using hpx::components::dataflow_object;

struct get_col_fun
{
    range_type range;
    size_type col;
    size_type cur;

    typedef std::vector<double> result_type;

    get_col_fun() {}
    get_col_fun(range_type const & r, size_type ro, size_type cur)
        : range(r), col(ro), cur(cur) {}
    
    template <typename Archive>
    void serialize(Archive & ar, unsigned)
    {
        ar & range;
        ar & col;
        ar & cur;
    }

    result_type operator()(std::vector<grid_type> & u) const
    {
        std::vector<double> result;
        result.reserve(range.second-range.first);
        for(size_type y = range.first; y < range.second; ++y)
        {
            result.push_back(u[cur](col, y));
        }
        return result;
    }
};

struct get_row_fun
{
    range_type range;
    size_type row;
    size_type cur;

    typedef std::vector<double> result_type;

    get_row_fun() {}
    get_row_fun(range_type const & r, size_type ro, size_type cur)
        : range(r), row(ro), cur(cur) {}

    template <typename Archive>
    void serialize(Archive & ar, unsigned)
    {
        ar & range;
        ar & row;
        ar & cur;
    }

    result_type operator()(std::vector<grid_type> & u) const
    {
        std::vector<double> result;
        
        result.reserve((range.second-range.first));

        for(size_type x = range.first; x < range.second; ++x)
        {
            result.push_back(u[cur](x, row));
        }

        return result;
    }
};

struct update_top_boundary_fun
{
    range_type range;
    dataflow_object<std::vector<grid_type> > neighbor;
    size_type cur;

    typedef void result_type;

    update_top_boundary_fun() {}
    update_top_boundary_fun(range_type const & r, dataflow_object<std::vector<grid_type> > n, size_type cur)
        : range(r)
        , neighbor(n)
        , cur(cur)
    {}

    template <typename Archive>
    void serialize(Archive & ar, unsigned version)
    {
        ar & range;
        ar & neighbor;
        ar & cur;
    }

    result_type operator()(std::vector<grid_type> & u) const
    {
    
        std::vector<double> b = neighbor.apply(get_row_fun(range, u[cur].y()-2, cur)).get();
        for(size_type x = range.first, i = 0; x < range.second; ++x, ++i)
        {
            u[cur](x, 0) = b.at(i);
        }
    }
};

struct update_bottom_boundary_fun
{
    range_type range;
    dataflow_object<std::vector<grid_type> > neighbor;
    size_type cur;

    typedef void result_type;

    update_bottom_boundary_fun() {}
    update_bottom_boundary_fun(range_type const & r, dataflow_object<std::vector<grid_type> > n, size_type cur)
        : range(r)
        , neighbor(n)
        , cur(cur)
    {}

    template <typename Archive>
    void serialize(Archive & ar, unsigned version)
    {
        ar & range;
        ar & neighbor;
        ar & cur;
    }

    result_type operator()(std::vector<grid_type> & u) const
    {
        std::vector<double> b = neighbor.apply(get_row_fun(range, 1, cur)).get();
        for(size_type x = range.first, i = 0; x < range.second; ++x, ++i)
        {
            u[cur](x, u[cur].y()-1) = b.at(i);
        }
    }
};

struct update_right_boundary_fun
{
    range_type range;
    dataflow_object<std::vector<grid_type>> neighbor;
    size_type cur;

    typedef void result_type;

    update_right_boundary_fun() {}
    update_right_boundary_fun(range_type const & r, dataflow_object<std::vector<grid_type>> n, size_type cur)
        : range(r)
        , neighbor(n)
        , cur(cur)
    {}

    template <typename Archive>
    void serialize(Archive & ar, unsigned version)
    {
        ar & range;
        ar & neighbor;
        ar & cur;
    }

    result_type operator()(std::vector<grid_type> & u) const
    {
        std::vector<double> b = neighbor.apply(get_col_fun(range, 1, cur)).get();
        for(size_type y = range.first, i = 0; y < range.second; ++y, ++i)
        {
            u[cur](u[cur].x()-1, y) = b.at(i);
        }
    }
};

struct update_left_boundary_fun
{
    range_type range;
    dataflow_object<std::vector<grid_type>> neighbor;
    size_type cur;

    typedef void result_type;

    update_left_boundary_fun() {}
    update_left_boundary_fun(range_type const & r, dataflow_object<std::vector<grid_type>> n, size_type cur)
        : range(r)
        , neighbor(n)
        , cur(cur)
    {}

    template <typename Archive>
    void serialize(Archive & ar, unsigned version)
    {
        ar & range;
        ar & neighbor;
        ar & cur;
    }

    result_type operator()(std::vector<grid_type> & u) const
    {
        std::vector<double> b = neighbor.apply(get_col_fun(range, u[cur].x()-2, cur)).get();
        for(size_type y = range.first, i = 0; y < range.second; ++y, ++i)
        {
            u[cur](0, y) = b.at(i);
        }
    }
};

struct update_fun
{
    typedef void result_type;
    
    range_type x_range;
    range_type y_range;
    size_type src;
    size_type dst;
    size_type cache_block;

    update_fun() {}

    update_fun(range_type x, range_type y, size_type old, size_type n, size_type c)
        : x_range(x)
        , y_range(y)
        , src(old)
        , dst(n)
        , cache_block(c)
    {}

    void operator()(std::vector<grid_type> & u) const
    {
        jacobi_kernel_simple(
            u
          , x_range
          , y_range
          , src, dst
          , cache_block
        );
    }
    
    template <typename Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & x_range;
        ar & y_range;
        ar & src;
        ar & dst;
        ar & cache_block;
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
    hpx::components::component_type type
        = hpx::components::get_component_type<hpx::components::server::remote_object>();

    std::vector<id_type> prefixes = find_all_localities(type);

    std::vector<std::size_t> dims = create_dim(prefixes.size(), 2);
    
    n_x = n_x -1;
    n_y = n_y -1;

    std::size_t n_x_local = n_x / dims[0];
    std::size_t n_y_local = n_y / dims[1];

    cout
        << "Locality Grid: " << dims[0] << "x" << dims[1] << "\n"
        << "Grid dimension: " << n_x << "x" << n_y << "\n"
        << "Local Grid dimension: " << n_x_local << "x" << n_y_local << "\n"
        << flush;

    typedef std::vector<grid_type> data_type;
    typedef dataflow_object<data_type> object_type;
    typedef dataflow_base<void> promise;
    typedef grid<promise> promise_grid_type;

    grid<object_type> object_grid(dims[0], dims[1]);
    {
        std::vector<hpx::lcos::promise<object<data_type> > >
            objects =
                distributed_new<data_type>(
                    dims[0] * dims[1]
                  , 2
                  , grid_type(
                        n_x_local + 1
                      , n_y_local + 1
                      , block_size
                      , 1
                    )
                );
        size_type x = 0;
        size_type y = 0;

        BOOST_FOREACH(hpx::lcos::promise<object<data_type> > const & o, objects)
        {
            using hpx::naming::id_type;
            using hpx::naming::strip_credit_from_gid;

            strip_credit_from_gid(o.get().gid_.get_gid());
            id_type
                id(
                    o.get().gid_.get_gid()
                  , id_type::unmanaged
                );
            object_grid(x, y) = id;

            if(++x > dims[0] - 1)
            {
                x = 0;
                ++y;
            }
        }
    }
    
    size_type n_x_local_block = (n_x_local - 2)/block_size + 1;
    size_type n_y_local_block = (n_y_local - 2)/block_size + 1;

    std::vector<grid<grid<promise> > >
        deps(
            2
          , grid<grid<promise> >(
                dims[0]
              , dims[1]
              , block_size
              , grid<promise>(
                    n_x_local_block
                  , n_y_local_block
                )
            )
        );

    std::size_t src = 0;
    std::size_t dst = 1;

    high_resolution_timer t;
    t.restart();
    for(std::size_t iter = 0; iter < max_iterations; ++iter)
    {
        for(size_type y_block = 0; y_block < dims[1]; ++y_block)
        {
            for(size_type x_block = 0; x_block < dims[0]; ++x_block)
            {
                promise_grid_type & cur_deps = deps[src](x_block, y_block);
                for(size_type y = 1, yy = 0; y < n_y_local; y += block_size, ++yy)
                {
                    size_type y_end = (std::min)(y + block_size, n_y_local);
                    for(size_type x = 1, xx = 0; x < n_x_local; x += block_size, ++xx)
                    {
                        size_type x_end = (std::min)(x + block_size, n_x_local);
                        if(iter > 0)
                        {
                            range_type x_range(x, x_end);
                            range_type y_range(y, y_end);
                            std::vector<promise > trigger;
                            trigger.reserve(9);
                            trigger.push_back(cur_deps(xx,yy));
                            if(xx + 1 < n_x_local_block)
                                trigger.push_back(cur_deps(xx+1, yy));
                            if(xx > 0)
                                trigger.push_back(cur_deps(xx-1, yy));
                            if(yy + 1 < n_y_local_block)
                                trigger.push_back(cur_deps(xx, yy+1));
                            if(yy > 0)
                                trigger.push_back(cur_deps(xx, yy-1));
                            
                            if(xx == 0 && x_block > 0)
                            {
                                trigger.push_back(
                                    object_grid(x_block, y_block).apply(
                                        update_left_boundary_fun(
                                            y_range
                                          , object_grid(x_block - 1, y_block)
                                          , src
                                        )
                                    )
                                );
                            }

                            if(xx + 1 == n_x_local && x_block + 1 < dims[0])
                            {
                                trigger.push_back(
                                    object_grid(x_block, y_block).apply(
                                        update_right_boundary_fun(
                                            y_range
                                          , object_grid(x_block + 1, y_block)
                                          , src
                                        )
                                    )
                                );
                            }

                            if(yy == 0 && y_block > 0)
                            {
                                trigger.push_back(
                                    object_grid(x_block, y_block).apply(
                                        update_top_boundary_fun(
                                            x_range
                                          , object_grid(x_block, y_block - 1)
                                          , src
                                        )
                                    )
                                );
                            }

                            if(yy + 1 == n_y_local && y_block + 1 < dims[1])
                            {
                                trigger.push_back(
                                    object_grid(x_block, y_block).apply(
                                        update_bottom_boundary_fun(
                                            x_range
                                          , object_grid(x_block, y_block + 1)
                                          , src
                                        )
                                    )
                                );
                            }

                            deps[dst](x_block, y_block)(xx, yy)
                                = object_grid(x_block, y_block).apply(
                                    update_fun(
                                        range_type(x, x_end)
                                      , range_type(y, y_end)
                                      , src
                                      , dst
                                      , cache_block
                                    )
                                  , dataflow_trigger(object_grid(x_block, y_block).gid_, trigger)
                                );
                        }
                        else
                        {
                            deps[dst](x_block, y_block)(xx, yy)
                                = object_grid(x_block, y_block).apply(
                                    update_fun(
                                        range_type(x, x_end)
                                      , range_type(y, y_end)
                                      , src
                                      , dst
                                      , cache_block
                                    )
                                );
                        }
                    }
                }
            }
        }
        std::swap(dst, src);
    }
    for(size_type y_block = 0; y_block < dims[1]; ++y_block)
    {
        for(size_type x_block = 0; x_block < dims[0]; ++x_block)
        {
            for(size_type y = 1, yy = 0; y < n_y_local; y += block_size, ++yy)
            {
                for(size_type x = 1, xx = 0; x < n_x_local; x += block_size, ++xx)
                {
                    deps[src](x_block, y_block)(xx, yy).get();
                }
            }
        }
    }

    double time_elapsed = t.elapsed();
    cout << n_x << "x" << n_y << " "
         << ((((n_x-2)*(n_y-2) * max_iterations)/1e6)/time_elapsed) << " MLUP/S\n" << flush;
}
