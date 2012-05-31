
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
#include <hpx/components/remote_object/distributed_new.hpp>
#include <hpx/components/dataflow/dataflow_object.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/include/async.hpp>
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
    std::size_t col;
    std::size_t cur;

    typedef std::vector<double> result_type;

    get_col_fun() {}
    get_col_fun(range_type const & r, std::size_t ro, std::size_t cu)
        : range(r), col(ro), cur(cu) {}

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
        for(std::size_t y = range.first; y < range.second; ++y)
        {
            result.push_back(u[cur](col, y));
        }
        return result;
    }
};

struct get_row_fun
{
    range_type range;
    std::size_t row;
    std::size_t cur;

    typedef std::vector<double> result_type;

    get_row_fun() {}
    get_row_fun(range_type const & r, std::size_t ro, std::size_t cu)
        : range(r), row(ro), cur(cu) {}

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

        for(std::size_t x = range.first; x < range.second; ++x)
        {
            result.push_back(u[cur](x, row));
        }

        return result;
    }
};

struct update_top_boundary_fun
{
    range_type range;
    std::size_t cur;

    typedef void result_type;

    update_top_boundary_fun() {}
    update_top_boundary_fun(range_type const & r, std::size_t cu)
        : range(r)
        , cur(cu)
    {}

    template <typename Archive>
    void serialize(Archive & ar, unsigned )
    {
        ar & range;
        ar & cur;
    }

    result_type operator()(std::vector<grid_type> & u, std::vector<double> const & b) const
    {
        for(std::size_t x = range.first, i = 0; x < range.second; ++x, ++i)
        {
            u[cur](x, 0) = b[i];
        }
    }
};

struct update_bottom_boundary_fun
{
    range_type range;
    std::size_t cur;

    typedef void result_type;

    update_bottom_boundary_fun() {}
    update_bottom_boundary_fun(range_type const & r, std::size_t cu)
        : range(r)
        , cur(cu)
    {}

    template <typename Archive>
    void serialize(Archive & ar, unsigned )
    {
        ar & range;
        ar & cur;
    }

    result_type operator()(std::vector<grid_type> & u, std::vector<double> const & b) const
    {
        for(std::size_t x = range.first, i = 0; x < range.second; ++x, ++i)
        {
            u[cur](x, u[cur].y()-1) = b[i];
        }
    }
};

struct update_right_boundary_fun
{
    range_type range;
    std::size_t cur;

    typedef void result_type;

    update_right_boundary_fun() {}
    update_right_boundary_fun(range_type const & r, std::size_t cu)
        : range(r)
        , cur(cu)
    {}

    template <typename Archive>
    void serialize(Archive & ar, unsigned )
    {
        ar & range;
        ar & cur;
    }

    result_type operator()(std::vector<grid_type> & u, std::vector<double> const & b) const
    {
        for(std::size_t y = range.first, i = 0; y < range.second; ++y, ++i)
        {
            u[cur](u[cur].x()-1, y) = b[i];
        }
    }
};

struct update_left_boundary_fun
{
    range_type range;
    std::size_t cur;

    typedef void result_type;

    update_left_boundary_fun() {}
    update_left_boundary_fun(range_type const & r, std::size_t cu)
        : range(r)
        , cur(cu)
    {}

    template <typename Archive>
    void serialize(Archive & ar, unsigned)
    {
        ar & range;
        ar & cur;
    }

    result_type operator()(std::vector<grid_type> & u, std::vector<double> const & b) const
    {
        for(std::size_t y = range.first, i = 0; y < range.second; ++y, ++i)
        {
            u[cur](0, y) = b[i];
        }
    }
};

struct update_fun
{
    typedef void result_type;

    range_type x_range;
    range_type y_range;
    std::size_t src;
    std::size_t dst;
    std::size_t cache_block;

    update_fun() {}

    update_fun(range_type x, range_type y, std::size_t old, std::size_t n, std::size_t c)
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
    void serialize(Archive & ar, const unsigned int )
    {
        ar & x_range;
        ar & y_range;
        ar & src;
        ar & dst;
        ar & cache_block;
    }

    private:
        BOOST_COPYABLE_AND_MOVABLE(update_fun)
};

void gs(
    std::size_t n_x
  , std::size_t n_y
  , double //hx_
  , double //hy_
  , double //k_
  , double //relaxation_
  , unsigned max_iterations
  , unsigned //iteration_block
  , unsigned block_size
  , std::size_t cache_block
  , std::string const & //output
)
{
    /*
    hpx::components::component_type type
        = hpx::components::get_component_type<hpx::components::server::remote_object>();
    */

    std::vector<id_type> prefixes = hpx::find_all_localities();

    cout << "Number of localities: " << prefixes.size() << "\n" << flush;

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
        std::vector<hpx::lcos::future<object<data_type> > >
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
        std::size_t x = 0;
        std::size_t y = 0;

        BOOST_FOREACH(hpx::lcos::future<object<data_type> > const & o, objects)
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

    std::size_t n_x_local_block = 0;
    for(std::size_t x = 1; x < n_x_local; x += block_size, ++n_x_local_block);
    std::size_t n_y_local_block = 0;
    for(std::size_t y = 1; y < n_y_local; y += block_size, ++n_y_local_block);

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
    grid<boost::array<promise, 4> > boundary_deps(dims[0], dims[1]);

    std::size_t src = 0;
    std::size_t dst = 1;

    high_resolution_timer t;
    t.restart();
    for(std::size_t iter = 0; iter < max_iterations; ++iter)
    {
        for(std::size_t y_block = 0; y_block < dims[1]; ++y_block)
        {
            for(std::size_t x_block = 0; x_block < dims[0]; ++x_block)
            {
                if(iter > 0)
                {
                    range_type y_range(0, n_y_local);
                    range_type x_range(0, n_x_local);
                    // left neighbor:
                    if(x_block > 0)
                    {
                        std::vector<promise > trigger;
                        trigger.reserve(n_y_local);
                        for(std::size_t y = 1, yy = 0; y < n_y_local; y += block_size, ++yy)
                        {
                            trigger.push_back(
                                deps[src](x_block-1, y_block)(n_x_local_block - 1, yy)
                            );
                        }
                        BOOST_ASSERT(trigger.size() > 0);
                        boundary_deps(x_block, y_block)[0] =
                        object_grid(x_block, y_block).apply3(
                            update_left_boundary_fun(
                                y_range
                              , src
                            )
                          , object_grid(x_block-1, y_block).apply(
                                get_col_fun(
                                    y_range
                                  , n_x_local-1
                                  , src
                                )
                            )
                          , dataflow_trigger(object_grid(x_block, y_block).gid_, trigger)
                        );
                    }
                    // right neighbor:
                    if(x_block + 1 < dims[0])
                    {
                        std::vector<promise > trigger;
                        trigger.reserve(n_y_local);
                        for(std::size_t y = 1, yy = 0; y < n_y_local; y += block_size, ++yy)
                        {
                            trigger.push_back(
                                deps[src](x_block+1, y_block)(0, yy)
                            );
                        }
                        boundary_deps(x_block, y_block)[1] =
                        object_grid(x_block, y_block).apply3(
                            update_right_boundary_fun(
                                y_range
                              , src
                            )
                          , object_grid(x_block + 1, y_block).apply(
                                get_col_fun(
                                    y_range
                                  , 1
                                  , src
                                )
                            )
                          , dataflow_trigger(object_grid(x_block, y_block).gid_, trigger)
                        );
                    }
                    // top neighbor:
                    if(y_block > 0)
                    {
                        std::vector<promise > trigger;
                        trigger.reserve(n_x_local);
                        for(std::size_t x = 1, xx = 0; x < n_x_local; x += block_size, ++xx)
                        {
                            trigger.push_back(
                                deps[src](x_block, y_block-1)(xx, n_y_local_block - 1)
                            );
                        }
                        BOOST_ASSERT(trigger.size() > 0);
                        boundary_deps(x_block, y_block)[2] =
                        object_grid(x_block, y_block).apply3(
                            update_top_boundary_fun(
                                x_range
                              , src
                            )
                          , object_grid(x_block, y_block - 1).apply(
                                get_row_fun(
                                    x_range
                                  , n_y_local-1
                                  , src
                                )
                            )
                          , dataflow_trigger(object_grid(x_block, y_block).gid_, trigger)
                        );
                        //boundary_deps(x_block, y_block)[2].get_future().get();
                    }
                    // bottom neighbor:
                    if(y_block + 1 < dims[1])
                    {
                        std::vector<promise > trigger;
                        trigger.reserve(n_x_local);
                        for(std::size_t x = 1, xx = 0; x < n_x_local; x += block_size, ++xx)
                        {
                            trigger.push_back(
                                deps[src](x_block, y_block+1)(xx, 0)
                            );
                        }
                        BOOST_ASSERT(trigger.size() > 0);
                        boundary_deps(x_block, y_block)[3] =
                        object_grid(x_block, y_block).apply3(
                            update_bottom_boundary_fun(
                                x_range
                              , src
                            )
                          , object_grid(x_block, y_block + 1).apply(
                                get_row_fun(
                                    x_range
                                  , 1
                                  , src
                                )
                            )
                          , dataflow_trigger(object_grid(x_block, y_block).gid_, trigger)
                        );
                        //boundary_deps(x_block, y_block)[3].get_future().get();
                    }
                }
                promise_grid_type & cur_deps = deps[dst](x_block, y_block);
                for(std::size_t y = 1, yy = 0; y < n_y_local; y += block_size, ++yy)
                {
                    std::size_t y_end = (std::min)(y + block_size, n_y_local);
                    for(std::size_t x = 1, xx = 0; x < n_x_local; x += block_size, ++xx)
                    {
                        std::size_t x_end = (std::min)(x + block_size, n_x_local);
                        if(iter > 0)
                        {
                            range_type x_range(x, x_end);
                            range_type y_range(y, y_end);
                            std::vector<promise > trigger;
                            trigger.reserve(9);
                            promise_grid_type & old_deps = deps[src](x_block, y_block);
                            trigger.push_back(old_deps(xx,yy));
                            if(xx + 1 < n_x_local_block)
                                trigger.push_back(old_deps(xx+1, yy));
                            if(xx > 0)
                                trigger.push_back(old_deps(xx-1, yy));
                            if(yy + 1 < n_y_local_block)
                                trigger.push_back(old_deps(xx, yy+1));
                            if(yy > 0)
                                trigger.push_back(old_deps(xx, yy-1));

                            if(xx == 0 && x_block > 0)
                            {
                                trigger.push_back(
                                    // left_boundary_dep
                                    boundary_deps(x_block, y_block)[0]
                                );
                            }

                            if(xx + 1 == n_x_local && x_block + 1 < dims[0])
                            {
                                trigger.push_back(
                                    // right_boundary_dep
                                    boundary_deps(x_block, y_block)[1]
                                );
                            }

                            if(yy == 0 && y_block > 0)
                            {
                                trigger.push_back(
                                    // top_boundary_dep
                                    boundary_deps(x_block, y_block)[2]
                                );
                            }

                            if(yy + 1 == n_y_local && y_block + 1 < dims[1])
                            {
                                trigger.push_back(
                                    // bottom_boundary_dep
                                    boundary_deps(x_block, y_block)[3]
                                );
                            }

                            cur_deps(xx, yy)
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
                            cur_deps(xx, yy)
                                = object_grid(x_block, y_block).apply(
                                    update_fun(
                                        range_type(x, x_end)
                                      , range_type(y, y_end)
                                      , src
                                      , dst
                                      , cache_block
                                    )
                                );
                            //cur_deps(xx, yy).get_future().get();
                        }
                    }
                }
            }
        }
        std::swap(dst, src);
    }
    for(std::size_t y_block = 0; y_block < dims[1]; ++y_block)
    {
        for(std::size_t x_block = 0; x_block < dims[0]; ++x_block)
        {
            for(std::size_t y = 1, yy = 0; y < n_y_local; y += block_size, ++yy)
            {
                for(std::size_t x = 1, xx = 0; x < n_x_local; x += block_size, ++xx)
                {
                    deps[src](x_block, y_block)(xx, yy).get_future().get();
                }
            }
        }
    }

    double time_elapsed = t.elapsed();
    cout << n_x << "x" << n_y << " "
         << ((double((n_x-2)*(n_y-2) * max_iterations)/1e6)/time_elapsed) << " MLUP/S\n" << flush;
}
