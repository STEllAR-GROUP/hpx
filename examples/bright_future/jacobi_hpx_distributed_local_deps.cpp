
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

typedef std::vector<grid_type> data_type;
typedef dataflow_object<data_type> object_type;
typedef dataflow_base<void> promise;
typedef grid<promise> promise_grid_type;

/////////////////////////////////////////////////////////////////////////////// 
struct jacobi_driver
{
  private:
    /////////////////////////////////////////////////////////////////////////// 
    // Application parameters.

    std::size_t block_size;  // Block size
    std::size_t cache_block;

    /////////////////////////////////////////////////////////////////////////// 
    // Dimensions, coordinates, etc.

    std::size_t x_block;   // Our locality's x coordinates in object_grid.
    std::size_t y_block;   // Our locality's x coordinates in object_grid.

    std::size_t n_x_local; // x dimension of our locality's submatrix.
    std::size_t n_y_local; // y dimension of our locality's submatrix.
    std::size_t n_x_local_block;
    std::size_t n_y_local_block;

    std::vector<std::size_t> dims; // Dimensions of object_grid.

    /////////////////////////////////////////////////////////////////////////// 
    // Local data.

    // Our local dependency data. This technically could be a boost::array<>.
    std::vector<promise_grid_type> deps; 

    // The matrix of submatrices.
    grid<object_type> object_grid; 
            
    std::size_t src;
    std::size_t dst;

    /////////////////////////////////////////////////////////////////////////// 

    BOOST_COPYABLE_AND_MOVABLE(jacobi_driver)

  public: 
    jacobi_driver(std::size_t block_size_,
        std::size_t cache_block_,
        std::size_t x_block_,
        std::size_t y_block_,
        std::size_t n_x_local_,
        std::size_t n_y_local_,
        std::vector<std::size_t> const& dims_,
        grid<object_type> const& object_grid_) 
      : block_size(block_size_),
        cache_block(cache_block_),
        x_block(x_block_),
        y_block(y_block_),
        n_x_local(n_x_local_),
        n_y_local(n_y_local_),
        n_x_local_block((n_x_local - 2)/block_size + 1),
        n_y_local_block((n_y_local - 2)/block_size + 1),
        dims(dims_),
        deps(),
        object_grid(object_grid_),
        src(0),
        dst(1)
    {}

    template <typename Archive>
    void serialize(Archive & ar, const unsigned int)
    {
        ar & block_size;
        ar & cache_block;
        ar & x_block;
        ar & y_block;
        ar & n_x_local;
        ar & n_y_local;
        ar & n_x_local_block;
        ar & n_y_local_block;
        ar & dims; 
        ar & object_grid;
        ar & deps;
        ar & src;
        ar & dst;
    }

    void iterate(std::size_t iter)
    { // {{{
        if(0 == iter)
        {
            BOOST_ASSERT(deps.empty());
            BOOST_ASSERT(src == 0);
            BOOST_ASSERT(dst == 1);

            deps.emplace_back(promise_grid_type(n_x_local_block, n_y_local_block));
            deps.emplace_back(promise_grid_type(n_x_local_block, n_y_local_block));
        }
 
        // Get a reference to our dependency grid for this iteration.
        promise_grid_type & cur_deps = deps[dst];
    
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
                    std::vector<promise> trigger;
                    trigger.reserve(9);
    
                    // Get a reference to the old dependency grid for this
                    // iteration. 
                    promise_grid_type & old_deps = deps[src];
    
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
                            object_grid(x_block, y_block).apply2(
                                update_left_boundary_fun(
                                    y_range
                                  , src
                                )
                              , object_grid(x_block - 1, y_block).apply(
                                  get_col_fun(y_range, n_x_local-1, src)
                                )
                            )
                        );
                    }
    
                    if(xx + 1 == n_x_local && x_block + 1 < dims[0])
                    {
                        trigger.push_back(
                            object_grid(x_block, y_block).apply2(
                                update_right_boundary_fun(
                                    y_range
                                  , src
                                )
                              , object_grid(x_block + 1, y_block).apply(
                                    get_col_fun(y_range, 1, src)
                                )
                            )
                        );
                    }
    
                    if(yy == 0 && y_block > 0)
                    {
                        trigger.push_back(
                            object_grid(x_block, y_block).apply2(
                                update_top_boundary_fun(
                                    x_range
                                  , src
                                )
                              , object_grid(x_block, y_block - 1).apply(
                                  get_row_fun(x_range, n_y_local-1, src)
                                )
                            )
                        );
                    }
    
                    if(yy + 1 == n_y_local && y_block + 1 < dims[1])
                    {
                        trigger.push_back(
                            object_grid(x_block, y_block).apply2(
                                update_bottom_boundary_fun(
                                    x_range
                                  , src
                                )
                              , object_grid(x_block, y_block + 1).apply(
                                  get_row_fun(x_range, 1, src)
                                )
                            )
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
                }
            }
        }

        std::swap(dst, src);
    } // }}}

    void wait()
    {
        BOOST_ASSERT(deps.size() == 2);

        for(std::size_t y = 1, yy = 0; y < n_y_local; y += block_size, ++yy)
        {
            for(std::size_t x = 1, xx = 0; x < n_x_local; x += block_size, ++xx)
            {
                deps[dst](xx, yy).get_future().get();
                deps[src](xx, yy).get_future().get();
            }
        }
    }
};

struct driver_iterate_fun
{
    typedef void result_type;

    std::size_t iter;

    driver_iterate_fun() {}

    driver_iterate_fun(std::size_t iter_)
        : iter(iter_)
    {}

    void operator()(jacobi_driver& d) const
    {
        d.iterate(iter);
    }

    template <typename Archive>
    void serialize(Archive & ar, const unsigned int)
    {
        ar & iter;
    }

    private:
        BOOST_COPYABLE_AND_MOVABLE(driver_iterate_fun)
};

struct driver_wait_fun
{
    typedef void result_type;

    void operator()(jacobi_driver& d) const
    {
        d.wait();
    }

    template <typename Archive>
    void serialize(Archive & ar, const unsigned int) {}

    private:
        BOOST_COPYABLE_AND_MOVABLE(driver_wait_fun)
};

typedef object<jacobi_driver> driver_object; 

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

    ///////////////////////////////////////////////////////////////////////////
    // Create object_grid.

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
            using hpx::naming::detail::strip_credit_from_gid;

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

    ///////////////////////////////////////////////////////////////////////////
    // Create driver_grid.

    grid<driver_object> driver_grid(dims[0], dims[1]);
    {
        std::vector<hpx::lcos::future<driver_object> > drivers;
        drivers.reserve(dims[0] * dims[1]);

        for(std::size_t y_block = 0; y_block < dims[1]; ++y_block)
        {
            for(std::size_t x_block = 0; x_block < dims[0]; ++x_block)
            {
                cout << "Creating driver " << x_block << "x" << y_block << "\n"
                     << flush;
                using hpx::components::new_;
                using hpx::naming::get_locality_from_id;
                drivers.push_back(
                        new_<jacobi_driver>(
                            get_locality_from_id(object_grid(x_block, y_block).gid_),
                            block_size,
                            cache_block,
                            x_block,
                            y_block,
                            n_x_local,
                            n_y_local,
                            dims,
                            object_grid
                        )
                    );
            }
        }

        std::size_t x_block = 0;
        std::size_t y_block = 0;

        BOOST_FOREACH(hpx::lcos::future<driver_object> const& d, drivers)
        {
            cout << "Created driver " << x_block << "x" << y_block << "\n"
                 << flush;

            driver_grid(x_block, y_block) = d.get();

            if(++x_block > dims[0] - 1)
            {
                x_block = 0;
                ++y_block;
            }
        }
    }

    high_resolution_timer t;
    t.restart();

    ///////////////////////////////////////////////////////////////////////////
    // Start creating the dependency graph, which implicitly starts the
    // computation.
    // 
    // We distribute the creation of the dependency graph across all the
    // localities (creating the dependencies for each local submatrix on the
    // locality where it resides).
    //
    // FIXME: ATM, the distributed construction of the dependency graph is done
    // sequentially, as this means we don't have to worry about concurrency in
    // the creation of the dependency graph itself. I believe we could do
    // this in parallel, though. 

    // This iteration is done in the same sequential order that was used inside
    // of the big for loop which used to create the dependencies and ran 
    // entirely on the head node.
    for(std::size_t iter = 0; iter < max_iterations; ++iter)
    {
        for(std::size_t y_block = 0; y_block < dims[1]; ++y_block)
        {
            for(std::size_t x_block = 0; x_block < dims[0]; ++x_block)
            {
                driver_grid(x_block, y_block).apply(
                    driver_iterate_fun(iter)
                ).get();
            }
        }        
    }

    ///////////////////////////////////////////////////////////////////////////
    // Wait for everything to finish. 
    for(std::size_t y_block = 0; y_block < dims[1]; ++y_block)
    {
        for(std::size_t x_block = 0; x_block < dims[0]; ++x_block)
        {
            driver_grid(x_block, y_block).apply(
                driver_wait_fun()
            ).get();
        }
    }        

    double time_elapsed = t.elapsed();
    cout << n_x << "x" << n_y << " "
         << ((double((n_x-2)*(n_y-2) * max_iterations)/1e6)/time_elapsed) << " MLUP/S\n" << flush;
}

