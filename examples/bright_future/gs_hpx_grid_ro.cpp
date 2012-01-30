//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>

#include "grid.hpp"
#include <cmath>

#include <hpx/hpx.hpp>
#include <hpx/components/remote_object/new.hpp>
#include <hpx/lcos/promise.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/util/high_resolution_timer.hpp>

using hpx::cout;
using hpx::flush;
using hpx::find_all_localities;
using hpx::find_here;
using hpx::lcos::promise;
using hpx::lcos::wait;
using hpx::naming::id_type;

using bright_future::grid;

typedef bright_future::grid<double> grid_type;
typedef grid_type::size_type size_type;

using hpx::components::new_;
using hpx::components::object;

using hpx::util::high_resolution_timer;

template <typename T>
struct lse_config
{
    typedef typename bright_future::grid<T>::size_type size_type;
    
    lse_config()
        : n_x(0.0)
        , n_y(0.0)
        , hx(0.0)
        , hy(0.0)
        , hx_sq(0.0)
        , hy_sq(0.0)
        , k(0.0)
        , div(1.0)
        , relaxation(0.0)
    {}

    lse_config(size_type n_x, size_type n_y, T hx_, T hy_, T k, T relaxation)
        : n_x(n_x)
        , n_y(n_y)
        , hx(hx_)
        , hy(hy_)
        , hx_sq(hx_*hx_)
        , hy_sq(hy_*hy_)
        , k(k)
        , div(2.0/(hx_*hx_) + 2.0/(hy_*hy_) + k*k)
        , relaxation(relaxation)
    {}
    
    size_type n_x;
    size_type n_y;
    T hx;
    T hy;
    T hx_sq;
    T hy_sq;
    T k;
    T div;
    T relaxation;

    template <typename Archive>
    void serialize(Archive & ar, unsigned)
    {
    }
};

struct init_u
{
    typedef void result_type;

    lse_config<double> c;
    size_type x;
    size_type y;
    size_type x_global;
    size_type y_global;
    init_u() {}
    init_u(
        lse_config<double> const & c
      , size_type x
      , size_type y
      , size_type x_global
      , size_type y_global
    )
        : c(c)
        , x(x)
        , y(y)
        , x_global(x_global)
        , y_global(y_global)
    {}

    result_type operator()(grid_type & u) const
    {
        u(x, y)
          = y_global == (c.n_y - 1)
          ? sin((x_global * c.hx) * 6.283) * sinh(6.283)
          : 0.0;
    }

    template <typename Archive>
    void serialize(Archive & ar, unsigned)
    {
    }
};

struct init_rhs
{
    typedef void result_type;

    lse_config<double> c;
    size_type x;
    size_type y;
    size_type x_global;
    size_type y_global;
    init_rhs() {}
    init_rhs(
        lse_config<double> const & c
      , size_type x
      , size_type y
      , size_type x_global
      , size_type y_global
    )
        : c(c)
        , x(x)
        , y(y)
        , x_global(x_global)
        , y_global(y_global)
    {}

    result_type operator()(grid_type & rhs) const
    {
        rhs(x, y) = 
            39.478 * sin((x_global * c.hx) * 6.283) * sinh((y_global * c.hy) * 6.283);
    }

    template <typename Archive>
    void serialize(Archive & ar, unsigned)
    {
    }
};

struct get
{
    typedef double result_type;
    
    get() {}
    get(size_type x, size_type y) : x(x), y(y) {}

    size_type x;
    size_type y;

    result_type operator()(grid_type const & g) const
    {
        return g(x, y);
    }

    template <typename Archive>
    void serialize(Archive & ar, unsigned)
    {
    }
};

struct update
{
    typedef void result_type;

    lse_config<double> c;
    size_type x;
    size_type y;
    object<grid_type> rhs;
    std::vector<promise<void> > deps;
    
    update() {}
    update(
        lse_config<double> const & c
      , size_type x
      , size_type y
      , object<grid_type> const & rhs
      , std::vector<promise<void> > const & deps
    )
        : c(c)
        , x(x)
        , y(y)
        , rhs(rhs)
        , deps(deps)
    {}

    void operator()(grid_type & u) const
    {
        promise<double> rhs_promise = rhs <= get(x, y);
        wait(deps);
        double rhs_value = rhs_promise.get();
        u(x, y) =
            u(x, y)
            + (
                (
                    (
                        (u(x - 1, y) + u(x + 1, y)) / c.hx_sq
                      + (u(x, y - 1) + u(x, y + 1)) / c.hy_sq
                      + rhs_value
                    )
                    / c.div
                )
                - u(x, y)
            )
            * c.relaxation
            ;
    }

    template <typename Archive>
    void serialize(Archive & ar, unsigned)
    {
    }
};

struct output_fun
{
    boost::shared_ptr<std::ofstream> file;

    lse_config<double> c;
    size_type x;
    size_type y;
    size_type x_global;
    size_type y_global;

    typedef void result_type;

    output_fun() {}
    output_fun(
        boost::shared_ptr<std::ofstream> const & file
      , lse_config<double> const & c
      , size_type x
      , size_type y
      , size_type x_global
      , size_type y_global
    )
        : file(file)
        , c(c)
        , x(x)
        , y(y)
        , x_global(x_global)
        , y_global(y_global)
    {}

    void operator()(grid_type const & u) const
    {
        (*file) << x_global * c.hx << " " << y_global * c.hy << " " << u(x, y) << "\n";
        if(y_global == c.n_y - 1)
        {
            (*file) << "\n";
        }
    }

    template <typename Archive>
    void serialize(Archive & ar, unsigned)
    {
    }
};

struct get_row
{
    typedef std::vector<double> result_type;

    size_type row;
    get_row() {}
    get_row(size_type row) : row(row) {}

    result_type operator()(grid<double> const & g) const
    {
        return result_type(g.begin() + row * g.y(), g.begin() + (row + 1) * g.y());
    }

    template <typename Archive>
    void serialize(Archive & ar, unsigned)
    {
    }
};

struct get_column
{
    typedef std::vector<double> result_type;

    size_type column;
    get_column() {}
    get_column(size_type column) : column(column) {}

    result_type operator()(grid<double> const & g) const
    {
        result_type result(g.y());
        for(size_type x = 0; x < g.x(); ++x)
        {
            result[x] = g(x, column);
        }
        return result;
    }

    template <typename Archive>
    void serialize(Archive & ar, unsigned)
    {
    }
};

struct update_top_boundary
{
    typedef void result_type;
    
    object<grid_type> other;

    update_top_boundary() {}
    update_top_boundary(object<grid_type> const & other) : other(other) {}

    result_type operator()(grid_type & g) const
    {
        std::vector<double> b((other <= get_row(g.y()-2)).get());

        std::copy(b.begin(), b.end(), g.begin() + g.y());
    }

    template <typename Archive>
    void serialize(Archive & ar, unsigned)
    {
    }
};

struct update_bottom_boundary
{
    typedef void result_type;
    
    object<grid_type> other;

    update_bottom_boundary() {}
    update_bottom_boundary(object<grid_type> const & other) : other(other) {}

    result_type operator()(grid_type & g) const
    {
        std::vector<double> b((other <= get_row(1)).get());

        std::copy(b.begin(), b.end(), g.begin() + (g.x()-2) * g.y());
    }

    template <typename Archive>
    void serialize(Archive & ar, unsigned)
    {
    }
};

struct update_left_boundary
{
    typedef void result_type;
    
    object<grid_type> other;

    update_left_boundary() {}
    update_left_boundary(object<grid_type> const & other) : other(other) {}

    result_type operator()(grid_type & g) const
    {
        std::vector<double> b((other <= get_column(g.y()-2)).get());

        for(size_type x = 0; x < g.x(); ++x)
        {
            g(x, 1) = b[x];
        }
    }

    template <typename Archive>
    void serialize(Archive & ar, unsigned)
    {
    }
};

struct update_right_boundary
{
    typedef void result_type;
    
    object<grid_type> other;

    update_right_boundary() {}
    update_right_boundary(object<grid_type> const & other) : other(other) {}

    result_type operator()(grid_type & g) const
    {
        std::vector<double> b((other <= get_column(1)).get());

        for(size_type x = 0; x < g.x(); ++x)
        {
            g(x, g.y()-2) = b[x];
        }
    }

    template <typename Archive>
    void serialize(Archive & ar, unsigned)
    {
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
  , unsigned
  , std::size_t block_size
  , std::string const & output
)
{
    typedef object<grid_type>    object_type;
    typedef promise<object_type> object_promise;
    typedef promise<void>        void_promise;

    typedef bright_future::grid<void_promise> promise_grid_type;
    typedef std::vector<promise_grid_type> iteration_dependencies_type;

    lse_config<double>
        config(
            n_x
          , n_y
          , hx
          , hy
          , k
          , relaxation
        );

    size_type n_x_block = n_x/block_size+1;
    size_type n_y_block = n_y/block_size+1;
    
    grid<object_type> u(n_x_block, n_y_block);
    grid<object_type> rhs(n_x_block, n_y_block);
    grid<object_promise> u_promise(n_x_block, n_y_block);
    grid<object_promise> rhs_promise(n_x_block, n_y_block);
    {
        {
            std::vector<id_type> prefixes = find_all_localities();
            size_type count = (n_x_block*n_y_block);
            std::vector<id_type>::size_type
                grids_per_loc = count / prefixes.size();
            size_type x = 0;
            size_type y = 0;
            std::size_t created_count = 0;
            std::size_t excess = count - grids_per_loc*prefixes.size();
            BOOST_FOREACH(id_type const & prefix, prefixes)
            {
                std::size_t numcreate = grids_per_loc;
                if (excess != 0) {
                    --excess;
                    ++numcreate;
                }

                if (created_count + numcreate > count)
                    numcreate = count - created_count;

                if (numcreate == 0)
                    break;

                for (std::size_t i = 0; i < numcreate; ++i) {
                    BOOST_ASSERT(x < n_x_block);
                    BOOST_ASSERT(y < n_y_block);

                    /////////////////////////////////////////////////////////////////
                    // actual allocation
                    rhs_promise(x, y)
                        = new_<grid_type>(prefix, block_size + 2, block_size + 2);

                    u_promise(x, y)
                        = new_<grid_type>(prefix, block_size + 2, block_size + 2);
                    /////////////////////////////////////////////////////////////////

                    if(++x > n_x_block - 1)
                    {
                        x = 0;
                        ++y;
                    }
                }

                created_count += numcreate;
                if (created_count >= count)
                    break;
            }
        }
    }

    iteration_dependencies_type
        iteration_dependencies(
            2
          , promise_grid_type(n_x/*_block*/, n_y/*_block*/)
        );
     
    {
        promise_grid_type rhs_promises(n_x, n_y);
        // initalize our right hand side and the initial grid
        for(size_type x_block = 0; x_block < n_x_block; ++x_block)
        {
            for(size_type y_block = 0; y_block < n_y_block; ++y_block)
            {
                size_type x_end
                    = x_block == n_x_block - 1
                    ? n_x - x_block*block_size + 2
                    : block_size + 2;
                size_type y_end
                    = y_block == n_y_block - 1
                    ? n_y - y_block * block_size + 2
                    : block_size + 2;

                rhs(x_block, y_block) = rhs_promise(x_block, y_block).get();
                u(x_block, y_block)   = u_promise(x_block, y_block).get();

                for(size_type y = 1; y < y_end - 1; ++y)
                {
                    for(size_type x = 1; x < x_end - 1; ++x)
                    {
                        size_type x_global = (x-1) + block_size * x_block;
                        size_type y_global = (y-1) + block_size * y_block;

                        // init rhs
                        rhs_promises(x_global, y_global) =
                            rhs(x_block, y_block)
                                <= init_rhs(
                                    config
                                  , x
                                  , y
                                  , x_global
                                  , y_global
                                );
                        // init u
                        iteration_dependencies[0](x_global, y_global) =
                            u(x_block, y_block)
                                <= init_u(
                                    config
                                  , x
                                  , y
                                  , x_global
                                  , y_global
                                );
                            
                            (u(x_block, y_block)
                                <= update_top_boundary(u(x_block, y_block-1))
                            ).get();
                            
                            (u(x_block, y_block)
                                <= update_left_boundary(u(x_block-1, y_block))
                            ).get();
                            
                            (u(x_block, y_block)
                                <= update_bottom_boundary(u(x_block, y_block+1))
                            ).get();
                            
                            (u(x_block, y_block)
                                <= update_right_boundary(u(x_block+1, y_block))
                            ).get();
                    }
                }
            }
        }
        
        // wait for the rhs initialization to finish.
        BOOST_FOREACH(void_promise const & promise, rhs_promises)
        {
            promise.get();
        }
    }
    
    high_resolution_timer t;
    for(unsigned iter = 0; iter < max_iterations; ++iter)//iter += iteration_block)
    {
        for(size_type x_block = 0; x_block < n_x_block; ++x_block)
        {
            for(size_type y_block = 0; y_block < n_y_block; ++y_block)
            {
                size_type x_end
                    = x_block == n_x_block - 1
                    ? n_x - x_block*block_size + 2
                    : block_size + 2;
                size_type y_end
                    = y_block == n_y_block - 1
                    ? n_y - y_block * block_size + 2
                    : block_size + 2;

                for(size_type y = 1; y < y_end - 1; ++y)
                {
                    for(size_type x = 1; x < x_end - 1; ++x)
                    {
                    }
                }
            }
        }
    }
    
    /*
    for(unsigned iter = 0; iter < max_iterations; ++iter)//iter += iteration_block)
    {
        promise_grid_type & prev = iteration_dependencies[iter % 2];
        promise_grid_type & current = iteration_dependencies[(iter + 1) % 2];
        for(size_type x = 1; x < n_x-1; ++x)
        {
            for(size_type y = 1; y < n_y-1; ++y)
            {
                std::vector<promise_type> deps;
                // we need to be sure to wait for the previous iteration.
                deps.push_back(prev(x,y));

                if(x + 1 < n_x - 1) // are we on the boundary?
                    deps.push_back(prev(x+1,y));

                if(y + 1 < n_y - 1) // are we on the boundary?
                    deps.push_back(prev(x,y+1));

                if(x > 0) // are we on the boundary?
                    deps.push_back(current(x-1,y));

                if(y > 0) // are we on the boundary?
                    deps.push_back(current(x,y-1));

                current(x, y) =
                    u <= update(config, x, y, rhs, deps);
            }
        }
    }
    */
        
    // add barrier to have some kind of time comparision to conventional systems
    BOOST_FOREACH(void_promise const & promise, iteration_dependencies[(max_iterations%2)])
    {
        promise.get();
    }

    double time_elapsed = t.elapsed();
    cout << (n_x*n_y) << " " << time_elapsed << "\n" << flush;
    
    if(!output.empty())
    {
        boost::shared_ptr<std::ofstream> file(new std::ofstream(output.c_str()));
        for(size_type x_block = 0; x_block < n_x_block; ++x_block)
        {
            for(size_type y_block = 0; y_block < n_y_block; ++y_block)
            {
                size_type x_end
                    = x_block == n_x_block - 1
                    ? n_x - x_block*block_size + 2
                    : block_size + 2;
                size_type y_end
                    = y_block == n_y_block - 1
                    ? n_y - y_block * block_size + 2
                    : block_size + 2;

                for(size_type x = 1; x < x_end - 1; ++x)
                {
                    for(size_type y = 1; y < y_end - 1; ++y)
                    {
                        size_type x_global = (x-1) + block_size * x_block;
                        size_type y_global = (y-1) + block_size * y_block;
                        wait(u(x_block, y_block) <= output_fun(file, config, x, y, x_global, y_global));
                    }
                }
            }
        }
    }
}
