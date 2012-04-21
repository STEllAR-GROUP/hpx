//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/runtime/components/plain_component_factory.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/include/iostreams.hpp>

#include <hpx/util/hardware/timestamp.hpp>

#include "remote_lse.hpp"
#include <cmath>

HPX_REGISTER_COMPONENT_MODULE();

namespace bright_future { namespace server {

    template <typename T>
    remote_lse<T>::remote_lse()
        : u(0,0)
        , rhs(0,0)
        , config(
            0u
          , 0u
          , 0.0
          , 0.0
          , 0.0
          , 1.0
        )
    {}

    template <typename T>
    void remote_lse<T>::init(
        typename remote_lse<T>::size_type size_x
      , typename remote_lse<T>::size_type size_y
      , typename remote_lse<T>::size_type nx
      , typename remote_lse<T>::size_type ny
      , double hx
      , double hy
    )
    {
        u = grid_type(size_x, size_y);
        rhs = grid_type(size_x, size_y);

        config = lse_config<T>(nx, ny, hx, hy, config.k, config.relaxation);
    }

    template <typename T>
    void remote_lse<T>::init_rhs(
        typename remote_lse<T>::init_func_type f
      , typename remote_lse<T>::size_type x
      , typename remote_lse<T>::size_type y
    )
    {
        rhs(x, y) = f(x, y, config);
    }

    template <typename T>
    void remote_lse<T>::init_rhs_blocked(
        typename remote_lse<T>::init_func_type f
      , typename remote_lse<T>::range_type x_range
      , typename remote_lse<T>::range_type y_range
    )
    {
        for(size_type y = y_range.first; y < y_range.second; ++y)
        {
            for(size_type x = x_range.first; x < x_range.second; ++x)
            {
                rhs(x, y) = f(x, y, config);
            }
        }
    }

    template <typename T>
    void remote_lse<T>::init_u(
        typename remote_lse<T>::init_func_type f
      , typename remote_lse<T>::size_type x
      , typename remote_lse<T>::size_type y
    )
    {
        u(x, y) = f(x, y, config);
    }

    template <typename T>
    void remote_lse<T>::init_u_blocked(
        typename remote_lse<T>::init_func_type f
      , typename remote_lse<T>::range_type x_range
      , typename remote_lse<T>::range_type y_range
    )
    {
        for(size_type y = y_range.first; y < y_range.second; ++y)
        {
            for(size_type x = x_range.first; x < x_range.second; ++x)
            {
                u(x, y) = f(x, y, config);
            }
        }
    }

    template <typename T>
    void remote_lse<T>::apply(
        typename remote_lse<T>::apply_func_type f
      , typename remote_lse<T>::size_type x
      , typename remote_lse<T>::size_type y
      , std::vector<hpx::lcos::future<void> > dependencies
    )
    {
        BOOST_FOREACH(hpx::lcos::future<void> const & promise, dependencies)
        {
            promise.get();
        }
        u(x, y) = f(u, rhs, x, y, config);
    }

    template <typename T>
    void remote_lse<T>::apply_region(
        typename remote_lse<T>::apply_func_type f
      , typename remote_lse<T>::range_type x_range
      , typename remote_lse<T>::range_type y_range
      , std::vector<hpx::lcos::future<void> *> dependencies
    )
    {
        BOOST_FOREACH(hpx::lcos::future<void> * promise, dependencies)
        {
            promise->get();
        }

        for(size_type y = y_range.first; y < y_range.second; ++y)
        {
            for(size_type x = x_range.first; x < x_range.second; ++x)
            {
                u(x, y) = f(u, rhs, x, y, config);
            }
        }
    }

    template <typename T>
    void remote_lse<T>::apply_region_df(
        typename remote_lse<T>::apply_func_type f
      , typename remote_lse<T>::range_type x_range
      , typename remote_lse<T>::range_type y_range
    )
    {
        //hpx::util::high_resolution_timer t;

        for(size_type y = y_range.first; y < y_range.second; ++y)
        {
            for(size_type x = x_range.first; x < x_range.second; ++x)
            {
                u(x, y) = f(u, rhs, x, y, config);
            }
        }
       
        /*
        double time = t.elapsed();
        {
            hpx::lcos::local::spinlock::scoped_lock l(mtx);
            timestamps.push_back(time);
        }
        */
    }
    
    template <typename T>
    std::vector<T> remote_lse<T>::get_row(size_type row, range_type range)
    {
        std::vector<T> result;
        
        result.reserve((range.second-range.first));

        for(size_type x = range.first; x < range.second; ++x)
        {
            result.push_back(u(x, row));
        }

        return result;
        /*
        return std::vector<T>(u.begin() + row * u.y(), u.begin() + (row + 1) * u.y());
        */
    }
    
    template <typename T>
    std::vector<T> remote_lse<T>::get_col(size_type col, range_type range)
    {
        std::vector<T> result;
        result.reserve(range.second-range.first);
        for(size_type y = range.first; y < range.second; ++y)
        {
            result.push_back(u(col, y));
        }
        return result;
    }

    template <typename T>
    void remote_lse<T>::update_top_boundary(std::vector<T> const & b, range_type range)
    {
        //std::copy(b.begin(), b.end(), u.begin() + u.y());
        for(size_type x = range.first, i = 0; x < range.second; ++x, ++i)
        {
            u(x, 0) = b.at(i);
        }
    }

    template <typename T>
    void remote_lse<T>::update_bottom_boundary(std::vector<T> const & b, range_type range)
    {
        //std::copy(b.begin(), b.end(), u.begin() + u.x() * u.y());
        for(size_type x = range.first, i = 0; x < range.second; ++x, ++i)
        {
            u(x, u.y()-1) = b.at(i);
        }
    }

    template <typename T>
    void remote_lse<T>::update_right_boundary(std::vector<T> const & b, range_type range)
    {
        for(size_type y = range.first, i = 0; y < range.second; ++y, ++i)
        {
            u(u.x()-1, y) = b.at(i);
        }
    }

    template <typename T>
    void remote_lse<T>::update_left_boundary(std::vector<T> const & b, range_type range)
    {
        for(size_type y = range.first, i = 0; y < range.second; ++y, ++i)
        {
            u(0, y) = b.at(i);
        }
    }

}}

template HPX_EXPORT class bright_future::server::remote_lse<double>;

HPX_REGISTER_ACTION_EX(
    bright_future::server::remote_lse<double>::init_action
  , remote_lse_init_action
);
HPX_REGISTER_ACTION_EX(
    bright_future::server::remote_lse<double>::init_u_action
  , remote_lse_init_u_action
);
HPX_REGISTER_ACTION_EX(
    bright_future::server::remote_lse<double>::init_u_blocked_action
  , remote_lse_init_u_blocked_action
);
HPX_REGISTER_ACTION_EX(
    bright_future::server::remote_lse<double>::init_rhs_action
  , remote_lse_init_rhs_action
);
HPX_REGISTER_ACTION_EX(
    bright_future::server::remote_lse<double>::init_rhs_blocked_action
  , remote_lse_init_rhs_blocked_action
);
HPX_REGISTER_ACTION_EX(
    bright_future::server::remote_lse<double>::apply_action
  , remote_lse_apply_action
);
HPX_REGISTER_ACTION_EX(
    bright_future::server::remote_lse<double>::apply_region_action
  , remote_lse_apply_region_action
);
HPX_REGISTER_ACTION_EX(
    bright_future::server::remote_lse<double>::apply_region_df_action
  , remote_lse_apply_region_df_action
);
/*
HPX_REGISTER_ACTION_EX(
    bright_future::server::remote_lse<double>::get_col_action
  , remote_lse_get_col_action
);
*/
HPX_REGISTER_ACTION_EX(
    hpx::lcos::base_lco_with_value<std::vector<double> >::set_value_action
  , remote_lse_base_lco_set_value_action
);
/*
HPX_REGISTER_ACTION_EX(
    bright_future::server::remote_lse<double>::get_row_action
  , remote_lse_get_row_action
);
HPX_REGISTER_ACTION_EX(
    bright_future::server::remote_lse<double>::update_top_boundary_action
  , remote_lse_update_top_boundary_action
);
HPX_REGISTER_ACTION_EX(
    bright_future::server::remote_lse<double>::update_bottom_boundary_action
  , remote_lse_update_bottom_boundary_action
);
HPX_REGISTER_ACTION_EX(
    bright_future::server::remote_lse<double>::update_left_boundary_action
  , remote_lse_update_left_boundary_action
);
HPX_REGISTER_ACTION_EX(
    bright_future::server::remote_lse<double>::update_right_boundary_action
  , remote_lse_update_right_boundary_action
);
*/

/*
BRIGHT_GRID_FUTURE_REGISTER_FUNCTOR(init, init_fun)
BRIGHT_GRID_FUTURE_REGISTER_FUNCTOR(update, update_fun)
*/

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
        hpx::components::simple_component<bright_future::server::remote_lse<double> >
      , gs_hpx
      );

