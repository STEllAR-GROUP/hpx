//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include <hpx/runtime/components/plain_component_factory.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/include/iostreams.hpp>

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
        typename remote_lse<T>::size_type nx
      , typename remote_lse<T>::size_type ny
      , double hx
      , double hy
    )
    {
        u = grid_type(nx, ny);
        rhs = grid_type(nx, ny);

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
        for(size_type y = y_range.first; y < (std::min)(config.n_y, y_range.second); ++y)
        {
            for(size_type x = x_range.first; x < (std::min)(config.n_x, x_range.second); ++x)
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
        for(size_type y = y_range.first; y < (std::min)(config.n_y, y_range.second); ++y)
        {
            for(size_type x = x_range.first; x < (std::min)(config.n_x, x_range.second); ++x)
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
      , std::vector<hpx::lcos::promise<void> > const & dependencies
    )
    {
        BOOST_FOREACH(hpx::lcos::promise<void> const & promise, dependencies)
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
      , std::vector<hpx::lcos::promise<void> > const & dependencies
    )
    {
        BOOST_FOREACH(hpx::lcos::promise<void> const & promise, dependencies)
        {
            promise.get();
        }

        for(size_type y = y_range.first; y < (std::min)(config.n_y, y_range.second); ++y)
        {
            for(size_type x = x_range.first; x < (std::min)(config.n_x, x_range.second); ++x)
            {
                u(x, y) = f(u, rhs, x, y, config);
            }
        }
    }

}}

template HPX_EXPORT class bright_future::server::remote_lse<double>;

HPX_REGISTER_ACTION_EX(bright_future::server::remote_lse<double>::init_action, remote_lse_init_action);
HPX_REGISTER_ACTION_EX(bright_future::server::remote_lse<double>::init_u_action, remote_lse_init_u_action);
HPX_REGISTER_ACTION_EX(bright_future::server::remote_lse<double>::init_u_blocked_action, remote_lse_init_u_blocked_action);
HPX_REGISTER_ACTION_EX(bright_future::server::remote_lse<double>::init_rhs_action, remote_lse_init_rhs_action);
HPX_REGISTER_ACTION_EX(bright_future::server::remote_lse<double>::init_rhs_blocked_action, remote_lse_init_rhs_blocked_action);
HPX_REGISTER_ACTION_EX(bright_future::server::remote_lse<double>::apply_action, remote_lse_apply_action);
HPX_REGISTER_ACTION_EX(bright_future::server::remote_lse<double>::apply_region_action, remote_lse_apply_region_action);

/*
BRIGHT_GRID_FUTURE_REGISTER_FUNCTOR(init, init_fun)
BRIGHT_GRID_FUTURE_REGISTER_FUNCTOR(update, update_fun)
*/

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
        hpx::components::simple_component<bright_future::server::remote_lse<double> >
      , gs_remote_lse_type
      );
HPX_DEFINE_GET_COMPONENT_TYPE(bright_future::server::remote_lse<double>);
