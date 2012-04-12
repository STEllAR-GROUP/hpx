//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#ifndef HPX_EXAMPLES_BRIGHT_FUTURES_REMOTE_LSE_HPP
#define HPX_EXAMPLES_BRIGHT_FUTURES_REMOTE_LSE_HPP

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>

#include <hpx/util/function.hpp>
#include "../grid.hpp"

#include <hpx/lcos/local/spinlock.hpp>

namespace boost { namespace serialization {
    template <typename Archive>
    void serialize(Archive & ar, hpx::lcos::future<void> &, unsigned)
    {
    }
}}


namespace bright_future {

    template <typename T> struct grid;

    template <typename T>
    struct lse_config
    {
        typedef typename grid<T>::size_type size_type;
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
    };

    namespace server {

    template <typename T>
    class HPX_COMPONENT_EXPORT remote_lse
        : public hpx::components::simple_component_base<remote_lse<T> >
    {
        private:
            typedef hpx::components::simple_component_base<remote_lse<T> > base_type;

            typedef bright_future::grid<T> grid_type;
            grid_type u;
            grid_type rhs;

            lse_config<T> config;

            // stuff for the stencil

        public:
            hpx::lcos::local::spinlock mtx;
            std::vector<double> timestamps;

            void clear_timestamps()
            {
                std::vector<double> t;
                std::swap(t, timestamps);
            }
            
            typedef
                hpx::actions::action0<
                    remote_lse<T>
                  , 0
                  , &remote_lse<T>::clear_timestamps
                >
                clear_timestamps_action;

            void print_timestamps()
            {
                double acc = 0;
                //std::accumulate(timestamps.begin(), timestamps.end(), acc);
                std::size_t n = timestamps.size();
                for(std::size_t i = 0; i < n; ++i)
                {
                    acc += timestamps[0]/n;
                }
                hpx::cout << "Average time per update: " << acc << "\n" << hpx::flush;
            }
            
            typedef
                hpx::actions::action0<
                    remote_lse<T>
                  , 0
                  , &remote_lse<T>::print_timestamps
                >
                print_timestamps_action;

            remote_lse();

            typedef typename grid_type::size_type size_type;
            typedef typename grid_type::value_type value_type;

            typedef std::pair<size_type, size_type> range_type;

            typedef remote_lse<T> wrapper_type;

            enum actions
            {
                remote_lse_init     = 0
              , remote_lse_init_rhs = 1
              , remote_lse_init_u   = 2
              , remote_lse_apply    = 3
              , remote_lse_apply_region = 4
              , remote_lse_init_u_blocked  = 5
              , remote_lse_init_rhs_blocked  = 6
            };

            void init(
                size_type size_x
              , size_type size_y
              , size_type nx
              , size_type ny
              , double hx
              , double hy
            );

            typedef
                hpx::actions::action6<
                    remote_lse<T>
                  , remote_lse_init
                  , size_type
                  , size_type
                  , size_type
                  , size_type
                  , double
                  , double
                  , &remote_lse<T>::init
                >
                init_action;

            typedef
                hpx::util::function<
                    double(
                        size_type
                      , size_type
                      , lse_config<T> const &
                    )
                >
                init_func_type;

            void init_rhs(
                init_func_type  f
              , typename remote_lse<T>::size_type x
              , typename remote_lse<T>::size_type y
            );

            typedef
                hpx::actions::action3<
                    remote_lse<T>
                  , remote_lse_init_rhs
                  , init_func_type
                  , size_type
                  , size_type
                  , &remote_lse<T>::init_rhs
                >
                init_rhs_action;

            void init_rhs_blocked(
                init_func_type  f
              , typename remote_lse<T>::range_type x
              , typename remote_lse<T>::range_type y
            );

            typedef
                hpx::actions::action3<
                    remote_lse<T>
                  , remote_lse_init_rhs_blocked
                  , init_func_type
                  , range_type
                  , range_type
                  , &remote_lse<T>::init_rhs_blocked
                >
                init_rhs_blocked_action;

            void init_u(
                init_func_type  f
              , typename remote_lse<T>::size_type x
              , typename remote_lse<T>::size_type y
            );

            typedef
                hpx::actions::action3<
                    remote_lse<T>
                  , remote_lse_init_u
                  , init_func_type
                  , size_type
                  , size_type
                  , &remote_lse<T>::init_u
                >
                init_u_action;

            void init_u_blocked(
                init_func_type  f
              , typename remote_lse<T>::range_type x
              , typename remote_lse<T>::range_type y
            );

            typedef
                hpx::actions::action3<
                    remote_lse<T>
                  , remote_lse_init_u_blocked
                  , init_func_type
                  , range_type
                  , range_type
                  , &remote_lse<T>::init_u_blocked
                >
                init_u_blocked_action;

            typedef
                hpx::util::function<
                    double(
                        grid<T> const &
                      , grid<T> const &
                      , size_type
                      , size_type
                      , lse_config<T> const &
                    )
                >
                apply_func_type;

            void apply(
                apply_func_type f
              , typename remote_lse<T>::size_type x
              , typename remote_lse<T>::size_type y
              , std::vector<hpx::lcos::future<void> > dependencies
            );

            typedef
                hpx::actions::action4<
                    remote_lse<T>
                  , remote_lse_apply
                  , apply_func_type
                  , size_type
                  , size_type
                  , std::vector<hpx::lcos::future<void> >
                  , &remote_lse<T>::apply
                >
                apply_action;

            void apply_region(
                apply_func_type f
              , range_type x_range
              , range_type y_range
              , std::vector<hpx::lcos::future<void> *> dependencies
            );

            typedef
                hpx::actions::action4<
                    remote_lse<T>
                  , remote_lse_apply_region
                  , apply_func_type
                  , range_type
                  , range_type
                  , std::vector<hpx::lcos::future<void> *>
                  , &remote_lse<T>::apply_region
                >
                apply_region_action;
            
            void apply_region_df(
                apply_func_type f
              , range_type x_range
              , range_type y_range
            );

            typedef
                hpx::actions::action3<
                    remote_lse<T>
                  , remote_lse_apply_region
                  , apply_func_type
                  , range_type
                  , range_type
                  , &remote_lse<T>::apply_region_df
                >
                apply_region_df_action;

            std::vector<T> get_row(size_type r, range_type);

            typedef
                hpx::actions::result_action2<
                    remote_lse<T>
                  , std::vector<T>
                  , 0
                  , size_type
                  , range_type
                  , &remote_lse<T>::get_row
                >
                get_row_action;
            
            std::vector<T> get_col(size_type r, range_type);

            typedef
                hpx::actions::result_action2<
                    remote_lse<T>
                  , std::vector<T>
                  , 0
                  , size_type
                  , range_type
                  , &remote_lse<T>::get_col
                >
                get_col_action;

            void update_top_boundary(std::vector<T> const &, range_type);
            typedef
                hpx::actions::action2<
                    remote_lse<T>
                  , 0
                  , std::vector<T> const &
                  , range_type
                  , &remote_lse<T>::update_top_boundary
                >
                update_top_boundary_action;

            void update_bottom_boundary(std::vector<T> const &, range_type);
            typedef
                hpx::actions::action2<
                    remote_lse<T>
                  , 0
                  , std::vector<T> const &
                  , range_type
                  , &remote_lse<T>::update_bottom_boundary
                >
                update_bottom_boundary_action;

            void update_left_boundary(std::vector<T> const &, range_type);
            typedef
                hpx::actions::action2<
                    remote_lse<T>
                  , 0
                  , std::vector<T> const &
                  , range_type
                  , &remote_lse<T>::update_left_boundary
                >
                update_left_boundary_action;

            void update_right_boundary(std::vector<T> const &, range_type);
            typedef
                hpx::actions::action2<
                    remote_lse<T>
                  , 0
                  , std::vector<T> const &
                  , range_type
                  , &remote_lse<T>::update_right_boundary
                >
                update_right_boundary_action;
    };
}}

HPX_REGISTER_ACTION_DECLARATION_EX(
    bright_future::server::remote_lse<double>::init_action
  , remote_lse_init_action
);
HPX_REGISTER_ACTION_DECLARATION_EX(
    bright_future::server::remote_lse<double>::init_u_action
  , remote_lse_init_u_action
);
HPX_REGISTER_ACTION_DECLARATION_EX(
    bright_future::server::remote_lse<double>::init_u_blocked_action
  , remote_lse_init_u_blocked_action
);
HPX_REGISTER_ACTION_DECLARATION_EX(
    bright_future::server::remote_lse<double>::init_rhs_action
  , remote_lse_init_rhs_action
);
HPX_REGISTER_ACTION_DECLARATION_EX(
    bright_future::server::remote_lse<double>::init_rhs_blocked_action
  , remote_lse_init_rhs_blocked_action
);
HPX_REGISTER_ACTION_DECLARATION_EX(
    bright_future::server::remote_lse<double>::apply_action
  , remote_lse_apply_action
);
HPX_REGISTER_ACTION_DECLARATION_EX(
    bright_future::server::remote_lse<double>::apply_region_action
  , remote_lse_apply_region_action
);
HPX_REGISTER_ACTION_DECLARATION_EX(
    bright_future::server::remote_lse<double>::apply_region_df_action
  , remote_lse_apply_region_df_action
);
/*
HPX_REGISTER_ACTION_DECLARATION_EX(
    bright_future::server::remote_lse<double>::get_col_action
  , remote_lse_get_col_action
);
*/
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::lcos::base_lco_with_value<std::vector<double> >::set_value_action
  , remote_lse_base_lco_set_value_action
);
/*
HPX_REGISTER_ACTION_DECLARATION_EX(
    bright_future::server::remote_lse<double>::get_row_action
  , remote_lse_get_row_action
);
HPX_REGISTER_ACTION_DECLARATION_EX(
    bright_future::server::remote_lse<double>::update_top_boundary_action
  , remote_lse_update_top_boundary_action
);
HPX_REGISTER_ACTION_DECLARATION_EX(
    bright_future::server::remote_lse<double>::update_bottom_boundary_action
  , remote_lse_update_bottom_boundary_action
);
HPX_REGISTER_ACTION_DECLARATION_EX(
    bright_future::server::remote_lse<double>::update_left_boundary_action
  , remote_lse_update_left_boundary_action
);
HPX_REGISTER_ACTION_DECLARATION_EX(
    bright_future::server::remote_lse<double>::update_right_boundary_action
  , remote_lse_update_right_boundary_action
);
*/

#endif
