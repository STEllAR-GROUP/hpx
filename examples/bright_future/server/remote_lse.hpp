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

#include <hpx/lcos/local_mutex.hpp>

template <typename Promise>
struct promise_wrapper
{
    promise_wrapper()
        //: finished(false)
    {}
    promise_wrapper(Promise const & p)
        : promise(p)
        //, finished(false)
    {}

    promise_wrapper(promise_wrapper const & o)
        : promise(o.promise)
        //, finished(o.finished)
    {
        //BOOST_ASSERT(o.finished == false);
    }

    /*
    promise_wrapper & operator=(promise_wrapper const & o)
    {
        BOOST_ASSERT(o.finished == false && finished == false);
        promise = o.promise;

        return *this;
    }
    */

    promise_wrapper & operator=(Promise const & p)
    {
        //BOOST_ASSERT(finished == false);
        promise = p;

        return *this;
    }


    void get()
    {
        //m.lock();
        //if(!finished)
        {
            promise.get();
            //finished = true;
        }
        //m.unlock();
    }

    Promise promise;
    bool finished;
    //hpx::lcos::local_mutex m;

    template <typename Archive>
    void serialize(Archive &, unsigned)
    {}
};

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
                size_type nx
              , size_type ny
              , double hx
              , double hy
            );

            typedef
                hpx::actions::action4<
                    remote_lse<T>
                  , remote_lse_init
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
              , std::vector<promise_wrapper<hpx::lcos::promise<void> > *> const & dependencies
            );

            typedef
                hpx::actions::action4<
                    remote_lse<T>
                  , remote_lse_apply
                  , apply_func_type
                  , size_type
                  , size_type
                  , std::vector<promise_wrapper<hpx::lcos::promise<void> > *> const &
                  , &remote_lse<T>::apply
                >
                apply_action;

            void apply_region(
                apply_func_type f
              , range_type x_range
              , range_type y_range
              , std::vector<promise_wrapper<hpx::lcos::promise<void> > *> const & dependencies
            );

            typedef
                hpx::actions::action4<
                    remote_lse<T>
                  , remote_lse_apply_region
                  , apply_func_type
                  , range_type
                  , range_type
                  , std::vector<promise_wrapper<hpx::lcos::promise<void> > *> const &
                  , &remote_lse<T>::apply_region
                >
                apply_region_action;
    };

}}

#endif
