// Copyright (c) 2007-2012 Hartmut Kaiser
// Copyright (c)      2012 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


        
        
        
        template <typename Action, typename A0>
        void init(
            naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 
        )
        {
            typedef
                detail::dataflow_impl<
                    Action
                  , typename boost::remove_const< typename hpx::util::detail::remove_reference< A0 >::type >::type
                >
                wrapped_type;
            typedef
                detail::component_wrapper<
                    wrapped_type
                >
                component_type;
            component_type * w = new component_type(target, mtx, targets);
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                component_ptr = w;
            }
            (*w)->init(boost::forward<A0>(a0));
            lcos::local::spinlock::scoped_lock
                l(detail::dataflow_counter_data_.mtx_);
            ++detail::dataflow_counter_data_.initialized_;
        }
        template <typename Action, typename A0>
        dataflow(
            component_type * back_ptr
          , Action
          , naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 
        )
            : base_type(back_ptr)
            , component_ptr(0)
        {
            
            init<typename Action::type>(target, boost::forward<A0>(a0));
        }
        
        template <typename Action, typename A0 , typename A1>
        void init(
            naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 
        )
        {
            typedef
                detail::dataflow_impl<
                    Action
                  , typename boost::remove_const< typename hpx::util::detail::remove_reference< A0 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A1 >::type >::type
                >
                wrapped_type;
            typedef
                detail::component_wrapper<
                    wrapped_type
                >
                component_type;
            component_type * w = new component_type(target, mtx, targets);
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                component_ptr = w;
            }
            (*w)->init(boost::forward<A0>(a0) , boost::forward<A1>(a1));
            lcos::local::spinlock::scoped_lock
                l(detail::dataflow_counter_data_.mtx_);
            ++detail::dataflow_counter_data_.initialized_;
        }
        template <typename Action, typename A0 , typename A1>
        dataflow(
            component_type * back_ptr
          , Action
          , naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 
        )
            : base_type(back_ptr)
            , component_ptr(0)
        {
            
            init<typename Action::type>(target, boost::forward<A0>(a0) , boost::forward<A1>(a1));
        }
        
        template <typename Action, typename A0 , typename A1 , typename A2>
        void init(
            naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 
        )
        {
            typedef
                detail::dataflow_impl<
                    Action
                  , typename boost::remove_const< typename hpx::util::detail::remove_reference< A0 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A1 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A2 >::type >::type
                >
                wrapped_type;
            typedef
                detail::component_wrapper<
                    wrapped_type
                >
                component_type;
            component_type * w = new component_type(target, mtx, targets);
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                component_ptr = w;
            }
            (*w)->init(boost::forward<A0>(a0) , boost::forward<A1>(a1) , boost::forward<A2>(a2));
            lcos::local::spinlock::scoped_lock
                l(detail::dataflow_counter_data_.mtx_);
            ++detail::dataflow_counter_data_.initialized_;
        }
        template <typename Action, typename A0 , typename A1 , typename A2>
        dataflow(
            component_type * back_ptr
          , Action
          , naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 
        )
            : base_type(back_ptr)
            , component_ptr(0)
        {
            
            init<typename Action::type>(target, boost::forward<A0>(a0) , boost::forward<A1>(a1) , boost::forward<A2>(a2));
        }
        
        template <typename Action, typename A0 , typename A1 , typename A2 , typename A3>
        void init(
            naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 
        )
        {
            typedef
                detail::dataflow_impl<
                    Action
                  , typename boost::remove_const< typename hpx::util::detail::remove_reference< A0 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A1 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A2 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A3 >::type >::type
                >
                wrapped_type;
            typedef
                detail::component_wrapper<
                    wrapped_type
                >
                component_type;
            component_type * w = new component_type(target, mtx, targets);
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                component_ptr = w;
            }
            (*w)->init(boost::forward<A0>(a0) , boost::forward<A1>(a1) , boost::forward<A2>(a2) , boost::forward<A3>(a3));
            lcos::local::spinlock::scoped_lock
                l(detail::dataflow_counter_data_.mtx_);
            ++detail::dataflow_counter_data_.initialized_;
        }
        template <typename Action, typename A0 , typename A1 , typename A2 , typename A3>
        dataflow(
            component_type * back_ptr
          , Action
          , naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 
        )
            : base_type(back_ptr)
            , component_ptr(0)
        {
            
            init<typename Action::type>(target, boost::forward<A0>(a0) , boost::forward<A1>(a1) , boost::forward<A2>(a2) , boost::forward<A3>(a3));
        }
        
        template <typename Action, typename A0 , typename A1 , typename A2 , typename A3 , typename A4>
        void init(
            naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 
        )
        {
            typedef
                detail::dataflow_impl<
                    Action
                  , typename boost::remove_const< typename hpx::util::detail::remove_reference< A0 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A1 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A2 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A3 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A4 >::type >::type
                >
                wrapped_type;
            typedef
                detail::component_wrapper<
                    wrapped_type
                >
                component_type;
            component_type * w = new component_type(target, mtx, targets);
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                component_ptr = w;
            }
            (*w)->init(boost::forward<A0>(a0) , boost::forward<A1>(a1) , boost::forward<A2>(a2) , boost::forward<A3>(a3) , boost::forward<A4>(a4));
            lcos::local::spinlock::scoped_lock
                l(detail::dataflow_counter_data_.mtx_);
            ++detail::dataflow_counter_data_.initialized_;
        }
        template <typename Action, typename A0 , typename A1 , typename A2 , typename A3 , typename A4>
        dataflow(
            component_type * back_ptr
          , Action
          , naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 
        )
            : base_type(back_ptr)
            , component_ptr(0)
        {
            
            init<typename Action::type>(target, boost::forward<A0>(a0) , boost::forward<A1>(a1) , boost::forward<A2>(a2) , boost::forward<A3>(a3) , boost::forward<A4>(a4));
        }
        
        template <typename Action, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5>
        void init(
            naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 
        )
        {
            typedef
                detail::dataflow_impl<
                    Action
                  , typename boost::remove_const< typename hpx::util::detail::remove_reference< A0 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A1 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A2 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A3 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A4 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A5 >::type >::type
                >
                wrapped_type;
            typedef
                detail::component_wrapper<
                    wrapped_type
                >
                component_type;
            component_type * w = new component_type(target, mtx, targets);
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                component_ptr = w;
            }
            (*w)->init(boost::forward<A0>(a0) , boost::forward<A1>(a1) , boost::forward<A2>(a2) , boost::forward<A3>(a3) , boost::forward<A4>(a4) , boost::forward<A5>(a5));
            lcos::local::spinlock::scoped_lock
                l(detail::dataflow_counter_data_.mtx_);
            ++detail::dataflow_counter_data_.initialized_;
        }
        template <typename Action, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5>
        dataflow(
            component_type * back_ptr
          , Action
          , naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 
        )
            : base_type(back_ptr)
            , component_ptr(0)
        {
            
            init<typename Action::type>(target, boost::forward<A0>(a0) , boost::forward<A1>(a1) , boost::forward<A2>(a2) , boost::forward<A3>(a3) , boost::forward<A4>(a4) , boost::forward<A5>(a5));
        }
        
        template <typename Action, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6>
        void init(
            naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 
        )
        {
            typedef
                detail::dataflow_impl<
                    Action
                  , typename boost::remove_const< typename hpx::util::detail::remove_reference< A0 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A1 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A2 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A3 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A4 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A5 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A6 >::type >::type
                >
                wrapped_type;
            typedef
                detail::component_wrapper<
                    wrapped_type
                >
                component_type;
            component_type * w = new component_type(target, mtx, targets);
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                component_ptr = w;
            }
            (*w)->init(boost::forward<A0>(a0) , boost::forward<A1>(a1) , boost::forward<A2>(a2) , boost::forward<A3>(a3) , boost::forward<A4>(a4) , boost::forward<A5>(a5) , boost::forward<A6>(a6));
            lcos::local::spinlock::scoped_lock
                l(detail::dataflow_counter_data_.mtx_);
            ++detail::dataflow_counter_data_.initialized_;
        }
        template <typename Action, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6>
        dataflow(
            component_type * back_ptr
          , Action
          , naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 
        )
            : base_type(back_ptr)
            , component_ptr(0)
        {
            
            init<typename Action::type>(target, boost::forward<A0>(a0) , boost::forward<A1>(a1) , boost::forward<A2>(a2) , boost::forward<A3>(a3) , boost::forward<A4>(a4) , boost::forward<A5>(a5) , boost::forward<A6>(a6));
        }
        
        template <typename Action, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7>
        void init(
            naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 
        )
        {
            typedef
                detail::dataflow_impl<
                    Action
                  , typename boost::remove_const< typename hpx::util::detail::remove_reference< A0 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A1 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A2 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A3 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A4 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A5 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A6 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A7 >::type >::type
                >
                wrapped_type;
            typedef
                detail::component_wrapper<
                    wrapped_type
                >
                component_type;
            component_type * w = new component_type(target, mtx, targets);
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                component_ptr = w;
            }
            (*w)->init(boost::forward<A0>(a0) , boost::forward<A1>(a1) , boost::forward<A2>(a2) , boost::forward<A3>(a3) , boost::forward<A4>(a4) , boost::forward<A5>(a5) , boost::forward<A6>(a6) , boost::forward<A7>(a7));
            lcos::local::spinlock::scoped_lock
                l(detail::dataflow_counter_data_.mtx_);
            ++detail::dataflow_counter_data_.initialized_;
        }
        template <typename Action, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7>
        dataflow(
            component_type * back_ptr
          , Action
          , naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 
        )
            : base_type(back_ptr)
            , component_ptr(0)
        {
            
            init<typename Action::type>(target, boost::forward<A0>(a0) , boost::forward<A1>(a1) , boost::forward<A2>(a2) , boost::forward<A3>(a3) , boost::forward<A4>(a4) , boost::forward<A5>(a5) , boost::forward<A6>(a6) , boost::forward<A7>(a7));
        }
        
        template <typename Action, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8>
        void init(
            naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 
        )
        {
            typedef
                detail::dataflow_impl<
                    Action
                  , typename boost::remove_const< typename hpx::util::detail::remove_reference< A0 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A1 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A2 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A3 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A4 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A5 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A6 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A7 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A8 >::type >::type
                >
                wrapped_type;
            typedef
                detail::component_wrapper<
                    wrapped_type
                >
                component_type;
            component_type * w = new component_type(target, mtx, targets);
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                component_ptr = w;
            }
            (*w)->init(boost::forward<A0>(a0) , boost::forward<A1>(a1) , boost::forward<A2>(a2) , boost::forward<A3>(a3) , boost::forward<A4>(a4) , boost::forward<A5>(a5) , boost::forward<A6>(a6) , boost::forward<A7>(a7) , boost::forward<A8>(a8));
            lcos::local::spinlock::scoped_lock
                l(detail::dataflow_counter_data_.mtx_);
            ++detail::dataflow_counter_data_.initialized_;
        }
        template <typename Action, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8>
        dataflow(
            component_type * back_ptr
          , Action
          , naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 
        )
            : base_type(back_ptr)
            , component_ptr(0)
        {
            
            init<typename Action::type>(target, boost::forward<A0>(a0) , boost::forward<A1>(a1) , boost::forward<A2>(a2) , boost::forward<A3>(a3) , boost::forward<A4>(a4) , boost::forward<A5>(a5) , boost::forward<A6>(a6) , boost::forward<A7>(a7) , boost::forward<A8>(a8));
        }
        
        template <typename Action, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9>
        void init(
            naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 
        )
        {
            typedef
                detail::dataflow_impl<
                    Action
                  , typename boost::remove_const< typename hpx::util::detail::remove_reference< A0 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A1 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A2 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A3 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A4 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A5 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A6 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A7 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A8 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A9 >::type >::type
                >
                wrapped_type;
            typedef
                detail::component_wrapper<
                    wrapped_type
                >
                component_type;
            component_type * w = new component_type(target, mtx, targets);
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                component_ptr = w;
            }
            (*w)->init(boost::forward<A0>(a0) , boost::forward<A1>(a1) , boost::forward<A2>(a2) , boost::forward<A3>(a3) , boost::forward<A4>(a4) , boost::forward<A5>(a5) , boost::forward<A6>(a6) , boost::forward<A7>(a7) , boost::forward<A8>(a8) , boost::forward<A9>(a9));
            lcos::local::spinlock::scoped_lock
                l(detail::dataflow_counter_data_.mtx_);
            ++detail::dataflow_counter_data_.initialized_;
        }
        template <typename Action, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9>
        dataflow(
            component_type * back_ptr
          , Action
          , naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 
        )
            : base_type(back_ptr)
            , component_ptr(0)
        {
            
            init<typename Action::type>(target, boost::forward<A0>(a0) , boost::forward<A1>(a1) , boost::forward<A2>(a2) , boost::forward<A3>(a3) , boost::forward<A4>(a4) , boost::forward<A5>(a5) , boost::forward<A6>(a6) , boost::forward<A7>(a7) , boost::forward<A8>(a8) , boost::forward<A9>(a9));
        }
        
        template <typename Action, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10>
        void init(
            naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 
        )
        {
            typedef
                detail::dataflow_impl<
                    Action
                  , typename boost::remove_const< typename hpx::util::detail::remove_reference< A0 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A1 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A2 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A3 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A4 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A5 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A6 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A7 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A8 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A9 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A10 >::type >::type
                >
                wrapped_type;
            typedef
                detail::component_wrapper<
                    wrapped_type
                >
                component_type;
            component_type * w = new component_type(target, mtx, targets);
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                component_ptr = w;
            }
            (*w)->init(boost::forward<A0>(a0) , boost::forward<A1>(a1) , boost::forward<A2>(a2) , boost::forward<A3>(a3) , boost::forward<A4>(a4) , boost::forward<A5>(a5) , boost::forward<A6>(a6) , boost::forward<A7>(a7) , boost::forward<A8>(a8) , boost::forward<A9>(a9) , boost::forward<A10>(a10));
            lcos::local::spinlock::scoped_lock
                l(detail::dataflow_counter_data_.mtx_);
            ++detail::dataflow_counter_data_.initialized_;
        }
        template <typename Action, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10>
        dataflow(
            component_type * back_ptr
          , Action
          , naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 
        )
            : base_type(back_ptr)
            , component_ptr(0)
        {
            
            init<typename Action::type>(target, boost::forward<A0>(a0) , boost::forward<A1>(a1) , boost::forward<A2>(a2) , boost::forward<A3>(a3) , boost::forward<A4>(a4) , boost::forward<A5>(a5) , boost::forward<A6>(a6) , boost::forward<A7>(a7) , boost::forward<A8>(a8) , boost::forward<A9>(a9) , boost::forward<A10>(a10));
        }
        
        template <typename Action, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11>
        void init(
            naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 
        )
        {
            typedef
                detail::dataflow_impl<
                    Action
                  , typename boost::remove_const< typename hpx::util::detail::remove_reference< A0 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A1 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A2 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A3 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A4 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A5 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A6 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A7 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A8 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A9 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A10 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A11 >::type >::type
                >
                wrapped_type;
            typedef
                detail::component_wrapper<
                    wrapped_type
                >
                component_type;
            component_type * w = new component_type(target, mtx, targets);
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                component_ptr = w;
            }
            (*w)->init(boost::forward<A0>(a0) , boost::forward<A1>(a1) , boost::forward<A2>(a2) , boost::forward<A3>(a3) , boost::forward<A4>(a4) , boost::forward<A5>(a5) , boost::forward<A6>(a6) , boost::forward<A7>(a7) , boost::forward<A8>(a8) , boost::forward<A9>(a9) , boost::forward<A10>(a10) , boost::forward<A11>(a11));
            lcos::local::spinlock::scoped_lock
                l(detail::dataflow_counter_data_.mtx_);
            ++detail::dataflow_counter_data_.initialized_;
        }
        template <typename Action, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11>
        dataflow(
            component_type * back_ptr
          , Action
          , naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 
        )
            : base_type(back_ptr)
            , component_ptr(0)
        {
            
            init<typename Action::type>(target, boost::forward<A0>(a0) , boost::forward<A1>(a1) , boost::forward<A2>(a2) , boost::forward<A3>(a3) , boost::forward<A4>(a4) , boost::forward<A5>(a5) , boost::forward<A6>(a6) , boost::forward<A7>(a7) , boost::forward<A8>(a8) , boost::forward<A9>(a9) , boost::forward<A10>(a10) , boost::forward<A11>(a11));
        }
        
        template <typename Action, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12>
        void init(
            naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 
        )
        {
            typedef
                detail::dataflow_impl<
                    Action
                  , typename boost::remove_const< typename hpx::util::detail::remove_reference< A0 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A1 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A2 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A3 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A4 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A5 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A6 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A7 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A8 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A9 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A10 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A11 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A12 >::type >::type
                >
                wrapped_type;
            typedef
                detail::component_wrapper<
                    wrapped_type
                >
                component_type;
            component_type * w = new component_type(target, mtx, targets);
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                component_ptr = w;
            }
            (*w)->init(boost::forward<A0>(a0) , boost::forward<A1>(a1) , boost::forward<A2>(a2) , boost::forward<A3>(a3) , boost::forward<A4>(a4) , boost::forward<A5>(a5) , boost::forward<A6>(a6) , boost::forward<A7>(a7) , boost::forward<A8>(a8) , boost::forward<A9>(a9) , boost::forward<A10>(a10) , boost::forward<A11>(a11) , boost::forward<A12>(a12));
            lcos::local::spinlock::scoped_lock
                l(detail::dataflow_counter_data_.mtx_);
            ++detail::dataflow_counter_data_.initialized_;
        }
        template <typename Action, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12>
        dataflow(
            component_type * back_ptr
          , Action
          , naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 
        )
            : base_type(back_ptr)
            , component_ptr(0)
        {
            
            init<typename Action::type>(target, boost::forward<A0>(a0) , boost::forward<A1>(a1) , boost::forward<A2>(a2) , boost::forward<A3>(a3) , boost::forward<A4>(a4) , boost::forward<A5>(a5) , boost::forward<A6>(a6) , boost::forward<A7>(a7) , boost::forward<A8>(a8) , boost::forward<A9>(a9) , boost::forward<A10>(a10) , boost::forward<A11>(a11) , boost::forward<A12>(a12));
        }
        
        template <typename Action, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13>
        void init(
            naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13 
        )
        {
            typedef
                detail::dataflow_impl<
                    Action
                  , typename boost::remove_const< typename hpx::util::detail::remove_reference< A0 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A1 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A2 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A3 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A4 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A5 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A6 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A7 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A8 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A9 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A10 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A11 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A12 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A13 >::type >::type
                >
                wrapped_type;
            typedef
                detail::component_wrapper<
                    wrapped_type
                >
                component_type;
            component_type * w = new component_type(target, mtx, targets);
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                component_ptr = w;
            }
            (*w)->init(boost::forward<A0>(a0) , boost::forward<A1>(a1) , boost::forward<A2>(a2) , boost::forward<A3>(a3) , boost::forward<A4>(a4) , boost::forward<A5>(a5) , boost::forward<A6>(a6) , boost::forward<A7>(a7) , boost::forward<A8>(a8) , boost::forward<A9>(a9) , boost::forward<A10>(a10) , boost::forward<A11>(a11) , boost::forward<A12>(a12) , boost::forward<A13>(a13));
            lcos::local::spinlock::scoped_lock
                l(detail::dataflow_counter_data_.mtx_);
            ++detail::dataflow_counter_data_.initialized_;
        }
        template <typename Action, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13>
        dataflow(
            component_type * back_ptr
          , Action
          , naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13 
        )
            : base_type(back_ptr)
            , component_ptr(0)
        {
            
            init<typename Action::type>(target, boost::forward<A0>(a0) , boost::forward<A1>(a1) , boost::forward<A2>(a2) , boost::forward<A3>(a3) , boost::forward<A4>(a4) , boost::forward<A5>(a5) , boost::forward<A6>(a6) , boost::forward<A7>(a7) , boost::forward<A8>(a8) , boost::forward<A9>(a9) , boost::forward<A10>(a10) , boost::forward<A11>(a11) , boost::forward<A12>(a12) , boost::forward<A13>(a13));
        }
        
        template <typename Action, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14>
        void init(
            naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13 , BOOST_FWD_REF(A14) a14 
        )
        {
            typedef
                detail::dataflow_impl<
                    Action
                  , typename boost::remove_const< typename hpx::util::detail::remove_reference< A0 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A1 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A2 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A3 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A4 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A5 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A6 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A7 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A8 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A9 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A10 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A11 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A12 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A13 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A14 >::type >::type
                >
                wrapped_type;
            typedef
                detail::component_wrapper<
                    wrapped_type
                >
                component_type;
            component_type * w = new component_type(target, mtx, targets);
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                component_ptr = w;
            }
            (*w)->init(boost::forward<A0>(a0) , boost::forward<A1>(a1) , boost::forward<A2>(a2) , boost::forward<A3>(a3) , boost::forward<A4>(a4) , boost::forward<A5>(a5) , boost::forward<A6>(a6) , boost::forward<A7>(a7) , boost::forward<A8>(a8) , boost::forward<A9>(a9) , boost::forward<A10>(a10) , boost::forward<A11>(a11) , boost::forward<A12>(a12) , boost::forward<A13>(a13) , boost::forward<A14>(a14));
            lcos::local::spinlock::scoped_lock
                l(detail::dataflow_counter_data_.mtx_);
            ++detail::dataflow_counter_data_.initialized_;
        }
        template <typename Action, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14>
        dataflow(
            component_type * back_ptr
          , Action
          , naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13 , BOOST_FWD_REF(A14) a14 
        )
            : base_type(back_ptr)
            , component_ptr(0)
        {
            
            init<typename Action::type>(target, boost::forward<A0>(a0) , boost::forward<A1>(a1) , boost::forward<A2>(a2) , boost::forward<A3>(a3) , boost::forward<A4>(a4) , boost::forward<A5>(a5) , boost::forward<A6>(a6) , boost::forward<A7>(a7) , boost::forward<A8>(a8) , boost::forward<A9>(a9) , boost::forward<A10>(a10) , boost::forward<A11>(a11) , boost::forward<A12>(a12) , boost::forward<A13>(a13) , boost::forward<A14>(a14));
        }
        
        template <typename Action, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15>
        void init(
            naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13 , BOOST_FWD_REF(A14) a14 , BOOST_FWD_REF(A15) a15 
        )
        {
            typedef
                detail::dataflow_impl<
                    Action
                  , typename boost::remove_const< typename hpx::util::detail::remove_reference< A0 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A1 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A2 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A3 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A4 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A5 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A6 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A7 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A8 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A9 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A10 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A11 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A12 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A13 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A14 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A15 >::type >::type
                >
                wrapped_type;
            typedef
                detail::component_wrapper<
                    wrapped_type
                >
                component_type;
            component_type * w = new component_type(target, mtx, targets);
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                component_ptr = w;
            }
            (*w)->init(boost::forward<A0>(a0) , boost::forward<A1>(a1) , boost::forward<A2>(a2) , boost::forward<A3>(a3) , boost::forward<A4>(a4) , boost::forward<A5>(a5) , boost::forward<A6>(a6) , boost::forward<A7>(a7) , boost::forward<A8>(a8) , boost::forward<A9>(a9) , boost::forward<A10>(a10) , boost::forward<A11>(a11) , boost::forward<A12>(a12) , boost::forward<A13>(a13) , boost::forward<A14>(a14) , boost::forward<A15>(a15));
            lcos::local::spinlock::scoped_lock
                l(detail::dataflow_counter_data_.mtx_);
            ++detail::dataflow_counter_data_.initialized_;
        }
        template <typename Action, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15>
        dataflow(
            component_type * back_ptr
          , Action
          , naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13 , BOOST_FWD_REF(A14) a14 , BOOST_FWD_REF(A15) a15 
        )
            : base_type(back_ptr)
            , component_ptr(0)
        {
            
            init<typename Action::type>(target, boost::forward<A0>(a0) , boost::forward<A1>(a1) , boost::forward<A2>(a2) , boost::forward<A3>(a3) , boost::forward<A4>(a4) , boost::forward<A5>(a5) , boost::forward<A6>(a6) , boost::forward<A7>(a7) , boost::forward<A8>(a8) , boost::forward<A9>(a9) , boost::forward<A10>(a10) , boost::forward<A11>(a11) , boost::forward<A12>(a12) , boost::forward<A13>(a13) , boost::forward<A14>(a14) , boost::forward<A15>(a15));
        }
        
        template <typename Action, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16>
        void init(
            naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13 , BOOST_FWD_REF(A14) a14 , BOOST_FWD_REF(A15) a15 , BOOST_FWD_REF(A16) a16 
        )
        {
            typedef
                detail::dataflow_impl<
                    Action
                  , typename boost::remove_const< typename hpx::util::detail::remove_reference< A0 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A1 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A2 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A3 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A4 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A5 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A6 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A7 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A8 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A9 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A10 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A11 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A12 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A13 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A14 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A15 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A16 >::type >::type
                >
                wrapped_type;
            typedef
                detail::component_wrapper<
                    wrapped_type
                >
                component_type;
            component_type * w = new component_type(target, mtx, targets);
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                component_ptr = w;
            }
            (*w)->init(boost::forward<A0>(a0) , boost::forward<A1>(a1) , boost::forward<A2>(a2) , boost::forward<A3>(a3) , boost::forward<A4>(a4) , boost::forward<A5>(a5) , boost::forward<A6>(a6) , boost::forward<A7>(a7) , boost::forward<A8>(a8) , boost::forward<A9>(a9) , boost::forward<A10>(a10) , boost::forward<A11>(a11) , boost::forward<A12>(a12) , boost::forward<A13>(a13) , boost::forward<A14>(a14) , boost::forward<A15>(a15) , boost::forward<A16>(a16));
            lcos::local::spinlock::scoped_lock
                l(detail::dataflow_counter_data_.mtx_);
            ++detail::dataflow_counter_data_.initialized_;
        }
        template <typename Action, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16>
        dataflow(
            component_type * back_ptr
          , Action
          , naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13 , BOOST_FWD_REF(A14) a14 , BOOST_FWD_REF(A15) a15 , BOOST_FWD_REF(A16) a16 
        )
            : base_type(back_ptr)
            , component_ptr(0)
        {
            
            init<typename Action::type>(target, boost::forward<A0>(a0) , boost::forward<A1>(a1) , boost::forward<A2>(a2) , boost::forward<A3>(a3) , boost::forward<A4>(a4) , boost::forward<A5>(a5) , boost::forward<A6>(a6) , boost::forward<A7>(a7) , boost::forward<A8>(a8) , boost::forward<A9>(a9) , boost::forward<A10>(a10) , boost::forward<A11>(a11) , boost::forward<A12>(a12) , boost::forward<A13>(a13) , boost::forward<A14>(a14) , boost::forward<A15>(a15) , boost::forward<A16>(a16));
        }
        
        template <typename Action, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17>
        void init(
            naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13 , BOOST_FWD_REF(A14) a14 , BOOST_FWD_REF(A15) a15 , BOOST_FWD_REF(A16) a16 , BOOST_FWD_REF(A17) a17 
        )
        {
            typedef
                detail::dataflow_impl<
                    Action
                  , typename boost::remove_const< typename hpx::util::detail::remove_reference< A0 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A1 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A2 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A3 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A4 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A5 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A6 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A7 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A8 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A9 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A10 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A11 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A12 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A13 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A14 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A15 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A16 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A17 >::type >::type
                >
                wrapped_type;
            typedef
                detail::component_wrapper<
                    wrapped_type
                >
                component_type;
            component_type * w = new component_type(target, mtx, targets);
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                component_ptr = w;
            }
            (*w)->init(boost::forward<A0>(a0) , boost::forward<A1>(a1) , boost::forward<A2>(a2) , boost::forward<A3>(a3) , boost::forward<A4>(a4) , boost::forward<A5>(a5) , boost::forward<A6>(a6) , boost::forward<A7>(a7) , boost::forward<A8>(a8) , boost::forward<A9>(a9) , boost::forward<A10>(a10) , boost::forward<A11>(a11) , boost::forward<A12>(a12) , boost::forward<A13>(a13) , boost::forward<A14>(a14) , boost::forward<A15>(a15) , boost::forward<A16>(a16) , boost::forward<A17>(a17));
            lcos::local::spinlock::scoped_lock
                l(detail::dataflow_counter_data_.mtx_);
            ++detail::dataflow_counter_data_.initialized_;
        }
        template <typename Action, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17>
        dataflow(
            component_type * back_ptr
          , Action
          , naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13 , BOOST_FWD_REF(A14) a14 , BOOST_FWD_REF(A15) a15 , BOOST_FWD_REF(A16) a16 , BOOST_FWD_REF(A17) a17 
        )
            : base_type(back_ptr)
            , component_ptr(0)
        {
            
            init<typename Action::type>(target, boost::forward<A0>(a0) , boost::forward<A1>(a1) , boost::forward<A2>(a2) , boost::forward<A3>(a3) , boost::forward<A4>(a4) , boost::forward<A5>(a5) , boost::forward<A6>(a6) , boost::forward<A7>(a7) , boost::forward<A8>(a8) , boost::forward<A9>(a9) , boost::forward<A10>(a10) , boost::forward<A11>(a11) , boost::forward<A12>(a12) , boost::forward<A13>(a13) , boost::forward<A14>(a14) , boost::forward<A15>(a15) , boost::forward<A16>(a16) , boost::forward<A17>(a17));
        }
        
        template <typename Action, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18>
        void init(
            naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13 , BOOST_FWD_REF(A14) a14 , BOOST_FWD_REF(A15) a15 , BOOST_FWD_REF(A16) a16 , BOOST_FWD_REF(A17) a17 , BOOST_FWD_REF(A18) a18 
        )
        {
            typedef
                detail::dataflow_impl<
                    Action
                  , typename boost::remove_const< typename hpx::util::detail::remove_reference< A0 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A1 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A2 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A3 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A4 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A5 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A6 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A7 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A8 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A9 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A10 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A11 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A12 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A13 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A14 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A15 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A16 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A17 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A18 >::type >::type
                >
                wrapped_type;
            typedef
                detail::component_wrapper<
                    wrapped_type
                >
                component_type;
            component_type * w = new component_type(target, mtx, targets);
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                component_ptr = w;
            }
            (*w)->init(boost::forward<A0>(a0) , boost::forward<A1>(a1) , boost::forward<A2>(a2) , boost::forward<A3>(a3) , boost::forward<A4>(a4) , boost::forward<A5>(a5) , boost::forward<A6>(a6) , boost::forward<A7>(a7) , boost::forward<A8>(a8) , boost::forward<A9>(a9) , boost::forward<A10>(a10) , boost::forward<A11>(a11) , boost::forward<A12>(a12) , boost::forward<A13>(a13) , boost::forward<A14>(a14) , boost::forward<A15>(a15) , boost::forward<A16>(a16) , boost::forward<A17>(a17) , boost::forward<A18>(a18));
            lcos::local::spinlock::scoped_lock
                l(detail::dataflow_counter_data_.mtx_);
            ++detail::dataflow_counter_data_.initialized_;
        }
        template <typename Action, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18>
        dataflow(
            component_type * back_ptr
          , Action
          , naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13 , BOOST_FWD_REF(A14) a14 , BOOST_FWD_REF(A15) a15 , BOOST_FWD_REF(A16) a16 , BOOST_FWD_REF(A17) a17 , BOOST_FWD_REF(A18) a18 
        )
            : base_type(back_ptr)
            , component_ptr(0)
        {
            
            init<typename Action::type>(target, boost::forward<A0>(a0) , boost::forward<A1>(a1) , boost::forward<A2>(a2) , boost::forward<A3>(a3) , boost::forward<A4>(a4) , boost::forward<A5>(a5) , boost::forward<A6>(a6) , boost::forward<A7>(a7) , boost::forward<A8>(a8) , boost::forward<A9>(a9) , boost::forward<A10>(a10) , boost::forward<A11>(a11) , boost::forward<A12>(a12) , boost::forward<A13>(a13) , boost::forward<A14>(a14) , boost::forward<A15>(a15) , boost::forward<A16>(a16) , boost::forward<A17>(a17) , boost::forward<A18>(a18));
        }
        
        template <typename Action, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18 , typename A19>
        void init(
            naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13 , BOOST_FWD_REF(A14) a14 , BOOST_FWD_REF(A15) a15 , BOOST_FWD_REF(A16) a16 , BOOST_FWD_REF(A17) a17 , BOOST_FWD_REF(A18) a18 , BOOST_FWD_REF(A19) a19 
        )
        {
            typedef
                detail::dataflow_impl<
                    Action
                  , typename boost::remove_const< typename hpx::util::detail::remove_reference< A0 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A1 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A2 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A3 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A4 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A5 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A6 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A7 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A8 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A9 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A10 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A11 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A12 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A13 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A14 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A15 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A16 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A17 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A18 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< A19 >::type >::type
                >
                wrapped_type;
            typedef
                detail::component_wrapper<
                    wrapped_type
                >
                component_type;
            component_type * w = new component_type(target, mtx, targets);
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                component_ptr = w;
            }
            (*w)->init(boost::forward<A0>(a0) , boost::forward<A1>(a1) , boost::forward<A2>(a2) , boost::forward<A3>(a3) , boost::forward<A4>(a4) , boost::forward<A5>(a5) , boost::forward<A6>(a6) , boost::forward<A7>(a7) , boost::forward<A8>(a8) , boost::forward<A9>(a9) , boost::forward<A10>(a10) , boost::forward<A11>(a11) , boost::forward<A12>(a12) , boost::forward<A13>(a13) , boost::forward<A14>(a14) , boost::forward<A15>(a15) , boost::forward<A16>(a16) , boost::forward<A17>(a17) , boost::forward<A18>(a18) , boost::forward<A19>(a19));
            lcos::local::spinlock::scoped_lock
                l(detail::dataflow_counter_data_.mtx_);
            ++detail::dataflow_counter_data_.initialized_;
        }
        template <typename Action, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18 , typename A19>
        dataflow(
            component_type * back_ptr
          , Action
          , naming::id_type const & target
          , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13 , BOOST_FWD_REF(A14) a14 , BOOST_FWD_REF(A15) a15 , BOOST_FWD_REF(A16) a16 , BOOST_FWD_REF(A17) a17 , BOOST_FWD_REF(A18) a18 , BOOST_FWD_REF(A19) a19 
        )
            : base_type(back_ptr)
            , component_ptr(0)
        {
            
            init<typename Action::type>(target, boost::forward<A0>(a0) , boost::forward<A1>(a1) , boost::forward<A2>(a2) , boost::forward<A3>(a3) , boost::forward<A4>(a4) , boost::forward<A5>(a5) , boost::forward<A6>(a6) , boost::forward<A7>(a7) , boost::forward<A8>(a8) , boost::forward<A9>(a9) , boost::forward<A10>(a10) , boost::forward<A11>(a11) , boost::forward<A12>(a12) , boost::forward<A13>(a13) , boost::forward<A14>(a14) , boost::forward<A15>(a15) , boost::forward<A16>(a16) , boost::forward<A17>(a17) , boost::forward<A18>(a18) , boost::forward<A19>(a19));
        }
