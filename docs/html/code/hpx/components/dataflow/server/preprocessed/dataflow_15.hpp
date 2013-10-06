// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
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
                  , typename util::decay<A0>::type
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
            detail::update_initialized_count();
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
                  , typename util::decay<A0>::type , typename util::decay<A1>::type
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
            detail::update_initialized_count();
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
                  , typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type
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
            detail::update_initialized_count();
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
                  , typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type
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
            detail::update_initialized_count();
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
                  , typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type
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
            detail::update_initialized_count();
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
                  , typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type
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
            detail::update_initialized_count();
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
                  , typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type
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
            detail::update_initialized_count();
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
                  , typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type
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
            detail::update_initialized_count();
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
                  , typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type
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
            detail::update_initialized_count();
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
                  , typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type
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
            detail::update_initialized_count();
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
                  , typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type
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
            detail::update_initialized_count();
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
                  , typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type
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
            detail::update_initialized_count();
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
                  , typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type
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
            detail::update_initialized_count();
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
                  , typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type , typename util::decay<A13>::type
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
            detail::update_initialized_count();
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
                  , typename util::decay<A0>::type , typename util::decay<A1>::type , typename util::decay<A2>::type , typename util::decay<A3>::type , typename util::decay<A4>::type , typename util::decay<A5>::type , typename util::decay<A6>::type , typename util::decay<A7>::type , typename util::decay<A8>::type , typename util::decay<A9>::type , typename util::decay<A10>::type , typename util::decay<A11>::type , typename util::decay<A12>::type , typename util::decay<A13>::type , typename util::decay<A14>::type
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
            detail::update_initialized_count();
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
