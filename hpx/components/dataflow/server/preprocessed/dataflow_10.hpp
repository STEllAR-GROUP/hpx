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
          , A0 && a0 
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
            (*w)->init(std::forward<A0>(a0));
            detail::update_initialized_count();
        }
        template <typename Action, typename A0>
        dataflow(
            component_type * back_ptr
          , Action
          , naming::id_type const & target
          , A0 && a0 
        )
            : base_type(back_ptr)
            , component_ptr(0)
        {
            
            init<typename Action::type>(target, std::forward<A0>(a0));
        }
        
        template <typename Action, typename A0 , typename A1>
        void init(
            naming::id_type const & target
          , A0 && a0 , A1 && a1 
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
            (*w)->init(std::forward<A0>(a0) , std::forward<A1>(a1));
            detail::update_initialized_count();
        }
        template <typename Action, typename A0 , typename A1>
        dataflow(
            component_type * back_ptr
          , Action
          , naming::id_type const & target
          , A0 && a0 , A1 && a1 
        )
            : base_type(back_ptr)
            , component_ptr(0)
        {
            
            init<typename Action::type>(target, std::forward<A0>(a0) , std::forward<A1>(a1));
        }
        
        template <typename Action, typename A0 , typename A1 , typename A2>
        void init(
            naming::id_type const & target
          , A0 && a0 , A1 && a1 , A2 && a2 
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
            (*w)->init(std::forward<A0>(a0) , std::forward<A1>(a1) , std::forward<A2>(a2));
            detail::update_initialized_count();
        }
        template <typename Action, typename A0 , typename A1 , typename A2>
        dataflow(
            component_type * back_ptr
          , Action
          , naming::id_type const & target
          , A0 && a0 , A1 && a1 , A2 && a2 
        )
            : base_type(back_ptr)
            , component_ptr(0)
        {
            
            init<typename Action::type>(target, std::forward<A0>(a0) , std::forward<A1>(a1) , std::forward<A2>(a2));
        }
        
        template <typename Action, typename A0 , typename A1 , typename A2 , typename A3>
        void init(
            naming::id_type const & target
          , A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 
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
            (*w)->init(std::forward<A0>(a0) , std::forward<A1>(a1) , std::forward<A2>(a2) , std::forward<A3>(a3));
            detail::update_initialized_count();
        }
        template <typename Action, typename A0 , typename A1 , typename A2 , typename A3>
        dataflow(
            component_type * back_ptr
          , Action
          , naming::id_type const & target
          , A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 
        )
            : base_type(back_ptr)
            , component_ptr(0)
        {
            
            init<typename Action::type>(target, std::forward<A0>(a0) , std::forward<A1>(a1) , std::forward<A2>(a2) , std::forward<A3>(a3));
        }
        
        template <typename Action, typename A0 , typename A1 , typename A2 , typename A3 , typename A4>
        void init(
            naming::id_type const & target
          , A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 
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
            (*w)->init(std::forward<A0>(a0) , std::forward<A1>(a1) , std::forward<A2>(a2) , std::forward<A3>(a3) , std::forward<A4>(a4));
            detail::update_initialized_count();
        }
        template <typename Action, typename A0 , typename A1 , typename A2 , typename A3 , typename A4>
        dataflow(
            component_type * back_ptr
          , Action
          , naming::id_type const & target
          , A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 
        )
            : base_type(back_ptr)
            , component_ptr(0)
        {
            
            init<typename Action::type>(target, std::forward<A0>(a0) , std::forward<A1>(a1) , std::forward<A2>(a2) , std::forward<A3>(a3) , std::forward<A4>(a4));
        }
        
        template <typename Action, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5>
        void init(
            naming::id_type const & target
          , A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 
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
            (*w)->init(std::forward<A0>(a0) , std::forward<A1>(a1) , std::forward<A2>(a2) , std::forward<A3>(a3) , std::forward<A4>(a4) , std::forward<A5>(a5));
            detail::update_initialized_count();
        }
        template <typename Action, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5>
        dataflow(
            component_type * back_ptr
          , Action
          , naming::id_type const & target
          , A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 
        )
            : base_type(back_ptr)
            , component_ptr(0)
        {
            
            init<typename Action::type>(target, std::forward<A0>(a0) , std::forward<A1>(a1) , std::forward<A2>(a2) , std::forward<A3>(a3) , std::forward<A4>(a4) , std::forward<A5>(a5));
        }
        
        template <typename Action, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6>
        void init(
            naming::id_type const & target
          , A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 
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
            (*w)->init(std::forward<A0>(a0) , std::forward<A1>(a1) , std::forward<A2>(a2) , std::forward<A3>(a3) , std::forward<A4>(a4) , std::forward<A5>(a5) , std::forward<A6>(a6));
            detail::update_initialized_count();
        }
        template <typename Action, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6>
        dataflow(
            component_type * back_ptr
          , Action
          , naming::id_type const & target
          , A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 
        )
            : base_type(back_ptr)
            , component_ptr(0)
        {
            
            init<typename Action::type>(target, std::forward<A0>(a0) , std::forward<A1>(a1) , std::forward<A2>(a2) , std::forward<A3>(a3) , std::forward<A4>(a4) , std::forward<A5>(a5) , std::forward<A6>(a6));
        }
        
        template <typename Action, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7>
        void init(
            naming::id_type const & target
          , A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 
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
            (*w)->init(std::forward<A0>(a0) , std::forward<A1>(a1) , std::forward<A2>(a2) , std::forward<A3>(a3) , std::forward<A4>(a4) , std::forward<A5>(a5) , std::forward<A6>(a6) , std::forward<A7>(a7));
            detail::update_initialized_count();
        }
        template <typename Action, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7>
        dataflow(
            component_type * back_ptr
          , Action
          , naming::id_type const & target
          , A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 
        )
            : base_type(back_ptr)
            , component_ptr(0)
        {
            
            init<typename Action::type>(target, std::forward<A0>(a0) , std::forward<A1>(a1) , std::forward<A2>(a2) , std::forward<A3>(a3) , std::forward<A4>(a4) , std::forward<A5>(a5) , std::forward<A6>(a6) , std::forward<A7>(a7));
        }
        
        template <typename Action, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8>
        void init(
            naming::id_type const & target
          , A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 , A8 && a8 
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
            (*w)->init(std::forward<A0>(a0) , std::forward<A1>(a1) , std::forward<A2>(a2) , std::forward<A3>(a3) , std::forward<A4>(a4) , std::forward<A5>(a5) , std::forward<A6>(a6) , std::forward<A7>(a7) , std::forward<A8>(a8));
            detail::update_initialized_count();
        }
        template <typename Action, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8>
        dataflow(
            component_type * back_ptr
          , Action
          , naming::id_type const & target
          , A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 , A8 && a8 
        )
            : base_type(back_ptr)
            , component_ptr(0)
        {
            
            init<typename Action::type>(target, std::forward<A0>(a0) , std::forward<A1>(a1) , std::forward<A2>(a2) , std::forward<A3>(a3) , std::forward<A4>(a4) , std::forward<A5>(a5) , std::forward<A6>(a6) , std::forward<A7>(a7) , std::forward<A8>(a8));
        }
        
        template <typename Action, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9>
        void init(
            naming::id_type const & target
          , A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 , A8 && a8 , A9 && a9 
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
            (*w)->init(std::forward<A0>(a0) , std::forward<A1>(a1) , std::forward<A2>(a2) , std::forward<A3>(a3) , std::forward<A4>(a4) , std::forward<A5>(a5) , std::forward<A6>(a6) , std::forward<A7>(a7) , std::forward<A8>(a8) , std::forward<A9>(a9));
            detail::update_initialized_count();
        }
        template <typename Action, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9>
        dataflow(
            component_type * back_ptr
          , Action
          , naming::id_type const & target
          , A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 , A8 && a8 , A9 && a9 
        )
            : base_type(back_ptr)
            , component_ptr(0)
        {
            
            init<typename Action::type>(target, std::forward<A0>(a0) , std::forward<A1>(a1) , std::forward<A2>(a2) , std::forward<A3>(a3) , std::forward<A4>(a4) , std::forward<A5>(a5) , std::forward<A6>(a6) , std::forward<A7>(a7) , std::forward<A8>(a8) , std::forward<A9>(a9));
        }
