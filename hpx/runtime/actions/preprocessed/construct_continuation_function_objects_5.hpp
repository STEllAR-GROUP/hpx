// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


namespace detail
{
    
    
    
    
    
    
    template <typename Action>
    struct continuation_thread_function_void_0
    {
        typedef threads::thread_state_enum result_type;
        template <typename Func
           >
        BOOST_FORCEINLINE result_type operator()(
            continuation_type cont, Func const& func
           ) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<Action>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                func();
                cont->trigger();
            }
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    
    
    
    template <typename Action>
    struct construct_continuation_thread_function_voidN<Action, 0>
    {
        template <typename Func, typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        call(continuation_type cont, BOOST_FWD_REF(Func) func,
            BOOST_FWD_REF(Arguments) args)
        {
            return HPX_STD_BIND(
                continuation_thread_function_void_0<Action>(),
                cont, boost::forward<Func>(func)
              
                    );
        }
    };
    
    template <typename Action>
    struct continuation_thread_function_0
    {
        typedef threads::thread_state_enum result_type;
        template <typename Func
           >
        BOOST_FORCEINLINE result_type operator()(
            continuation_type cont, Func const& func
           ) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<Action>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                cont->trigger(boost::move(
                    func()
                ));
            }
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    template <typename Action>
    struct construct_continuation_thread_functionN<Action, 0>
    {
        template <typename Func, typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        call(continuation_type cont, BOOST_FWD_REF(Func) func,
            BOOST_FWD_REF(Arguments) args)
        {
            return HPX_STD_BIND(
                continuation_thread_function_0<Action>(),
                cont, boost::forward<Func>(func)
              
                    );
        }
    };
}
namespace detail
{
    
    
    
    
    
    
    template <typename Action>
    struct continuation_thread_function_void_1
    {
        typedef threads::thread_state_enum result_type;
        template <typename Func
          , typename Arg0>
        BOOST_FORCEINLINE result_type operator()(
            continuation_type cont, Func const& func
          , BOOST_FWD_REF(Arg0) arg0) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<Action>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                func(boost::move(arg0));
                cont->trigger();
            }
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    
    
    
    template <typename Action>
    struct construct_continuation_thread_function_voidN<Action, 1>
    {
        template <typename Func, typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        call(continuation_type cont, BOOST_FWD_REF(Func) func,
            BOOST_FWD_REF(Arguments) args)
        {
            return HPX_STD_BIND(
                continuation_thread_function_void_1<Action>(),
                cont, boost::forward<Func>(func)
              ,
                    util::detail::move_if_no_ref< typename util::remove_reference<Arguments>::type:: member_type0>::call( args. a0));
        }
    };
    
    template <typename Action>
    struct continuation_thread_function_1
    {
        typedef threads::thread_state_enum result_type;
        template <typename Func
          , typename Arg0>
        BOOST_FORCEINLINE result_type operator()(
            continuation_type cont, Func const& func
          , BOOST_FWD_REF(Arg0) arg0) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<Action>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                cont->trigger(boost::move(
                    func(boost::move(arg0))
                ));
            }
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    template <typename Action>
    struct construct_continuation_thread_functionN<Action, 1>
    {
        template <typename Func, typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        call(continuation_type cont, BOOST_FWD_REF(Func) func,
            BOOST_FWD_REF(Arguments) args)
        {
            return HPX_STD_BIND(
                continuation_thread_function_1<Action>(),
                cont, boost::forward<Func>(func)
              ,
                    util::detail::move_if_no_ref< typename util::remove_reference<Arguments>::type:: member_type0>::call( args. a0));
        }
    };
}
namespace detail
{
    
    
    
    
    
    
    template <typename Action>
    struct continuation_thread_function_void_2
    {
        typedef threads::thread_state_enum result_type;
        template <typename Func
          , typename Arg0 , typename Arg1>
        BOOST_FORCEINLINE result_type operator()(
            continuation_type cont, Func const& func
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<Action>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                func(boost::move(arg0) , boost::move(arg1));
                cont->trigger();
            }
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    
    
    
    template <typename Action>
    struct construct_continuation_thread_function_voidN<Action, 2>
    {
        template <typename Func, typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        call(continuation_type cont, BOOST_FWD_REF(Func) func,
            BOOST_FWD_REF(Arguments) args)
        {
            return HPX_STD_BIND(
                continuation_thread_function_void_2<Action>(),
                cont, boost::forward<Func>(func)
              ,
                    util::detail::move_if_no_ref< typename util::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::remove_reference<Arguments>::type:: member_type1>::call( args. a1));
        }
    };
    
    template <typename Action>
    struct continuation_thread_function_2
    {
        typedef threads::thread_state_enum result_type;
        template <typename Func
          , typename Arg0 , typename Arg1>
        BOOST_FORCEINLINE result_type operator()(
            continuation_type cont, Func const& func
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<Action>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                cont->trigger(boost::move(
                    func(boost::move(arg0) , boost::move(arg1))
                ));
            }
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    template <typename Action>
    struct construct_continuation_thread_functionN<Action, 2>
    {
        template <typename Func, typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        call(continuation_type cont, BOOST_FWD_REF(Func) func,
            BOOST_FWD_REF(Arguments) args)
        {
            return HPX_STD_BIND(
                continuation_thread_function_2<Action>(),
                cont, boost::forward<Func>(func)
              ,
                    util::detail::move_if_no_ref< typename util::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::remove_reference<Arguments>::type:: member_type1>::call( args. a1));
        }
    };
}
namespace detail
{
    
    
    
    
    
    
    template <typename Action>
    struct continuation_thread_function_void_3
    {
        typedef threads::thread_state_enum result_type;
        template <typename Func
          , typename Arg0 , typename Arg1 , typename Arg2>
        BOOST_FORCEINLINE result_type operator()(
            continuation_type cont, Func const& func
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<Action>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                func(boost::move(arg0) , boost::move(arg1) , boost::move(arg2));
                cont->trigger();
            }
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    
    
    
    template <typename Action>
    struct construct_continuation_thread_function_voidN<Action, 3>
    {
        template <typename Func, typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        call(continuation_type cont, BOOST_FWD_REF(Func) func,
            BOOST_FWD_REF(Arguments) args)
        {
            return HPX_STD_BIND(
                continuation_thread_function_void_3<Action>(),
                cont, boost::forward<Func>(func)
              ,
                    util::detail::move_if_no_ref< typename util::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::remove_reference<Arguments>::type:: member_type2>::call( args. a2));
        }
    };
    
    template <typename Action>
    struct continuation_thread_function_3
    {
        typedef threads::thread_state_enum result_type;
        template <typename Func
          , typename Arg0 , typename Arg1 , typename Arg2>
        BOOST_FORCEINLINE result_type operator()(
            continuation_type cont, Func const& func
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<Action>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                cont->trigger(boost::move(
                    func(boost::move(arg0) , boost::move(arg1) , boost::move(arg2))
                ));
            }
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    template <typename Action>
    struct construct_continuation_thread_functionN<Action, 3>
    {
        template <typename Func, typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        call(continuation_type cont, BOOST_FWD_REF(Func) func,
            BOOST_FWD_REF(Arguments) args)
        {
            return HPX_STD_BIND(
                continuation_thread_function_3<Action>(),
                cont, boost::forward<Func>(func)
              ,
                    util::detail::move_if_no_ref< typename util::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::remove_reference<Arguments>::type:: member_type2>::call( args. a2));
        }
    };
}
namespace detail
{
    
    
    
    
    
    
    template <typename Action>
    struct continuation_thread_function_void_4
    {
        typedef threads::thread_state_enum result_type;
        template <typename Func
          , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
        BOOST_FORCEINLINE result_type operator()(
            continuation_type cont, Func const& func
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<Action>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                func(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3));
                cont->trigger();
            }
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    
    
    
    template <typename Action>
    struct construct_continuation_thread_function_voidN<Action, 4>
    {
        template <typename Func, typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        call(continuation_type cont, BOOST_FWD_REF(Func) func,
            BOOST_FWD_REF(Arguments) args)
        {
            return HPX_STD_BIND(
                continuation_thread_function_void_4<Action>(),
                cont, boost::forward<Func>(func)
              ,
                    util::detail::move_if_no_ref< typename util::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::remove_reference<Arguments>::type:: member_type3>::call( args. a3));
        }
    };
    
    template <typename Action>
    struct continuation_thread_function_4
    {
        typedef threads::thread_state_enum result_type;
        template <typename Func
          , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
        BOOST_FORCEINLINE result_type operator()(
            continuation_type cont, Func const& func
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<Action>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                cont->trigger(boost::move(
                    func(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3))
                ));
            }
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    template <typename Action>
    struct construct_continuation_thread_functionN<Action, 4>
    {
        template <typename Func, typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        call(continuation_type cont, BOOST_FWD_REF(Func) func,
            BOOST_FWD_REF(Arguments) args)
        {
            return HPX_STD_BIND(
                continuation_thread_function_4<Action>(),
                cont, boost::forward<Func>(func)
              ,
                    util::detail::move_if_no_ref< typename util::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::remove_reference<Arguments>::type:: member_type3>::call( args. a3));
        }
    };
}
namespace detail
{
    
    
    
    
    
    
    template <typename Action>
    struct continuation_thread_function_void_5
    {
        typedef threads::thread_state_enum result_type;
        template <typename Func
          , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
        BOOST_FORCEINLINE result_type operator()(
            continuation_type cont, Func const& func
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<Action>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                func(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4));
                cont->trigger();
            }
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    
    
    
    template <typename Action>
    struct construct_continuation_thread_function_voidN<Action, 5>
    {
        template <typename Func, typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        call(continuation_type cont, BOOST_FWD_REF(Func) func,
            BOOST_FWD_REF(Arguments) args)
        {
            return HPX_STD_BIND(
                continuation_thread_function_void_5<Action>(),
                cont, boost::forward<Func>(func)
              ,
                    util::detail::move_if_no_ref< typename util::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::remove_reference<Arguments>::type:: member_type4>::call( args. a4));
        }
    };
    
    template <typename Action>
    struct continuation_thread_function_5
    {
        typedef threads::thread_state_enum result_type;
        template <typename Func
          , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
        BOOST_FORCEINLINE result_type operator()(
            continuation_type cont, Func const& func
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<Action>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                cont->trigger(boost::move(
                    func(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4))
                ));
            }
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    template <typename Action>
    struct construct_continuation_thread_functionN<Action, 5>
    {
        template <typename Func, typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        call(continuation_type cont, BOOST_FWD_REF(Func) func,
            BOOST_FWD_REF(Arguments) args)
        {
            return HPX_STD_BIND(
                continuation_thread_function_5<Action>(),
                cont, boost::forward<Func>(func)
              ,
                    util::detail::move_if_no_ref< typename util::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::remove_reference<Arguments>::type:: member_type4>::call( args. a4));
        }
    };
}
