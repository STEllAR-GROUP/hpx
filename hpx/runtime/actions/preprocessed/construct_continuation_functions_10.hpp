// Copyright (c) 2007-2012 Hartmut Kaiser
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


    
    
    struct continuation_thread_object_function_void_0
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
             
             >
        result_type operator()(continuation_type cont,
            void (Object::* func)(),
            Object* obj
           ) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                (obj->*func)();
                cont->trigger();
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
             
             >
        result_type operator()(continuation_type cont,
            void (Object::* const func)(
                ) const,
            Component* obj
           ) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                (obj->*func)();
                cont->trigger();
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    template <typename Object, typename Arguments_
         >
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* func)(), Object* obj,
        BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_void_0(),
            cont, func, obj
          
                );
    }
    template <typename Object, typename Arguments_
         >
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* const func)() const,
        Component* obj, BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_void_0(),
            cont, func, obj
          
                );
    }
    
    struct continuation_thread_object_function_0
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
             
             >
        result_type operator()(continuation_type cont,
            Result (Object::* func)(),
            Component* obj
           ) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                cont->trigger(boost::move(
                    (obj->*func)()
                ));
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
             
             >
        result_type operator()(continuation_type cont,
            Result (Object::* const func)(
                ) const,
            Component* obj
           ) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                cont->trigger(boost::move(
                    (obj->*func)()
                ));
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    template <typename Object, typename Arguments_
         >
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* func)(), Component* obj,
        BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_0(),
            cont, func, obj
          
                );
    }
    template <typename Object, typename Arguments_
         >
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* const func)() const,
        Component* obj, BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_0(),
            cont, func, obj
          
                );
    }
    
    
    struct continuation_thread_object_function_void_1
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0
            , typename FArg0>
        result_type operator()(continuation_type cont,
            void (Object::* func)(FArg0 arg0),
            Object* obj
          , BOOST_FWD_REF(Arg0) arg0) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                (obj->*func)(boost::move(arg0));
                cont->trigger();
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0
            , typename FArg0>
        result_type operator()(continuation_type cont,
            void (Object::* const func)(
                FArg0 arg0) const,
            Component* obj
          , BOOST_FWD_REF(Arg0) arg0) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                (obj->*func)(boost::move(arg0));
                cont->trigger();
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    template <typename Object, typename Arguments_
        , typename FArg0>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* func)(FArg0), Object* obj,
        BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_void_1(),
            cont, func, obj
          ,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0));
    }
    template <typename Object, typename Arguments_
        , typename FArg0>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* const func)(FArg0) const,
        Component* obj, BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_void_1(),
            cont, func, obj
          ,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0));
    }
    
    struct continuation_thread_object_function_1
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0
            , typename FArg0>
        result_type operator()(continuation_type cont,
            Result (Object::* func)(FArg0 arg0),
            Component* obj
          , BOOST_FWD_REF(Arg0) arg0) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                cont->trigger(boost::move(
                    (obj->*func)(boost::move(arg0))
                ));
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0
            , typename FArg0>
        result_type operator()(continuation_type cont,
            Result (Object::* const func)(
                FArg0 arg0) const,
            Component* obj
          , BOOST_FWD_REF(Arg0) arg0) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                cont->trigger(boost::move(
                    (obj->*func)(boost::move(arg0))
                ));
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    template <typename Object, typename Arguments_
        , typename FArg0>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* func)(FArg0), Component* obj,
        BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_1(),
            cont, func, obj
          ,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0));
    }
    template <typename Object, typename Arguments_
        , typename FArg0>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* const func)(FArg0) const,
        Component* obj, BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_1(),
            cont, func, obj
          ,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0));
    }
    
    
    struct continuation_thread_object_function_void_2
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1
            , typename FArg0 , typename FArg1>
        result_type operator()(continuation_type cont,
            void (Object::* func)(FArg0 arg0 , FArg1 arg1),
            Object* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                (obj->*func)(boost::move(arg0) , boost::move(arg1));
                cont->trigger();
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1
            , typename FArg0 , typename FArg1>
        result_type operator()(continuation_type cont,
            void (Object::* const func)(
                FArg0 arg0 , FArg1 arg1) const,
            Component* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                (obj->*func)(boost::move(arg0) , boost::move(arg1));
                cont->trigger();
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* func)(FArg0 , FArg1), Object* obj,
        BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_void_2(),
            cont, func, obj
          ,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* const func)(FArg0 , FArg1) const,
        Component* obj, BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_void_2(),
            cont, func, obj
          ,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1));
    }
    
    struct continuation_thread_object_function_2
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1
            , typename FArg0 , typename FArg1>
        result_type operator()(continuation_type cont,
            Result (Object::* func)(FArg0 arg0 , FArg1 arg1),
            Component* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                cont->trigger(boost::move(
                    (obj->*func)(boost::move(arg0) , boost::move(arg1))
                ));
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1
            , typename FArg0 , typename FArg1>
        result_type operator()(continuation_type cont,
            Result (Object::* const func)(
                FArg0 arg0 , FArg1 arg1) const,
            Component* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                cont->trigger(boost::move(
                    (obj->*func)(boost::move(arg0) , boost::move(arg1))
                ));
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* func)(FArg0 , FArg1), Component* obj,
        BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_2(),
            cont, func, obj
          ,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* const func)(FArg0 , FArg1) const,
        Component* obj, BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_2(),
            cont, func, obj
          ,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1));
    }
    
    
    struct continuation_thread_object_function_void_3
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2
            , typename FArg0 , typename FArg1 , typename FArg2>
        result_type operator()(continuation_type cont,
            void (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2),
            Object* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2));
                cont->trigger();
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2
            , typename FArg0 , typename FArg1 , typename FArg2>
        result_type operator()(continuation_type cont,
            void (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2) const,
            Component* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2));
                cont->trigger();
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* func)(FArg0 , FArg1 , FArg2), Object* obj,
        BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_void_3(),
            cont, func, obj
          ,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* const func)(FArg0 , FArg1 , FArg2) const,
        Component* obj, BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_void_3(),
            cont, func, obj
          ,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2));
    }
    
    struct continuation_thread_object_function_3
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2
            , typename FArg0 , typename FArg1 , typename FArg2>
        result_type operator()(continuation_type cont,
            Result (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2),
            Component* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                cont->trigger(boost::move(
                    (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2))
                ));
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2
            , typename FArg0 , typename FArg1 , typename FArg2>
        result_type operator()(continuation_type cont,
            Result (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2) const,
            Component* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                cont->trigger(boost::move(
                    (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2))
                ));
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* func)(FArg0 , FArg1 , FArg2), Component* obj,
        BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_3(),
            cont, func, obj
          ,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* const func)(FArg0 , FArg1 , FArg2) const,
        Component* obj, BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_3(),
            cont, func, obj
          ,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2));
    }
    
    
    struct continuation_thread_object_function_void_4
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3>
        result_type operator()(continuation_type cont,
            void (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3),
            Object* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3));
                cont->trigger();
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3>
        result_type operator()(continuation_type cont,
            void (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3) const,
            Component* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3));
                cont->trigger();
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3), Object* obj,
        BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_void_4(),
            cont, func, obj
          ,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3) const,
        Component* obj, BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_void_4(),
            cont, func, obj
          ,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3));
    }
    
    struct continuation_thread_object_function_4
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3>
        result_type operator()(continuation_type cont,
            Result (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3),
            Component* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                cont->trigger(boost::move(
                    (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3))
                ));
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3>
        result_type operator()(continuation_type cont,
            Result (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3) const,
            Component* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                cont->trigger(boost::move(
                    (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3))
                ));
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3), Component* obj,
        BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_4(),
            cont, func, obj
          ,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3) const,
        Component* obj, BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_4(),
            cont, func, obj
          ,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3));
    }
    
    
    struct continuation_thread_object_function_void_5
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4>
        result_type operator()(continuation_type cont,
            void (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4),
            Object* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4));
                cont->trigger();
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4>
        result_type operator()(continuation_type cont,
            void (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4) const,
            Component* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4));
                cont->trigger();
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4), Object* obj,
        BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_void_5(),
            cont, func, obj
          ,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4) const,
        Component* obj, BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_void_5(),
            cont, func, obj
          ,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4));
    }
    
    struct continuation_thread_object_function_5
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4>
        result_type operator()(continuation_type cont,
            Result (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4),
            Component* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                cont->trigger(boost::move(
                    (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4))
                ));
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4>
        result_type operator()(continuation_type cont,
            Result (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4) const,
            Component* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                cont->trigger(boost::move(
                    (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4))
                ));
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4), Component* obj,
        BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_5(),
            cont, func, obj
          ,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4) const,
        Component* obj, BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_5(),
            cont, func, obj
          ,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4));
    }
    
    
    struct continuation_thread_object_function_void_6
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5>
        result_type operator()(continuation_type cont,
            void (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5),
            Object* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5));
                cont->trigger();
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5>
        result_type operator()(continuation_type cont,
            void (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5) const,
            Component* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5));
                cont->trigger();
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5), Object* obj,
        BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_void_6(),
            cont, func, obj
          ,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type5>::call( args. a5));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5) const,
        Component* obj, BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_void_6(),
            cont, func, obj
          ,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type5>::call( args. a5));
    }
    
    struct continuation_thread_object_function_6
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5>
        result_type operator()(continuation_type cont,
            Result (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5),
            Component* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                cont->trigger(boost::move(
                    (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5))
                ));
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5>
        result_type operator()(continuation_type cont,
            Result (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5) const,
            Component* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                cont->trigger(boost::move(
                    (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5))
                ));
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5), Component* obj,
        BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_6(),
            cont, func, obj
          ,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type5>::call( args. a5));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5) const,
        Component* obj, BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_6(),
            cont, func, obj
          ,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type5>::call( args. a5));
    }
    
    
    struct continuation_thread_object_function_void_7
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6>
        result_type operator()(continuation_type cont,
            void (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6),
            Object* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6));
                cont->trigger();
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6>
        result_type operator()(continuation_type cont,
            void (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6) const,
            Component* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6));
                cont->trigger();
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6), Object* obj,
        BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_void_7(),
            cont, func, obj
          ,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type6>::call( args. a6));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6) const,
        Component* obj, BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_void_7(),
            cont, func, obj
          ,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type6>::call( args. a6));
    }
    
    struct continuation_thread_object_function_7
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6>
        result_type operator()(continuation_type cont,
            Result (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6),
            Component* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                cont->trigger(boost::move(
                    (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6))
                ));
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6>
        result_type operator()(continuation_type cont,
            Result (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6) const,
            Component* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                cont->trigger(boost::move(
                    (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6))
                ));
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6), Component* obj,
        BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_7(),
            cont, func, obj
          ,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type6>::call( args. a6));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6) const,
        Component* obj, BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_7(),
            cont, func, obj
          ,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type6>::call( args. a6));
    }
    
    
    struct continuation_thread_object_function_void_8
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7>
        result_type operator()(continuation_type cont,
            void (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7),
            Object* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7));
                cont->trigger();
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7>
        result_type operator()(continuation_type cont,
            void (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7) const,
            Component* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7));
                cont->trigger();
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7), Object* obj,
        BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_void_8(),
            cont, func, obj
          ,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type7>::call( args. a7));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7) const,
        Component* obj, BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_void_8(),
            cont, func, obj
          ,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type7>::call( args. a7));
    }
    
    struct continuation_thread_object_function_8
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7>
        result_type operator()(continuation_type cont,
            Result (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7),
            Component* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                cont->trigger(boost::move(
                    (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7))
                ));
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7>
        result_type operator()(continuation_type cont,
            Result (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7) const,
            Component* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                cont->trigger(boost::move(
                    (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7))
                ));
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7), Component* obj,
        BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_8(),
            cont, func, obj
          ,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type7>::call( args. a7));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7) const,
        Component* obj, BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_8(),
            cont, func, obj
          ,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type7>::call( args. a7));
    }
    
    
    struct continuation_thread_object_function_void_9
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8>
        result_type operator()(continuation_type cont,
            void (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8),
            Object* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8));
                cont->trigger();
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8>
        result_type operator()(continuation_type cont,
            void (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8) const,
            Component* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8));
                cont->trigger();
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8), Object* obj,
        BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_void_9(),
            cont, func, obj
          ,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type8>::call( args. a8));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8) const,
        Component* obj, BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_void_9(),
            cont, func, obj
          ,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type8>::call( args. a8));
    }
    
    struct continuation_thread_object_function_9
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8>
        result_type operator()(continuation_type cont,
            Result (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8),
            Component* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                cont->trigger(boost::move(
                    (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8))
                ));
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8>
        result_type operator()(continuation_type cont,
            Result (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8) const,
            Component* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                cont->trigger(boost::move(
                    (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8))
                ));
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8), Component* obj,
        BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_9(),
            cont, func, obj
          ,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type8>::call( args. a8));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8) const,
        Component* obj, BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_9(),
            cont, func, obj
          ,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type8>::call( args. a8));
    }
    
    
    struct continuation_thread_object_function_void_10
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9>
        result_type operator()(continuation_type cont,
            void (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9),
            Object* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8) , boost::move(arg9));
                cont->trigger();
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9>
        result_type operator()(continuation_type cont,
            void (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9) const,
            Component* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8) , boost::move(arg9));
                cont->trigger();
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9), Object* obj,
        BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_void_10(),
            cont, func, obj
          ,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type9>::call( args. a9));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9) const,
        Component* obj, BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_void_10(),
            cont, func, obj
          ,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type9>::call( args. a9));
    }
    
    struct continuation_thread_object_function_10
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9>
        result_type operator()(continuation_type cont,
            Result (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9),
            Component* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                cont->trigger(boost::move(
                    (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8) , boost::move(arg9))
                ));
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9>
        result_type operator()(continuation_type cont,
            Result (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9) const,
            Component* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                cont->trigger(boost::move(
                    (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8) , boost::move(arg9))
                ));
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9), Component* obj,
        BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_10(),
            cont, func, obj
          ,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type9>::call( args. a9));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9) const,
        Component* obj, BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_10(),
            cont, func, obj
          ,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type9>::call( args. a9));
    }
