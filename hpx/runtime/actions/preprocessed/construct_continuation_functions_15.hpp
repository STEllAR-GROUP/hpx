// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
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
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
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
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
             
             >
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
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
            catch (...) {
                
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
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
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
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
             
             >
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
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
            catch (...) {
                
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
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
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
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0
            , typename FArg0>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
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
            catch (...) {
                
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
                boost::forward< typename util::remove_reference<Arguments_>::type:: member_type0>( args. a0));
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
                boost::forward< typename util::remove_reference<Arguments_>::type:: member_type0>( args. a0));
    }
    
    struct continuation_thread_object_function_1
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0
            , typename FArg0>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
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
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0
            , typename FArg0>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
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
            catch (...) {
                
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
                boost::forward< typename util::remove_reference<Arguments_>::type:: member_type0>( args. a0));
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
                boost::forward< typename util::remove_reference<Arguments_>::type:: member_type0>( args. a0));
    }
    
    
    struct continuation_thread_object_function_void_2
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1
            , typename FArg0 , typename FArg1>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
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
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1
            , typename FArg0 , typename FArg1>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
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
            catch (...) {
                
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
                boost::forward< typename util::remove_reference<Arguments_>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type1>( args. a1));
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
                boost::forward< typename util::remove_reference<Arguments_>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type1>( args. a1));
    }
    
    struct continuation_thread_object_function_2
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1
            , typename FArg0 , typename FArg1>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
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
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1
            , typename FArg0 , typename FArg1>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
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
            catch (...) {
                
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
                boost::forward< typename util::remove_reference<Arguments_>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type1>( args. a1));
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
                boost::forward< typename util::remove_reference<Arguments_>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type1>( args. a1));
    }
    
    
    struct continuation_thread_object_function_void_3
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2
            , typename FArg0 , typename FArg1 , typename FArg2>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
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
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2
            , typename FArg0 , typename FArg1 , typename FArg2>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
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
            catch (...) {
                
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
                boost::forward< typename util::remove_reference<Arguments_>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type2>( args. a2));
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
                boost::forward< typename util::remove_reference<Arguments_>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type2>( args. a2));
    }
    
    struct continuation_thread_object_function_3
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2
            , typename FArg0 , typename FArg1 , typename FArg2>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
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
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2
            , typename FArg0 , typename FArg1 , typename FArg2>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
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
            catch (...) {
                
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
                boost::forward< typename util::remove_reference<Arguments_>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type2>( args. a2));
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
                boost::forward< typename util::remove_reference<Arguments_>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type2>( args. a2));
    }
    
    
    struct continuation_thread_object_function_void_4
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
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
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
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
            catch (...) {
                
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
                boost::forward< typename util::remove_reference<Arguments_>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type2>( args. a2) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type3>( args. a3));
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
                boost::forward< typename util::remove_reference<Arguments_>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type2>( args. a2) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type3>( args. a3));
    }
    
    struct continuation_thread_object_function_4
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
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
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
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
            catch (...) {
                
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
                boost::forward< typename util::remove_reference<Arguments_>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type2>( args. a2) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type3>( args. a3));
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
                boost::forward< typename util::remove_reference<Arguments_>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type2>( args. a2) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type3>( args. a3));
    }
    
    
    struct continuation_thread_object_function_void_5
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
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
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
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
            catch (...) {
                
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
                boost::forward< typename util::remove_reference<Arguments_>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type2>( args. a2) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type3>( args. a3) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type4>( args. a4));
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
                boost::forward< typename util::remove_reference<Arguments_>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type2>( args. a2) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type3>( args. a3) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type4>( args. a4));
    }
    
    struct continuation_thread_object_function_5
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
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
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
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
            catch (...) {
                
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
                boost::forward< typename util::remove_reference<Arguments_>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type2>( args. a2) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type3>( args. a3) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type4>( args. a4));
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
                boost::forward< typename util::remove_reference<Arguments_>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type2>( args. a2) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type3>( args. a3) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type4>( args. a4));
    }
    
    
    struct continuation_thread_object_function_void_6
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
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
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
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
            catch (...) {
                
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
                boost::forward< typename util::remove_reference<Arguments_>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type2>( args. a2) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type3>( args. a3) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type4>( args. a4) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type5>( args. a5));
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
                boost::forward< typename util::remove_reference<Arguments_>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type2>( args. a2) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type3>( args. a3) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type4>( args. a4) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type5>( args. a5));
    }
    
    struct continuation_thread_object_function_6
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
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
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
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
            catch (...) {
                
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
                boost::forward< typename util::remove_reference<Arguments_>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type2>( args. a2) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type3>( args. a3) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type4>( args. a4) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type5>( args. a5));
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
                boost::forward< typename util::remove_reference<Arguments_>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type2>( args. a2) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type3>( args. a3) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type4>( args. a4) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type5>( args. a5));
    }
    
    
    struct continuation_thread_object_function_void_7
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
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
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
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
            catch (...) {
                
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
                boost::forward< typename util::remove_reference<Arguments_>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type2>( args. a2) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type3>( args. a3) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type4>( args. a4) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type5>( args. a5) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type6>( args. a6));
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
                boost::forward< typename util::remove_reference<Arguments_>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type2>( args. a2) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type3>( args. a3) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type4>( args. a4) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type5>( args. a5) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type6>( args. a6));
    }
    
    struct continuation_thread_object_function_7
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
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
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
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
            catch (...) {
                
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
                boost::forward< typename util::remove_reference<Arguments_>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type2>( args. a2) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type3>( args. a3) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type4>( args. a4) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type5>( args. a5) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type6>( args. a6));
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
                boost::forward< typename util::remove_reference<Arguments_>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type2>( args. a2) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type3>( args. a3) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type4>( args. a4) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type5>( args. a5) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type6>( args. a6));
    }
    
    
    struct continuation_thread_object_function_void_8
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
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
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
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
            catch (...) {
                
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
                boost::forward< typename util::remove_reference<Arguments_>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type2>( args. a2) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type3>( args. a3) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type4>( args. a4) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type5>( args. a5) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type6>( args. a6) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type7>( args. a7));
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
                boost::forward< typename util::remove_reference<Arguments_>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type2>( args. a2) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type3>( args. a3) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type4>( args. a4) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type5>( args. a5) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type6>( args. a6) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type7>( args. a7));
    }
    
    struct continuation_thread_object_function_8
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
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
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
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
            catch (...) {
                
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
                boost::forward< typename util::remove_reference<Arguments_>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type2>( args. a2) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type3>( args. a3) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type4>( args. a4) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type5>( args. a5) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type6>( args. a6) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type7>( args. a7));
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
                boost::forward< typename util::remove_reference<Arguments_>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type2>( args. a2) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type3>( args. a3) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type4>( args. a4) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type5>( args. a5) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type6>( args. a6) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type7>( args. a7));
    }
    
    
    struct continuation_thread_object_function_void_9
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
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
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
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
            catch (...) {
                
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
                boost::forward< typename util::remove_reference<Arguments_>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type2>( args. a2) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type3>( args. a3) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type4>( args. a4) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type5>( args. a5) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type6>( args. a6) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type7>( args. a7) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type8>( args. a8));
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
                boost::forward< typename util::remove_reference<Arguments_>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type2>( args. a2) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type3>( args. a3) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type4>( args. a4) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type5>( args. a5) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type6>( args. a6) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type7>( args. a7) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type8>( args. a8));
    }
    
    struct continuation_thread_object_function_9
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
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
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
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
            catch (...) {
                
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
                boost::forward< typename util::remove_reference<Arguments_>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type2>( args. a2) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type3>( args. a3) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type4>( args. a4) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type5>( args. a5) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type6>( args. a6) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type7>( args. a7) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type8>( args. a8));
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
                boost::forward< typename util::remove_reference<Arguments_>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type2>( args. a2) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type3>( args. a3) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type4>( args. a4) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type5>( args. a5) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type6>( args. a6) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type7>( args. a7) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type8>( args. a8));
    }
    
    
    struct continuation_thread_object_function_void_10
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
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
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
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
            catch (...) {
                
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
                boost::forward< typename util::remove_reference<Arguments_>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type2>( args. a2) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type3>( args. a3) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type4>( args. a4) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type5>( args. a5) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type6>( args. a6) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type7>( args. a7) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type8>( args. a8) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type9>( args. a9));
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
                boost::forward< typename util::remove_reference<Arguments_>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type2>( args. a2) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type3>( args. a3) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type4>( args. a4) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type5>( args. a5) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type6>( args. a6) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type7>( args. a7) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type8>( args. a8) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type9>( args. a9));
    }
    
    struct continuation_thread_object_function_10
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
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
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
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
            catch (...) {
                
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
                boost::forward< typename util::remove_reference<Arguments_>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type2>( args. a2) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type3>( args. a3) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type4>( args. a4) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type5>( args. a5) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type6>( args. a6) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type7>( args. a7) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type8>( args. a8) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type9>( args. a9));
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
                boost::forward< typename util::remove_reference<Arguments_>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type2>( args. a2) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type3>( args. a3) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type4>( args. a4) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type5>( args. a5) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type6>( args. a6) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type7>( args. a7) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type8>( args. a8) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type9>( args. a9));
    }
    
    
    struct continuation_thread_object_function_void_11
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
            void (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10),
            Object* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8) , boost::move(arg9) , boost::move(arg10));
                cont->trigger();
            }
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
            void (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10) const,
            Component* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8) , boost::move(arg9) , boost::move(arg10));
                cont->trigger();
            }
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10), Object* obj,
        BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_void_11(),
            cont, func, obj
          ,
                boost::forward< typename util::remove_reference<Arguments_>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type2>( args. a2) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type3>( args. a3) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type4>( args. a4) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type5>( args. a5) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type6>( args. a6) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type7>( args. a7) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type8>( args. a8) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type9>( args. a9) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type10>( args. a10));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10) const,
        Component* obj, BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_void_11(),
            cont, func, obj
          ,
                boost::forward< typename util::remove_reference<Arguments_>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type2>( args. a2) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type3>( args. a3) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type4>( args. a4) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type5>( args. a5) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type6>( args. a6) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type7>( args. a7) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type8>( args. a8) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type9>( args. a9) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type10>( args. a10));
    }
    
    struct continuation_thread_object_function_11
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
            Result (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10),
            Component* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                cont->trigger(boost::move(
                    (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8) , boost::move(arg9) , boost::move(arg10))
                ));
            }
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
            Result (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10) const,
            Component* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                cont->trigger(boost::move(
                    (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8) , boost::move(arg9) , boost::move(arg10))
                ));
            }
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10), Component* obj,
        BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_11(),
            cont, func, obj
          ,
                boost::forward< typename util::remove_reference<Arguments_>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type2>( args. a2) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type3>( args. a3) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type4>( args. a4) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type5>( args. a5) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type6>( args. a6) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type7>( args. a7) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type8>( args. a8) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type9>( args. a9) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type10>( args. a10));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10) const,
        Component* obj, BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_11(),
            cont, func, obj
          ,
                boost::forward< typename util::remove_reference<Arguments_>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type2>( args. a2) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type3>( args. a3) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type4>( args. a4) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type5>( args. a5) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type6>( args. a6) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type7>( args. a7) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type8>( args. a8) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type9>( args. a9) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type10>( args. a10));
    }
    
    
    struct continuation_thread_object_function_void_12
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
            void (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11),
            Object* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8) , boost::move(arg9) , boost::move(arg10) , boost::move(arg11));
                cont->trigger();
            }
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
            void (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11) const,
            Component* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8) , boost::move(arg9) , boost::move(arg10) , boost::move(arg11));
                cont->trigger();
            }
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11), Object* obj,
        BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_void_12(),
            cont, func, obj
          ,
                boost::forward< typename util::remove_reference<Arguments_>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type2>( args. a2) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type3>( args. a3) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type4>( args. a4) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type5>( args. a5) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type6>( args. a6) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type7>( args. a7) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type8>( args. a8) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type9>( args. a9) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type10>( args. a10) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type11>( args. a11));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11) const,
        Component* obj, BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_void_12(),
            cont, func, obj
          ,
                boost::forward< typename util::remove_reference<Arguments_>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type2>( args. a2) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type3>( args. a3) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type4>( args. a4) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type5>( args. a5) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type6>( args. a6) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type7>( args. a7) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type8>( args. a8) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type9>( args. a9) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type10>( args. a10) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type11>( args. a11));
    }
    
    struct continuation_thread_object_function_12
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
            Result (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11),
            Component* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                cont->trigger(boost::move(
                    (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8) , boost::move(arg9) , boost::move(arg10) , boost::move(arg11))
                ));
            }
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
            Result (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11) const,
            Component* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                cont->trigger(boost::move(
                    (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8) , boost::move(arg9) , boost::move(arg10) , boost::move(arg11))
                ));
            }
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11), Component* obj,
        BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_12(),
            cont, func, obj
          ,
                boost::forward< typename util::remove_reference<Arguments_>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type2>( args. a2) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type3>( args. a3) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type4>( args. a4) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type5>( args. a5) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type6>( args. a6) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type7>( args. a7) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type8>( args. a8) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type9>( args. a9) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type10>( args. a10) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type11>( args. a11));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11) const,
        Component* obj, BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_12(),
            cont, func, obj
          ,
                boost::forward< typename util::remove_reference<Arguments_>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type2>( args. a2) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type3>( args. a3) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type4>( args. a4) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type5>( args. a5) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type6>( args. a6) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type7>( args. a7) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type8>( args. a8) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type9>( args. a9) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type10>( args. a10) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type11>( args. a11));
    }
    
    
    struct continuation_thread_object_function_void_13
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
            void (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12),
            Object* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8) , boost::move(arg9) , boost::move(arg10) , boost::move(arg11) , boost::move(arg12));
                cont->trigger();
            }
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
            void (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12) const,
            Component* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8) , boost::move(arg9) , boost::move(arg10) , boost::move(arg11) , boost::move(arg12));
                cont->trigger();
            }
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12), Object* obj,
        BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_void_13(),
            cont, func, obj
          ,
                boost::forward< typename util::remove_reference<Arguments_>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type2>( args. a2) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type3>( args. a3) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type4>( args. a4) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type5>( args. a5) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type6>( args. a6) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type7>( args. a7) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type8>( args. a8) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type9>( args. a9) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type10>( args. a10) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type11>( args. a11) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type12>( args. a12));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12) const,
        Component* obj, BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_void_13(),
            cont, func, obj
          ,
                boost::forward< typename util::remove_reference<Arguments_>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type2>( args. a2) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type3>( args. a3) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type4>( args. a4) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type5>( args. a5) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type6>( args. a6) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type7>( args. a7) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type8>( args. a8) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type9>( args. a9) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type10>( args. a10) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type11>( args. a11) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type12>( args. a12));
    }
    
    struct continuation_thread_object_function_13
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
            Result (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12),
            Component* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                cont->trigger(boost::move(
                    (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8) , boost::move(arg9) , boost::move(arg10) , boost::move(arg11) , boost::move(arg12))
                ));
            }
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
            Result (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12) const,
            Component* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                cont->trigger(boost::move(
                    (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8) , boost::move(arg9) , boost::move(arg10) , boost::move(arg11) , boost::move(arg12))
                ));
            }
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12), Component* obj,
        BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_13(),
            cont, func, obj
          ,
                boost::forward< typename util::remove_reference<Arguments_>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type2>( args. a2) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type3>( args. a3) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type4>( args. a4) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type5>( args. a5) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type6>( args. a6) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type7>( args. a7) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type8>( args. a8) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type9>( args. a9) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type10>( args. a10) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type11>( args. a11) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type12>( args. a12));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12) const,
        Component* obj, BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_13(),
            cont, func, obj
          ,
                boost::forward< typename util::remove_reference<Arguments_>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type2>( args. a2) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type3>( args. a3) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type4>( args. a4) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type5>( args. a5) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type6>( args. a6) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type7>( args. a7) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type8>( args. a8) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type9>( args. a9) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type10>( args. a10) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type11>( args. a11) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type12>( args. a12));
    }
    
    
    struct continuation_thread_object_function_void_14
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
            void (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12 , FArg13 arg13),
            Object* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8) , boost::move(arg9) , boost::move(arg10) , boost::move(arg11) , boost::move(arg12) , boost::move(arg13));
                cont->trigger();
            }
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
            void (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12 , FArg13 arg13) const,
            Component* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8) , boost::move(arg9) , boost::move(arg10) , boost::move(arg11) , boost::move(arg12) , boost::move(arg13));
                cont->trigger();
            }
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12 , FArg13), Object* obj,
        BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_void_14(),
            cont, func, obj
          ,
                boost::forward< typename util::remove_reference<Arguments_>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type2>( args. a2) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type3>( args. a3) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type4>( args. a4) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type5>( args. a5) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type6>( args. a6) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type7>( args. a7) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type8>( args. a8) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type9>( args. a9) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type10>( args. a10) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type11>( args. a11) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type12>( args. a12) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type13>( args. a13));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12 , FArg13) const,
        Component* obj, BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_void_14(),
            cont, func, obj
          ,
                boost::forward< typename util::remove_reference<Arguments_>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type2>( args. a2) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type3>( args. a3) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type4>( args. a4) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type5>( args. a5) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type6>( args. a6) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type7>( args. a7) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type8>( args. a8) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type9>( args. a9) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type10>( args. a10) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type11>( args. a11) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type12>( args. a12) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type13>( args. a13));
    }
    
    struct continuation_thread_object_function_14
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
            Result (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12 , FArg13 arg13),
            Component* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                cont->trigger(boost::move(
                    (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8) , boost::move(arg9) , boost::move(arg10) , boost::move(arg11) , boost::move(arg12) , boost::move(arg13))
                ));
            }
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
            Result (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12 , FArg13 arg13) const,
            Component* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                cont->trigger(boost::move(
                    (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8) , boost::move(arg9) , boost::move(arg10) , boost::move(arg11) , boost::move(arg12) , boost::move(arg13))
                ));
            }
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12 , FArg13), Component* obj,
        BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_14(),
            cont, func, obj
          ,
                boost::forward< typename util::remove_reference<Arguments_>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type2>( args. a2) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type3>( args. a3) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type4>( args. a4) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type5>( args. a5) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type6>( args. a6) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type7>( args. a7) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type8>( args. a8) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type9>( args. a9) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type10>( args. a10) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type11>( args. a11) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type12>( args. a12) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type13>( args. a13));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12 , FArg13) const,
        Component* obj, BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_14(),
            cont, func, obj
          ,
                boost::forward< typename util::remove_reference<Arguments_>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type2>( args. a2) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type3>( args. a3) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type4>( args. a4) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type5>( args. a5) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type6>( args. a6) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type7>( args. a7) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type8>( args. a8) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type9>( args. a9) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type10>( args. a10) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type11>( args. a11) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type12>( args. a12) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type13>( args. a13));
    }
    
    
    struct continuation_thread_object_function_void_15
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
            void (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12 , FArg13 arg13 , FArg14 arg14),
            Object* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8) , boost::move(arg9) , boost::move(arg10) , boost::move(arg11) , boost::move(arg12) , boost::move(arg13) , boost::move(arg14));
                cont->trigger();
            }
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
            void (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12 , FArg13 arg13 , FArg14 arg14) const,
            Component* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8) , boost::move(arg9) , boost::move(arg10) , boost::move(arg11) , boost::move(arg12) , boost::move(arg13) , boost::move(arg14));
                cont->trigger();
            }
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12 , FArg13 , FArg14), Object* obj,
        BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_void_15(),
            cont, func, obj
          ,
                boost::forward< typename util::remove_reference<Arguments_>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type2>( args. a2) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type3>( args. a3) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type4>( args. a4) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type5>( args. a5) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type6>( args. a6) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type7>( args. a7) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type8>( args. a8) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type9>( args. a9) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type10>( args. a10) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type11>( args. a11) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type12>( args. a12) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type13>( args. a13) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type14>( args. a14));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12 , FArg13 , FArg14) const,
        Component* obj, BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_void_15(),
            cont, func, obj
          ,
                boost::forward< typename util::remove_reference<Arguments_>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type2>( args. a2) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type3>( args. a3) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type4>( args. a4) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type5>( args. a5) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type6>( args. a6) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type7>( args. a7) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type8>( args. a8) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type9>( args. a9) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type10>( args. a10) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type11>( args. a11) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type12>( args. a12) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type13>( args. a13) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type14>( args. a14));
    }
    
    struct continuation_thread_object_function_15
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
            Result (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12 , FArg13 arg13 , FArg14 arg14),
            Component* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                cont->trigger(boost::move(
                    (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8) , boost::move(arg9) , boost::move(arg10) , boost::move(arg11) , boost::move(arg12) , boost::move(arg13) , boost::move(arg14))
                ));
            }
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14>
        BOOST_FORCEINLINE result_type operator()(continuation_type cont,
            Result (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12 , FArg13 arg13 , FArg14 arg14) const,
            Component* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                cont->trigger(boost::move(
                    (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8) , boost::move(arg9) , boost::move(arg10) , boost::move(arg11) , boost::move(arg12) , boost::move(arg13) , boost::move(arg14))
                ));
            }
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12 , FArg13 , FArg14), Component* obj,
        BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_15(),
            cont, func, obj
          ,
                boost::forward< typename util::remove_reference<Arguments_>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type2>( args. a2) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type3>( args. a3) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type4>( args. a4) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type5>( args. a5) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type6>( args. a6) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type7>( args. a7) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type8>( args. a8) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type9>( args. a9) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type10>( args. a10) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type11>( args. a11) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type12>( args. a12) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type13>( args. a13) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type14>( args. a14));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12 , FArg13 , FArg14) const,
        Component* obj, BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_15(),
            cont, func, obj
          ,
                boost::forward< typename util::remove_reference<Arguments_>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type2>( args. a2) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type3>( args. a3) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type4>( args. a4) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type5>( args. a5) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type6>( args. a6) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type7>( args. a7) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type8>( args. a8) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type9>( args. a9) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type10>( args. a10) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type11>( args. a11) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type12>( args. a12) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type13>( args. a13) , boost::forward< typename util::remove_reference<Arguments_>::type:: member_type14>( args. a14));
    }
