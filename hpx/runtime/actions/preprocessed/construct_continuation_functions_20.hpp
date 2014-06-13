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
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
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
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
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
    static threads::thread_function_type
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* func)(), Component* obj,
        Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_void_0()),
            cont, func, obj
          
                );
    }
    template <typename Object, typename Arguments_
         >
    static threads::thread_function_type
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* const func)() const,
        Component* obj, Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_void_0()),
            cont, func, obj
          
                );
    }
    
    struct continuation_thread_object_function_0
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
             
             >
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            Result (Object::* func)(),
            Component* obj
           ) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                cont->trigger(std::forward<Result>(
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
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            Result (Object::* const func)(
                ) const,
            Component* obj
           ) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                cont->trigger(std::forward<Result>(
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
    static threads::thread_function_type
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* func)(), Component* obj,
        Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_0()),
            cont, func, obj
          
                );
    }
    template <typename Object, typename Arguments_
         >
    static threads::thread_function_type
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* const func)() const,
        Component* obj, Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_0()),
            cont, func, obj
          
                );
    }
    
    
    struct continuation_thread_object_function_void_1
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0
            , typename FArg0>
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            void (Object::* func)(FArg0 arg0),
            Object* obj
          , Arg0 && arg0) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                (obj->*func)(std::forward<Arg0>( arg0 ));
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
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            void (Object::* const func)(
                FArg0 arg0) const,
            Component* obj
          , Arg0 && arg0) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                (obj->*func)(std::forward<Arg0>( arg0 ));
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
    static threads::thread_function_type
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* func)(FArg0), Component* obj,
        Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_void_1()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)));
    }
    template <typename Object, typename Arguments_
        , typename FArg0>
    static threads::thread_function_type
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* const func)(FArg0) const,
        Component* obj, Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_void_1()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)));
    }
    
    struct continuation_thread_object_function_1
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0
            , typename FArg0>
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            Result (Object::* func)(FArg0 arg0),
            Component* obj
          , Arg0 && arg0) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                cont->trigger(std::forward<Result>(
                    (obj->*func)(std::forward<Arg0>( arg0 ))
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
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            Result (Object::* const func)(
                FArg0 arg0) const,
            Component* obj
          , Arg0 && arg0) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                cont->trigger(std::forward<Result>(
                    (obj->*func)(std::forward<Arg0>( arg0 ))
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
    static threads::thread_function_type
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* func)(FArg0), Component* obj,
        Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_1()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)));
    }
    template <typename Object, typename Arguments_
        , typename FArg0>
    static threads::thread_function_type
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* const func)(FArg0) const,
        Component* obj, Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_1()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)));
    }
    
    
    struct continuation_thread_object_function_void_2
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1
            , typename FArg0 , typename FArg1>
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            void (Object::* func)(FArg0 arg0 , FArg1 arg1),
            Object* obj
          , Arg0 && arg0 , Arg1 && arg1) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
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
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            void (Object::* const func)(
                FArg0 arg0 , FArg1 arg1) const,
            Component* obj
          , Arg0 && arg0 , Arg1 && arg1) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
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
    static threads::thread_function_type
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* func)(FArg0 , FArg1), Component* obj,
        Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_void_2()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1>
    static threads::thread_function_type
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* const func)(FArg0 , FArg1) const,
        Component* obj, Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_void_2()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)));
    }
    
    struct continuation_thread_object_function_2
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1
            , typename FArg0 , typename FArg1>
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            Result (Object::* func)(FArg0 arg0 , FArg1 arg1),
            Component* obj
          , Arg0 && arg0 , Arg1 && arg1) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                cont->trigger(std::forward<Result>(
                    (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ))
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
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            Result (Object::* const func)(
                FArg0 arg0 , FArg1 arg1) const,
            Component* obj
          , Arg0 && arg0 , Arg1 && arg1) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                cont->trigger(std::forward<Result>(
                    (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ))
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
    static threads::thread_function_type
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* func)(FArg0 , FArg1), Component* obj,
        Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_2()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1>
    static threads::thread_function_type
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* const func)(FArg0 , FArg1) const,
        Component* obj, Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_2()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)));
    }
    
    
    struct continuation_thread_object_function_void_3
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2
            , typename FArg0 , typename FArg1 , typename FArg2>
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            void (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2),
            Object* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
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
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            void (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2) const,
            Component* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
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
    static threads::thread_function_type
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* func)(FArg0 , FArg1 , FArg2), Component* obj,
        Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_void_3()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2>
    static threads::thread_function_type
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* const func)(FArg0 , FArg1 , FArg2) const,
        Component* obj, Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_void_3()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)));
    }
    
    struct continuation_thread_object_function_3
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2
            , typename FArg0 , typename FArg1 , typename FArg2>
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            Result (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2),
            Component* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                cont->trigger(std::forward<Result>(
                    (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ))
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
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            Result (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2) const,
            Component* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                cont->trigger(std::forward<Result>(
                    (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ))
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
    static threads::thread_function_type
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* func)(FArg0 , FArg1 , FArg2), Component* obj,
        Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_3()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2>
    static threads::thread_function_type
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* const func)(FArg0 , FArg1 , FArg2) const,
        Component* obj, Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_3()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)));
    }
    
    
    struct continuation_thread_object_function_void_4
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3>
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            void (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3),
            Object* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
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
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            void (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3) const,
            Component* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
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
    static threads::thread_function_type
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3), Component* obj,
        Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_void_4()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3>
    static threads::thread_function_type
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3) const,
        Component* obj, Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_void_4()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)));
    }
    
    struct continuation_thread_object_function_4
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3>
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            Result (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3),
            Component* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                cont->trigger(std::forward<Result>(
                    (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ))
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
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            Result (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3) const,
            Component* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                cont->trigger(std::forward<Result>(
                    (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ))
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
    static threads::thread_function_type
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3), Component* obj,
        Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_4()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3>
    static threads::thread_function_type
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3) const,
        Component* obj, Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_4()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)));
    }
    
    
    struct continuation_thread_object_function_void_5
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4>
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            void (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4),
            Object* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
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
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            void (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4) const,
            Component* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
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
    static threads::thread_function_type
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4), Component* obj,
        Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_void_5()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4>
    static threads::thread_function_type
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4) const,
        Component* obj, Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_void_5()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)));
    }
    
    struct continuation_thread_object_function_5
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4>
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            Result (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4),
            Component* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                cont->trigger(std::forward<Result>(
                    (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ))
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
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            Result (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4) const,
            Component* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                cont->trigger(std::forward<Result>(
                    (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ))
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
    static threads::thread_function_type
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4), Component* obj,
        Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_5()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4>
    static threads::thread_function_type
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4) const,
        Component* obj, Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_5()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)));
    }
    
    
    struct continuation_thread_object_function_void_6
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5>
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            void (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5),
            Object* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ));
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
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            void (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5) const,
            Component* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ));
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
    static threads::thread_function_type
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5), Component* obj,
        Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_void_6()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)) , util::get< 5>(std::forward<Arguments_>( args)));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5>
    static threads::thread_function_type
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5) const,
        Component* obj, Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_void_6()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)) , util::get< 5>(std::forward<Arguments_>( args)));
    }
    
    struct continuation_thread_object_function_6
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5>
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            Result (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5),
            Component* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                cont->trigger(std::forward<Result>(
                    (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ))
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
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            Result (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5) const,
            Component* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                cont->trigger(std::forward<Result>(
                    (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ))
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
    static threads::thread_function_type
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5), Component* obj,
        Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_6()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)) , util::get< 5>(std::forward<Arguments_>( args)));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5>
    static threads::thread_function_type
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5) const,
        Component* obj, Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_6()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)) , util::get< 5>(std::forward<Arguments_>( args)));
    }
    
    
    struct continuation_thread_object_function_void_7
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6>
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            void (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6),
            Object* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ));
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
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            void (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6) const,
            Component* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ));
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
    static threads::thread_function_type
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6), Component* obj,
        Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_void_7()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)) , util::get< 5>(std::forward<Arguments_>( args)) , util::get< 6>(std::forward<Arguments_>( args)));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6>
    static threads::thread_function_type
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6) const,
        Component* obj, Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_void_7()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)) , util::get< 5>(std::forward<Arguments_>( args)) , util::get< 6>(std::forward<Arguments_>( args)));
    }
    
    struct continuation_thread_object_function_7
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6>
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            Result (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6),
            Component* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                cont->trigger(std::forward<Result>(
                    (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ))
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
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            Result (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6) const,
            Component* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                cont->trigger(std::forward<Result>(
                    (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ))
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
    static threads::thread_function_type
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6), Component* obj,
        Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_7()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)) , util::get< 5>(std::forward<Arguments_>( args)) , util::get< 6>(std::forward<Arguments_>( args)));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6>
    static threads::thread_function_type
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6) const,
        Component* obj, Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_7()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)) , util::get< 5>(std::forward<Arguments_>( args)) , util::get< 6>(std::forward<Arguments_>( args)));
    }
    
    
    struct continuation_thread_object_function_void_8
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7>
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            void (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7),
            Object* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ));
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
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            void (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7) const,
            Component* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ));
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
    static threads::thread_function_type
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7), Component* obj,
        Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_void_8()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)) , util::get< 5>(std::forward<Arguments_>( args)) , util::get< 6>(std::forward<Arguments_>( args)) , util::get< 7>(std::forward<Arguments_>( args)));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7>
    static threads::thread_function_type
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7) const,
        Component* obj, Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_void_8()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)) , util::get< 5>(std::forward<Arguments_>( args)) , util::get< 6>(std::forward<Arguments_>( args)) , util::get< 7>(std::forward<Arguments_>( args)));
    }
    
    struct continuation_thread_object_function_8
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7>
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            Result (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7),
            Component* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                cont->trigger(std::forward<Result>(
                    (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ))
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
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            Result (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7) const,
            Component* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                cont->trigger(std::forward<Result>(
                    (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ))
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
    static threads::thread_function_type
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7), Component* obj,
        Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_8()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)) , util::get< 5>(std::forward<Arguments_>( args)) , util::get< 6>(std::forward<Arguments_>( args)) , util::get< 7>(std::forward<Arguments_>( args)));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7>
    static threads::thread_function_type
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7) const,
        Component* obj, Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_8()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)) , util::get< 5>(std::forward<Arguments_>( args)) , util::get< 6>(std::forward<Arguments_>( args)) , util::get< 7>(std::forward<Arguments_>( args)));
    }
    
    
    struct continuation_thread_object_function_void_9
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8>
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            void (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8),
            Object* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ));
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
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            void (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8) const,
            Component* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ));
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
    static threads::thread_function_type
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8), Component* obj,
        Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_void_9()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)) , util::get< 5>(std::forward<Arguments_>( args)) , util::get< 6>(std::forward<Arguments_>( args)) , util::get< 7>(std::forward<Arguments_>( args)) , util::get< 8>(std::forward<Arguments_>( args)));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8>
    static threads::thread_function_type
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8) const,
        Component* obj, Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_void_9()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)) , util::get< 5>(std::forward<Arguments_>( args)) , util::get< 6>(std::forward<Arguments_>( args)) , util::get< 7>(std::forward<Arguments_>( args)) , util::get< 8>(std::forward<Arguments_>( args)));
    }
    
    struct continuation_thread_object_function_9
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8>
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            Result (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8),
            Component* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                cont->trigger(std::forward<Result>(
                    (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ))
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
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            Result (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8) const,
            Component* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                cont->trigger(std::forward<Result>(
                    (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ))
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
    static threads::thread_function_type
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8), Component* obj,
        Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_9()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)) , util::get< 5>(std::forward<Arguments_>( args)) , util::get< 6>(std::forward<Arguments_>( args)) , util::get< 7>(std::forward<Arguments_>( args)) , util::get< 8>(std::forward<Arguments_>( args)));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8>
    static threads::thread_function_type
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8) const,
        Component* obj, Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_9()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)) , util::get< 5>(std::forward<Arguments_>( args)) , util::get< 6>(std::forward<Arguments_>( args)) , util::get< 7>(std::forward<Arguments_>( args)) , util::get< 8>(std::forward<Arguments_>( args)));
    }
    
    
    struct continuation_thread_object_function_void_10
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9>
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            void (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9),
            Object* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ));
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
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            void (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9) const,
            Component* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ));
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
    static threads::thread_function_type
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9), Component* obj,
        Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_void_10()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)) , util::get< 5>(std::forward<Arguments_>( args)) , util::get< 6>(std::forward<Arguments_>( args)) , util::get< 7>(std::forward<Arguments_>( args)) , util::get< 8>(std::forward<Arguments_>( args)) , util::get< 9>(std::forward<Arguments_>( args)));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9>
    static threads::thread_function_type
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9) const,
        Component* obj, Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_void_10()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)) , util::get< 5>(std::forward<Arguments_>( args)) , util::get< 6>(std::forward<Arguments_>( args)) , util::get< 7>(std::forward<Arguments_>( args)) , util::get< 8>(std::forward<Arguments_>( args)) , util::get< 9>(std::forward<Arguments_>( args)));
    }
    
    struct continuation_thread_object_function_10
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9>
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            Result (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9),
            Component* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                cont->trigger(std::forward<Result>(
                    (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ))
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
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            Result (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9) const,
            Component* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                cont->trigger(std::forward<Result>(
                    (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ))
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
    static threads::thread_function_type
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9), Component* obj,
        Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_10()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)) , util::get< 5>(std::forward<Arguments_>( args)) , util::get< 6>(std::forward<Arguments_>( args)) , util::get< 7>(std::forward<Arguments_>( args)) , util::get< 8>(std::forward<Arguments_>( args)) , util::get< 9>(std::forward<Arguments_>( args)));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9>
    static threads::thread_function_type
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9) const,
        Component* obj, Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_10()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)) , util::get< 5>(std::forward<Arguments_>( args)) , util::get< 6>(std::forward<Arguments_>( args)) , util::get< 7>(std::forward<Arguments_>( args)) , util::get< 8>(std::forward<Arguments_>( args)) , util::get< 9>(std::forward<Arguments_>( args)));
    }
    
    
    struct continuation_thread_object_function_void_11
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10>
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            void (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10),
            Object* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ));
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
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            void (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10) const,
            Component* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ));
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
    static threads::thread_function_type
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10), Component* obj,
        Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_void_11()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)) , util::get< 5>(std::forward<Arguments_>( args)) , util::get< 6>(std::forward<Arguments_>( args)) , util::get< 7>(std::forward<Arguments_>( args)) , util::get< 8>(std::forward<Arguments_>( args)) , util::get< 9>(std::forward<Arguments_>( args)) , util::get< 10>(std::forward<Arguments_>( args)));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10>
    static threads::thread_function_type
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10) const,
        Component* obj, Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_void_11()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)) , util::get< 5>(std::forward<Arguments_>( args)) , util::get< 6>(std::forward<Arguments_>( args)) , util::get< 7>(std::forward<Arguments_>( args)) , util::get< 8>(std::forward<Arguments_>( args)) , util::get< 9>(std::forward<Arguments_>( args)) , util::get< 10>(std::forward<Arguments_>( args)));
    }
    
    struct continuation_thread_object_function_11
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10>
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            Result (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10),
            Component* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                cont->trigger(std::forward<Result>(
                    (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ))
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
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            Result (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10) const,
            Component* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                cont->trigger(std::forward<Result>(
                    (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ))
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
    static threads::thread_function_type
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10), Component* obj,
        Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_11()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)) , util::get< 5>(std::forward<Arguments_>( args)) , util::get< 6>(std::forward<Arguments_>( args)) , util::get< 7>(std::forward<Arguments_>( args)) , util::get< 8>(std::forward<Arguments_>( args)) , util::get< 9>(std::forward<Arguments_>( args)) , util::get< 10>(std::forward<Arguments_>( args)));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10>
    static threads::thread_function_type
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10) const,
        Component* obj, Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_11()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)) , util::get< 5>(std::forward<Arguments_>( args)) , util::get< 6>(std::forward<Arguments_>( args)) , util::get< 7>(std::forward<Arguments_>( args)) , util::get< 8>(std::forward<Arguments_>( args)) , util::get< 9>(std::forward<Arguments_>( args)) , util::get< 10>(std::forward<Arguments_>( args)));
    }
    
    
    struct continuation_thread_object_function_void_12
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11>
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            void (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11),
            Object* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ));
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
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            void (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11) const,
            Component* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ));
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
    static threads::thread_function_type
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11), Component* obj,
        Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_void_12()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)) , util::get< 5>(std::forward<Arguments_>( args)) , util::get< 6>(std::forward<Arguments_>( args)) , util::get< 7>(std::forward<Arguments_>( args)) , util::get< 8>(std::forward<Arguments_>( args)) , util::get< 9>(std::forward<Arguments_>( args)) , util::get< 10>(std::forward<Arguments_>( args)) , util::get< 11>(std::forward<Arguments_>( args)));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11>
    static threads::thread_function_type
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11) const,
        Component* obj, Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_void_12()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)) , util::get< 5>(std::forward<Arguments_>( args)) , util::get< 6>(std::forward<Arguments_>( args)) , util::get< 7>(std::forward<Arguments_>( args)) , util::get< 8>(std::forward<Arguments_>( args)) , util::get< 9>(std::forward<Arguments_>( args)) , util::get< 10>(std::forward<Arguments_>( args)) , util::get< 11>(std::forward<Arguments_>( args)));
    }
    
    struct continuation_thread_object_function_12
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11>
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            Result (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11),
            Component* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                cont->trigger(std::forward<Result>(
                    (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ))
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
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            Result (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11) const,
            Component* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                cont->trigger(std::forward<Result>(
                    (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ))
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
    static threads::thread_function_type
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11), Component* obj,
        Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_12()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)) , util::get< 5>(std::forward<Arguments_>( args)) , util::get< 6>(std::forward<Arguments_>( args)) , util::get< 7>(std::forward<Arguments_>( args)) , util::get< 8>(std::forward<Arguments_>( args)) , util::get< 9>(std::forward<Arguments_>( args)) , util::get< 10>(std::forward<Arguments_>( args)) , util::get< 11>(std::forward<Arguments_>( args)));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11>
    static threads::thread_function_type
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11) const,
        Component* obj, Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_12()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)) , util::get< 5>(std::forward<Arguments_>( args)) , util::get< 6>(std::forward<Arguments_>( args)) , util::get< 7>(std::forward<Arguments_>( args)) , util::get< 8>(std::forward<Arguments_>( args)) , util::get< 9>(std::forward<Arguments_>( args)) , util::get< 10>(std::forward<Arguments_>( args)) , util::get< 11>(std::forward<Arguments_>( args)));
    }
    
    
    struct continuation_thread_object_function_void_13
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12>
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            void (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12),
            Object* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ));
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
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            void (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12) const,
            Component* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ));
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
    static threads::thread_function_type
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12), Component* obj,
        Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_void_13()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)) , util::get< 5>(std::forward<Arguments_>( args)) , util::get< 6>(std::forward<Arguments_>( args)) , util::get< 7>(std::forward<Arguments_>( args)) , util::get< 8>(std::forward<Arguments_>( args)) , util::get< 9>(std::forward<Arguments_>( args)) , util::get< 10>(std::forward<Arguments_>( args)) , util::get< 11>(std::forward<Arguments_>( args)) , util::get< 12>(std::forward<Arguments_>( args)));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12>
    static threads::thread_function_type
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12) const,
        Component* obj, Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_void_13()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)) , util::get< 5>(std::forward<Arguments_>( args)) , util::get< 6>(std::forward<Arguments_>( args)) , util::get< 7>(std::forward<Arguments_>( args)) , util::get< 8>(std::forward<Arguments_>( args)) , util::get< 9>(std::forward<Arguments_>( args)) , util::get< 10>(std::forward<Arguments_>( args)) , util::get< 11>(std::forward<Arguments_>( args)) , util::get< 12>(std::forward<Arguments_>( args)));
    }
    
    struct continuation_thread_object_function_13
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12>
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            Result (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12),
            Component* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                cont->trigger(std::forward<Result>(
                    (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ))
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
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            Result (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12) const,
            Component* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                cont->trigger(std::forward<Result>(
                    (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ))
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
    static threads::thread_function_type
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12), Component* obj,
        Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_13()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)) , util::get< 5>(std::forward<Arguments_>( args)) , util::get< 6>(std::forward<Arguments_>( args)) , util::get< 7>(std::forward<Arguments_>( args)) , util::get< 8>(std::forward<Arguments_>( args)) , util::get< 9>(std::forward<Arguments_>( args)) , util::get< 10>(std::forward<Arguments_>( args)) , util::get< 11>(std::forward<Arguments_>( args)) , util::get< 12>(std::forward<Arguments_>( args)));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12>
    static threads::thread_function_type
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12) const,
        Component* obj, Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_13()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)) , util::get< 5>(std::forward<Arguments_>( args)) , util::get< 6>(std::forward<Arguments_>( args)) , util::get< 7>(std::forward<Arguments_>( args)) , util::get< 8>(std::forward<Arguments_>( args)) , util::get< 9>(std::forward<Arguments_>( args)) , util::get< 10>(std::forward<Arguments_>( args)) , util::get< 11>(std::forward<Arguments_>( args)) , util::get< 12>(std::forward<Arguments_>( args)));
    }
    
    
    struct continuation_thread_object_function_void_14
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13>
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            void (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12 , FArg13 arg13),
            Object* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ));
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
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            void (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12 , FArg13 arg13) const,
            Component* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ));
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
    static threads::thread_function_type
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12 , FArg13), Component* obj,
        Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_void_14()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)) , util::get< 5>(std::forward<Arguments_>( args)) , util::get< 6>(std::forward<Arguments_>( args)) , util::get< 7>(std::forward<Arguments_>( args)) , util::get< 8>(std::forward<Arguments_>( args)) , util::get< 9>(std::forward<Arguments_>( args)) , util::get< 10>(std::forward<Arguments_>( args)) , util::get< 11>(std::forward<Arguments_>( args)) , util::get< 12>(std::forward<Arguments_>( args)) , util::get< 13>(std::forward<Arguments_>( args)));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13>
    static threads::thread_function_type
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12 , FArg13) const,
        Component* obj, Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_void_14()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)) , util::get< 5>(std::forward<Arguments_>( args)) , util::get< 6>(std::forward<Arguments_>( args)) , util::get< 7>(std::forward<Arguments_>( args)) , util::get< 8>(std::forward<Arguments_>( args)) , util::get< 9>(std::forward<Arguments_>( args)) , util::get< 10>(std::forward<Arguments_>( args)) , util::get< 11>(std::forward<Arguments_>( args)) , util::get< 12>(std::forward<Arguments_>( args)) , util::get< 13>(std::forward<Arguments_>( args)));
    }
    
    struct continuation_thread_object_function_14
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13>
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            Result (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12 , FArg13 arg13),
            Component* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                cont->trigger(std::forward<Result>(
                    (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ))
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
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            Result (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12 , FArg13 arg13) const,
            Component* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                cont->trigger(std::forward<Result>(
                    (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ))
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
    static threads::thread_function_type
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12 , FArg13), Component* obj,
        Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_14()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)) , util::get< 5>(std::forward<Arguments_>( args)) , util::get< 6>(std::forward<Arguments_>( args)) , util::get< 7>(std::forward<Arguments_>( args)) , util::get< 8>(std::forward<Arguments_>( args)) , util::get< 9>(std::forward<Arguments_>( args)) , util::get< 10>(std::forward<Arguments_>( args)) , util::get< 11>(std::forward<Arguments_>( args)) , util::get< 12>(std::forward<Arguments_>( args)) , util::get< 13>(std::forward<Arguments_>( args)));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13>
    static threads::thread_function_type
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12 , FArg13) const,
        Component* obj, Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_14()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)) , util::get< 5>(std::forward<Arguments_>( args)) , util::get< 6>(std::forward<Arguments_>( args)) , util::get< 7>(std::forward<Arguments_>( args)) , util::get< 8>(std::forward<Arguments_>( args)) , util::get< 9>(std::forward<Arguments_>( args)) , util::get< 10>(std::forward<Arguments_>( args)) , util::get< 11>(std::forward<Arguments_>( args)) , util::get< 12>(std::forward<Arguments_>( args)) , util::get< 13>(std::forward<Arguments_>( args)));
    }
    
    
    struct continuation_thread_object_function_void_15
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14>
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            void (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12 , FArg13 arg13 , FArg14 arg14),
            Object* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13 , Arg14 && arg14) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ) , std::forward<Arg14>( arg14 ));
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
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            void (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12 , FArg13 arg13 , FArg14 arg14) const,
            Component* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13 , Arg14 && arg14) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ) , std::forward<Arg14>( arg14 ));
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
    static threads::thread_function_type
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12 , FArg13 , FArg14), Component* obj,
        Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_void_15()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)) , util::get< 5>(std::forward<Arguments_>( args)) , util::get< 6>(std::forward<Arguments_>( args)) , util::get< 7>(std::forward<Arguments_>( args)) , util::get< 8>(std::forward<Arguments_>( args)) , util::get< 9>(std::forward<Arguments_>( args)) , util::get< 10>(std::forward<Arguments_>( args)) , util::get< 11>(std::forward<Arguments_>( args)) , util::get< 12>(std::forward<Arguments_>( args)) , util::get< 13>(std::forward<Arguments_>( args)) , util::get< 14>(std::forward<Arguments_>( args)));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14>
    static threads::thread_function_type
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12 , FArg13 , FArg14) const,
        Component* obj, Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_void_15()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)) , util::get< 5>(std::forward<Arguments_>( args)) , util::get< 6>(std::forward<Arguments_>( args)) , util::get< 7>(std::forward<Arguments_>( args)) , util::get< 8>(std::forward<Arguments_>( args)) , util::get< 9>(std::forward<Arguments_>( args)) , util::get< 10>(std::forward<Arguments_>( args)) , util::get< 11>(std::forward<Arguments_>( args)) , util::get< 12>(std::forward<Arguments_>( args)) , util::get< 13>(std::forward<Arguments_>( args)) , util::get< 14>(std::forward<Arguments_>( args)));
    }
    
    struct continuation_thread_object_function_15
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14>
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            Result (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12 , FArg13 arg13 , FArg14 arg14),
            Component* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13 , Arg14 && arg14) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                cont->trigger(std::forward<Result>(
                    (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ) , std::forward<Arg14>( arg14 ))
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
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            Result (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12 , FArg13 arg13 , FArg14 arg14) const,
            Component* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13 , Arg14 && arg14) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                cont->trigger(std::forward<Result>(
                    (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ) , std::forward<Arg14>( arg14 ))
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
    static threads::thread_function_type
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12 , FArg13 , FArg14), Component* obj,
        Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_15()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)) , util::get< 5>(std::forward<Arguments_>( args)) , util::get< 6>(std::forward<Arguments_>( args)) , util::get< 7>(std::forward<Arguments_>( args)) , util::get< 8>(std::forward<Arguments_>( args)) , util::get< 9>(std::forward<Arguments_>( args)) , util::get< 10>(std::forward<Arguments_>( args)) , util::get< 11>(std::forward<Arguments_>( args)) , util::get< 12>(std::forward<Arguments_>( args)) , util::get< 13>(std::forward<Arguments_>( args)) , util::get< 14>(std::forward<Arguments_>( args)));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14>
    static threads::thread_function_type
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12 , FArg13 , FArg14) const,
        Component* obj, Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_15()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)) , util::get< 5>(std::forward<Arguments_>( args)) , util::get< 6>(std::forward<Arguments_>( args)) , util::get< 7>(std::forward<Arguments_>( args)) , util::get< 8>(std::forward<Arguments_>( args)) , util::get< 9>(std::forward<Arguments_>( args)) , util::get< 10>(std::forward<Arguments_>( args)) , util::get< 11>(std::forward<Arguments_>( args)) , util::get< 12>(std::forward<Arguments_>( args)) , util::get< 13>(std::forward<Arguments_>( args)) , util::get< 14>(std::forward<Arguments_>( args)));
    }
    
    
    struct continuation_thread_object_function_void_16
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15>
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            void (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12 , FArg13 arg13 , FArg14 arg14 , FArg15 arg15),
            Object* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13 , Arg14 && arg14 , Arg15 && arg15) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ) , std::forward<Arg14>( arg14 ) , std::forward<Arg15>( arg15 ));
                cont->trigger();
            }
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15>
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            void (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12 , FArg13 arg13 , FArg14 arg14 , FArg15 arg15) const,
            Component* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13 , Arg14 && arg14 , Arg15 && arg15) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ) , std::forward<Arg14>( arg14 ) , std::forward<Arg15>( arg15 ));
                cont->trigger();
            }
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15>
    static threads::thread_function_type
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12 , FArg13 , FArg14 , FArg15), Component* obj,
        Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_void_16()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)) , util::get< 5>(std::forward<Arguments_>( args)) , util::get< 6>(std::forward<Arguments_>( args)) , util::get< 7>(std::forward<Arguments_>( args)) , util::get< 8>(std::forward<Arguments_>( args)) , util::get< 9>(std::forward<Arguments_>( args)) , util::get< 10>(std::forward<Arguments_>( args)) , util::get< 11>(std::forward<Arguments_>( args)) , util::get< 12>(std::forward<Arguments_>( args)) , util::get< 13>(std::forward<Arguments_>( args)) , util::get< 14>(std::forward<Arguments_>( args)) , util::get< 15>(std::forward<Arguments_>( args)));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15>
    static threads::thread_function_type
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12 , FArg13 , FArg14 , FArg15) const,
        Component* obj, Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_void_16()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)) , util::get< 5>(std::forward<Arguments_>( args)) , util::get< 6>(std::forward<Arguments_>( args)) , util::get< 7>(std::forward<Arguments_>( args)) , util::get< 8>(std::forward<Arguments_>( args)) , util::get< 9>(std::forward<Arguments_>( args)) , util::get< 10>(std::forward<Arguments_>( args)) , util::get< 11>(std::forward<Arguments_>( args)) , util::get< 12>(std::forward<Arguments_>( args)) , util::get< 13>(std::forward<Arguments_>( args)) , util::get< 14>(std::forward<Arguments_>( args)) , util::get< 15>(std::forward<Arguments_>( args)));
    }
    
    struct continuation_thread_object_function_16
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15>
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            Result (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12 , FArg13 arg13 , FArg14 arg14 , FArg15 arg15),
            Component* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13 , Arg14 && arg14 , Arg15 && arg15) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                cont->trigger(std::forward<Result>(
                    (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ) , std::forward<Arg14>( arg14 ) , std::forward<Arg15>( arg15 ))
                ));
            }
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15>
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            Result (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12 , FArg13 arg13 , FArg14 arg14 , FArg15 arg15) const,
            Component* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13 , Arg14 && arg14 , Arg15 && arg15) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                cont->trigger(std::forward<Result>(
                    (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ) , std::forward<Arg14>( arg14 ) , std::forward<Arg15>( arg15 ))
                ));
            }
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15>
    static threads::thread_function_type
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12 , FArg13 , FArg14 , FArg15), Component* obj,
        Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_16()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)) , util::get< 5>(std::forward<Arguments_>( args)) , util::get< 6>(std::forward<Arguments_>( args)) , util::get< 7>(std::forward<Arguments_>( args)) , util::get< 8>(std::forward<Arguments_>( args)) , util::get< 9>(std::forward<Arguments_>( args)) , util::get< 10>(std::forward<Arguments_>( args)) , util::get< 11>(std::forward<Arguments_>( args)) , util::get< 12>(std::forward<Arguments_>( args)) , util::get< 13>(std::forward<Arguments_>( args)) , util::get< 14>(std::forward<Arguments_>( args)) , util::get< 15>(std::forward<Arguments_>( args)));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15>
    static threads::thread_function_type
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12 , FArg13 , FArg14 , FArg15) const,
        Component* obj, Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_16()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)) , util::get< 5>(std::forward<Arguments_>( args)) , util::get< 6>(std::forward<Arguments_>( args)) , util::get< 7>(std::forward<Arguments_>( args)) , util::get< 8>(std::forward<Arguments_>( args)) , util::get< 9>(std::forward<Arguments_>( args)) , util::get< 10>(std::forward<Arguments_>( args)) , util::get< 11>(std::forward<Arguments_>( args)) , util::get< 12>(std::forward<Arguments_>( args)) , util::get< 13>(std::forward<Arguments_>( args)) , util::get< 14>(std::forward<Arguments_>( args)) , util::get< 15>(std::forward<Arguments_>( args)));
    }
    
    
    struct continuation_thread_object_function_void_17
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16>
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            void (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12 , FArg13 arg13 , FArg14 arg14 , FArg15 arg15 , FArg16 arg16),
            Object* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13 , Arg14 && arg14 , Arg15 && arg15 , Arg16 && arg16) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ) , std::forward<Arg14>( arg14 ) , std::forward<Arg15>( arg15 ) , std::forward<Arg16>( arg16 ));
                cont->trigger();
            }
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16>
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            void (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12 , FArg13 arg13 , FArg14 arg14 , FArg15 arg15 , FArg16 arg16) const,
            Component* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13 , Arg14 && arg14 , Arg15 && arg15 , Arg16 && arg16) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ) , std::forward<Arg14>( arg14 ) , std::forward<Arg15>( arg15 ) , std::forward<Arg16>( arg16 ));
                cont->trigger();
            }
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16>
    static threads::thread_function_type
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12 , FArg13 , FArg14 , FArg15 , FArg16), Component* obj,
        Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_void_17()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)) , util::get< 5>(std::forward<Arguments_>( args)) , util::get< 6>(std::forward<Arguments_>( args)) , util::get< 7>(std::forward<Arguments_>( args)) , util::get< 8>(std::forward<Arguments_>( args)) , util::get< 9>(std::forward<Arguments_>( args)) , util::get< 10>(std::forward<Arguments_>( args)) , util::get< 11>(std::forward<Arguments_>( args)) , util::get< 12>(std::forward<Arguments_>( args)) , util::get< 13>(std::forward<Arguments_>( args)) , util::get< 14>(std::forward<Arguments_>( args)) , util::get< 15>(std::forward<Arguments_>( args)) , util::get< 16>(std::forward<Arguments_>( args)));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16>
    static threads::thread_function_type
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12 , FArg13 , FArg14 , FArg15 , FArg16) const,
        Component* obj, Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_void_17()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)) , util::get< 5>(std::forward<Arguments_>( args)) , util::get< 6>(std::forward<Arguments_>( args)) , util::get< 7>(std::forward<Arguments_>( args)) , util::get< 8>(std::forward<Arguments_>( args)) , util::get< 9>(std::forward<Arguments_>( args)) , util::get< 10>(std::forward<Arguments_>( args)) , util::get< 11>(std::forward<Arguments_>( args)) , util::get< 12>(std::forward<Arguments_>( args)) , util::get< 13>(std::forward<Arguments_>( args)) , util::get< 14>(std::forward<Arguments_>( args)) , util::get< 15>(std::forward<Arguments_>( args)) , util::get< 16>(std::forward<Arguments_>( args)));
    }
    
    struct continuation_thread_object_function_17
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16>
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            Result (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12 , FArg13 arg13 , FArg14 arg14 , FArg15 arg15 , FArg16 arg16),
            Component* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13 , Arg14 && arg14 , Arg15 && arg15 , Arg16 && arg16) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                cont->trigger(std::forward<Result>(
                    (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ) , std::forward<Arg14>( arg14 ) , std::forward<Arg15>( arg15 ) , std::forward<Arg16>( arg16 ))
                ));
            }
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16>
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            Result (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12 , FArg13 arg13 , FArg14 arg14 , FArg15 arg15 , FArg16 arg16) const,
            Component* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13 , Arg14 && arg14 , Arg15 && arg15 , Arg16 && arg16) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                cont->trigger(std::forward<Result>(
                    (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ) , std::forward<Arg14>( arg14 ) , std::forward<Arg15>( arg15 ) , std::forward<Arg16>( arg16 ))
                ));
            }
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16>
    static threads::thread_function_type
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12 , FArg13 , FArg14 , FArg15 , FArg16), Component* obj,
        Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_17()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)) , util::get< 5>(std::forward<Arguments_>( args)) , util::get< 6>(std::forward<Arguments_>( args)) , util::get< 7>(std::forward<Arguments_>( args)) , util::get< 8>(std::forward<Arguments_>( args)) , util::get< 9>(std::forward<Arguments_>( args)) , util::get< 10>(std::forward<Arguments_>( args)) , util::get< 11>(std::forward<Arguments_>( args)) , util::get< 12>(std::forward<Arguments_>( args)) , util::get< 13>(std::forward<Arguments_>( args)) , util::get< 14>(std::forward<Arguments_>( args)) , util::get< 15>(std::forward<Arguments_>( args)) , util::get< 16>(std::forward<Arguments_>( args)));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16>
    static threads::thread_function_type
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12 , FArg13 , FArg14 , FArg15 , FArg16) const,
        Component* obj, Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_17()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)) , util::get< 5>(std::forward<Arguments_>( args)) , util::get< 6>(std::forward<Arguments_>( args)) , util::get< 7>(std::forward<Arguments_>( args)) , util::get< 8>(std::forward<Arguments_>( args)) , util::get< 9>(std::forward<Arguments_>( args)) , util::get< 10>(std::forward<Arguments_>( args)) , util::get< 11>(std::forward<Arguments_>( args)) , util::get< 12>(std::forward<Arguments_>( args)) , util::get< 13>(std::forward<Arguments_>( args)) , util::get< 14>(std::forward<Arguments_>( args)) , util::get< 15>(std::forward<Arguments_>( args)) , util::get< 16>(std::forward<Arguments_>( args)));
    }
    
    
    struct continuation_thread_object_function_void_18
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16 , typename FArg17>
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            void (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12 , FArg13 arg13 , FArg14 arg14 , FArg15 arg15 , FArg16 arg16 , FArg17 arg17),
            Object* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13 , Arg14 && arg14 , Arg15 && arg15 , Arg16 && arg16 , Arg17 && arg17) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ) , std::forward<Arg14>( arg14 ) , std::forward<Arg15>( arg15 ) , std::forward<Arg16>( arg16 ) , std::forward<Arg17>( arg17 ));
                cont->trigger();
            }
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16 , typename FArg17>
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            void (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12 , FArg13 arg13 , FArg14 arg14 , FArg15 arg15 , FArg16 arg16 , FArg17 arg17) const,
            Component* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13 , Arg14 && arg14 , Arg15 && arg15 , Arg16 && arg16 , Arg17 && arg17) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ) , std::forward<Arg14>( arg14 ) , std::forward<Arg15>( arg15 ) , std::forward<Arg16>( arg16 ) , std::forward<Arg17>( arg17 ));
                cont->trigger();
            }
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16 , typename FArg17>
    static threads::thread_function_type
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12 , FArg13 , FArg14 , FArg15 , FArg16 , FArg17), Component* obj,
        Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_void_18()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)) , util::get< 5>(std::forward<Arguments_>( args)) , util::get< 6>(std::forward<Arguments_>( args)) , util::get< 7>(std::forward<Arguments_>( args)) , util::get< 8>(std::forward<Arguments_>( args)) , util::get< 9>(std::forward<Arguments_>( args)) , util::get< 10>(std::forward<Arguments_>( args)) , util::get< 11>(std::forward<Arguments_>( args)) , util::get< 12>(std::forward<Arguments_>( args)) , util::get< 13>(std::forward<Arguments_>( args)) , util::get< 14>(std::forward<Arguments_>( args)) , util::get< 15>(std::forward<Arguments_>( args)) , util::get< 16>(std::forward<Arguments_>( args)) , util::get< 17>(std::forward<Arguments_>( args)));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16 , typename FArg17>
    static threads::thread_function_type
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12 , FArg13 , FArg14 , FArg15 , FArg16 , FArg17) const,
        Component* obj, Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_void_18()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)) , util::get< 5>(std::forward<Arguments_>( args)) , util::get< 6>(std::forward<Arguments_>( args)) , util::get< 7>(std::forward<Arguments_>( args)) , util::get< 8>(std::forward<Arguments_>( args)) , util::get< 9>(std::forward<Arguments_>( args)) , util::get< 10>(std::forward<Arguments_>( args)) , util::get< 11>(std::forward<Arguments_>( args)) , util::get< 12>(std::forward<Arguments_>( args)) , util::get< 13>(std::forward<Arguments_>( args)) , util::get< 14>(std::forward<Arguments_>( args)) , util::get< 15>(std::forward<Arguments_>( args)) , util::get< 16>(std::forward<Arguments_>( args)) , util::get< 17>(std::forward<Arguments_>( args)));
    }
    
    struct continuation_thread_object_function_18
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16 , typename FArg17>
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            Result (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12 , FArg13 arg13 , FArg14 arg14 , FArg15 arg15 , FArg16 arg16 , FArg17 arg17),
            Component* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13 , Arg14 && arg14 , Arg15 && arg15 , Arg16 && arg16 , Arg17 && arg17) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                cont->trigger(std::forward<Result>(
                    (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ) , std::forward<Arg14>( arg14 ) , std::forward<Arg15>( arg15 ) , std::forward<Arg16>( arg16 ) , std::forward<Arg17>( arg17 ))
                ));
            }
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16 , typename FArg17>
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            Result (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12 , FArg13 arg13 , FArg14 arg14 , FArg15 arg15 , FArg16 arg16 , FArg17 arg17) const,
            Component* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13 , Arg14 && arg14 , Arg15 && arg15 , Arg16 && arg16 , Arg17 && arg17) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                cont->trigger(std::forward<Result>(
                    (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ) , std::forward<Arg14>( arg14 ) , std::forward<Arg15>( arg15 ) , std::forward<Arg16>( arg16 ) , std::forward<Arg17>( arg17 ))
                ));
            }
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16 , typename FArg17>
    static threads::thread_function_type
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12 , FArg13 , FArg14 , FArg15 , FArg16 , FArg17), Component* obj,
        Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_18()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)) , util::get< 5>(std::forward<Arguments_>( args)) , util::get< 6>(std::forward<Arguments_>( args)) , util::get< 7>(std::forward<Arguments_>( args)) , util::get< 8>(std::forward<Arguments_>( args)) , util::get< 9>(std::forward<Arguments_>( args)) , util::get< 10>(std::forward<Arguments_>( args)) , util::get< 11>(std::forward<Arguments_>( args)) , util::get< 12>(std::forward<Arguments_>( args)) , util::get< 13>(std::forward<Arguments_>( args)) , util::get< 14>(std::forward<Arguments_>( args)) , util::get< 15>(std::forward<Arguments_>( args)) , util::get< 16>(std::forward<Arguments_>( args)) , util::get< 17>(std::forward<Arguments_>( args)));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16 , typename FArg17>
    static threads::thread_function_type
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12 , FArg13 , FArg14 , FArg15 , FArg16 , FArg17) const,
        Component* obj, Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_18()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)) , util::get< 5>(std::forward<Arguments_>( args)) , util::get< 6>(std::forward<Arguments_>( args)) , util::get< 7>(std::forward<Arguments_>( args)) , util::get< 8>(std::forward<Arguments_>( args)) , util::get< 9>(std::forward<Arguments_>( args)) , util::get< 10>(std::forward<Arguments_>( args)) , util::get< 11>(std::forward<Arguments_>( args)) , util::get< 12>(std::forward<Arguments_>( args)) , util::get< 13>(std::forward<Arguments_>( args)) , util::get< 14>(std::forward<Arguments_>( args)) , util::get< 15>(std::forward<Arguments_>( args)) , util::get< 16>(std::forward<Arguments_>( args)) , util::get< 17>(std::forward<Arguments_>( args)));
    }
    
    
    struct continuation_thread_object_function_void_19
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17 , typename Arg18
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16 , typename FArg17 , typename FArg18>
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            void (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12 , FArg13 arg13 , FArg14 arg14 , FArg15 arg15 , FArg16 arg16 , FArg17 arg17 , FArg18 arg18),
            Object* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13 , Arg14 && arg14 , Arg15 && arg15 , Arg16 && arg16 , Arg17 && arg17 , Arg18 && arg18) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ) , std::forward<Arg14>( arg14 ) , std::forward<Arg15>( arg15 ) , std::forward<Arg16>( arg16 ) , std::forward<Arg17>( arg17 ) , std::forward<Arg18>( arg18 ));
                cont->trigger();
            }
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17 , typename Arg18
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16 , typename FArg17 , typename FArg18>
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            void (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12 , FArg13 arg13 , FArg14 arg14 , FArg15 arg15 , FArg16 arg16 , FArg17 arg17 , FArg18 arg18) const,
            Component* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13 , Arg14 && arg14 , Arg15 && arg15 , Arg16 && arg16 , Arg17 && arg17 , Arg18 && arg18) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ) , std::forward<Arg14>( arg14 ) , std::forward<Arg15>( arg15 ) , std::forward<Arg16>( arg16 ) , std::forward<Arg17>( arg17 ) , std::forward<Arg18>( arg18 ));
                cont->trigger();
            }
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16 , typename FArg17 , typename FArg18>
    static threads::thread_function_type
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12 , FArg13 , FArg14 , FArg15 , FArg16 , FArg17 , FArg18), Component* obj,
        Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_void_19()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)) , util::get< 5>(std::forward<Arguments_>( args)) , util::get< 6>(std::forward<Arguments_>( args)) , util::get< 7>(std::forward<Arguments_>( args)) , util::get< 8>(std::forward<Arguments_>( args)) , util::get< 9>(std::forward<Arguments_>( args)) , util::get< 10>(std::forward<Arguments_>( args)) , util::get< 11>(std::forward<Arguments_>( args)) , util::get< 12>(std::forward<Arguments_>( args)) , util::get< 13>(std::forward<Arguments_>( args)) , util::get< 14>(std::forward<Arguments_>( args)) , util::get< 15>(std::forward<Arguments_>( args)) , util::get< 16>(std::forward<Arguments_>( args)) , util::get< 17>(std::forward<Arguments_>( args)) , util::get< 18>(std::forward<Arguments_>( args)));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16 , typename FArg17 , typename FArg18>
    static threads::thread_function_type
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12 , FArg13 , FArg14 , FArg15 , FArg16 , FArg17 , FArg18) const,
        Component* obj, Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_void_19()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)) , util::get< 5>(std::forward<Arguments_>( args)) , util::get< 6>(std::forward<Arguments_>( args)) , util::get< 7>(std::forward<Arguments_>( args)) , util::get< 8>(std::forward<Arguments_>( args)) , util::get< 9>(std::forward<Arguments_>( args)) , util::get< 10>(std::forward<Arguments_>( args)) , util::get< 11>(std::forward<Arguments_>( args)) , util::get< 12>(std::forward<Arguments_>( args)) , util::get< 13>(std::forward<Arguments_>( args)) , util::get< 14>(std::forward<Arguments_>( args)) , util::get< 15>(std::forward<Arguments_>( args)) , util::get< 16>(std::forward<Arguments_>( args)) , util::get< 17>(std::forward<Arguments_>( args)) , util::get< 18>(std::forward<Arguments_>( args)));
    }
    
    struct continuation_thread_object_function_19
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17 , typename Arg18
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16 , typename FArg17 , typename FArg18>
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            Result (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12 , FArg13 arg13 , FArg14 arg14 , FArg15 arg15 , FArg16 arg16 , FArg17 arg17 , FArg18 arg18),
            Component* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13 , Arg14 && arg14 , Arg15 && arg15 , Arg16 && arg16 , Arg17 && arg17 , Arg18 && arg18) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                cont->trigger(std::forward<Result>(
                    (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ) , std::forward<Arg14>( arg14 ) , std::forward<Arg15>( arg15 ) , std::forward<Arg16>( arg16 ) , std::forward<Arg17>( arg17 ) , std::forward<Arg18>( arg18 ))
                ));
            }
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17 , typename Arg18
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16 , typename FArg17 , typename FArg18>
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            Result (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12 , FArg13 arg13 , FArg14 arg14 , FArg15 arg15 , FArg16 arg16 , FArg17 arg17 , FArg18 arg18) const,
            Component* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13 , Arg14 && arg14 , Arg15 && arg15 , Arg16 && arg16 , Arg17 && arg17 , Arg18 && arg18) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                cont->trigger(std::forward<Result>(
                    (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ) , std::forward<Arg14>( arg14 ) , std::forward<Arg15>( arg15 ) , std::forward<Arg16>( arg16 ) , std::forward<Arg17>( arg17 ) , std::forward<Arg18>( arg18 ))
                ));
            }
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16 , typename FArg17 , typename FArg18>
    static threads::thread_function_type
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12 , FArg13 , FArg14 , FArg15 , FArg16 , FArg17 , FArg18), Component* obj,
        Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_19()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)) , util::get< 5>(std::forward<Arguments_>( args)) , util::get< 6>(std::forward<Arguments_>( args)) , util::get< 7>(std::forward<Arguments_>( args)) , util::get< 8>(std::forward<Arguments_>( args)) , util::get< 9>(std::forward<Arguments_>( args)) , util::get< 10>(std::forward<Arguments_>( args)) , util::get< 11>(std::forward<Arguments_>( args)) , util::get< 12>(std::forward<Arguments_>( args)) , util::get< 13>(std::forward<Arguments_>( args)) , util::get< 14>(std::forward<Arguments_>( args)) , util::get< 15>(std::forward<Arguments_>( args)) , util::get< 16>(std::forward<Arguments_>( args)) , util::get< 17>(std::forward<Arguments_>( args)) , util::get< 18>(std::forward<Arguments_>( args)));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16 , typename FArg17 , typename FArg18>
    static threads::thread_function_type
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12 , FArg13 , FArg14 , FArg15 , FArg16 , FArg17 , FArg18) const,
        Component* obj, Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_19()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)) , util::get< 5>(std::forward<Arguments_>( args)) , util::get< 6>(std::forward<Arguments_>( args)) , util::get< 7>(std::forward<Arguments_>( args)) , util::get< 8>(std::forward<Arguments_>( args)) , util::get< 9>(std::forward<Arguments_>( args)) , util::get< 10>(std::forward<Arguments_>( args)) , util::get< 11>(std::forward<Arguments_>( args)) , util::get< 12>(std::forward<Arguments_>( args)) , util::get< 13>(std::forward<Arguments_>( args)) , util::get< 14>(std::forward<Arguments_>( args)) , util::get< 15>(std::forward<Arguments_>( args)) , util::get< 16>(std::forward<Arguments_>( args)) , util::get< 17>(std::forward<Arguments_>( args)) , util::get< 18>(std::forward<Arguments_>( args)));
    }
    
    
    struct continuation_thread_object_function_void_20
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17 , typename Arg18 , typename Arg19
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16 , typename FArg17 , typename FArg18 , typename FArg19>
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            void (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12 , FArg13 arg13 , FArg14 arg14 , FArg15 arg15 , FArg16 arg16 , FArg17 arg17 , FArg18 arg18 , FArg19 arg19),
            Object* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13 , Arg14 && arg14 , Arg15 && arg15 , Arg16 && arg16 , Arg17 && arg17 , Arg18 && arg18 , Arg19 && arg19) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ) , std::forward<Arg14>( arg14 ) , std::forward<Arg15>( arg15 ) , std::forward<Arg16>( arg16 ) , std::forward<Arg17>( arg17 ) , std::forward<Arg18>( arg18 ) , std::forward<Arg19>( arg19 ));
                cont->trigger();
            }
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17 , typename Arg18 , typename Arg19
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16 , typename FArg17 , typename FArg18 , typename FArg19>
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            void (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12 , FArg13 arg13 , FArg14 arg14 , FArg15 arg15 , FArg16 arg16 , FArg17 arg17 , FArg18 arg18 , FArg19 arg19) const,
            Component* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13 , Arg14 && arg14 , Arg15 && arg15 , Arg16 && arg16 , Arg17 && arg17 , Arg18 && arg18 , Arg19 && arg19) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ) , std::forward<Arg14>( arg14 ) , std::forward<Arg15>( arg15 ) , std::forward<Arg16>( arg16 ) , std::forward<Arg17>( arg17 ) , std::forward<Arg18>( arg18 ) , std::forward<Arg19>( arg19 ));
                cont->trigger();
            }
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16 , typename FArg17 , typename FArg18 , typename FArg19>
    static threads::thread_function_type
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12 , FArg13 , FArg14 , FArg15 , FArg16 , FArg17 , FArg18 , FArg19), Component* obj,
        Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_void_20()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)) , util::get< 5>(std::forward<Arguments_>( args)) , util::get< 6>(std::forward<Arguments_>( args)) , util::get< 7>(std::forward<Arguments_>( args)) , util::get< 8>(std::forward<Arguments_>( args)) , util::get< 9>(std::forward<Arguments_>( args)) , util::get< 10>(std::forward<Arguments_>( args)) , util::get< 11>(std::forward<Arguments_>( args)) , util::get< 12>(std::forward<Arguments_>( args)) , util::get< 13>(std::forward<Arguments_>( args)) , util::get< 14>(std::forward<Arguments_>( args)) , util::get< 15>(std::forward<Arguments_>( args)) , util::get< 16>(std::forward<Arguments_>( args)) , util::get< 17>(std::forward<Arguments_>( args)) , util::get< 18>(std::forward<Arguments_>( args)) , util::get< 19>(std::forward<Arguments_>( args)));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16 , typename FArg17 , typename FArg18 , typename FArg19>
    static threads::thread_function_type
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12 , FArg13 , FArg14 , FArg15 , FArg16 , FArg17 , FArg18 , FArg19) const,
        Component* obj, Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_void_20()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)) , util::get< 5>(std::forward<Arguments_>( args)) , util::get< 6>(std::forward<Arguments_>( args)) , util::get< 7>(std::forward<Arguments_>( args)) , util::get< 8>(std::forward<Arguments_>( args)) , util::get< 9>(std::forward<Arguments_>( args)) , util::get< 10>(std::forward<Arguments_>( args)) , util::get< 11>(std::forward<Arguments_>( args)) , util::get< 12>(std::forward<Arguments_>( args)) , util::get< 13>(std::forward<Arguments_>( args)) , util::get< 14>(std::forward<Arguments_>( args)) , util::get< 15>(std::forward<Arguments_>( args)) , util::get< 16>(std::forward<Arguments_>( args)) , util::get< 17>(std::forward<Arguments_>( args)) , util::get< 18>(std::forward<Arguments_>( args)) , util::get< 19>(std::forward<Arguments_>( args)));
    }
    
    struct continuation_thread_object_function_20
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17 , typename Arg18 , typename Arg19
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16 , typename FArg17 , typename FArg18 , typename FArg19>
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            Result (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12 , FArg13 arg13 , FArg14 arg14 , FArg15 arg15 , FArg16 arg16 , FArg17 arg17 , FArg18 arg18 , FArg19 arg19),
            Component* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13 , Arg14 && arg14 , Arg15 && arg15 , Arg16 && arg16 , Arg17 && arg17 , Arg18 && arg18 , Arg19 && arg19) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                cont->trigger(std::forward<Result>(
                    (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ) , std::forward<Arg14>( arg14 ) , std::forward<Arg15>( arg15 ) , std::forward<Arg16>( arg16 ) , std::forward<Arg17>( arg17 ) , std::forward<Arg18>( arg18 ) , std::forward<Arg19>( arg19 ))
                ));
            }
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17 , typename Arg18 , typename Arg19
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16 , typename FArg17 , typename FArg18 , typename FArg19>
        HPX_MAYBE_FORCEINLINE result_type operator()(continuation_type cont,
            Result (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12 , FArg13 arg13 , FArg14 arg14 , FArg15 arg15 , FArg16 arg16 , FArg17 arg17 , FArg18 arg18 , FArg19 arg19) const,
            Component* obj
          , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13 , Arg14 && arg14 , Arg15 && arg15 , Arg16 && arg16 , Arg17 && arg17 , Arg18 && arg18 , Arg19 && arg19) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                cont->trigger(std::forward<Result>(
                    (obj->*func)(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ) , std::forward<Arg14>( arg14 ) , std::forward<Arg15>( arg15 ) , std::forward<Arg16>( arg16 ) , std::forward<Arg17>( arg17 ) , std::forward<Arg18>( arg18 ) , std::forward<Arg19>( arg19 ))
                ));
            }
            catch (...) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16 , typename FArg17 , typename FArg18 , typename FArg19>
    static threads::thread_function_type
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12 , FArg13 , FArg14 , FArg15 , FArg16 , FArg17 , FArg18 , FArg19), Component* obj,
        Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_20()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)) , util::get< 5>(std::forward<Arguments_>( args)) , util::get< 6>(std::forward<Arguments_>( args)) , util::get< 7>(std::forward<Arguments_>( args)) , util::get< 8>(std::forward<Arguments_>( args)) , util::get< 9>(std::forward<Arguments_>( args)) , util::get< 10>(std::forward<Arguments_>( args)) , util::get< 11>(std::forward<Arguments_>( args)) , util::get< 12>(std::forward<Arguments_>( args)) , util::get< 13>(std::forward<Arguments_>( args)) , util::get< 14>(std::forward<Arguments_>( args)) , util::get< 15>(std::forward<Arguments_>( args)) , util::get< 16>(std::forward<Arguments_>( args)) , util::get< 17>(std::forward<Arguments_>( args)) , util::get< 18>(std::forward<Arguments_>( args)) , util::get< 19>(std::forward<Arguments_>( args)));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16 , typename FArg17 , typename FArg18 , typename FArg19>
    static threads::thread_function_type
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12 , FArg13 , FArg14 , FArg15 , FArg16 , FArg17 , FArg18 , FArg19) const,
        Component* obj, Arguments_ && args)
    {
        return util::bind(util::one_shot(
            continuation_thread_object_function_20()),
            cont, func, obj
          ,
                util::get< 0>(std::forward<Arguments_>( args)) , util::get< 1>(std::forward<Arguments_>( args)) , util::get< 2>(std::forward<Arguments_>( args)) , util::get< 3>(std::forward<Arguments_>( args)) , util::get< 4>(std::forward<Arguments_>( args)) , util::get< 5>(std::forward<Arguments_>( args)) , util::get< 6>(std::forward<Arguments_>( args)) , util::get< 7>(std::forward<Arguments_>( args)) , util::get< 8>(std::forward<Arguments_>( args)) , util::get< 9>(std::forward<Arguments_>( args)) , util::get< 10>(std::forward<Arguments_>( args)) , util::get< 11>(std::forward<Arguments_>( args)) , util::get< 12>(std::forward<Arguments_>( args)) , util::get< 13>(std::forward<Arguments_>( args)) , util::get< 14>(std::forward<Arguments_>( args)) , util::get< 15>(std::forward<Arguments_>( args)) , util::get< 16>(std::forward<Arguments_>( args)) , util::get< 17>(std::forward<Arguments_>( args)) , util::get< 18>(std::forward<Arguments_>( args)) , util::get< 19>(std::forward<Arguments_>( args)));
    }
