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
