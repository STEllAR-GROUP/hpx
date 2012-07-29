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
    
    
    struct continuation_thread_object_function_void_11
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10>
        result_type operator()(continuation_type cont,
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
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10>
        result_type operator()(continuation_type cont,
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
            catch (hpx::exception const&) {
                
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
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type10>::call( args. a10));
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
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type10>::call( args. a10));
    }
    
    struct continuation_thread_object_function_11
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10>
        result_type operator()(continuation_type cont,
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
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10>
        result_type operator()(continuation_type cont,
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
            catch (hpx::exception const&) {
                
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
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type10>::call( args. a10));
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
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type10>::call( args. a10));
    }
    
    
    struct continuation_thread_object_function_void_12
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11>
        result_type operator()(continuation_type cont,
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
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11>
        result_type operator()(continuation_type cont,
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
            catch (hpx::exception const&) {
                
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
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type11>::call( args. a11));
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
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type11>::call( args. a11));
    }
    
    struct continuation_thread_object_function_12
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11>
        result_type operator()(continuation_type cont,
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
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11>
        result_type operator()(continuation_type cont,
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
            catch (hpx::exception const&) {
                
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
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type11>::call( args. a11));
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
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type11>::call( args. a11));
    }
    
    
    struct continuation_thread_object_function_void_13
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12>
        result_type operator()(continuation_type cont,
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
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12>
        result_type operator()(continuation_type cont,
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
            catch (hpx::exception const&) {
                
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
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type12>::call( args. a12));
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
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type12>::call( args. a12));
    }
    
    struct continuation_thread_object_function_13
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12>
        result_type operator()(continuation_type cont,
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
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12>
        result_type operator()(continuation_type cont,
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
            catch (hpx::exception const&) {
                
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
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type12>::call( args. a12));
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
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type12>::call( args. a12));
    }
    
    
    struct continuation_thread_object_function_void_14
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13>
        result_type operator()(continuation_type cont,
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
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13>
        result_type operator()(continuation_type cont,
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
            catch (hpx::exception const&) {
                
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
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type12>::call( args. a12) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type13>::call( args. a13));
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
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type12>::call( args. a12) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type13>::call( args. a13));
    }
    
    struct continuation_thread_object_function_14
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13>
        result_type operator()(continuation_type cont,
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
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13>
        result_type operator()(continuation_type cont,
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
            catch (hpx::exception const&) {
                
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
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type12>::call( args. a12) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type13>::call( args. a13));
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
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type12>::call( args. a12) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type13>::call( args. a13));
    }
    
    
    struct continuation_thread_object_function_void_15
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14>
        result_type operator()(continuation_type cont,
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
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14>
        result_type operator()(continuation_type cont,
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
            catch (hpx::exception const&) {
                
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
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type12>::call( args. a12) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type13>::call( args. a13) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type14>::call( args. a14));
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
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type12>::call( args. a12) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type13>::call( args. a13) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type14>::call( args. a14));
    }
    
    struct continuation_thread_object_function_15
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14>
        result_type operator()(continuation_type cont,
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
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14>
        result_type operator()(continuation_type cont,
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
            catch (hpx::exception const&) {
                
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
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type12>::call( args. a12) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type13>::call( args. a13) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type14>::call( args. a14));
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
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type12>::call( args. a12) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type13>::call( args. a13) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type14>::call( args. a14));
    }
    
    
    struct continuation_thread_object_function_void_16
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15>
        result_type operator()(continuation_type cont,
            void (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12 , FArg13 arg13 , FArg14 arg14 , FArg15 arg15),
            Object* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14 , BOOST_FWD_REF(Arg15) arg15) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8) , boost::move(arg9) , boost::move(arg10) , boost::move(arg11) , boost::move(arg12) , boost::move(arg13) , boost::move(arg14) , boost::move(arg15));
                cont->trigger();
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15>
        result_type operator()(continuation_type cont,
            void (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12 , FArg13 arg13 , FArg14 arg14 , FArg15 arg15) const,
            Component* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14 , BOOST_FWD_REF(Arg15) arg15) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8) , boost::move(arg9) , boost::move(arg10) , boost::move(arg11) , boost::move(arg12) , boost::move(arg13) , boost::move(arg14) , boost::move(arg15));
                cont->trigger();
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12 , FArg13 , FArg14 , FArg15), Object* obj,
        BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_void_16(),
            cont, func, obj
          ,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type12>::call( args. a12) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type13>::call( args. a13) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type14>::call( args. a14) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type15>::call( args. a15));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12 , FArg13 , FArg14 , FArg15) const,
        Component* obj, BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_void_16(),
            cont, func, obj
          ,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type12>::call( args. a12) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type13>::call( args. a13) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type14>::call( args. a14) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type15>::call( args. a15));
    }
    
    struct continuation_thread_object_function_16
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15>
        result_type operator()(continuation_type cont,
            Result (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12 , FArg13 arg13 , FArg14 arg14 , FArg15 arg15),
            Component* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14 , BOOST_FWD_REF(Arg15) arg15) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                cont->trigger(boost::move(
                    (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8) , boost::move(arg9) , boost::move(arg10) , boost::move(arg11) , boost::move(arg12) , boost::move(arg13) , boost::move(arg14) , boost::move(arg15))
                ));
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15>
        result_type operator()(continuation_type cont,
            Result (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12 , FArg13 arg13 , FArg14 arg14 , FArg15 arg15) const,
            Component* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14 , BOOST_FWD_REF(Arg15) arg15) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                cont->trigger(boost::move(
                    (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8) , boost::move(arg9) , boost::move(arg10) , boost::move(arg11) , boost::move(arg12) , boost::move(arg13) , boost::move(arg14) , boost::move(arg15))
                ));
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12 , FArg13 , FArg14 , FArg15), Component* obj,
        BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_16(),
            cont, func, obj
          ,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type12>::call( args. a12) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type13>::call( args. a13) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type14>::call( args. a14) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type15>::call( args. a15));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12 , FArg13 , FArg14 , FArg15) const,
        Component* obj, BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_16(),
            cont, func, obj
          ,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type12>::call( args. a12) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type13>::call( args. a13) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type14>::call( args. a14) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type15>::call( args. a15));
    }
    
    
    struct continuation_thread_object_function_void_17
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16>
        result_type operator()(continuation_type cont,
            void (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12 , FArg13 arg13 , FArg14 arg14 , FArg15 arg15 , FArg16 arg16),
            Object* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14 , BOOST_FWD_REF(Arg15) arg15 , BOOST_FWD_REF(Arg16) arg16) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8) , boost::move(arg9) , boost::move(arg10) , boost::move(arg11) , boost::move(arg12) , boost::move(arg13) , boost::move(arg14) , boost::move(arg15) , boost::move(arg16));
                cont->trigger();
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16>
        result_type operator()(continuation_type cont,
            void (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12 , FArg13 arg13 , FArg14 arg14 , FArg15 arg15 , FArg16 arg16) const,
            Component* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14 , BOOST_FWD_REF(Arg15) arg15 , BOOST_FWD_REF(Arg16) arg16) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8) , boost::move(arg9) , boost::move(arg10) , boost::move(arg11) , boost::move(arg12) , boost::move(arg13) , boost::move(arg14) , boost::move(arg15) , boost::move(arg16));
                cont->trigger();
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12 , FArg13 , FArg14 , FArg15 , FArg16), Object* obj,
        BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_void_17(),
            cont, func, obj
          ,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type12>::call( args. a12) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type13>::call( args. a13) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type14>::call( args. a14) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type15>::call( args. a15) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type16>::call( args. a16));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12 , FArg13 , FArg14 , FArg15 , FArg16) const,
        Component* obj, BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_void_17(),
            cont, func, obj
          ,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type12>::call( args. a12) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type13>::call( args. a13) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type14>::call( args. a14) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type15>::call( args. a15) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type16>::call( args. a16));
    }
    
    struct continuation_thread_object_function_17
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16>
        result_type operator()(continuation_type cont,
            Result (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12 , FArg13 arg13 , FArg14 arg14 , FArg15 arg15 , FArg16 arg16),
            Component* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14 , BOOST_FWD_REF(Arg15) arg15 , BOOST_FWD_REF(Arg16) arg16) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                cont->trigger(boost::move(
                    (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8) , boost::move(arg9) , boost::move(arg10) , boost::move(arg11) , boost::move(arg12) , boost::move(arg13) , boost::move(arg14) , boost::move(arg15) , boost::move(arg16))
                ));
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16>
        result_type operator()(continuation_type cont,
            Result (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12 , FArg13 arg13 , FArg14 arg14 , FArg15 arg15 , FArg16 arg16) const,
            Component* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14 , BOOST_FWD_REF(Arg15) arg15 , BOOST_FWD_REF(Arg16) arg16) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                cont->trigger(boost::move(
                    (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8) , boost::move(arg9) , boost::move(arg10) , boost::move(arg11) , boost::move(arg12) , boost::move(arg13) , boost::move(arg14) , boost::move(arg15) , boost::move(arg16))
                ));
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12 , FArg13 , FArg14 , FArg15 , FArg16), Component* obj,
        BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_17(),
            cont, func, obj
          ,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type12>::call( args. a12) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type13>::call( args. a13) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type14>::call( args. a14) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type15>::call( args. a15) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type16>::call( args. a16));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12 , FArg13 , FArg14 , FArg15 , FArg16) const,
        Component* obj, BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_17(),
            cont, func, obj
          ,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type12>::call( args. a12) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type13>::call( args. a13) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type14>::call( args. a14) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type15>::call( args. a15) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type16>::call( args. a16));
    }
    
    
    struct continuation_thread_object_function_void_18
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16 , typename FArg17>
        result_type operator()(continuation_type cont,
            void (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12 , FArg13 arg13 , FArg14 arg14 , FArg15 arg15 , FArg16 arg16 , FArg17 arg17),
            Object* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14 , BOOST_FWD_REF(Arg15) arg15 , BOOST_FWD_REF(Arg16) arg16 , BOOST_FWD_REF(Arg17) arg17) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8) , boost::move(arg9) , boost::move(arg10) , boost::move(arg11) , boost::move(arg12) , boost::move(arg13) , boost::move(arg14) , boost::move(arg15) , boost::move(arg16) , boost::move(arg17));
                cont->trigger();
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16 , typename FArg17>
        result_type operator()(continuation_type cont,
            void (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12 , FArg13 arg13 , FArg14 arg14 , FArg15 arg15 , FArg16 arg16 , FArg17 arg17) const,
            Component* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14 , BOOST_FWD_REF(Arg15) arg15 , BOOST_FWD_REF(Arg16) arg16 , BOOST_FWD_REF(Arg17) arg17) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8) , boost::move(arg9) , boost::move(arg10) , boost::move(arg11) , boost::move(arg12) , boost::move(arg13) , boost::move(arg14) , boost::move(arg15) , boost::move(arg16) , boost::move(arg17));
                cont->trigger();
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16 , typename FArg17>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12 , FArg13 , FArg14 , FArg15 , FArg16 , FArg17), Object* obj,
        BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_void_18(),
            cont, func, obj
          ,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type12>::call( args. a12) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type13>::call( args. a13) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type14>::call( args. a14) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type15>::call( args. a15) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type16>::call( args. a16) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type17>::call( args. a17));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16 , typename FArg17>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12 , FArg13 , FArg14 , FArg15 , FArg16 , FArg17) const,
        Component* obj, BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_void_18(),
            cont, func, obj
          ,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type12>::call( args. a12) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type13>::call( args. a13) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type14>::call( args. a14) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type15>::call( args. a15) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type16>::call( args. a16) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type17>::call( args. a17));
    }
    
    struct continuation_thread_object_function_18
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16 , typename FArg17>
        result_type operator()(continuation_type cont,
            Result (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12 , FArg13 arg13 , FArg14 arg14 , FArg15 arg15 , FArg16 arg16 , FArg17 arg17),
            Component* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14 , BOOST_FWD_REF(Arg15) arg15 , BOOST_FWD_REF(Arg16) arg16 , BOOST_FWD_REF(Arg17) arg17) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                cont->trigger(boost::move(
                    (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8) , boost::move(arg9) , boost::move(arg10) , boost::move(arg11) , boost::move(arg12) , boost::move(arg13) , boost::move(arg14) , boost::move(arg15) , boost::move(arg16) , boost::move(arg17))
                ));
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16 , typename FArg17>
        result_type operator()(continuation_type cont,
            Result (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12 , FArg13 arg13 , FArg14 arg14 , FArg15 arg15 , FArg16 arg16 , FArg17 arg17) const,
            Component* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14 , BOOST_FWD_REF(Arg15) arg15 , BOOST_FWD_REF(Arg16) arg16 , BOOST_FWD_REF(Arg17) arg17) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                cont->trigger(boost::move(
                    (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8) , boost::move(arg9) , boost::move(arg10) , boost::move(arg11) , boost::move(arg12) , boost::move(arg13) , boost::move(arg14) , boost::move(arg15) , boost::move(arg16) , boost::move(arg17))
                ));
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16 , typename FArg17>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12 , FArg13 , FArg14 , FArg15 , FArg16 , FArg17), Component* obj,
        BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_18(),
            cont, func, obj
          ,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type12>::call( args. a12) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type13>::call( args. a13) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type14>::call( args. a14) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type15>::call( args. a15) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type16>::call( args. a16) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type17>::call( args. a17));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16 , typename FArg17>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12 , FArg13 , FArg14 , FArg15 , FArg16 , FArg17) const,
        Component* obj, BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_18(),
            cont, func, obj
          ,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type12>::call( args. a12) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type13>::call( args. a13) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type14>::call( args. a14) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type15>::call( args. a15) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type16>::call( args. a16) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type17>::call( args. a17));
    }
    
    
    struct continuation_thread_object_function_void_19
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17 , typename Arg18
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16 , typename FArg17 , typename FArg18>
        result_type operator()(continuation_type cont,
            void (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12 , FArg13 arg13 , FArg14 arg14 , FArg15 arg15 , FArg16 arg16 , FArg17 arg17 , FArg18 arg18),
            Object* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14 , BOOST_FWD_REF(Arg15) arg15 , BOOST_FWD_REF(Arg16) arg16 , BOOST_FWD_REF(Arg17) arg17 , BOOST_FWD_REF(Arg18) arg18) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8) , boost::move(arg9) , boost::move(arg10) , boost::move(arg11) , boost::move(arg12) , boost::move(arg13) , boost::move(arg14) , boost::move(arg15) , boost::move(arg16) , boost::move(arg17) , boost::move(arg18));
                cont->trigger();
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17 , typename Arg18
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16 , typename FArg17 , typename FArg18>
        result_type operator()(continuation_type cont,
            void (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12 , FArg13 arg13 , FArg14 arg14 , FArg15 arg15 , FArg16 arg16 , FArg17 arg17 , FArg18 arg18) const,
            Component* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14 , BOOST_FWD_REF(Arg15) arg15 , BOOST_FWD_REF(Arg16) arg16 , BOOST_FWD_REF(Arg17) arg17 , BOOST_FWD_REF(Arg18) arg18) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8) , boost::move(arg9) , boost::move(arg10) , boost::move(arg11) , boost::move(arg12) , boost::move(arg13) , boost::move(arg14) , boost::move(arg15) , boost::move(arg16) , boost::move(arg17) , boost::move(arg18));
                cont->trigger();
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16 , typename FArg17 , typename FArg18>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12 , FArg13 , FArg14 , FArg15 , FArg16 , FArg17 , FArg18), Object* obj,
        BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_void_19(),
            cont, func, obj
          ,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type12>::call( args. a12) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type13>::call( args. a13) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type14>::call( args. a14) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type15>::call( args. a15) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type16>::call( args. a16) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type17>::call( args. a17) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type18>::call( args. a18));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16 , typename FArg17 , typename FArg18>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12 , FArg13 , FArg14 , FArg15 , FArg16 , FArg17 , FArg18) const,
        Component* obj, BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_void_19(),
            cont, func, obj
          ,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type12>::call( args. a12) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type13>::call( args. a13) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type14>::call( args. a14) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type15>::call( args. a15) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type16>::call( args. a16) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type17>::call( args. a17) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type18>::call( args. a18));
    }
    
    struct continuation_thread_object_function_19
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17 , typename Arg18
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16 , typename FArg17 , typename FArg18>
        result_type operator()(continuation_type cont,
            Result (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12 , FArg13 arg13 , FArg14 arg14 , FArg15 arg15 , FArg16 arg16 , FArg17 arg17 , FArg18 arg18),
            Component* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14 , BOOST_FWD_REF(Arg15) arg15 , BOOST_FWD_REF(Arg16) arg16 , BOOST_FWD_REF(Arg17) arg17 , BOOST_FWD_REF(Arg18) arg18) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                cont->trigger(boost::move(
                    (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8) , boost::move(arg9) , boost::move(arg10) , boost::move(arg11) , boost::move(arg12) , boost::move(arg13) , boost::move(arg14) , boost::move(arg15) , boost::move(arg16) , boost::move(arg17) , boost::move(arg18))
                ));
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17 , typename Arg18
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16 , typename FArg17 , typename FArg18>
        result_type operator()(continuation_type cont,
            Result (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12 , FArg13 arg13 , FArg14 arg14 , FArg15 arg15 , FArg16 arg16 , FArg17 arg17 , FArg18 arg18) const,
            Component* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14 , BOOST_FWD_REF(Arg15) arg15 , BOOST_FWD_REF(Arg16) arg16 , BOOST_FWD_REF(Arg17) arg17 , BOOST_FWD_REF(Arg18) arg18) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                cont->trigger(boost::move(
                    (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8) , boost::move(arg9) , boost::move(arg10) , boost::move(arg11) , boost::move(arg12) , boost::move(arg13) , boost::move(arg14) , boost::move(arg15) , boost::move(arg16) , boost::move(arg17) , boost::move(arg18))
                ));
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16 , typename FArg17 , typename FArg18>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12 , FArg13 , FArg14 , FArg15 , FArg16 , FArg17 , FArg18), Component* obj,
        BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_19(),
            cont, func, obj
          ,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type12>::call( args. a12) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type13>::call( args. a13) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type14>::call( args. a14) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type15>::call( args. a15) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type16>::call( args. a16) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type17>::call( args. a17) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type18>::call( args. a18));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16 , typename FArg17 , typename FArg18>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12 , FArg13 , FArg14 , FArg15 , FArg16 , FArg17 , FArg18) const,
        Component* obj, BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_19(),
            cont, func, obj
          ,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type12>::call( args. a12) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type13>::call( args. a13) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type14>::call( args. a14) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type15>::call( args. a15) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type16>::call( args. a16) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type17>::call( args. a17) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type18>::call( args. a18));
    }
    
    
    struct continuation_thread_object_function_void_20
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17 , typename Arg18 , typename Arg19
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16 , typename FArg17 , typename FArg18 , typename FArg19>
        result_type operator()(continuation_type cont,
            void (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12 , FArg13 arg13 , FArg14 arg14 , FArg15 arg15 , FArg16 arg16 , FArg17 arg17 , FArg18 arg18 , FArg19 arg19),
            Object* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14 , BOOST_FWD_REF(Arg15) arg15 , BOOST_FWD_REF(Arg16) arg16 , BOOST_FWD_REF(Arg17) arg17 , BOOST_FWD_REF(Arg18) arg18 , BOOST_FWD_REF(Arg19) arg19) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8) , boost::move(arg9) , boost::move(arg10) , boost::move(arg11) , boost::move(arg12) , boost::move(arg13) , boost::move(arg14) , boost::move(arg15) , boost::move(arg16) , boost::move(arg17) , boost::move(arg18) , boost::move(arg19));
                cont->trigger();
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17 , typename Arg18 , typename Arg19
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16 , typename FArg17 , typename FArg18 , typename FArg19>
        result_type operator()(continuation_type cont,
            void (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12 , FArg13 arg13 , FArg14 arg14 , FArg15 arg15 , FArg16 arg16 , FArg17 arg17 , FArg18 arg18 , FArg19 arg19) const,
            Component* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14 , BOOST_FWD_REF(Arg15) arg15 , BOOST_FWD_REF(Arg16) arg16 , BOOST_FWD_REF(Arg17) arg17 , BOOST_FWD_REF(Arg18) arg18 , BOOST_FWD_REF(Arg19) arg19) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8) , boost::move(arg9) , boost::move(arg10) , boost::move(arg11) , boost::move(arg12) , boost::move(arg13) , boost::move(arg14) , boost::move(arg15) , boost::move(arg16) , boost::move(arg17) , boost::move(arg18) , boost::move(arg19));
                cont->trigger();
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16 , typename FArg17 , typename FArg18 , typename FArg19>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12 , FArg13 , FArg14 , FArg15 , FArg16 , FArg17 , FArg18 , FArg19), Object* obj,
        BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_void_20(),
            cont, func, obj
          ,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type12>::call( args. a12) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type13>::call( args. a13) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type14>::call( args. a14) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type15>::call( args. a15) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type16>::call( args. a16) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type17>::call( args. a17) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type18>::call( args. a18) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type19>::call( args. a19));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16 , typename FArg17 , typename FArg18 , typename FArg19>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function_void(
        continuation_type cont,
        void (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12 , FArg13 , FArg14 , FArg15 , FArg16 , FArg17 , FArg18 , FArg19) const,
        Component* obj, BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_void_20(),
            cont, func, obj
          ,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type12>::call( args. a12) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type13>::call( args. a13) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type14>::call( args. a14) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type15>::call( args. a15) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type16>::call( args. a16) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type17>::call( args. a17) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type18>::call( args. a18) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type19>::call( args. a19));
    }
    
    struct continuation_thread_object_function_20
    {
        typedef threads::thread_state_enum result_type;
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17 , typename Arg18 , typename Arg19
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16 , typename FArg17 , typename FArg18 , typename FArg19>
        result_type operator()(continuation_type cont,
            Result (Object::* func)(FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12 , FArg13 arg13 , FArg14 arg14 , FArg15 arg15 , FArg16 arg16 , FArg17 arg17 , FArg18 arg18 , FArg19 arg19),
            Component* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14 , BOOST_FWD_REF(Arg15) arg15 , BOOST_FWD_REF(Arg16) arg16 , BOOST_FWD_REF(Arg17) arg17 , BOOST_FWD_REF(Arg18) arg18 , BOOST_FWD_REF(Arg19) arg19) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                cont->trigger(boost::move(
                    (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8) , boost::move(arg9) , boost::move(arg10) , boost::move(arg11) , boost::move(arg12) , boost::move(arg13) , boost::move(arg14) , boost::move(arg15) , boost::move(arg16) , boost::move(arg17) , boost::move(arg18) , boost::move(arg19))
                ));
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
        template <typename Object
            , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17 , typename Arg18 , typename Arg19
            , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16 , typename FArg17 , typename FArg18 , typename FArg19>
        result_type operator()(continuation_type cont,
            Result (Object::* const func)(
                FArg0 arg0 , FArg1 arg1 , FArg2 arg2 , FArg3 arg3 , FArg4 arg4 , FArg5 arg5 , FArg6 arg6 , FArg7 arg7 , FArg8 arg8 , FArg9 arg9 , FArg10 arg10 , FArg11 arg11 , FArg12 arg12 , FArg13 arg13 , FArg14 arg14 , FArg15 arg15 , FArg16 arg16 , FArg17 arg17 , FArg18 arg18 , FArg19 arg19) const,
            Component* obj
          , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14 , BOOST_FWD_REF(Arg15) arg15 , BOOST_FWD_REF(Arg16) arg16 , BOOST_FWD_REF(Arg17) arg17 , BOOST_FWD_REF(Arg18) arg18 , BOOST_FWD_REF(Arg19) arg19) const
        {
            try {
                LTM_(debug) << "Executing action("
                    << detail::get_action_name<derived_type>()
                    << ") with continuation(" << cont->get_gid() << ")";
                
                
                
                
                cont->trigger(boost::move(
                    (obj->*func)(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8) , boost::move(arg9) , boost::move(arg10) , boost::move(arg11) , boost::move(arg12) , boost::move(arg13) , boost::move(arg14) , boost::move(arg15) , boost::move(arg16) , boost::move(arg17) , boost::move(arg18) , boost::move(arg19))
                ));
            }
            catch (hpx::exception const&) {
                
                cont->trigger_error(boost::current_exception());
            }
            return threads::terminated;
        }
    };
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16 , typename FArg17 , typename FArg18 , typename FArg19>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12 , FArg13 , FArg14 , FArg15 , FArg16 , FArg17 , FArg18 , FArg19), Component* obj,
        BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_20(),
            cont, func, obj
          ,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type12>::call( args. a12) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type13>::call( args. a13) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type14>::call( args. a14) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type15>::call( args. a15) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type16>::call( args. a16) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type17>::call( args. a17) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type18>::call( args. a18) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type19>::call( args. a19));
    }
    template <typename Object, typename Arguments_
        , typename FArg0 , typename FArg1 , typename FArg2 , typename FArg3 , typename FArg4 , typename FArg5 , typename FArg6 , typename FArg7 , typename FArg8 , typename FArg9 , typename FArg10 , typename FArg11 , typename FArg12 , typename FArg13 , typename FArg14 , typename FArg15 , typename FArg16 , typename FArg17 , typename FArg18 , typename FArg19>
    static HPX_STD_FUNCTION<threads::thread_function_type>
    construct_continuation_thread_object_function(
        continuation_type cont,
        Result (Object::* const func)(FArg0 , FArg1 , FArg2 , FArg3 , FArg4 , FArg5 , FArg6 , FArg7 , FArg8 , FArg9 , FArg10 , FArg11 , FArg12 , FArg13 , FArg14 , FArg15 , FArg16 , FArg17 , FArg18 , FArg19) const,
        Component* obj, BOOST_FWD_REF(Arguments_) args)
    {
        return HPX_STD_BIND(
            continuation_thread_object_function_20(),
            cont, func, obj
          ,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type12>::call( args. a12) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type13>::call( args. a13) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type14>::call( args. a14) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type15>::call( args. a15) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type16>::call( args. a16) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type17>::call( args. a17) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type18>::call( args. a18) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments_>::type:: member_type19>::call( args. a19));
    }
