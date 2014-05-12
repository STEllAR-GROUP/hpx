// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


namespace hpx { namespace actions
{
    
    
    template <
        typename Component, typename Result, typename T0,
        Result (Component::*F)(T0), typename Derived>
    class base_result_action1<
            Result (Component::*)(T0), F, Derived>
      : public action<
            Component, Result,
            hpx::util::tuple<typename util::decay<T0>::type>,
            Derived>
    {
    public:
        typedef Result result_type;
        typedef typename detail::remote_action_result<Result>::type
            remote_result_type;
        typedef hpx::util::tuple<
            typename util::decay<T0>::type> arguments_type;
        typedef action<Component, result_type, arguments_type, Derived>
            base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                Arg0 && arg0) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component>::call(lva)) << ")";
                    (get_lva<Component>::call(lva)->*F)(
                        std::forward<Arg0>( arg0 ));
                }
                catch (hpx::thread_interrupted const&) {
                     
                }
                catch (hpx::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component>::call(lva)) << "): " << e.what();
                    
                    hpx::report_error(boost::current_exception());
                }
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component>::call(lva)) << ")";
                    
                    hpx::report_error(boost::current_exception());
                }
                
                
                
                util::force_error_on_lock();
                return threads::terminated;
            }
        };
    public:
        typedef boost::mpl::false_ direct_execution;
        
        
        
        
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(naming::address::address_type lva,
            Arguments && args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                util::bind(util::one_shot(typename Derived::thread_function()),
                    lva, util::get< 0>(std::forward<Arguments>( args))));
        }
        
        
        
        
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, Arguments && args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                base_type::construct_continuation_thread_object_function(
                    cont, F, get_lva<Component>::call(lva),
                    std::forward<Arguments>(args)));
        }
        
        template <typename Arguments>
        BOOST_FORCEINLINE static Result
        execute_function(naming::address::address_type lva,
            Arguments && args)
        {
            LTM_(debug)
                << "base_result_action" << 1
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            return (get_lva<Component>::call(lva)->*F)(
                util::get< 0>(std::forward<Arguments>( args)));
        }
    };
    
    
    template <
        typename Component, typename Result, typename T0,
        Result (Component::*F)(T0), typename Derived>
    struct result_action1<
            Result (Component::*)(T0), F, Derived>
      : base_result_action1<
            Result (Component::*)(T0), F,
            typename detail::action_type<
                result_action1<
                    Result (Component::*)(T0), F, Derived>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            result_action1, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    
    template <
        typename Component, typename Result, typename T0,
        Result (Component::*F)(T0), typename Derived>
    struct make_action<Result (Component::*)(T0),
            F, Derived, boost::mpl::false_>
      : result_action1<
            Result (Component::*)(T0), F, Derived>
    {
        typedef result_action1<
            Result (Component::*)(T0), F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename Result,
        typename T0,
        Result (Component::*F)(T0),
        typename Derived>
    struct direct_result_action1<
            Result (Component::*)(T0), F, Derived>
      : base_result_action1<
            Result (Component::*)(T0), F,
            typename detail::action_type<
                direct_result_action1<
                    Result (Component::*)(T0), F, Derived>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_result_action1, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    
    template <
        typename Component, typename Result, typename T0,
        Result (Component::*F)(T0), typename Derived>
    struct make_action<Result (Component::*)(T0),
            F, Derived, boost::mpl::true_>
      : direct_result_action1<
            Result (Component::*)(T0), F, Derived>
    {
        typedef direct_result_action1<
            Result (Component::*)(T0), F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename T0,
        void (Component::*F)(T0), typename Derived>
    class base_action1<
            void (Component::*)(T0), F, Derived>
      : public action<
            Component, util::unused_type,
            hpx::util::tuple<typename util::decay<T0>::type>,
            Derived>
    {
    public:
        typedef util::unused_type result_type;
        typedef util::unused_type remote_result_type;
        typedef hpx::util::tuple<
            typename util::decay<T0>::type> arguments_type;
        typedef action<Component, result_type, arguments_type, Derived>
            base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                Arg0 && arg0) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component>::call(lva)) << ")";
                    (get_lva<Component>::call(lva)->*F)(
                        std::forward<Arg0>( arg0 ));
                }
                catch (hpx::thread_interrupted const&) {
                     
                }
                catch (hpx::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component>::call(lva)) << "): " << e.what();
                    
                    hpx::report_error(boost::current_exception());
                }
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component>::call(lva)) << ")";
                    
                    hpx::report_error(boost::current_exception());
                }
                
                
                
                util::force_error_on_lock();
                return threads::terminated;
            }
        };
    public:
        
        
        
        
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(naming::address::address_type lva,
            Arguments && args)
        {
            
            
            return traits::action_decorate_function<Derived>::call(lva,
                util::bind(util::one_shot(typename Derived::thread_function()), lva,
                    util::get< 0>(std::forward<Arguments>( args))));
        }
        
        
        
        
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, Arguments && args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                base_type::construct_continuation_thread_object_function_void(
                    cont, F, get_lva<Component>::call(lva),
                    std::forward<Arguments>(args)));
        }
        
        template <typename Arguments>
        BOOST_FORCEINLINE static util::unused_type
        execute_function(naming::address::address_type lva,
            Arguments && args)
        {
            LTM_(debug)
                << "base_action" << 1
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            (get_lva<Component>::call(lva)->*F)(
                util::get< 0>(std::forward<Arguments>( args)));
            return util::unused;
        }
    };
    
    template <
        typename Component, typename T0,
        void (Component::*F)(T0),
        typename Derived>
    struct action1<
            void (Component::*)(T0), F, Derived>
      : base_action1<
            void (Component::*)(T0), F,
            typename detail::action_type<
                action1<
                    void (Component::*)(T0), F, Derived>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            action1, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    
    template <
        typename Component, typename T0,
        void (Component::*F)(T0), typename Derived>
    struct make_action<void (Component::*)(T0),
            F, Derived, boost::mpl::false_>
      : action1<
            void (Component::*)(T0), F, Derived>
    {
        typedef action1<
            void (Component::*)(T0), F, Derived
        > type;
    };
    
    template <
        typename Component, typename T0,
        void (Component::*F)(T0),
        typename Derived>
    struct direct_action1<
            void (Component::*)(T0), F, Derived>
      : base_action1<
            void (Component::*)(T0), F,
            typename detail::action_type<
                direct_action1<
                    void (Component::*)(T0), F, Derived>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_action1, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    
    template <
        typename Component, typename T0,
        void (Component::*F)(T0), typename Derived>
    struct make_action<void (Component::*)(T0),
            F, Derived, boost::mpl::true_>
      : direct_action1<
            void (Component::*)(T0), F, Derived>
    {
        typedef direct_action1<
            void (Component::*)(T0), F, Derived
        > type;
    };
    
    
    template <
        typename Component,
        typename T0,
        void (Component::*F)(T0),
        typename Derived>
    struct result_action1<
            void (Component::*)(T0), F, Derived>
      : action1<
            void (Component::*)(T0), F, Derived>
    {};
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Component, typename Result, typename T0 , typename T1,
        Result (Component::*F)(T0 , T1), typename Derived>
    class base_result_action2<
            Result (Component::*)(T0 , T1), F, Derived>
      : public action<
            Component, Result,
            hpx::util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type>,
            Derived>
    {
    public:
        typedef Result result_type;
        typedef typename detail::remote_action_result<Result>::type
            remote_result_type;
        typedef hpx::util::tuple<
            typename util::decay<T0>::type , typename util::decay<T1>::type> arguments_type;
        typedef action<Component, result_type, arguments_type, Derived>
            base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                Arg0 && arg0 , Arg1 && arg1) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component>::call(lva)) << ")";
                    (get_lva<Component>::call(lva)->*F)(
                        std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
                }
                catch (hpx::thread_interrupted const&) {
                     
                }
                catch (hpx::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component>::call(lva)) << "): " << e.what();
                    
                    hpx::report_error(boost::current_exception());
                }
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component>::call(lva)) << ")";
                    
                    hpx::report_error(boost::current_exception());
                }
                
                
                
                util::force_error_on_lock();
                return threads::terminated;
            }
        };
    public:
        typedef boost::mpl::false_ direct_execution;
        
        
        
        
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(naming::address::address_type lva,
            Arguments && args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                util::bind(util::one_shot(typename Derived::thread_function()),
                    lva, util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args))));
        }
        
        
        
        
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, Arguments && args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                base_type::construct_continuation_thread_object_function(
                    cont, F, get_lva<Component>::call(lva),
                    std::forward<Arguments>(args)));
        }
        
        template <typename Arguments>
        BOOST_FORCEINLINE static Result
        execute_function(naming::address::address_type lva,
            Arguments && args)
        {
            LTM_(debug)
                << "base_result_action" << 2
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            return (get_lva<Component>::call(lva)->*F)(
                util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)));
        }
    };
    
    
    template <
        typename Component, typename Result, typename T0 , typename T1,
        Result (Component::*F)(T0 , T1), typename Derived>
    struct result_action2<
            Result (Component::*)(T0 , T1), F, Derived>
      : base_result_action2<
            Result (Component::*)(T0 , T1), F,
            typename detail::action_type<
                result_action2<
                    Result (Component::*)(T0 , T1), F, Derived>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            result_action2, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    
    template <
        typename Component, typename Result, typename T0 , typename T1,
        Result (Component::*F)(T0 , T1), typename Derived>
    struct make_action<Result (Component::*)(T0 , T1),
            F, Derived, boost::mpl::false_>
      : result_action2<
            Result (Component::*)(T0 , T1), F, Derived>
    {
        typedef result_action2<
            Result (Component::*)(T0 , T1), F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename Result,
        typename T0 , typename T1,
        Result (Component::*F)(T0 , T1),
        typename Derived>
    struct direct_result_action2<
            Result (Component::*)(T0 , T1), F, Derived>
      : base_result_action2<
            Result (Component::*)(T0 , T1), F,
            typename detail::action_type<
                direct_result_action2<
                    Result (Component::*)(T0 , T1), F, Derived>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_result_action2, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    
    template <
        typename Component, typename Result, typename T0 , typename T1,
        Result (Component::*F)(T0 , T1), typename Derived>
    struct make_action<Result (Component::*)(T0 , T1),
            F, Derived, boost::mpl::true_>
      : direct_result_action2<
            Result (Component::*)(T0 , T1), F, Derived>
    {
        typedef direct_result_action2<
            Result (Component::*)(T0 , T1), F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename T0 , typename T1,
        void (Component::*F)(T0 , T1), typename Derived>
    class base_action2<
            void (Component::*)(T0 , T1), F, Derived>
      : public action<
            Component, util::unused_type,
            hpx::util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type>,
            Derived>
    {
    public:
        typedef util::unused_type result_type;
        typedef util::unused_type remote_result_type;
        typedef hpx::util::tuple<
            typename util::decay<T0>::type , typename util::decay<T1>::type> arguments_type;
        typedef action<Component, result_type, arguments_type, Derived>
            base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                Arg0 && arg0 , Arg1 && arg1) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component>::call(lva)) << ")";
                    (get_lva<Component>::call(lva)->*F)(
                        std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
                }
                catch (hpx::thread_interrupted const&) {
                     
                }
                catch (hpx::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component>::call(lva)) << "): " << e.what();
                    
                    hpx::report_error(boost::current_exception());
                }
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component>::call(lva)) << ")";
                    
                    hpx::report_error(boost::current_exception());
                }
                
                
                
                util::force_error_on_lock();
                return threads::terminated;
            }
        };
    public:
        
        
        
        
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(naming::address::address_type lva,
            Arguments && args)
        {
            
            
            return traits::action_decorate_function<Derived>::call(lva,
                util::bind(util::one_shot(typename Derived::thread_function()), lva,
                    util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args))));
        }
        
        
        
        
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, Arguments && args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                base_type::construct_continuation_thread_object_function_void(
                    cont, F, get_lva<Component>::call(lva),
                    std::forward<Arguments>(args)));
        }
        
        template <typename Arguments>
        BOOST_FORCEINLINE static util::unused_type
        execute_function(naming::address::address_type lva,
            Arguments && args)
        {
            LTM_(debug)
                << "base_action" << 2
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            (get_lva<Component>::call(lva)->*F)(
                util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)));
            return util::unused;
        }
    };
    
    template <
        typename Component, typename T0 , typename T1,
        void (Component::*F)(T0 , T1),
        typename Derived>
    struct action2<
            void (Component::*)(T0 , T1), F, Derived>
      : base_action2<
            void (Component::*)(T0 , T1), F,
            typename detail::action_type<
                action2<
                    void (Component::*)(T0 , T1), F, Derived>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            action2, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    
    template <
        typename Component, typename T0 , typename T1,
        void (Component::*F)(T0 , T1), typename Derived>
    struct make_action<void (Component::*)(T0 , T1),
            F, Derived, boost::mpl::false_>
      : action2<
            void (Component::*)(T0 , T1), F, Derived>
    {
        typedef action2<
            void (Component::*)(T0 , T1), F, Derived
        > type;
    };
    
    template <
        typename Component, typename T0 , typename T1,
        void (Component::*F)(T0 , T1),
        typename Derived>
    struct direct_action2<
            void (Component::*)(T0 , T1), F, Derived>
      : base_action2<
            void (Component::*)(T0 , T1), F,
            typename detail::action_type<
                direct_action2<
                    void (Component::*)(T0 , T1), F, Derived>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_action2, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    
    template <
        typename Component, typename T0 , typename T1,
        void (Component::*F)(T0 , T1), typename Derived>
    struct make_action<void (Component::*)(T0 , T1),
            F, Derived, boost::mpl::true_>
      : direct_action2<
            void (Component::*)(T0 , T1), F, Derived>
    {
        typedef direct_action2<
            void (Component::*)(T0 , T1), F, Derived
        > type;
    };
    
    
    template <
        typename Component,
        typename T0 , typename T1,
        void (Component::*F)(T0 , T1),
        typename Derived>
    struct result_action2<
            void (Component::*)(T0 , T1), F, Derived>
      : action2<
            void (Component::*)(T0 , T1), F, Derived>
    {};
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Component, typename Result, typename T0 , typename T1 , typename T2,
        Result (Component::*F)(T0 , T1 , T2), typename Derived>
    class base_result_action3<
            Result (Component::*)(T0 , T1 , T2), F, Derived>
      : public action<
            Component, Result,
            hpx::util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type>,
            Derived>
    {
    public:
        typedef Result result_type;
        typedef typename detail::remote_action_result<Result>::type
            remote_result_type;
        typedef hpx::util::tuple<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type> arguments_type;
        typedef action<Component, result_type, arguments_type, Derived>
            base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component>::call(lva)) << ")";
                    (get_lva<Component>::call(lva)->*F)(
                        std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
                }
                catch (hpx::thread_interrupted const&) {
                     
                }
                catch (hpx::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component>::call(lva)) << "): " << e.what();
                    
                    hpx::report_error(boost::current_exception());
                }
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component>::call(lva)) << ")";
                    
                    hpx::report_error(boost::current_exception());
                }
                
                
                
                util::force_error_on_lock();
                return threads::terminated;
            }
        };
    public:
        typedef boost::mpl::false_ direct_execution;
        
        
        
        
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(naming::address::address_type lva,
            Arguments && args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                util::bind(util::one_shot(typename Derived::thread_function()),
                    lva, util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args))));
        }
        
        
        
        
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, Arguments && args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                base_type::construct_continuation_thread_object_function(
                    cont, F, get_lva<Component>::call(lva),
                    std::forward<Arguments>(args)));
        }
        
        template <typename Arguments>
        BOOST_FORCEINLINE static Result
        execute_function(naming::address::address_type lva,
            Arguments && args)
        {
            LTM_(debug)
                << "base_result_action" << 3
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            return (get_lva<Component>::call(lva)->*F)(
                util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)));
        }
    };
    
    
    template <
        typename Component, typename Result, typename T0 , typename T1 , typename T2,
        Result (Component::*F)(T0 , T1 , T2), typename Derived>
    struct result_action3<
            Result (Component::*)(T0 , T1 , T2), F, Derived>
      : base_result_action3<
            Result (Component::*)(T0 , T1 , T2), F,
            typename detail::action_type<
                result_action3<
                    Result (Component::*)(T0 , T1 , T2), F, Derived>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            result_action3, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    
    template <
        typename Component, typename Result, typename T0 , typename T1 , typename T2,
        Result (Component::*F)(T0 , T1 , T2), typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2),
            F, Derived, boost::mpl::false_>
      : result_action3<
            Result (Component::*)(T0 , T1 , T2), F, Derived>
    {
        typedef result_action3<
            Result (Component::*)(T0 , T1 , T2), F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename Result,
        typename T0 , typename T1 , typename T2,
        Result (Component::*F)(T0 , T1 , T2),
        typename Derived>
    struct direct_result_action3<
            Result (Component::*)(T0 , T1 , T2), F, Derived>
      : base_result_action3<
            Result (Component::*)(T0 , T1 , T2), F,
            typename detail::action_type<
                direct_result_action3<
                    Result (Component::*)(T0 , T1 , T2), F, Derived>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_result_action3, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    
    template <
        typename Component, typename Result, typename T0 , typename T1 , typename T2,
        Result (Component::*F)(T0 , T1 , T2), typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2),
            F, Derived, boost::mpl::true_>
      : direct_result_action3<
            Result (Component::*)(T0 , T1 , T2), F, Derived>
    {
        typedef direct_result_action3<
            Result (Component::*)(T0 , T1 , T2), F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename T0 , typename T1 , typename T2,
        void (Component::*F)(T0 , T1 , T2), typename Derived>
    class base_action3<
            void (Component::*)(T0 , T1 , T2), F, Derived>
      : public action<
            Component, util::unused_type,
            hpx::util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type>,
            Derived>
    {
    public:
        typedef util::unused_type result_type;
        typedef util::unused_type remote_result_type;
        typedef hpx::util::tuple<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type> arguments_type;
        typedef action<Component, result_type, arguments_type, Derived>
            base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component>::call(lva)) << ")";
                    (get_lva<Component>::call(lva)->*F)(
                        std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
                }
                catch (hpx::thread_interrupted const&) {
                     
                }
                catch (hpx::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component>::call(lva)) << "): " << e.what();
                    
                    hpx::report_error(boost::current_exception());
                }
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component>::call(lva)) << ")";
                    
                    hpx::report_error(boost::current_exception());
                }
                
                
                
                util::force_error_on_lock();
                return threads::terminated;
            }
        };
    public:
        
        
        
        
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(naming::address::address_type lva,
            Arguments && args)
        {
            
            
            return traits::action_decorate_function<Derived>::call(lva,
                util::bind(util::one_shot(typename Derived::thread_function()), lva,
                    util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args))));
        }
        
        
        
        
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, Arguments && args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                base_type::construct_continuation_thread_object_function_void(
                    cont, F, get_lva<Component>::call(lva),
                    std::forward<Arguments>(args)));
        }
        
        template <typename Arguments>
        BOOST_FORCEINLINE static util::unused_type
        execute_function(naming::address::address_type lva,
            Arguments && args)
        {
            LTM_(debug)
                << "base_action" << 3
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            (get_lva<Component>::call(lva)->*F)(
                util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)));
            return util::unused;
        }
    };
    
    template <
        typename Component, typename T0 , typename T1 , typename T2,
        void (Component::*F)(T0 , T1 , T2),
        typename Derived>
    struct action3<
            void (Component::*)(T0 , T1 , T2), F, Derived>
      : base_action3<
            void (Component::*)(T0 , T1 , T2), F,
            typename detail::action_type<
                action3<
                    void (Component::*)(T0 , T1 , T2), F, Derived>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            action3, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    
    template <
        typename Component, typename T0 , typename T1 , typename T2,
        void (Component::*F)(T0 , T1 , T2), typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2),
            F, Derived, boost::mpl::false_>
      : action3<
            void (Component::*)(T0 , T1 , T2), F, Derived>
    {
        typedef action3<
            void (Component::*)(T0 , T1 , T2), F, Derived
        > type;
    };
    
    template <
        typename Component, typename T0 , typename T1 , typename T2,
        void (Component::*F)(T0 , T1 , T2),
        typename Derived>
    struct direct_action3<
            void (Component::*)(T0 , T1 , T2), F, Derived>
      : base_action3<
            void (Component::*)(T0 , T1 , T2), F,
            typename detail::action_type<
                direct_action3<
                    void (Component::*)(T0 , T1 , T2), F, Derived>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_action3, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    
    template <
        typename Component, typename T0 , typename T1 , typename T2,
        void (Component::*F)(T0 , T1 , T2), typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2),
            F, Derived, boost::mpl::true_>
      : direct_action3<
            void (Component::*)(T0 , T1 , T2), F, Derived>
    {
        typedef direct_action3<
            void (Component::*)(T0 , T1 , T2), F, Derived
        > type;
    };
    
    
    template <
        typename Component,
        typename T0 , typename T1 , typename T2,
        void (Component::*F)(T0 , T1 , T2),
        typename Derived>
    struct result_action3<
            void (Component::*)(T0 , T1 , T2), F, Derived>
      : action3<
            void (Component::*)(T0 , T1 , T2), F, Derived>
    {};
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Component, typename Result, typename T0 , typename T1 , typename T2 , typename T3,
        Result (Component::*F)(T0 , T1 , T2 , T3), typename Derived>
    class base_result_action4<
            Result (Component::*)(T0 , T1 , T2 , T3), F, Derived>
      : public action<
            Component, Result,
            hpx::util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type>,
            Derived>
    {
    public:
        typedef Result result_type;
        typedef typename detail::remote_action_result<Result>::type
            remote_result_type;
        typedef hpx::util::tuple<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type> arguments_type;
        typedef action<Component, result_type, arguments_type, Derived>
            base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component>::call(lva)) << ")";
                    (get_lva<Component>::call(lva)->*F)(
                        std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
                }
                catch (hpx::thread_interrupted const&) {
                     
                }
                catch (hpx::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component>::call(lva)) << "): " << e.what();
                    
                    hpx::report_error(boost::current_exception());
                }
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component>::call(lva)) << ")";
                    
                    hpx::report_error(boost::current_exception());
                }
                
                
                
                util::force_error_on_lock();
                return threads::terminated;
            }
        };
    public:
        typedef boost::mpl::false_ direct_execution;
        
        
        
        
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(naming::address::address_type lva,
            Arguments && args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                util::bind(util::one_shot(typename Derived::thread_function()),
                    lva, util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args))));
        }
        
        
        
        
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, Arguments && args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                base_type::construct_continuation_thread_object_function(
                    cont, F, get_lva<Component>::call(lva),
                    std::forward<Arguments>(args)));
        }
        
        template <typename Arguments>
        BOOST_FORCEINLINE static Result
        execute_function(naming::address::address_type lva,
            Arguments && args)
        {
            LTM_(debug)
                << "base_result_action" << 4
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            return (get_lva<Component>::call(lva)->*F)(
                util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)));
        }
    };
    
    
    template <
        typename Component, typename Result, typename T0 , typename T1 , typename T2 , typename T3,
        Result (Component::*F)(T0 , T1 , T2 , T3), typename Derived>
    struct result_action4<
            Result (Component::*)(T0 , T1 , T2 , T3), F, Derived>
      : base_result_action4<
            Result (Component::*)(T0 , T1 , T2 , T3), F,
            typename detail::action_type<
                result_action4<
                    Result (Component::*)(T0 , T1 , T2 , T3), F, Derived>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            result_action4, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    
    template <
        typename Component, typename Result, typename T0 , typename T1 , typename T2 , typename T3,
        Result (Component::*F)(T0 , T1 , T2 , T3), typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3),
            F, Derived, boost::mpl::false_>
      : result_action4<
            Result (Component::*)(T0 , T1 , T2 , T3), F, Derived>
    {
        typedef result_action4<
            Result (Component::*)(T0 , T1 , T2 , T3), F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3,
        Result (Component::*F)(T0 , T1 , T2 , T3),
        typename Derived>
    struct direct_result_action4<
            Result (Component::*)(T0 , T1 , T2 , T3), F, Derived>
      : base_result_action4<
            Result (Component::*)(T0 , T1 , T2 , T3), F,
            typename detail::action_type<
                direct_result_action4<
                    Result (Component::*)(T0 , T1 , T2 , T3), F, Derived>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_result_action4, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    
    template <
        typename Component, typename Result, typename T0 , typename T1 , typename T2 , typename T3,
        Result (Component::*F)(T0 , T1 , T2 , T3), typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3),
            F, Derived, boost::mpl::true_>
      : direct_result_action4<
            Result (Component::*)(T0 , T1 , T2 , T3), F, Derived>
    {
        typedef direct_result_action4<
            Result (Component::*)(T0 , T1 , T2 , T3), F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3,
        void (Component::*F)(T0 , T1 , T2 , T3), typename Derived>
    class base_action4<
            void (Component::*)(T0 , T1 , T2 , T3), F, Derived>
      : public action<
            Component, util::unused_type,
            hpx::util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type>,
            Derived>
    {
    public:
        typedef util::unused_type result_type;
        typedef util::unused_type remote_result_type;
        typedef hpx::util::tuple<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type> arguments_type;
        typedef action<Component, result_type, arguments_type, Derived>
            base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component>::call(lva)) << ")";
                    (get_lva<Component>::call(lva)->*F)(
                        std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
                }
                catch (hpx::thread_interrupted const&) {
                     
                }
                catch (hpx::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component>::call(lva)) << "): " << e.what();
                    
                    hpx::report_error(boost::current_exception());
                }
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component>::call(lva)) << ")";
                    
                    hpx::report_error(boost::current_exception());
                }
                
                
                
                util::force_error_on_lock();
                return threads::terminated;
            }
        };
    public:
        
        
        
        
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(naming::address::address_type lva,
            Arguments && args)
        {
            
            
            return traits::action_decorate_function<Derived>::call(lva,
                util::bind(util::one_shot(typename Derived::thread_function()), lva,
                    util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args))));
        }
        
        
        
        
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, Arguments && args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                base_type::construct_continuation_thread_object_function_void(
                    cont, F, get_lva<Component>::call(lva),
                    std::forward<Arguments>(args)));
        }
        
        template <typename Arguments>
        BOOST_FORCEINLINE static util::unused_type
        execute_function(naming::address::address_type lva,
            Arguments && args)
        {
            LTM_(debug)
                << "base_action" << 4
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            (get_lva<Component>::call(lva)->*F)(
                util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)));
            return util::unused;
        }
    };
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3,
        void (Component::*F)(T0 , T1 , T2 , T3),
        typename Derived>
    struct action4<
            void (Component::*)(T0 , T1 , T2 , T3), F, Derived>
      : base_action4<
            void (Component::*)(T0 , T1 , T2 , T3), F,
            typename detail::action_type<
                action4<
                    void (Component::*)(T0 , T1 , T2 , T3), F, Derived>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            action4, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3,
        void (Component::*F)(T0 , T1 , T2 , T3), typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3),
            F, Derived, boost::mpl::false_>
      : action4<
            void (Component::*)(T0 , T1 , T2 , T3), F, Derived>
    {
        typedef action4<
            void (Component::*)(T0 , T1 , T2 , T3), F, Derived
        > type;
    };
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3,
        void (Component::*F)(T0 , T1 , T2 , T3),
        typename Derived>
    struct direct_action4<
            void (Component::*)(T0 , T1 , T2 , T3), F, Derived>
      : base_action4<
            void (Component::*)(T0 , T1 , T2 , T3), F,
            typename detail::action_type<
                direct_action4<
                    void (Component::*)(T0 , T1 , T2 , T3), F, Derived>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_action4, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3,
        void (Component::*F)(T0 , T1 , T2 , T3), typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3),
            F, Derived, boost::mpl::true_>
      : direct_action4<
            void (Component::*)(T0 , T1 , T2 , T3), F, Derived>
    {
        typedef direct_action4<
            void (Component::*)(T0 , T1 , T2 , T3), F, Derived
        > type;
    };
    
    
    template <
        typename Component,
        typename T0 , typename T1 , typename T2 , typename T3,
        void (Component::*F)(T0 , T1 , T2 , T3),
        typename Derived>
    struct result_action4<
            void (Component::*)(T0 , T1 , T2 , T3), F, Derived>
      : action4<
            void (Component::*)(T0 , T1 , T2 , T3), F, Derived>
    {};
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Component, typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4), typename Derived>
    class base_result_action5<
            Result (Component::*)(T0 , T1 , T2 , T3 , T4), F, Derived>
      : public action<
            Component, Result,
            hpx::util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type>,
            Derived>
    {
    public:
        typedef Result result_type;
        typedef typename detail::remote_action_result<Result>::type
            remote_result_type;
        typedef hpx::util::tuple<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type> arguments_type;
        typedef action<Component, result_type, arguments_type, Derived>
            base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component>::call(lva)) << ")";
                    (get_lva<Component>::call(lva)->*F)(
                        std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
                }
                catch (hpx::thread_interrupted const&) {
                     
                }
                catch (hpx::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component>::call(lva)) << "): " << e.what();
                    
                    hpx::report_error(boost::current_exception());
                }
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component>::call(lva)) << ")";
                    
                    hpx::report_error(boost::current_exception());
                }
                
                
                
                util::force_error_on_lock();
                return threads::terminated;
            }
        };
    public:
        typedef boost::mpl::false_ direct_execution;
        
        
        
        
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(naming::address::address_type lva,
            Arguments && args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                util::bind(util::one_shot(typename Derived::thread_function()),
                    lva, util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args))));
        }
        
        
        
        
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, Arguments && args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                base_type::construct_continuation_thread_object_function(
                    cont, F, get_lva<Component>::call(lva),
                    std::forward<Arguments>(args)));
        }
        
        template <typename Arguments>
        BOOST_FORCEINLINE static Result
        execute_function(naming::address::address_type lva,
            Arguments && args)
        {
            LTM_(debug)
                << "base_result_action" << 5
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            return (get_lva<Component>::call(lva)->*F)(
                util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)));
        }
    };
    
    
    template <
        typename Component, typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4), typename Derived>
    struct result_action5<
            Result (Component::*)(T0 , T1 , T2 , T3 , T4), F, Derived>
      : base_result_action5<
            Result (Component::*)(T0 , T1 , T2 , T3 , T4), F,
            typename detail::action_type<
                result_action5<
                    Result (Component::*)(T0 , T1 , T2 , T3 , T4), F, Derived>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            result_action5, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    
    template <
        typename Component, typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4), typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4),
            F, Derived, boost::mpl::false_>
      : result_action5<
            Result (Component::*)(T0 , T1 , T2 , T3 , T4), F, Derived>
    {
        typedef result_action5<
            Result (Component::*)(T0 , T1 , T2 , T3 , T4), F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4),
        typename Derived>
    struct direct_result_action5<
            Result (Component::*)(T0 , T1 , T2 , T3 , T4), F, Derived>
      : base_result_action5<
            Result (Component::*)(T0 , T1 , T2 , T3 , T4), F,
            typename detail::action_type<
                direct_result_action5<
                    Result (Component::*)(T0 , T1 , T2 , T3 , T4), F, Derived>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_result_action5, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    
    template <
        typename Component, typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4), typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4),
            F, Derived, boost::mpl::true_>
      : direct_result_action5<
            Result (Component::*)(T0 , T1 , T2 , T3 , T4), F, Derived>
    {
        typedef direct_result_action5<
            Result (Component::*)(T0 , T1 , T2 , T3 , T4), F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4), typename Derived>
    class base_action5<
            void (Component::*)(T0 , T1 , T2 , T3 , T4), F, Derived>
      : public action<
            Component, util::unused_type,
            hpx::util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type>,
            Derived>
    {
    public:
        typedef util::unused_type result_type;
        typedef util::unused_type remote_result_type;
        typedef hpx::util::tuple<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type> arguments_type;
        typedef action<Component, result_type, arguments_type, Derived>
            base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component>::call(lva)) << ")";
                    (get_lva<Component>::call(lva)->*F)(
                        std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
                }
                catch (hpx::thread_interrupted const&) {
                     
                }
                catch (hpx::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component>::call(lva)) << "): " << e.what();
                    
                    hpx::report_error(boost::current_exception());
                }
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component>::call(lva)) << ")";
                    
                    hpx::report_error(boost::current_exception());
                }
                
                
                
                util::force_error_on_lock();
                return threads::terminated;
            }
        };
    public:
        
        
        
        
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(naming::address::address_type lva,
            Arguments && args)
        {
            
            
            return traits::action_decorate_function<Derived>::call(lva,
                util::bind(util::one_shot(typename Derived::thread_function()), lva,
                    util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args))));
        }
        
        
        
        
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, Arguments && args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                base_type::construct_continuation_thread_object_function_void(
                    cont, F, get_lva<Component>::call(lva),
                    std::forward<Arguments>(args)));
        }
        
        template <typename Arguments>
        BOOST_FORCEINLINE static util::unused_type
        execute_function(naming::address::address_type lva,
            Arguments && args)
        {
            LTM_(debug)
                << "base_action" << 5
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            (get_lva<Component>::call(lva)->*F)(
                util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)));
            return util::unused;
        }
    };
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4),
        typename Derived>
    struct action5<
            void (Component::*)(T0 , T1 , T2 , T3 , T4), F, Derived>
      : base_action5<
            void (Component::*)(T0 , T1 , T2 , T3 , T4), F,
            typename detail::action_type<
                action5<
                    void (Component::*)(T0 , T1 , T2 , T3 , T4), F, Derived>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            action5, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4), typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4),
            F, Derived, boost::mpl::false_>
      : action5<
            void (Component::*)(T0 , T1 , T2 , T3 , T4), F, Derived>
    {
        typedef action5<
            void (Component::*)(T0 , T1 , T2 , T3 , T4), F, Derived
        > type;
    };
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4),
        typename Derived>
    struct direct_action5<
            void (Component::*)(T0 , T1 , T2 , T3 , T4), F, Derived>
      : base_action5<
            void (Component::*)(T0 , T1 , T2 , T3 , T4), F,
            typename detail::action_type<
                direct_action5<
                    void (Component::*)(T0 , T1 , T2 , T3 , T4), F, Derived>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_action5, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4), typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4),
            F, Derived, boost::mpl::true_>
      : direct_action5<
            void (Component::*)(T0 , T1 , T2 , T3 , T4), F, Derived>
    {
        typedef direct_action5<
            void (Component::*)(T0 , T1 , T2 , T3 , T4), F, Derived
        > type;
    };
    
    
    template <
        typename Component,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4),
        typename Derived>
    struct result_action5<
            void (Component::*)(T0 , T1 , T2 , T3 , T4), F, Derived>
      : action5<
            void (Component::*)(T0 , T1 , T2 , T3 , T4), F, Derived>
    {};
}}
