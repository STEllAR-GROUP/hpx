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
        typename Component, typename R,
        typename T0,
        R (Component::*F)(T0) const, typename Derived>
    class component_base_action<
            R (Component::*)(T0) const, F, Derived>
      : public basic_action<
            Component const, R(typename util::decay<T0>::type),
            Derived>
    {
    public:
        typedef basic_action<Component const,
            R(typename util::decay<T0>::type), Derived>
            base_type;
        
        static bool is_target_valid(naming::id_type const& id)
        {
            return Component::is_target_valid(id);
        }
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
                                    (get_lva<Component const>::call(lva)) << ")";
                    (get_lva<Component const>::call(lva)->*F)(
                        std::forward<Arg0>( arg0 ));
                }
                catch (hpx::thread_interrupted const&) {
                     
                }
                catch (hpx::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << "): " << e.what();
                    
                    hpx::report_error(boost::current_exception());
                }
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << ")";
                    
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
                    cont, F, get_lva<Component const>::call(lva),
                    std::forward<Arguments>(args)));
        }
        
        template <typename Arguments>
        BOOST_FORCEINLINE static R
        execute_function(naming::address::address_type lva,
            Arguments && args)
        {
            LTM_(debug)
                << "component_base_action" << 1
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component const>::call(lva)) << ")";
            return (get_lva<Component const>::call(lva)->*F)(
                util::get< 0>(std::forward<Arguments>( args)));
        }
    };
    
    
    template <
        typename Component, typename R,
        typename T0,
        R (Component::*F)(T0) const,
        typename Derived>
    struct component_action<
            R (Component::*)(T0) const, F, Derived>
      : component_base_action<
            R (Component::*)(T0) const, F,
            typename detail::action_type<
                component_action<
                    R (Component::*)(T0) const, F, Derived>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            component_action, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    
    template <
        typename Component, typename R, typename T0,
        R (Component::*F)(T0) const, typename Derived>
    struct make_action<R (Component::*)(T0) const,
            F, Derived, boost::mpl::false_>
      : component_action<
            R (Component::*)(T0) const, F, Derived>
    {
        typedef component_action<
            R (Component::*)(T0) const, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename R,
        typename T0,
        R (Component::*F)(T0) const,
        typename Derived>
    struct component_direct_action<
            R (Component::*)(T0) const, F, Derived>
      : component_base_action<
            R (Component::*)(T0) const, F,
            typename detail::action_type<
                component_direct_action<
                    R (Component::*)(T0) const, F, Derived>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            component_direct_action, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    
    template <
        typename Component, typename R, typename T0,
        R (Component::*F)(T0) const, typename Derived>
    struct make_action<R (Component::*)(T0) const,
            F, Derived, boost::mpl::true_>
      : component_direct_action<
            R (Component::*)(T0) const, F, Derived>
    {
        typedef component_direct_action<
            R (Component::*)(T0) const, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename T0,
        void (Component::*F)(T0) const, typename Derived>
    class component_base_action<
            void (Component::*)(T0) const, F, Derived>
      : public basic_action<
            Component const, util::unused_type(typename util::decay<T0>::type),
            Derived>
    {
    public:
        typedef basic_action<Component const,
            util::unused_type(typename util::decay<T0>::type), Derived>
            base_type;
        
        static bool is_target_valid(naming::id_type const& id)
        {
            return Component::is_target_valid(id);
        }
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
                                    (get_lva<Component const>::call(lva)) << ")";
                    (get_lva<Component const>::call(lva)->*F)(
                        std::forward<Arg0>( arg0 ));
                }
                catch (hpx::thread_interrupted const&) {
                     
                }
                catch (hpx::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << "): " << e.what();
                    
                    hpx::report_error(boost::current_exception());
                }
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << ")";
                    
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
                    cont, F, get_lva<Component const>::call(lva),
                    std::forward<Arguments>(args)));
        }
        
        template <typename Arguments>
        BOOST_FORCEINLINE static util::unused_type
        execute_function(naming::address::address_type lva,
            Arguments && args)
        {
            LTM_(debug)
                << "component_base_action" << 1
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component const>::call(lva)) << ")";
            (get_lva<Component const>::call(lva)->*F)(
                util::get< 0>(std::forward<Arguments>( args)));
            return util::unused;
        }
    };
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1,
        R (Component::*F)(T0 , T1) const, typename Derived>
    class component_base_action<
            R (Component::*)(T0 , T1) const, F, Derived>
      : public basic_action<
            Component const, R(typename util::decay<T0>::type , typename util::decay<T1>::type),
            Derived>
    {
    public:
        typedef basic_action<Component const,
            R(typename util::decay<T0>::type , typename util::decay<T1>::type), Derived>
            base_type;
        
        static bool is_target_valid(naming::id_type const& id)
        {
            return Component::is_target_valid(id);
        }
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
                                    (get_lva<Component const>::call(lva)) << ")";
                    (get_lva<Component const>::call(lva)->*F)(
                        std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
                }
                catch (hpx::thread_interrupted const&) {
                     
                }
                catch (hpx::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << "): " << e.what();
                    
                    hpx::report_error(boost::current_exception());
                }
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << ")";
                    
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
                    cont, F, get_lva<Component const>::call(lva),
                    std::forward<Arguments>(args)));
        }
        
        template <typename Arguments>
        BOOST_FORCEINLINE static R
        execute_function(naming::address::address_type lva,
            Arguments && args)
        {
            LTM_(debug)
                << "component_base_action" << 2
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component const>::call(lva)) << ")";
            return (get_lva<Component const>::call(lva)->*F)(
                util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)));
        }
    };
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1,
        R (Component::*F)(T0 , T1) const,
        typename Derived>
    struct component_action<
            R (Component::*)(T0 , T1) const, F, Derived>
      : component_base_action<
            R (Component::*)(T0 , T1) const, F,
            typename detail::action_type<
                component_action<
                    R (Component::*)(T0 , T1) const, F, Derived>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            component_action, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    
    template <
        typename Component, typename R, typename T0 , typename T1,
        R (Component::*F)(T0 , T1) const, typename Derived>
    struct make_action<R (Component::*)(T0 , T1) const,
            F, Derived, boost::mpl::false_>
      : component_action<
            R (Component::*)(T0 , T1) const, F, Derived>
    {
        typedef component_action<
            R (Component::*)(T0 , T1) const, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1,
        R (Component::*F)(T0 , T1) const,
        typename Derived>
    struct component_direct_action<
            R (Component::*)(T0 , T1) const, F, Derived>
      : component_base_action<
            R (Component::*)(T0 , T1) const, F,
            typename detail::action_type<
                component_direct_action<
                    R (Component::*)(T0 , T1) const, F, Derived>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            component_direct_action, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    
    template <
        typename Component, typename R, typename T0 , typename T1,
        R (Component::*F)(T0 , T1) const, typename Derived>
    struct make_action<R (Component::*)(T0 , T1) const,
            F, Derived, boost::mpl::true_>
      : component_direct_action<
            R (Component::*)(T0 , T1) const, F, Derived>
    {
        typedef component_direct_action<
            R (Component::*)(T0 , T1) const, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename T0 , typename T1,
        void (Component::*F)(T0 , T1) const, typename Derived>
    class component_base_action<
            void (Component::*)(T0 , T1) const, F, Derived>
      : public basic_action<
            Component const, util::unused_type(typename util::decay<T0>::type , typename util::decay<T1>::type),
            Derived>
    {
    public:
        typedef basic_action<Component const,
            util::unused_type(typename util::decay<T0>::type , typename util::decay<T1>::type), Derived>
            base_type;
        
        static bool is_target_valid(naming::id_type const& id)
        {
            return Component::is_target_valid(id);
        }
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
                                    (get_lva<Component const>::call(lva)) << ")";
                    (get_lva<Component const>::call(lva)->*F)(
                        std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
                }
                catch (hpx::thread_interrupted const&) {
                     
                }
                catch (hpx::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << "): " << e.what();
                    
                    hpx::report_error(boost::current_exception());
                }
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << ")";
                    
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
                    cont, F, get_lva<Component const>::call(lva),
                    std::forward<Arguments>(args)));
        }
        
        template <typename Arguments>
        BOOST_FORCEINLINE static util::unused_type
        execute_function(naming::address::address_type lva,
            Arguments && args)
        {
            LTM_(debug)
                << "component_base_action" << 2
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component const>::call(lva)) << ")";
            (get_lva<Component const>::call(lva)->*F)(
                util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)));
            return util::unused;
        }
    };
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1 , typename T2,
        R (Component::*F)(T0 , T1 , T2) const, typename Derived>
    class component_base_action<
            R (Component::*)(T0 , T1 , T2) const, F, Derived>
      : public basic_action<
            Component const, R(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type),
            Derived>
    {
    public:
        typedef basic_action<Component const,
            R(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type), Derived>
            base_type;
        
        static bool is_target_valid(naming::id_type const& id)
        {
            return Component::is_target_valid(id);
        }
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
                                    (get_lva<Component const>::call(lva)) << ")";
                    (get_lva<Component const>::call(lva)->*F)(
                        std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
                }
                catch (hpx::thread_interrupted const&) {
                     
                }
                catch (hpx::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << "): " << e.what();
                    
                    hpx::report_error(boost::current_exception());
                }
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << ")";
                    
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
                    cont, F, get_lva<Component const>::call(lva),
                    std::forward<Arguments>(args)));
        }
        
        template <typename Arguments>
        BOOST_FORCEINLINE static R
        execute_function(naming::address::address_type lva,
            Arguments && args)
        {
            LTM_(debug)
                << "component_base_action" << 3
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component const>::call(lva)) << ")";
            return (get_lva<Component const>::call(lva)->*F)(
                util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)));
        }
    };
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1 , typename T2,
        R (Component::*F)(T0 , T1 , T2) const,
        typename Derived>
    struct component_action<
            R (Component::*)(T0 , T1 , T2) const, F, Derived>
      : component_base_action<
            R (Component::*)(T0 , T1 , T2) const, F,
            typename detail::action_type<
                component_action<
                    R (Component::*)(T0 , T1 , T2) const, F, Derived>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            component_action, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    
    template <
        typename Component, typename R, typename T0 , typename T1 , typename T2,
        R (Component::*F)(T0 , T1 , T2) const, typename Derived>
    struct make_action<R (Component::*)(T0 , T1 , T2) const,
            F, Derived, boost::mpl::false_>
      : component_action<
            R (Component::*)(T0 , T1 , T2) const, F, Derived>
    {
        typedef component_action<
            R (Component::*)(T0 , T1 , T2) const, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1 , typename T2,
        R (Component::*F)(T0 , T1 , T2) const,
        typename Derived>
    struct component_direct_action<
            R (Component::*)(T0 , T1 , T2) const, F, Derived>
      : component_base_action<
            R (Component::*)(T0 , T1 , T2) const, F,
            typename detail::action_type<
                component_direct_action<
                    R (Component::*)(T0 , T1 , T2) const, F, Derived>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            component_direct_action, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    
    template <
        typename Component, typename R, typename T0 , typename T1 , typename T2,
        R (Component::*F)(T0 , T1 , T2) const, typename Derived>
    struct make_action<R (Component::*)(T0 , T1 , T2) const,
            F, Derived, boost::mpl::true_>
      : component_direct_action<
            R (Component::*)(T0 , T1 , T2) const, F, Derived>
    {
        typedef component_direct_action<
            R (Component::*)(T0 , T1 , T2) const, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename T0 , typename T1 , typename T2,
        void (Component::*F)(T0 , T1 , T2) const, typename Derived>
    class component_base_action<
            void (Component::*)(T0 , T1 , T2) const, F, Derived>
      : public basic_action<
            Component const, util::unused_type(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type),
            Derived>
    {
    public:
        typedef basic_action<Component const,
            util::unused_type(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type), Derived>
            base_type;
        
        static bool is_target_valid(naming::id_type const& id)
        {
            return Component::is_target_valid(id);
        }
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
                                    (get_lva<Component const>::call(lva)) << ")";
                    (get_lva<Component const>::call(lva)->*F)(
                        std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
                }
                catch (hpx::thread_interrupted const&) {
                     
                }
                catch (hpx::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << "): " << e.what();
                    
                    hpx::report_error(boost::current_exception());
                }
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << ")";
                    
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
                    cont, F, get_lva<Component const>::call(lva),
                    std::forward<Arguments>(args)));
        }
        
        template <typename Arguments>
        BOOST_FORCEINLINE static util::unused_type
        execute_function(naming::address::address_type lva,
            Arguments && args)
        {
            LTM_(debug)
                << "component_base_action" << 3
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component const>::call(lva)) << ")";
            (get_lva<Component const>::call(lva)->*F)(
                util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)));
            return util::unused;
        }
    };
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1 , typename T2 , typename T3,
        R (Component::*F)(T0 , T1 , T2 , T3) const, typename Derived>
    class component_base_action<
            R (Component::*)(T0 , T1 , T2 , T3) const, F, Derived>
      : public basic_action<
            Component const, R(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type),
            Derived>
    {
    public:
        typedef basic_action<Component const,
            R(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type), Derived>
            base_type;
        
        static bool is_target_valid(naming::id_type const& id)
        {
            return Component::is_target_valid(id);
        }
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
                                    (get_lva<Component const>::call(lva)) << ")";
                    (get_lva<Component const>::call(lva)->*F)(
                        std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
                }
                catch (hpx::thread_interrupted const&) {
                     
                }
                catch (hpx::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << "): " << e.what();
                    
                    hpx::report_error(boost::current_exception());
                }
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << ")";
                    
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
                    cont, F, get_lva<Component const>::call(lva),
                    std::forward<Arguments>(args)));
        }
        
        template <typename Arguments>
        BOOST_FORCEINLINE static R
        execute_function(naming::address::address_type lva,
            Arguments && args)
        {
            LTM_(debug)
                << "component_base_action" << 4
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component const>::call(lva)) << ")";
            return (get_lva<Component const>::call(lva)->*F)(
                util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)));
        }
    };
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1 , typename T2 , typename T3,
        R (Component::*F)(T0 , T1 , T2 , T3) const,
        typename Derived>
    struct component_action<
            R (Component::*)(T0 , T1 , T2 , T3) const, F, Derived>
      : component_base_action<
            R (Component::*)(T0 , T1 , T2 , T3) const, F,
            typename detail::action_type<
                component_action<
                    R (Component::*)(T0 , T1 , T2 , T3) const, F, Derived>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            component_action, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    
    template <
        typename Component, typename R, typename T0 , typename T1 , typename T2 , typename T3,
        R (Component::*F)(T0 , T1 , T2 , T3) const, typename Derived>
    struct make_action<R (Component::*)(T0 , T1 , T2 , T3) const,
            F, Derived, boost::mpl::false_>
      : component_action<
            R (Component::*)(T0 , T1 , T2 , T3) const, F, Derived>
    {
        typedef component_action<
            R (Component::*)(T0 , T1 , T2 , T3) const, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1 , typename T2 , typename T3,
        R (Component::*F)(T0 , T1 , T2 , T3) const,
        typename Derived>
    struct component_direct_action<
            R (Component::*)(T0 , T1 , T2 , T3) const, F, Derived>
      : component_base_action<
            R (Component::*)(T0 , T1 , T2 , T3) const, F,
            typename detail::action_type<
                component_direct_action<
                    R (Component::*)(T0 , T1 , T2 , T3) const, F, Derived>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            component_direct_action, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    
    template <
        typename Component, typename R, typename T0 , typename T1 , typename T2 , typename T3,
        R (Component::*F)(T0 , T1 , T2 , T3) const, typename Derived>
    struct make_action<R (Component::*)(T0 , T1 , T2 , T3) const,
            F, Derived, boost::mpl::true_>
      : component_direct_action<
            R (Component::*)(T0 , T1 , T2 , T3) const, F, Derived>
    {
        typedef component_direct_action<
            R (Component::*)(T0 , T1 , T2 , T3) const, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3,
        void (Component::*F)(T0 , T1 , T2 , T3) const, typename Derived>
    class component_base_action<
            void (Component::*)(T0 , T1 , T2 , T3) const, F, Derived>
      : public basic_action<
            Component const, util::unused_type(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type),
            Derived>
    {
    public:
        typedef basic_action<Component const,
            util::unused_type(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type), Derived>
            base_type;
        
        static bool is_target_valid(naming::id_type const& id)
        {
            return Component::is_target_valid(id);
        }
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
                                    (get_lva<Component const>::call(lva)) << ")";
                    (get_lva<Component const>::call(lva)->*F)(
                        std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
                }
                catch (hpx::thread_interrupted const&) {
                     
                }
                catch (hpx::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << "): " << e.what();
                    
                    hpx::report_error(boost::current_exception());
                }
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << ")";
                    
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
                    cont, F, get_lva<Component const>::call(lva),
                    std::forward<Arguments>(args)));
        }
        
        template <typename Arguments>
        BOOST_FORCEINLINE static util::unused_type
        execute_function(naming::address::address_type lva,
            Arguments && args)
        {
            LTM_(debug)
                << "component_base_action" << 4
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component const>::call(lva)) << ")";
            (get_lva<Component const>::call(lva)->*F)(
                util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)));
            return util::unused;
        }
    };
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4) const, typename Derived>
    class component_base_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4) const, F, Derived>
      : public basic_action<
            Component const, R(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type),
            Derived>
    {
    public:
        typedef basic_action<Component const,
            R(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type), Derived>
            base_type;
        
        static bool is_target_valid(naming::id_type const& id)
        {
            return Component::is_target_valid(id);
        }
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
                                    (get_lva<Component const>::call(lva)) << ")";
                    (get_lva<Component const>::call(lva)->*F)(
                        std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
                }
                catch (hpx::thread_interrupted const&) {
                     
                }
                catch (hpx::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << "): " << e.what();
                    
                    hpx::report_error(boost::current_exception());
                }
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << ")";
                    
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
                    cont, F, get_lva<Component const>::call(lva),
                    std::forward<Arguments>(args)));
        }
        
        template <typename Arguments>
        BOOST_FORCEINLINE static R
        execute_function(naming::address::address_type lva,
            Arguments && args)
        {
            LTM_(debug)
                << "component_base_action" << 5
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component const>::call(lva)) << ")";
            return (get_lva<Component const>::call(lva)->*F)(
                util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)));
        }
    };
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4) const,
        typename Derived>
    struct component_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4) const, F, Derived>
      : component_base_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4) const, F,
            typename detail::action_type<
                component_action<
                    R (Component::*)(T0 , T1 , T2 , T3 , T4) const, F, Derived>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            component_action, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    
    template <
        typename Component, typename R, typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4) const, typename Derived>
    struct make_action<R (Component::*)(T0 , T1 , T2 , T3 , T4) const,
            F, Derived, boost::mpl::false_>
      : component_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4) const, F, Derived>
    {
        typedef component_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4) const, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4) const,
        typename Derived>
    struct component_direct_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4) const, F, Derived>
      : component_base_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4) const, F,
            typename detail::action_type<
                component_direct_action<
                    R (Component::*)(T0 , T1 , T2 , T3 , T4) const, F, Derived>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            component_direct_action, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    
    template <
        typename Component, typename R, typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4) const, typename Derived>
    struct make_action<R (Component::*)(T0 , T1 , T2 , T3 , T4) const,
            F, Derived, boost::mpl::true_>
      : component_direct_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4) const, F, Derived>
    {
        typedef component_direct_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4) const, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4) const, typename Derived>
    class component_base_action<
            void (Component::*)(T0 , T1 , T2 , T3 , T4) const, F, Derived>
      : public basic_action<
            Component const, util::unused_type(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type),
            Derived>
    {
    public:
        typedef basic_action<Component const,
            util::unused_type(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type), Derived>
            base_type;
        
        static bool is_target_valid(naming::id_type const& id)
        {
            return Component::is_target_valid(id);
        }
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
                                    (get_lva<Component const>::call(lva)) << ")";
                    (get_lva<Component const>::call(lva)->*F)(
                        std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
                }
                catch (hpx::thread_interrupted const&) {
                     
                }
                catch (hpx::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << "): " << e.what();
                    
                    hpx::report_error(boost::current_exception());
                }
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << ")";
                    
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
                    cont, F, get_lva<Component const>::call(lva),
                    std::forward<Arguments>(args)));
        }
        
        template <typename Arguments>
        BOOST_FORCEINLINE static util::unused_type
        execute_function(naming::address::address_type lva,
            Arguments && args)
        {
            LTM_(debug)
                << "component_base_action" << 5
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component const>::call(lva)) << ")";
            (get_lva<Component const>::call(lva)->*F)(
                util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)));
            return util::unused;
        }
    };
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5) const, typename Derived>
    class component_base_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5) const, F, Derived>
      : public basic_action<
            Component const, R(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type),
            Derived>
    {
    public:
        typedef basic_action<Component const,
            R(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type), Derived>
            base_type;
        
        static bool is_target_valid(naming::id_type const& id)
        {
            return Component::is_target_valid(id);
        }
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component const>::call(lva)) << ")";
                    (get_lva<Component const>::call(lva)->*F)(
                        std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ));
                }
                catch (hpx::thread_interrupted const&) {
                     
                }
                catch (hpx::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << "): " << e.what();
                    
                    hpx::report_error(boost::current_exception());
                }
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << ")";
                    
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
                    lva, util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)) , util::get< 5>(std::forward<Arguments>( args))));
        }
        
        
        
        
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, Arguments && args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                base_type::construct_continuation_thread_object_function(
                    cont, F, get_lva<Component const>::call(lva),
                    std::forward<Arguments>(args)));
        }
        
        template <typename Arguments>
        BOOST_FORCEINLINE static R
        execute_function(naming::address::address_type lva,
            Arguments && args)
        {
            LTM_(debug)
                << "component_base_action" << 6
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component const>::call(lva)) << ")";
            return (get_lva<Component const>::call(lva)->*F)(
                util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)) , util::get< 5>(std::forward<Arguments>( args)));
        }
    };
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5) const,
        typename Derived>
    struct component_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5) const, F, Derived>
      : component_base_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5) const, F,
            typename detail::action_type<
                component_action<
                    R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5) const, F, Derived>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            component_action, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    
    template <
        typename Component, typename R, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5) const, typename Derived>
    struct make_action<R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5) const,
            F, Derived, boost::mpl::false_>
      : component_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5) const, F, Derived>
    {
        typedef component_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5) const, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5) const,
        typename Derived>
    struct component_direct_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5) const, F, Derived>
      : component_base_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5) const, F,
            typename detail::action_type<
                component_direct_action<
                    R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5) const, F, Derived>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            component_direct_action, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    
    template <
        typename Component, typename R, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5) const, typename Derived>
    struct make_action<R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5) const,
            F, Derived, boost::mpl::true_>
      : component_direct_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5) const, F, Derived>
    {
        typedef component_direct_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5) const, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5) const, typename Derived>
    class component_base_action<
            void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5) const, F, Derived>
      : public basic_action<
            Component const, util::unused_type(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type),
            Derived>
    {
    public:
        typedef basic_action<Component const,
            util::unused_type(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type), Derived>
            base_type;
        
        static bool is_target_valid(naming::id_type const& id)
        {
            return Component::is_target_valid(id);
        }
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component const>::call(lva)) << ")";
                    (get_lva<Component const>::call(lva)->*F)(
                        std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ));
                }
                catch (hpx::thread_interrupted const&) {
                     
                }
                catch (hpx::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << "): " << e.what();
                    
                    hpx::report_error(boost::current_exception());
                }
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << ")";
                    
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
                    util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)) , util::get< 5>(std::forward<Arguments>( args))));
        }
        
        
        
        
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, Arguments && args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                base_type::construct_continuation_thread_object_function_void(
                    cont, F, get_lva<Component const>::call(lva),
                    std::forward<Arguments>(args)));
        }
        
        template <typename Arguments>
        BOOST_FORCEINLINE static util::unused_type
        execute_function(naming::address::address_type lva,
            Arguments && args)
        {
            LTM_(debug)
                << "component_base_action" << 6
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component const>::call(lva)) << ")";
            (get_lva<Component const>::call(lva)->*F)(
                util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)) , util::get< 5>(std::forward<Arguments>( args)));
            return util::unused;
        }
    };
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6) const, typename Derived>
    class component_base_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6) const, F, Derived>
      : public basic_action<
            Component const, R(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type),
            Derived>
    {
    public:
        typedef basic_action<Component const,
            R(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type), Derived>
            base_type;
        
        static bool is_target_valid(naming::id_type const& id)
        {
            return Component::is_target_valid(id);
        }
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component const>::call(lva)) << ")";
                    (get_lva<Component const>::call(lva)->*F)(
                        std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ));
                }
                catch (hpx::thread_interrupted const&) {
                     
                }
                catch (hpx::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << "): " << e.what();
                    
                    hpx::report_error(boost::current_exception());
                }
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << ")";
                    
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
                    lva, util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)) , util::get< 5>(std::forward<Arguments>( args)) , util::get< 6>(std::forward<Arguments>( args))));
        }
        
        
        
        
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, Arguments && args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                base_type::construct_continuation_thread_object_function(
                    cont, F, get_lva<Component const>::call(lva),
                    std::forward<Arguments>(args)));
        }
        
        template <typename Arguments>
        BOOST_FORCEINLINE static R
        execute_function(naming::address::address_type lva,
            Arguments && args)
        {
            LTM_(debug)
                << "component_base_action" << 7
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component const>::call(lva)) << ")";
            return (get_lva<Component const>::call(lva)->*F)(
                util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)) , util::get< 5>(std::forward<Arguments>( args)) , util::get< 6>(std::forward<Arguments>( args)));
        }
    };
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6) const,
        typename Derived>
    struct component_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6) const, F, Derived>
      : component_base_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6) const, F,
            typename detail::action_type<
                component_action<
                    R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6) const, F, Derived>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            component_action, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    
    template <
        typename Component, typename R, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6) const, typename Derived>
    struct make_action<R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6) const,
            F, Derived, boost::mpl::false_>
      : component_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6) const, F, Derived>
    {
        typedef component_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6) const, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6) const,
        typename Derived>
    struct component_direct_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6) const, F, Derived>
      : component_base_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6) const, F,
            typename detail::action_type<
                component_direct_action<
                    R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6) const, F, Derived>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            component_direct_action, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    
    template <
        typename Component, typename R, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6) const, typename Derived>
    struct make_action<R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6) const,
            F, Derived, boost::mpl::true_>
      : component_direct_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6) const, F, Derived>
    {
        typedef component_direct_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6) const, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6) const, typename Derived>
    class component_base_action<
            void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6) const, F, Derived>
      : public basic_action<
            Component const, util::unused_type(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type),
            Derived>
    {
    public:
        typedef basic_action<Component const,
            util::unused_type(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type), Derived>
            base_type;
        
        static bool is_target_valid(naming::id_type const& id)
        {
            return Component::is_target_valid(id);
        }
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component const>::call(lva)) << ")";
                    (get_lva<Component const>::call(lva)->*F)(
                        std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ));
                }
                catch (hpx::thread_interrupted const&) {
                     
                }
                catch (hpx::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << "): " << e.what();
                    
                    hpx::report_error(boost::current_exception());
                }
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << ")";
                    
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
                    util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)) , util::get< 5>(std::forward<Arguments>( args)) , util::get< 6>(std::forward<Arguments>( args))));
        }
        
        
        
        
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, Arguments && args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                base_type::construct_continuation_thread_object_function_void(
                    cont, F, get_lva<Component const>::call(lva),
                    std::forward<Arguments>(args)));
        }
        
        template <typename Arguments>
        BOOST_FORCEINLINE static util::unused_type
        execute_function(naming::address::address_type lva,
            Arguments && args)
        {
            LTM_(debug)
                << "component_base_action" << 7
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component const>::call(lva)) << ")";
            (get_lva<Component const>::call(lva)->*F)(
                util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)) , util::get< 5>(std::forward<Arguments>( args)) , util::get< 6>(std::forward<Arguments>( args)));
            return util::unused;
        }
    };
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7) const, typename Derived>
    class component_base_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7) const, F, Derived>
      : public basic_action<
            Component const, R(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type),
            Derived>
    {
    public:
        typedef basic_action<Component const,
            R(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type), Derived>
            base_type;
        
        static bool is_target_valid(naming::id_type const& id)
        {
            return Component::is_target_valid(id);
        }
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component const>::call(lva)) << ")";
                    (get_lva<Component const>::call(lva)->*F)(
                        std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ));
                }
                catch (hpx::thread_interrupted const&) {
                     
                }
                catch (hpx::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << "): " << e.what();
                    
                    hpx::report_error(boost::current_exception());
                }
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << ")";
                    
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
                    lva, util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)) , util::get< 5>(std::forward<Arguments>( args)) , util::get< 6>(std::forward<Arguments>( args)) , util::get< 7>(std::forward<Arguments>( args))));
        }
        
        
        
        
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, Arguments && args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                base_type::construct_continuation_thread_object_function(
                    cont, F, get_lva<Component const>::call(lva),
                    std::forward<Arguments>(args)));
        }
        
        template <typename Arguments>
        BOOST_FORCEINLINE static R
        execute_function(naming::address::address_type lva,
            Arguments && args)
        {
            LTM_(debug)
                << "component_base_action" << 8
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component const>::call(lva)) << ")";
            return (get_lva<Component const>::call(lva)->*F)(
                util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)) , util::get< 5>(std::forward<Arguments>( args)) , util::get< 6>(std::forward<Arguments>( args)) , util::get< 7>(std::forward<Arguments>( args)));
        }
    };
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7) const,
        typename Derived>
    struct component_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7) const, F, Derived>
      : component_base_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7) const, F,
            typename detail::action_type<
                component_action<
                    R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7) const, F, Derived>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            component_action, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    
    template <
        typename Component, typename R, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7) const, typename Derived>
    struct make_action<R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7) const,
            F, Derived, boost::mpl::false_>
      : component_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7) const, F, Derived>
    {
        typedef component_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7) const, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7) const,
        typename Derived>
    struct component_direct_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7) const, F, Derived>
      : component_base_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7) const, F,
            typename detail::action_type<
                component_direct_action<
                    R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7) const, F, Derived>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            component_direct_action, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    
    template <
        typename Component, typename R, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7) const, typename Derived>
    struct make_action<R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7) const,
            F, Derived, boost::mpl::true_>
      : component_direct_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7) const, F, Derived>
    {
        typedef component_direct_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7) const, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7) const, typename Derived>
    class component_base_action<
            void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7) const, F, Derived>
      : public basic_action<
            Component const, util::unused_type(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type),
            Derived>
    {
    public:
        typedef basic_action<Component const,
            util::unused_type(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type), Derived>
            base_type;
        
        static bool is_target_valid(naming::id_type const& id)
        {
            return Component::is_target_valid(id);
        }
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component const>::call(lva)) << ")";
                    (get_lva<Component const>::call(lva)->*F)(
                        std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ));
                }
                catch (hpx::thread_interrupted const&) {
                     
                }
                catch (hpx::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << "): " << e.what();
                    
                    hpx::report_error(boost::current_exception());
                }
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << ")";
                    
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
                    util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)) , util::get< 5>(std::forward<Arguments>( args)) , util::get< 6>(std::forward<Arguments>( args)) , util::get< 7>(std::forward<Arguments>( args))));
        }
        
        
        
        
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, Arguments && args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                base_type::construct_continuation_thread_object_function_void(
                    cont, F, get_lva<Component const>::call(lva),
                    std::forward<Arguments>(args)));
        }
        
        template <typename Arguments>
        BOOST_FORCEINLINE static util::unused_type
        execute_function(naming::address::address_type lva,
            Arguments && args)
        {
            LTM_(debug)
                << "component_base_action" << 8
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component const>::call(lva)) << ")";
            (get_lva<Component const>::call(lva)->*F)(
                util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)) , util::get< 5>(std::forward<Arguments>( args)) , util::get< 6>(std::forward<Arguments>( args)) , util::get< 7>(std::forward<Arguments>( args)));
            return util::unused;
        }
    };
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8) const, typename Derived>
    class component_base_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8) const, F, Derived>
      : public basic_action<
            Component const, R(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type),
            Derived>
    {
    public:
        typedef basic_action<Component const,
            R(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type), Derived>
            base_type;
        
        static bool is_target_valid(naming::id_type const& id)
        {
            return Component::is_target_valid(id);
        }
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component const>::call(lva)) << ")";
                    (get_lva<Component const>::call(lva)->*F)(
                        std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ));
                }
                catch (hpx::thread_interrupted const&) {
                     
                }
                catch (hpx::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << "): " << e.what();
                    
                    hpx::report_error(boost::current_exception());
                }
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << ")";
                    
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
                    lva, util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)) , util::get< 5>(std::forward<Arguments>( args)) , util::get< 6>(std::forward<Arguments>( args)) , util::get< 7>(std::forward<Arguments>( args)) , util::get< 8>(std::forward<Arguments>( args))));
        }
        
        
        
        
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, Arguments && args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                base_type::construct_continuation_thread_object_function(
                    cont, F, get_lva<Component const>::call(lva),
                    std::forward<Arguments>(args)));
        }
        
        template <typename Arguments>
        BOOST_FORCEINLINE static R
        execute_function(naming::address::address_type lva,
            Arguments && args)
        {
            LTM_(debug)
                << "component_base_action" << 9
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component const>::call(lva)) << ")";
            return (get_lva<Component const>::call(lva)->*F)(
                util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)) , util::get< 5>(std::forward<Arguments>( args)) , util::get< 6>(std::forward<Arguments>( args)) , util::get< 7>(std::forward<Arguments>( args)) , util::get< 8>(std::forward<Arguments>( args)));
        }
    };
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8) const,
        typename Derived>
    struct component_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8) const, F, Derived>
      : component_base_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8) const, F,
            typename detail::action_type<
                component_action<
                    R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8) const, F, Derived>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            component_action, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    
    template <
        typename Component, typename R, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8) const, typename Derived>
    struct make_action<R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8) const,
            F, Derived, boost::mpl::false_>
      : component_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8) const, F, Derived>
    {
        typedef component_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8) const, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8) const,
        typename Derived>
    struct component_direct_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8) const, F, Derived>
      : component_base_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8) const, F,
            typename detail::action_type<
                component_direct_action<
                    R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8) const, F, Derived>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            component_direct_action, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    
    template <
        typename Component, typename R, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8) const, typename Derived>
    struct make_action<R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8) const,
            F, Derived, boost::mpl::true_>
      : component_direct_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8) const, F, Derived>
    {
        typedef component_direct_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8) const, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8) const, typename Derived>
    class component_base_action<
            void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8) const, F, Derived>
      : public basic_action<
            Component const, util::unused_type(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type),
            Derived>
    {
    public:
        typedef basic_action<Component const,
            util::unused_type(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type), Derived>
            base_type;
        
        static bool is_target_valid(naming::id_type const& id)
        {
            return Component::is_target_valid(id);
        }
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component const>::call(lva)) << ")";
                    (get_lva<Component const>::call(lva)->*F)(
                        std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ));
                }
                catch (hpx::thread_interrupted const&) {
                     
                }
                catch (hpx::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << "): " << e.what();
                    
                    hpx::report_error(boost::current_exception());
                }
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << ")";
                    
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
                    util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)) , util::get< 5>(std::forward<Arguments>( args)) , util::get< 6>(std::forward<Arguments>( args)) , util::get< 7>(std::forward<Arguments>( args)) , util::get< 8>(std::forward<Arguments>( args))));
        }
        
        
        
        
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, Arguments && args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                base_type::construct_continuation_thread_object_function_void(
                    cont, F, get_lva<Component const>::call(lva),
                    std::forward<Arguments>(args)));
        }
        
        template <typename Arguments>
        BOOST_FORCEINLINE static util::unused_type
        execute_function(naming::address::address_type lva,
            Arguments && args)
        {
            LTM_(debug)
                << "component_base_action" << 9
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component const>::call(lva)) << ")";
            (get_lva<Component const>::call(lva)->*F)(
                util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)) , util::get< 5>(std::forward<Arguments>( args)) , util::get< 6>(std::forward<Arguments>( args)) , util::get< 7>(std::forward<Arguments>( args)) , util::get< 8>(std::forward<Arguments>( args)));
            return util::unused;
        }
    };
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9) const, typename Derived>
    class component_base_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9) const, F, Derived>
      : public basic_action<
            Component const, R(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type),
            Derived>
    {
    public:
        typedef basic_action<Component const,
            R(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type), Derived>
            base_type;
        
        static bool is_target_valid(naming::id_type const& id)
        {
            return Component::is_target_valid(id);
        }
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component const>::call(lva)) << ")";
                    (get_lva<Component const>::call(lva)->*F)(
                        std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ));
                }
                catch (hpx::thread_interrupted const&) {
                     
                }
                catch (hpx::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << "): " << e.what();
                    
                    hpx::report_error(boost::current_exception());
                }
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << ")";
                    
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
                    lva, util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)) , util::get< 5>(std::forward<Arguments>( args)) , util::get< 6>(std::forward<Arguments>( args)) , util::get< 7>(std::forward<Arguments>( args)) , util::get< 8>(std::forward<Arguments>( args)) , util::get< 9>(std::forward<Arguments>( args))));
        }
        
        
        
        
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, Arguments && args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                base_type::construct_continuation_thread_object_function(
                    cont, F, get_lva<Component const>::call(lva),
                    std::forward<Arguments>(args)));
        }
        
        template <typename Arguments>
        BOOST_FORCEINLINE static R
        execute_function(naming::address::address_type lva,
            Arguments && args)
        {
            LTM_(debug)
                << "component_base_action" << 10
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component const>::call(lva)) << ")";
            return (get_lva<Component const>::call(lva)->*F)(
                util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)) , util::get< 5>(std::forward<Arguments>( args)) , util::get< 6>(std::forward<Arguments>( args)) , util::get< 7>(std::forward<Arguments>( args)) , util::get< 8>(std::forward<Arguments>( args)) , util::get< 9>(std::forward<Arguments>( args)));
        }
    };
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9) const,
        typename Derived>
    struct component_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9) const, F, Derived>
      : component_base_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9) const, F,
            typename detail::action_type<
                component_action<
                    R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9) const, F, Derived>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            component_action, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    
    template <
        typename Component, typename R, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9) const, typename Derived>
    struct make_action<R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9) const,
            F, Derived, boost::mpl::false_>
      : component_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9) const, F, Derived>
    {
        typedef component_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9) const, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9) const,
        typename Derived>
    struct component_direct_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9) const, F, Derived>
      : component_base_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9) const, F,
            typename detail::action_type<
                component_direct_action<
                    R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9) const, F, Derived>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            component_direct_action, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    
    template <
        typename Component, typename R, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9) const, typename Derived>
    struct make_action<R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9) const,
            F, Derived, boost::mpl::true_>
      : component_direct_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9) const, F, Derived>
    {
        typedef component_direct_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9) const, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9) const, typename Derived>
    class component_base_action<
            void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9) const, F, Derived>
      : public basic_action<
            Component const, util::unused_type(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type),
            Derived>
    {
    public:
        typedef basic_action<Component const,
            util::unused_type(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type), Derived>
            base_type;
        
        static bool is_target_valid(naming::id_type const& id)
        {
            return Component::is_target_valid(id);
        }
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component const>::call(lva)) << ")";
                    (get_lva<Component const>::call(lva)->*F)(
                        std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ));
                }
                catch (hpx::thread_interrupted const&) {
                     
                }
                catch (hpx::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << "): " << e.what();
                    
                    hpx::report_error(boost::current_exception());
                }
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << ")";
                    
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
                    util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)) , util::get< 5>(std::forward<Arguments>( args)) , util::get< 6>(std::forward<Arguments>( args)) , util::get< 7>(std::forward<Arguments>( args)) , util::get< 8>(std::forward<Arguments>( args)) , util::get< 9>(std::forward<Arguments>( args))));
        }
        
        
        
        
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, Arguments && args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                base_type::construct_continuation_thread_object_function_void(
                    cont, F, get_lva<Component const>::call(lva),
                    std::forward<Arguments>(args)));
        }
        
        template <typename Arguments>
        BOOST_FORCEINLINE static util::unused_type
        execute_function(naming::address::address_type lva,
            Arguments && args)
        {
            LTM_(debug)
                << "component_base_action" << 10
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component const>::call(lva)) << ")";
            (get_lva<Component const>::call(lva)->*F)(
                util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)) , util::get< 5>(std::forward<Arguments>( args)) , util::get< 6>(std::forward<Arguments>( args)) , util::get< 7>(std::forward<Arguments>( args)) , util::get< 8>(std::forward<Arguments>( args)) , util::get< 9>(std::forward<Arguments>( args)));
            return util::unused;
        }
    };
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10) const, typename Derived>
    class component_base_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10) const, F, Derived>
      : public basic_action<
            Component const, R(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type),
            Derived>
    {
    public:
        typedef basic_action<Component const,
            R(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type), Derived>
            base_type;
        
        static bool is_target_valid(naming::id_type const& id)
        {
            return Component::is_target_valid(id);
        }
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component const>::call(lva)) << ")";
                    (get_lva<Component const>::call(lva)->*F)(
                        std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ));
                }
                catch (hpx::thread_interrupted const&) {
                     
                }
                catch (hpx::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << "): " << e.what();
                    
                    hpx::report_error(boost::current_exception());
                }
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << ")";
                    
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
                    lva, util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)) , util::get< 5>(std::forward<Arguments>( args)) , util::get< 6>(std::forward<Arguments>( args)) , util::get< 7>(std::forward<Arguments>( args)) , util::get< 8>(std::forward<Arguments>( args)) , util::get< 9>(std::forward<Arguments>( args)) , util::get< 10>(std::forward<Arguments>( args))));
        }
        
        
        
        
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, Arguments && args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                base_type::construct_continuation_thread_object_function(
                    cont, F, get_lva<Component const>::call(lva),
                    std::forward<Arguments>(args)));
        }
        
        template <typename Arguments>
        BOOST_FORCEINLINE static R
        execute_function(naming::address::address_type lva,
            Arguments && args)
        {
            LTM_(debug)
                << "component_base_action" << 11
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component const>::call(lva)) << ")";
            return (get_lva<Component const>::call(lva)->*F)(
                util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)) , util::get< 5>(std::forward<Arguments>( args)) , util::get< 6>(std::forward<Arguments>( args)) , util::get< 7>(std::forward<Arguments>( args)) , util::get< 8>(std::forward<Arguments>( args)) , util::get< 9>(std::forward<Arguments>( args)) , util::get< 10>(std::forward<Arguments>( args)));
        }
    };
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10) const,
        typename Derived>
    struct component_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10) const, F, Derived>
      : component_base_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10) const, F,
            typename detail::action_type<
                component_action<
                    R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10) const, F, Derived>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            component_action, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    
    template <
        typename Component, typename R, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10) const, typename Derived>
    struct make_action<R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10) const,
            F, Derived, boost::mpl::false_>
      : component_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10) const, F, Derived>
    {
        typedef component_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10) const, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10) const,
        typename Derived>
    struct component_direct_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10) const, F, Derived>
      : component_base_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10) const, F,
            typename detail::action_type<
                component_direct_action<
                    R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10) const, F, Derived>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            component_direct_action, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    
    template <
        typename Component, typename R, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10) const, typename Derived>
    struct make_action<R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10) const,
            F, Derived, boost::mpl::true_>
      : component_direct_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10) const, F, Derived>
    {
        typedef component_direct_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10) const, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10) const, typename Derived>
    class component_base_action<
            void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10) const, F, Derived>
      : public basic_action<
            Component const, util::unused_type(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type),
            Derived>
    {
    public:
        typedef basic_action<Component const,
            util::unused_type(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type), Derived>
            base_type;
        
        static bool is_target_valid(naming::id_type const& id)
        {
            return Component::is_target_valid(id);
        }
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component const>::call(lva)) << ")";
                    (get_lva<Component const>::call(lva)->*F)(
                        std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ));
                }
                catch (hpx::thread_interrupted const&) {
                     
                }
                catch (hpx::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << "): " << e.what();
                    
                    hpx::report_error(boost::current_exception());
                }
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << ")";
                    
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
                    util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)) , util::get< 5>(std::forward<Arguments>( args)) , util::get< 6>(std::forward<Arguments>( args)) , util::get< 7>(std::forward<Arguments>( args)) , util::get< 8>(std::forward<Arguments>( args)) , util::get< 9>(std::forward<Arguments>( args)) , util::get< 10>(std::forward<Arguments>( args))));
        }
        
        
        
        
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, Arguments && args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                base_type::construct_continuation_thread_object_function_void(
                    cont, F, get_lva<Component const>::call(lva),
                    std::forward<Arguments>(args)));
        }
        
        template <typename Arguments>
        BOOST_FORCEINLINE static util::unused_type
        execute_function(naming::address::address_type lva,
            Arguments && args)
        {
            LTM_(debug)
                << "component_base_action" << 11
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component const>::call(lva)) << ")";
            (get_lva<Component const>::call(lva)->*F)(
                util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)) , util::get< 5>(std::forward<Arguments>( args)) , util::get< 6>(std::forward<Arguments>( args)) , util::get< 7>(std::forward<Arguments>( args)) , util::get< 8>(std::forward<Arguments>( args)) , util::get< 9>(std::forward<Arguments>( args)) , util::get< 10>(std::forward<Arguments>( args)));
            return util::unused;
        }
    };
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11) const, typename Derived>
    class component_base_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11) const, F, Derived>
      : public basic_action<
            Component const, R(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type),
            Derived>
    {
    public:
        typedef basic_action<Component const,
            R(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type), Derived>
            base_type;
        
        static bool is_target_valid(naming::id_type const& id)
        {
            return Component::is_target_valid(id);
        }
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component const>::call(lva)) << ")";
                    (get_lva<Component const>::call(lva)->*F)(
                        std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ));
                }
                catch (hpx::thread_interrupted const&) {
                     
                }
                catch (hpx::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << "): " << e.what();
                    
                    hpx::report_error(boost::current_exception());
                }
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << ")";
                    
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
                    lva, util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)) , util::get< 5>(std::forward<Arguments>( args)) , util::get< 6>(std::forward<Arguments>( args)) , util::get< 7>(std::forward<Arguments>( args)) , util::get< 8>(std::forward<Arguments>( args)) , util::get< 9>(std::forward<Arguments>( args)) , util::get< 10>(std::forward<Arguments>( args)) , util::get< 11>(std::forward<Arguments>( args))));
        }
        
        
        
        
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, Arguments && args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                base_type::construct_continuation_thread_object_function(
                    cont, F, get_lva<Component const>::call(lva),
                    std::forward<Arguments>(args)));
        }
        
        template <typename Arguments>
        BOOST_FORCEINLINE static R
        execute_function(naming::address::address_type lva,
            Arguments && args)
        {
            LTM_(debug)
                << "component_base_action" << 12
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component const>::call(lva)) << ")";
            return (get_lva<Component const>::call(lva)->*F)(
                util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)) , util::get< 5>(std::forward<Arguments>( args)) , util::get< 6>(std::forward<Arguments>( args)) , util::get< 7>(std::forward<Arguments>( args)) , util::get< 8>(std::forward<Arguments>( args)) , util::get< 9>(std::forward<Arguments>( args)) , util::get< 10>(std::forward<Arguments>( args)) , util::get< 11>(std::forward<Arguments>( args)));
        }
    };
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11) const,
        typename Derived>
    struct component_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11) const, F, Derived>
      : component_base_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11) const, F,
            typename detail::action_type<
                component_action<
                    R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11) const, F, Derived>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            component_action, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    
    template <
        typename Component, typename R, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11) const, typename Derived>
    struct make_action<R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11) const,
            F, Derived, boost::mpl::false_>
      : component_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11) const, F, Derived>
    {
        typedef component_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11) const, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11) const,
        typename Derived>
    struct component_direct_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11) const, F, Derived>
      : component_base_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11) const, F,
            typename detail::action_type<
                component_direct_action<
                    R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11) const, F, Derived>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            component_direct_action, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    
    template <
        typename Component, typename R, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11) const, typename Derived>
    struct make_action<R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11) const,
            F, Derived, boost::mpl::true_>
      : component_direct_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11) const, F, Derived>
    {
        typedef component_direct_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11) const, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11) const, typename Derived>
    class component_base_action<
            void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11) const, F, Derived>
      : public basic_action<
            Component const, util::unused_type(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type),
            Derived>
    {
    public:
        typedef basic_action<Component const,
            util::unused_type(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type), Derived>
            base_type;
        
        static bool is_target_valid(naming::id_type const& id)
        {
            return Component::is_target_valid(id);
        }
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component const>::call(lva)) << ")";
                    (get_lva<Component const>::call(lva)->*F)(
                        std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ));
                }
                catch (hpx::thread_interrupted const&) {
                     
                }
                catch (hpx::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << "): " << e.what();
                    
                    hpx::report_error(boost::current_exception());
                }
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << ")";
                    
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
                    util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)) , util::get< 5>(std::forward<Arguments>( args)) , util::get< 6>(std::forward<Arguments>( args)) , util::get< 7>(std::forward<Arguments>( args)) , util::get< 8>(std::forward<Arguments>( args)) , util::get< 9>(std::forward<Arguments>( args)) , util::get< 10>(std::forward<Arguments>( args)) , util::get< 11>(std::forward<Arguments>( args))));
        }
        
        
        
        
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, Arguments && args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                base_type::construct_continuation_thread_object_function_void(
                    cont, F, get_lva<Component const>::call(lva),
                    std::forward<Arguments>(args)));
        }
        
        template <typename Arguments>
        BOOST_FORCEINLINE static util::unused_type
        execute_function(naming::address::address_type lva,
            Arguments && args)
        {
            LTM_(debug)
                << "component_base_action" << 12
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component const>::call(lva)) << ")";
            (get_lva<Component const>::call(lva)->*F)(
                util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)) , util::get< 5>(std::forward<Arguments>( args)) , util::get< 6>(std::forward<Arguments>( args)) , util::get< 7>(std::forward<Arguments>( args)) , util::get< 8>(std::forward<Arguments>( args)) , util::get< 9>(std::forward<Arguments>( args)) , util::get< 10>(std::forward<Arguments>( args)) , util::get< 11>(std::forward<Arguments>( args)));
            return util::unused;
        }
    };
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12) const, typename Derived>
    class component_base_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12) const, F, Derived>
      : public basic_action<
            Component const, R(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type),
            Derived>
    {
    public:
        typedef basic_action<Component const,
            R(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type), Derived>
            base_type;
        
        static bool is_target_valid(naming::id_type const& id)
        {
            return Component::is_target_valid(id);
        }
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component const>::call(lva)) << ")";
                    (get_lva<Component const>::call(lva)->*F)(
                        std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ));
                }
                catch (hpx::thread_interrupted const&) {
                     
                }
                catch (hpx::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << "): " << e.what();
                    
                    hpx::report_error(boost::current_exception());
                }
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << ")";
                    
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
                    lva, util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)) , util::get< 5>(std::forward<Arguments>( args)) , util::get< 6>(std::forward<Arguments>( args)) , util::get< 7>(std::forward<Arguments>( args)) , util::get< 8>(std::forward<Arguments>( args)) , util::get< 9>(std::forward<Arguments>( args)) , util::get< 10>(std::forward<Arguments>( args)) , util::get< 11>(std::forward<Arguments>( args)) , util::get< 12>(std::forward<Arguments>( args))));
        }
        
        
        
        
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, Arguments && args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                base_type::construct_continuation_thread_object_function(
                    cont, F, get_lva<Component const>::call(lva),
                    std::forward<Arguments>(args)));
        }
        
        template <typename Arguments>
        BOOST_FORCEINLINE static R
        execute_function(naming::address::address_type lva,
            Arguments && args)
        {
            LTM_(debug)
                << "component_base_action" << 13
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component const>::call(lva)) << ")";
            return (get_lva<Component const>::call(lva)->*F)(
                util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)) , util::get< 5>(std::forward<Arguments>( args)) , util::get< 6>(std::forward<Arguments>( args)) , util::get< 7>(std::forward<Arguments>( args)) , util::get< 8>(std::forward<Arguments>( args)) , util::get< 9>(std::forward<Arguments>( args)) , util::get< 10>(std::forward<Arguments>( args)) , util::get< 11>(std::forward<Arguments>( args)) , util::get< 12>(std::forward<Arguments>( args)));
        }
    };
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12) const,
        typename Derived>
    struct component_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12) const, F, Derived>
      : component_base_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12) const, F,
            typename detail::action_type<
                component_action<
                    R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12) const, F, Derived>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            component_action, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    
    template <
        typename Component, typename R, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12) const, typename Derived>
    struct make_action<R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12) const,
            F, Derived, boost::mpl::false_>
      : component_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12) const, F, Derived>
    {
        typedef component_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12) const, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12) const,
        typename Derived>
    struct component_direct_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12) const, F, Derived>
      : component_base_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12) const, F,
            typename detail::action_type<
                component_direct_action<
                    R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12) const, F, Derived>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            component_direct_action, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    
    template <
        typename Component, typename R, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12) const, typename Derived>
    struct make_action<R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12) const,
            F, Derived, boost::mpl::true_>
      : component_direct_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12) const, F, Derived>
    {
        typedef component_direct_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12) const, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12) const, typename Derived>
    class component_base_action<
            void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12) const, F, Derived>
      : public basic_action<
            Component const, util::unused_type(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type),
            Derived>
    {
    public:
        typedef basic_action<Component const,
            util::unused_type(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type), Derived>
            base_type;
        
        static bool is_target_valid(naming::id_type const& id)
        {
            return Component::is_target_valid(id);
        }
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component const>::call(lva)) << ")";
                    (get_lva<Component const>::call(lva)->*F)(
                        std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ));
                }
                catch (hpx::thread_interrupted const&) {
                     
                }
                catch (hpx::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << "): " << e.what();
                    
                    hpx::report_error(boost::current_exception());
                }
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << ")";
                    
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
                    util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)) , util::get< 5>(std::forward<Arguments>( args)) , util::get< 6>(std::forward<Arguments>( args)) , util::get< 7>(std::forward<Arguments>( args)) , util::get< 8>(std::forward<Arguments>( args)) , util::get< 9>(std::forward<Arguments>( args)) , util::get< 10>(std::forward<Arguments>( args)) , util::get< 11>(std::forward<Arguments>( args)) , util::get< 12>(std::forward<Arguments>( args))));
        }
        
        
        
        
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, Arguments && args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                base_type::construct_continuation_thread_object_function_void(
                    cont, F, get_lva<Component const>::call(lva),
                    std::forward<Arguments>(args)));
        }
        
        template <typename Arguments>
        BOOST_FORCEINLINE static util::unused_type
        execute_function(naming::address::address_type lva,
            Arguments && args)
        {
            LTM_(debug)
                << "component_base_action" << 13
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component const>::call(lva)) << ")";
            (get_lva<Component const>::call(lva)->*F)(
                util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)) , util::get< 5>(std::forward<Arguments>( args)) , util::get< 6>(std::forward<Arguments>( args)) , util::get< 7>(std::forward<Arguments>( args)) , util::get< 8>(std::forward<Arguments>( args)) , util::get< 9>(std::forward<Arguments>( args)) , util::get< 10>(std::forward<Arguments>( args)) , util::get< 11>(std::forward<Arguments>( args)) , util::get< 12>(std::forward<Arguments>( args)));
            return util::unused;
        }
    };
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13) const, typename Derived>
    class component_base_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13) const, F, Derived>
      : public basic_action<
            Component const, R(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type),
            Derived>
    {
    public:
        typedef basic_action<Component const,
            R(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type), Derived>
            base_type;
        
        static bool is_target_valid(naming::id_type const& id)
        {
            return Component::is_target_valid(id);
        }
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component const>::call(lva)) << ")";
                    (get_lva<Component const>::call(lva)->*F)(
                        std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ));
                }
                catch (hpx::thread_interrupted const&) {
                     
                }
                catch (hpx::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << "): " << e.what();
                    
                    hpx::report_error(boost::current_exception());
                }
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << ")";
                    
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
                    lva, util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)) , util::get< 5>(std::forward<Arguments>( args)) , util::get< 6>(std::forward<Arguments>( args)) , util::get< 7>(std::forward<Arguments>( args)) , util::get< 8>(std::forward<Arguments>( args)) , util::get< 9>(std::forward<Arguments>( args)) , util::get< 10>(std::forward<Arguments>( args)) , util::get< 11>(std::forward<Arguments>( args)) , util::get< 12>(std::forward<Arguments>( args)) , util::get< 13>(std::forward<Arguments>( args))));
        }
        
        
        
        
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, Arguments && args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                base_type::construct_continuation_thread_object_function(
                    cont, F, get_lva<Component const>::call(lva),
                    std::forward<Arguments>(args)));
        }
        
        template <typename Arguments>
        BOOST_FORCEINLINE static R
        execute_function(naming::address::address_type lva,
            Arguments && args)
        {
            LTM_(debug)
                << "component_base_action" << 14
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component const>::call(lva)) << ")";
            return (get_lva<Component const>::call(lva)->*F)(
                util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)) , util::get< 5>(std::forward<Arguments>( args)) , util::get< 6>(std::forward<Arguments>( args)) , util::get< 7>(std::forward<Arguments>( args)) , util::get< 8>(std::forward<Arguments>( args)) , util::get< 9>(std::forward<Arguments>( args)) , util::get< 10>(std::forward<Arguments>( args)) , util::get< 11>(std::forward<Arguments>( args)) , util::get< 12>(std::forward<Arguments>( args)) , util::get< 13>(std::forward<Arguments>( args)));
        }
    };
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13) const,
        typename Derived>
    struct component_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13) const, F, Derived>
      : component_base_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13) const, F,
            typename detail::action_type<
                component_action<
                    R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13) const, F, Derived>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            component_action, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    
    template <
        typename Component, typename R, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13) const, typename Derived>
    struct make_action<R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13) const,
            F, Derived, boost::mpl::false_>
      : component_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13) const, F, Derived>
    {
        typedef component_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13) const, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13) const,
        typename Derived>
    struct component_direct_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13) const, F, Derived>
      : component_base_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13) const, F,
            typename detail::action_type<
                component_direct_action<
                    R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13) const, F, Derived>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            component_direct_action, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    
    template <
        typename Component, typename R, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13) const, typename Derived>
    struct make_action<R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13) const,
            F, Derived, boost::mpl::true_>
      : component_direct_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13) const, F, Derived>
    {
        typedef component_direct_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13) const, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13) const, typename Derived>
    class component_base_action<
            void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13) const, F, Derived>
      : public basic_action<
            Component const, util::unused_type(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type),
            Derived>
    {
    public:
        typedef basic_action<Component const,
            util::unused_type(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type), Derived>
            base_type;
        
        static bool is_target_valid(naming::id_type const& id)
        {
            return Component::is_target_valid(id);
        }
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component const>::call(lva)) << ")";
                    (get_lva<Component const>::call(lva)->*F)(
                        std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ));
                }
                catch (hpx::thread_interrupted const&) {
                     
                }
                catch (hpx::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << "): " << e.what();
                    
                    hpx::report_error(boost::current_exception());
                }
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << ")";
                    
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
                    util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)) , util::get< 5>(std::forward<Arguments>( args)) , util::get< 6>(std::forward<Arguments>( args)) , util::get< 7>(std::forward<Arguments>( args)) , util::get< 8>(std::forward<Arguments>( args)) , util::get< 9>(std::forward<Arguments>( args)) , util::get< 10>(std::forward<Arguments>( args)) , util::get< 11>(std::forward<Arguments>( args)) , util::get< 12>(std::forward<Arguments>( args)) , util::get< 13>(std::forward<Arguments>( args))));
        }
        
        
        
        
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, Arguments && args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                base_type::construct_continuation_thread_object_function_void(
                    cont, F, get_lva<Component const>::call(lva),
                    std::forward<Arguments>(args)));
        }
        
        template <typename Arguments>
        BOOST_FORCEINLINE static util::unused_type
        execute_function(naming::address::address_type lva,
            Arguments && args)
        {
            LTM_(debug)
                << "component_base_action" << 14
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component const>::call(lva)) << ")";
            (get_lva<Component const>::call(lva)->*F)(
                util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)) , util::get< 5>(std::forward<Arguments>( args)) , util::get< 6>(std::forward<Arguments>( args)) , util::get< 7>(std::forward<Arguments>( args)) , util::get< 8>(std::forward<Arguments>( args)) , util::get< 9>(std::forward<Arguments>( args)) , util::get< 10>(std::forward<Arguments>( args)) , util::get< 11>(std::forward<Arguments>( args)) , util::get< 12>(std::forward<Arguments>( args)) , util::get< 13>(std::forward<Arguments>( args)));
            return util::unused;
        }
    };
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14) const, typename Derived>
    class component_base_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14) const, F, Derived>
      : public basic_action<
            Component const, R(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type),
            Derived>
    {
    public:
        typedef basic_action<Component const,
            R(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type), Derived>
            base_type;
        
        static bool is_target_valid(naming::id_type const& id)
        {
            return Component::is_target_valid(id);
        }
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13 , Arg14 && arg14) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component const>::call(lva)) << ")";
                    (get_lva<Component const>::call(lva)->*F)(
                        std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ) , std::forward<Arg14>( arg14 ));
                }
                catch (hpx::thread_interrupted const&) {
                     
                }
                catch (hpx::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << "): " << e.what();
                    
                    hpx::report_error(boost::current_exception());
                }
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << ")";
                    
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
                    lva, util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)) , util::get< 5>(std::forward<Arguments>( args)) , util::get< 6>(std::forward<Arguments>( args)) , util::get< 7>(std::forward<Arguments>( args)) , util::get< 8>(std::forward<Arguments>( args)) , util::get< 9>(std::forward<Arguments>( args)) , util::get< 10>(std::forward<Arguments>( args)) , util::get< 11>(std::forward<Arguments>( args)) , util::get< 12>(std::forward<Arguments>( args)) , util::get< 13>(std::forward<Arguments>( args)) , util::get< 14>(std::forward<Arguments>( args))));
        }
        
        
        
        
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, Arguments && args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                base_type::construct_continuation_thread_object_function(
                    cont, F, get_lva<Component const>::call(lva),
                    std::forward<Arguments>(args)));
        }
        
        template <typename Arguments>
        BOOST_FORCEINLINE static R
        execute_function(naming::address::address_type lva,
            Arguments && args)
        {
            LTM_(debug)
                << "component_base_action" << 15
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component const>::call(lva)) << ")";
            return (get_lva<Component const>::call(lva)->*F)(
                util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)) , util::get< 5>(std::forward<Arguments>( args)) , util::get< 6>(std::forward<Arguments>( args)) , util::get< 7>(std::forward<Arguments>( args)) , util::get< 8>(std::forward<Arguments>( args)) , util::get< 9>(std::forward<Arguments>( args)) , util::get< 10>(std::forward<Arguments>( args)) , util::get< 11>(std::forward<Arguments>( args)) , util::get< 12>(std::forward<Arguments>( args)) , util::get< 13>(std::forward<Arguments>( args)) , util::get< 14>(std::forward<Arguments>( args)));
        }
    };
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14) const,
        typename Derived>
    struct component_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14) const, F, Derived>
      : component_base_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14) const, F,
            typename detail::action_type<
                component_action<
                    R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14) const, F, Derived>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            component_action, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    
    template <
        typename Component, typename R, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14) const, typename Derived>
    struct make_action<R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14) const,
            F, Derived, boost::mpl::false_>
      : component_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14) const, F, Derived>
    {
        typedef component_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14) const, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14) const,
        typename Derived>
    struct component_direct_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14) const, F, Derived>
      : component_base_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14) const, F,
            typename detail::action_type<
                component_direct_action<
                    R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14) const, F, Derived>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            component_direct_action, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    
    template <
        typename Component, typename R, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14) const, typename Derived>
    struct make_action<R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14) const,
            F, Derived, boost::mpl::true_>
      : component_direct_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14) const, F, Derived>
    {
        typedef component_direct_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14) const, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14) const, typename Derived>
    class component_base_action<
            void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14) const, F, Derived>
      : public basic_action<
            Component const, util::unused_type(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type),
            Derived>
    {
    public:
        typedef basic_action<Component const,
            util::unused_type(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type), Derived>
            base_type;
        
        static bool is_target_valid(naming::id_type const& id)
        {
            return Component::is_target_valid(id);
        }
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13 , Arg14 && arg14) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component const>::call(lva)) << ")";
                    (get_lva<Component const>::call(lva)->*F)(
                        std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ) , std::forward<Arg14>( arg14 ));
                }
                catch (hpx::thread_interrupted const&) {
                     
                }
                catch (hpx::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << "): " << e.what();
                    
                    hpx::report_error(boost::current_exception());
                }
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << ")";
                    
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
                    util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)) , util::get< 5>(std::forward<Arguments>( args)) , util::get< 6>(std::forward<Arguments>( args)) , util::get< 7>(std::forward<Arguments>( args)) , util::get< 8>(std::forward<Arguments>( args)) , util::get< 9>(std::forward<Arguments>( args)) , util::get< 10>(std::forward<Arguments>( args)) , util::get< 11>(std::forward<Arguments>( args)) , util::get< 12>(std::forward<Arguments>( args)) , util::get< 13>(std::forward<Arguments>( args)) , util::get< 14>(std::forward<Arguments>( args))));
        }
        
        
        
        
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, Arguments && args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                base_type::construct_continuation_thread_object_function_void(
                    cont, F, get_lva<Component const>::call(lva),
                    std::forward<Arguments>(args)));
        }
        
        template <typename Arguments>
        BOOST_FORCEINLINE static util::unused_type
        execute_function(naming::address::address_type lva,
            Arguments && args)
        {
            LTM_(debug)
                << "component_base_action" << 15
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component const>::call(lva)) << ")";
            (get_lva<Component const>::call(lva)->*F)(
                util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)) , util::get< 5>(std::forward<Arguments>( args)) , util::get< 6>(std::forward<Arguments>( args)) , util::get< 7>(std::forward<Arguments>( args)) , util::get< 8>(std::forward<Arguments>( args)) , util::get< 9>(std::forward<Arguments>( args)) , util::get< 10>(std::forward<Arguments>( args)) , util::get< 11>(std::forward<Arguments>( args)) , util::get< 12>(std::forward<Arguments>( args)) , util::get< 13>(std::forward<Arguments>( args)) , util::get< 14>(std::forward<Arguments>( args)));
            return util::unused;
        }
    };
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15) const, typename Derived>
    class component_base_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15) const, F, Derived>
      : public basic_action<
            Component const, R(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type),
            Derived>
    {
    public:
        typedef basic_action<Component const,
            R(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type), Derived>
            base_type;
        
        static bool is_target_valid(naming::id_type const& id)
        {
            return Component::is_target_valid(id);
        }
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13 , Arg14 && arg14 , Arg15 && arg15) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component const>::call(lva)) << ")";
                    (get_lva<Component const>::call(lva)->*F)(
                        std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ) , std::forward<Arg14>( arg14 ) , std::forward<Arg15>( arg15 ));
                }
                catch (hpx::thread_interrupted const&) {
                     
                }
                catch (hpx::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << "): " << e.what();
                    
                    hpx::report_error(boost::current_exception());
                }
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << ")";
                    
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
                    lva, util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)) , util::get< 5>(std::forward<Arguments>( args)) , util::get< 6>(std::forward<Arguments>( args)) , util::get< 7>(std::forward<Arguments>( args)) , util::get< 8>(std::forward<Arguments>( args)) , util::get< 9>(std::forward<Arguments>( args)) , util::get< 10>(std::forward<Arguments>( args)) , util::get< 11>(std::forward<Arguments>( args)) , util::get< 12>(std::forward<Arguments>( args)) , util::get< 13>(std::forward<Arguments>( args)) , util::get< 14>(std::forward<Arguments>( args)) , util::get< 15>(std::forward<Arguments>( args))));
        }
        
        
        
        
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, Arguments && args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                base_type::construct_continuation_thread_object_function(
                    cont, F, get_lva<Component const>::call(lva),
                    std::forward<Arguments>(args)));
        }
        
        template <typename Arguments>
        BOOST_FORCEINLINE static R
        execute_function(naming::address::address_type lva,
            Arguments && args)
        {
            LTM_(debug)
                << "component_base_action" << 16
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component const>::call(lva)) << ")";
            return (get_lva<Component const>::call(lva)->*F)(
                util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)) , util::get< 5>(std::forward<Arguments>( args)) , util::get< 6>(std::forward<Arguments>( args)) , util::get< 7>(std::forward<Arguments>( args)) , util::get< 8>(std::forward<Arguments>( args)) , util::get< 9>(std::forward<Arguments>( args)) , util::get< 10>(std::forward<Arguments>( args)) , util::get< 11>(std::forward<Arguments>( args)) , util::get< 12>(std::forward<Arguments>( args)) , util::get< 13>(std::forward<Arguments>( args)) , util::get< 14>(std::forward<Arguments>( args)) , util::get< 15>(std::forward<Arguments>( args)));
        }
    };
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15) const,
        typename Derived>
    struct component_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15) const, F, Derived>
      : component_base_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15) const, F,
            typename detail::action_type<
                component_action<
                    R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15) const, F, Derived>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            component_action, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    
    template <
        typename Component, typename R, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15) const, typename Derived>
    struct make_action<R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15) const,
            F, Derived, boost::mpl::false_>
      : component_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15) const, F, Derived>
    {
        typedef component_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15) const, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15) const,
        typename Derived>
    struct component_direct_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15) const, F, Derived>
      : component_base_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15) const, F,
            typename detail::action_type<
                component_direct_action<
                    R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15) const, F, Derived>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            component_direct_action, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    
    template <
        typename Component, typename R, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15) const, typename Derived>
    struct make_action<R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15) const,
            F, Derived, boost::mpl::true_>
      : component_direct_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15) const, F, Derived>
    {
        typedef component_direct_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15) const, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15) const, typename Derived>
    class component_base_action<
            void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15) const, F, Derived>
      : public basic_action<
            Component const, util::unused_type(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type),
            Derived>
    {
    public:
        typedef basic_action<Component const,
            util::unused_type(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type), Derived>
            base_type;
        
        static bool is_target_valid(naming::id_type const& id)
        {
            return Component::is_target_valid(id);
        }
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13 , Arg14 && arg14 , Arg15 && arg15) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component const>::call(lva)) << ")";
                    (get_lva<Component const>::call(lva)->*F)(
                        std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ) , std::forward<Arg14>( arg14 ) , std::forward<Arg15>( arg15 ));
                }
                catch (hpx::thread_interrupted const&) {
                     
                }
                catch (hpx::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << "): " << e.what();
                    
                    hpx::report_error(boost::current_exception());
                }
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << ")";
                    
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
                    util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)) , util::get< 5>(std::forward<Arguments>( args)) , util::get< 6>(std::forward<Arguments>( args)) , util::get< 7>(std::forward<Arguments>( args)) , util::get< 8>(std::forward<Arguments>( args)) , util::get< 9>(std::forward<Arguments>( args)) , util::get< 10>(std::forward<Arguments>( args)) , util::get< 11>(std::forward<Arguments>( args)) , util::get< 12>(std::forward<Arguments>( args)) , util::get< 13>(std::forward<Arguments>( args)) , util::get< 14>(std::forward<Arguments>( args)) , util::get< 15>(std::forward<Arguments>( args))));
        }
        
        
        
        
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, Arguments && args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                base_type::construct_continuation_thread_object_function_void(
                    cont, F, get_lva<Component const>::call(lva),
                    std::forward<Arguments>(args)));
        }
        
        template <typename Arguments>
        BOOST_FORCEINLINE static util::unused_type
        execute_function(naming::address::address_type lva,
            Arguments && args)
        {
            LTM_(debug)
                << "component_base_action" << 16
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component const>::call(lva)) << ")";
            (get_lva<Component const>::call(lva)->*F)(
                util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)) , util::get< 5>(std::forward<Arguments>( args)) , util::get< 6>(std::forward<Arguments>( args)) , util::get< 7>(std::forward<Arguments>( args)) , util::get< 8>(std::forward<Arguments>( args)) , util::get< 9>(std::forward<Arguments>( args)) , util::get< 10>(std::forward<Arguments>( args)) , util::get< 11>(std::forward<Arguments>( args)) , util::get< 12>(std::forward<Arguments>( args)) , util::get< 13>(std::forward<Arguments>( args)) , util::get< 14>(std::forward<Arguments>( args)) , util::get< 15>(std::forward<Arguments>( args)));
            return util::unused;
        }
    };
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16) const, typename Derived>
    class component_base_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16) const, F, Derived>
      : public basic_action<
            Component const, R(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type),
            Derived>
    {
    public:
        typedef basic_action<Component const,
            R(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type), Derived>
            base_type;
        
        static bool is_target_valid(naming::id_type const& id)
        {
            return Component::is_target_valid(id);
        }
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13 , Arg14 && arg14 , Arg15 && arg15 , Arg16 && arg16) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component const>::call(lva)) << ")";
                    (get_lva<Component const>::call(lva)->*F)(
                        std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ) , std::forward<Arg14>( arg14 ) , std::forward<Arg15>( arg15 ) , std::forward<Arg16>( arg16 ));
                }
                catch (hpx::thread_interrupted const&) {
                     
                }
                catch (hpx::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << "): " << e.what();
                    
                    hpx::report_error(boost::current_exception());
                }
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << ")";
                    
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
                    lva, util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)) , util::get< 5>(std::forward<Arguments>( args)) , util::get< 6>(std::forward<Arguments>( args)) , util::get< 7>(std::forward<Arguments>( args)) , util::get< 8>(std::forward<Arguments>( args)) , util::get< 9>(std::forward<Arguments>( args)) , util::get< 10>(std::forward<Arguments>( args)) , util::get< 11>(std::forward<Arguments>( args)) , util::get< 12>(std::forward<Arguments>( args)) , util::get< 13>(std::forward<Arguments>( args)) , util::get< 14>(std::forward<Arguments>( args)) , util::get< 15>(std::forward<Arguments>( args)) , util::get< 16>(std::forward<Arguments>( args))));
        }
        
        
        
        
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, Arguments && args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                base_type::construct_continuation_thread_object_function(
                    cont, F, get_lva<Component const>::call(lva),
                    std::forward<Arguments>(args)));
        }
        
        template <typename Arguments>
        BOOST_FORCEINLINE static R
        execute_function(naming::address::address_type lva,
            Arguments && args)
        {
            LTM_(debug)
                << "component_base_action" << 17
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component const>::call(lva)) << ")";
            return (get_lva<Component const>::call(lva)->*F)(
                util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)) , util::get< 5>(std::forward<Arguments>( args)) , util::get< 6>(std::forward<Arguments>( args)) , util::get< 7>(std::forward<Arguments>( args)) , util::get< 8>(std::forward<Arguments>( args)) , util::get< 9>(std::forward<Arguments>( args)) , util::get< 10>(std::forward<Arguments>( args)) , util::get< 11>(std::forward<Arguments>( args)) , util::get< 12>(std::forward<Arguments>( args)) , util::get< 13>(std::forward<Arguments>( args)) , util::get< 14>(std::forward<Arguments>( args)) , util::get< 15>(std::forward<Arguments>( args)) , util::get< 16>(std::forward<Arguments>( args)));
        }
    };
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16) const,
        typename Derived>
    struct component_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16) const, F, Derived>
      : component_base_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16) const, F,
            typename detail::action_type<
                component_action<
                    R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16) const, F, Derived>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            component_action, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    
    template <
        typename Component, typename R, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16) const, typename Derived>
    struct make_action<R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16) const,
            F, Derived, boost::mpl::false_>
      : component_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16) const, F, Derived>
    {
        typedef component_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16) const, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16) const,
        typename Derived>
    struct component_direct_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16) const, F, Derived>
      : component_base_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16) const, F,
            typename detail::action_type<
                component_direct_action<
                    R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16) const, F, Derived>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            component_direct_action, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    
    template <
        typename Component, typename R, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16) const, typename Derived>
    struct make_action<R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16) const,
            F, Derived, boost::mpl::true_>
      : component_direct_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16) const, F, Derived>
    {
        typedef component_direct_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16) const, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16) const, typename Derived>
    class component_base_action<
            void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16) const, F, Derived>
      : public basic_action<
            Component const, util::unused_type(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type),
            Derived>
    {
    public:
        typedef basic_action<Component const,
            util::unused_type(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type), Derived>
            base_type;
        
        static bool is_target_valid(naming::id_type const& id)
        {
            return Component::is_target_valid(id);
        }
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13 , Arg14 && arg14 , Arg15 && arg15 , Arg16 && arg16) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component const>::call(lva)) << ")";
                    (get_lva<Component const>::call(lva)->*F)(
                        std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ) , std::forward<Arg14>( arg14 ) , std::forward<Arg15>( arg15 ) , std::forward<Arg16>( arg16 ));
                }
                catch (hpx::thread_interrupted const&) {
                     
                }
                catch (hpx::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << "): " << e.what();
                    
                    hpx::report_error(boost::current_exception());
                }
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << ")";
                    
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
                    util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)) , util::get< 5>(std::forward<Arguments>( args)) , util::get< 6>(std::forward<Arguments>( args)) , util::get< 7>(std::forward<Arguments>( args)) , util::get< 8>(std::forward<Arguments>( args)) , util::get< 9>(std::forward<Arguments>( args)) , util::get< 10>(std::forward<Arguments>( args)) , util::get< 11>(std::forward<Arguments>( args)) , util::get< 12>(std::forward<Arguments>( args)) , util::get< 13>(std::forward<Arguments>( args)) , util::get< 14>(std::forward<Arguments>( args)) , util::get< 15>(std::forward<Arguments>( args)) , util::get< 16>(std::forward<Arguments>( args))));
        }
        
        
        
        
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, Arguments && args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                base_type::construct_continuation_thread_object_function_void(
                    cont, F, get_lva<Component const>::call(lva),
                    std::forward<Arguments>(args)));
        }
        
        template <typename Arguments>
        BOOST_FORCEINLINE static util::unused_type
        execute_function(naming::address::address_type lva,
            Arguments && args)
        {
            LTM_(debug)
                << "component_base_action" << 17
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component const>::call(lva)) << ")";
            (get_lva<Component const>::call(lva)->*F)(
                util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)) , util::get< 5>(std::forward<Arguments>( args)) , util::get< 6>(std::forward<Arguments>( args)) , util::get< 7>(std::forward<Arguments>( args)) , util::get< 8>(std::forward<Arguments>( args)) , util::get< 9>(std::forward<Arguments>( args)) , util::get< 10>(std::forward<Arguments>( args)) , util::get< 11>(std::forward<Arguments>( args)) , util::get< 12>(std::forward<Arguments>( args)) , util::get< 13>(std::forward<Arguments>( args)) , util::get< 14>(std::forward<Arguments>( args)) , util::get< 15>(std::forward<Arguments>( args)) , util::get< 16>(std::forward<Arguments>( args)));
            return util::unused;
        }
    };
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17) const, typename Derived>
    class component_base_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17) const, F, Derived>
      : public basic_action<
            Component const, R(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type),
            Derived>
    {
    public:
        typedef basic_action<Component const,
            R(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type), Derived>
            base_type;
        
        static bool is_target_valid(naming::id_type const& id)
        {
            return Component::is_target_valid(id);
        }
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13 , Arg14 && arg14 , Arg15 && arg15 , Arg16 && arg16 , Arg17 && arg17) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component const>::call(lva)) << ")";
                    (get_lva<Component const>::call(lva)->*F)(
                        std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ) , std::forward<Arg14>( arg14 ) , std::forward<Arg15>( arg15 ) , std::forward<Arg16>( arg16 ) , std::forward<Arg17>( arg17 ));
                }
                catch (hpx::thread_interrupted const&) {
                     
                }
                catch (hpx::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << "): " << e.what();
                    
                    hpx::report_error(boost::current_exception());
                }
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << ")";
                    
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
                    lva, util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)) , util::get< 5>(std::forward<Arguments>( args)) , util::get< 6>(std::forward<Arguments>( args)) , util::get< 7>(std::forward<Arguments>( args)) , util::get< 8>(std::forward<Arguments>( args)) , util::get< 9>(std::forward<Arguments>( args)) , util::get< 10>(std::forward<Arguments>( args)) , util::get< 11>(std::forward<Arguments>( args)) , util::get< 12>(std::forward<Arguments>( args)) , util::get< 13>(std::forward<Arguments>( args)) , util::get< 14>(std::forward<Arguments>( args)) , util::get< 15>(std::forward<Arguments>( args)) , util::get< 16>(std::forward<Arguments>( args)) , util::get< 17>(std::forward<Arguments>( args))));
        }
        
        
        
        
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, Arguments && args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                base_type::construct_continuation_thread_object_function(
                    cont, F, get_lva<Component const>::call(lva),
                    std::forward<Arguments>(args)));
        }
        
        template <typename Arguments>
        BOOST_FORCEINLINE static R
        execute_function(naming::address::address_type lva,
            Arguments && args)
        {
            LTM_(debug)
                << "component_base_action" << 18
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component const>::call(lva)) << ")";
            return (get_lva<Component const>::call(lva)->*F)(
                util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)) , util::get< 5>(std::forward<Arguments>( args)) , util::get< 6>(std::forward<Arguments>( args)) , util::get< 7>(std::forward<Arguments>( args)) , util::get< 8>(std::forward<Arguments>( args)) , util::get< 9>(std::forward<Arguments>( args)) , util::get< 10>(std::forward<Arguments>( args)) , util::get< 11>(std::forward<Arguments>( args)) , util::get< 12>(std::forward<Arguments>( args)) , util::get< 13>(std::forward<Arguments>( args)) , util::get< 14>(std::forward<Arguments>( args)) , util::get< 15>(std::forward<Arguments>( args)) , util::get< 16>(std::forward<Arguments>( args)) , util::get< 17>(std::forward<Arguments>( args)));
        }
    };
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17) const,
        typename Derived>
    struct component_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17) const, F, Derived>
      : component_base_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17) const, F,
            typename detail::action_type<
                component_action<
                    R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17) const, F, Derived>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            component_action, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    
    template <
        typename Component, typename R, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17) const, typename Derived>
    struct make_action<R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17) const,
            F, Derived, boost::mpl::false_>
      : component_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17) const, F, Derived>
    {
        typedef component_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17) const, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17) const,
        typename Derived>
    struct component_direct_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17) const, F, Derived>
      : component_base_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17) const, F,
            typename detail::action_type<
                component_direct_action<
                    R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17) const, F, Derived>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            component_direct_action, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    
    template <
        typename Component, typename R, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17) const, typename Derived>
    struct make_action<R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17) const,
            F, Derived, boost::mpl::true_>
      : component_direct_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17) const, F, Derived>
    {
        typedef component_direct_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17) const, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17) const, typename Derived>
    class component_base_action<
            void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17) const, F, Derived>
      : public basic_action<
            Component const, util::unused_type(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type),
            Derived>
    {
    public:
        typedef basic_action<Component const,
            util::unused_type(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type), Derived>
            base_type;
        
        static bool is_target_valid(naming::id_type const& id)
        {
            return Component::is_target_valid(id);
        }
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13 , Arg14 && arg14 , Arg15 && arg15 , Arg16 && arg16 , Arg17 && arg17) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component const>::call(lva)) << ")";
                    (get_lva<Component const>::call(lva)->*F)(
                        std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ) , std::forward<Arg14>( arg14 ) , std::forward<Arg15>( arg15 ) , std::forward<Arg16>( arg16 ) , std::forward<Arg17>( arg17 ));
                }
                catch (hpx::thread_interrupted const&) {
                     
                }
                catch (hpx::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << "): " << e.what();
                    
                    hpx::report_error(boost::current_exception());
                }
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << ")";
                    
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
                    util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)) , util::get< 5>(std::forward<Arguments>( args)) , util::get< 6>(std::forward<Arguments>( args)) , util::get< 7>(std::forward<Arguments>( args)) , util::get< 8>(std::forward<Arguments>( args)) , util::get< 9>(std::forward<Arguments>( args)) , util::get< 10>(std::forward<Arguments>( args)) , util::get< 11>(std::forward<Arguments>( args)) , util::get< 12>(std::forward<Arguments>( args)) , util::get< 13>(std::forward<Arguments>( args)) , util::get< 14>(std::forward<Arguments>( args)) , util::get< 15>(std::forward<Arguments>( args)) , util::get< 16>(std::forward<Arguments>( args)) , util::get< 17>(std::forward<Arguments>( args))));
        }
        
        
        
        
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, Arguments && args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                base_type::construct_continuation_thread_object_function_void(
                    cont, F, get_lva<Component const>::call(lva),
                    std::forward<Arguments>(args)));
        }
        
        template <typename Arguments>
        BOOST_FORCEINLINE static util::unused_type
        execute_function(naming::address::address_type lva,
            Arguments && args)
        {
            LTM_(debug)
                << "component_base_action" << 18
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component const>::call(lva)) << ")";
            (get_lva<Component const>::call(lva)->*F)(
                util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)) , util::get< 5>(std::forward<Arguments>( args)) , util::get< 6>(std::forward<Arguments>( args)) , util::get< 7>(std::forward<Arguments>( args)) , util::get< 8>(std::forward<Arguments>( args)) , util::get< 9>(std::forward<Arguments>( args)) , util::get< 10>(std::forward<Arguments>( args)) , util::get< 11>(std::forward<Arguments>( args)) , util::get< 12>(std::forward<Arguments>( args)) , util::get< 13>(std::forward<Arguments>( args)) , util::get< 14>(std::forward<Arguments>( args)) , util::get< 15>(std::forward<Arguments>( args)) , util::get< 16>(std::forward<Arguments>( args)) , util::get< 17>(std::forward<Arguments>( args)));
            return util::unused;
        }
    };
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18) const, typename Derived>
    class component_base_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18) const, F, Derived>
      : public basic_action<
            Component const, R(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type , typename util::decay<T18>::type),
            Derived>
    {
    public:
        typedef basic_action<Component const,
            R(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type , typename util::decay<T18>::type), Derived>
            base_type;
        
        static bool is_target_valid(naming::id_type const& id)
        {
            return Component::is_target_valid(id);
        }
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17 , typename Arg18>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13 , Arg14 && arg14 , Arg15 && arg15 , Arg16 && arg16 , Arg17 && arg17 , Arg18 && arg18) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component const>::call(lva)) << ")";
                    (get_lva<Component const>::call(lva)->*F)(
                        std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ) , std::forward<Arg14>( arg14 ) , std::forward<Arg15>( arg15 ) , std::forward<Arg16>( arg16 ) , std::forward<Arg17>( arg17 ) , std::forward<Arg18>( arg18 ));
                }
                catch (hpx::thread_interrupted const&) {
                     
                }
                catch (hpx::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << "): " << e.what();
                    
                    hpx::report_error(boost::current_exception());
                }
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << ")";
                    
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
                    lva, util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)) , util::get< 5>(std::forward<Arguments>( args)) , util::get< 6>(std::forward<Arguments>( args)) , util::get< 7>(std::forward<Arguments>( args)) , util::get< 8>(std::forward<Arguments>( args)) , util::get< 9>(std::forward<Arguments>( args)) , util::get< 10>(std::forward<Arguments>( args)) , util::get< 11>(std::forward<Arguments>( args)) , util::get< 12>(std::forward<Arguments>( args)) , util::get< 13>(std::forward<Arguments>( args)) , util::get< 14>(std::forward<Arguments>( args)) , util::get< 15>(std::forward<Arguments>( args)) , util::get< 16>(std::forward<Arguments>( args)) , util::get< 17>(std::forward<Arguments>( args)) , util::get< 18>(std::forward<Arguments>( args))));
        }
        
        
        
        
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, Arguments && args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                base_type::construct_continuation_thread_object_function(
                    cont, F, get_lva<Component const>::call(lva),
                    std::forward<Arguments>(args)));
        }
        
        template <typename Arguments>
        BOOST_FORCEINLINE static R
        execute_function(naming::address::address_type lva,
            Arguments && args)
        {
            LTM_(debug)
                << "component_base_action" << 19
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component const>::call(lva)) << ")";
            return (get_lva<Component const>::call(lva)->*F)(
                util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)) , util::get< 5>(std::forward<Arguments>( args)) , util::get< 6>(std::forward<Arguments>( args)) , util::get< 7>(std::forward<Arguments>( args)) , util::get< 8>(std::forward<Arguments>( args)) , util::get< 9>(std::forward<Arguments>( args)) , util::get< 10>(std::forward<Arguments>( args)) , util::get< 11>(std::forward<Arguments>( args)) , util::get< 12>(std::forward<Arguments>( args)) , util::get< 13>(std::forward<Arguments>( args)) , util::get< 14>(std::forward<Arguments>( args)) , util::get< 15>(std::forward<Arguments>( args)) , util::get< 16>(std::forward<Arguments>( args)) , util::get< 17>(std::forward<Arguments>( args)) , util::get< 18>(std::forward<Arguments>( args)));
        }
    };
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18) const,
        typename Derived>
    struct component_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18) const, F, Derived>
      : component_base_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18) const, F,
            typename detail::action_type<
                component_action<
                    R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18) const, F, Derived>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            component_action, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    
    template <
        typename Component, typename R, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18) const, typename Derived>
    struct make_action<R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18) const,
            F, Derived, boost::mpl::false_>
      : component_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18) const, F, Derived>
    {
        typedef component_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18) const, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18) const,
        typename Derived>
    struct component_direct_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18) const, F, Derived>
      : component_base_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18) const, F,
            typename detail::action_type<
                component_direct_action<
                    R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18) const, F, Derived>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            component_direct_action, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    
    template <
        typename Component, typename R, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18) const, typename Derived>
    struct make_action<R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18) const,
            F, Derived, boost::mpl::true_>
      : component_direct_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18) const, F, Derived>
    {
        typedef component_direct_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18) const, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18) const, typename Derived>
    class component_base_action<
            void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18) const, F, Derived>
      : public basic_action<
            Component const, util::unused_type(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type , typename util::decay<T18>::type),
            Derived>
    {
    public:
        typedef basic_action<Component const,
            util::unused_type(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type , typename util::decay<T18>::type), Derived>
            base_type;
        
        static bool is_target_valid(naming::id_type const& id)
        {
            return Component::is_target_valid(id);
        }
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17 , typename Arg18>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13 , Arg14 && arg14 , Arg15 && arg15 , Arg16 && arg16 , Arg17 && arg17 , Arg18 && arg18) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component const>::call(lva)) << ")";
                    (get_lva<Component const>::call(lva)->*F)(
                        std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ) , std::forward<Arg14>( arg14 ) , std::forward<Arg15>( arg15 ) , std::forward<Arg16>( arg16 ) , std::forward<Arg17>( arg17 ) , std::forward<Arg18>( arg18 ));
                }
                catch (hpx::thread_interrupted const&) {
                     
                }
                catch (hpx::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << "): " << e.what();
                    
                    hpx::report_error(boost::current_exception());
                }
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << ")";
                    
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
                    util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)) , util::get< 5>(std::forward<Arguments>( args)) , util::get< 6>(std::forward<Arguments>( args)) , util::get< 7>(std::forward<Arguments>( args)) , util::get< 8>(std::forward<Arguments>( args)) , util::get< 9>(std::forward<Arguments>( args)) , util::get< 10>(std::forward<Arguments>( args)) , util::get< 11>(std::forward<Arguments>( args)) , util::get< 12>(std::forward<Arguments>( args)) , util::get< 13>(std::forward<Arguments>( args)) , util::get< 14>(std::forward<Arguments>( args)) , util::get< 15>(std::forward<Arguments>( args)) , util::get< 16>(std::forward<Arguments>( args)) , util::get< 17>(std::forward<Arguments>( args)) , util::get< 18>(std::forward<Arguments>( args))));
        }
        
        
        
        
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, Arguments && args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                base_type::construct_continuation_thread_object_function_void(
                    cont, F, get_lva<Component const>::call(lva),
                    std::forward<Arguments>(args)));
        }
        
        template <typename Arguments>
        BOOST_FORCEINLINE static util::unused_type
        execute_function(naming::address::address_type lva,
            Arguments && args)
        {
            LTM_(debug)
                << "component_base_action" << 19
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component const>::call(lva)) << ")";
            (get_lva<Component const>::call(lva)->*F)(
                util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)) , util::get< 5>(std::forward<Arguments>( args)) , util::get< 6>(std::forward<Arguments>( args)) , util::get< 7>(std::forward<Arguments>( args)) , util::get< 8>(std::forward<Arguments>( args)) , util::get< 9>(std::forward<Arguments>( args)) , util::get< 10>(std::forward<Arguments>( args)) , util::get< 11>(std::forward<Arguments>( args)) , util::get< 12>(std::forward<Arguments>( args)) , util::get< 13>(std::forward<Arguments>( args)) , util::get< 14>(std::forward<Arguments>( args)) , util::get< 15>(std::forward<Arguments>( args)) , util::get< 16>(std::forward<Arguments>( args)) , util::get< 17>(std::forward<Arguments>( args)) , util::get< 18>(std::forward<Arguments>( args)));
            return util::unused;
        }
    };
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19) const, typename Derived>
    class component_base_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19) const, F, Derived>
      : public basic_action<
            Component const, R(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type , typename util::decay<T18>::type , typename util::decay<T19>::type),
            Derived>
    {
    public:
        typedef basic_action<Component const,
            R(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type , typename util::decay<T18>::type , typename util::decay<T19>::type), Derived>
            base_type;
        
        static bool is_target_valid(naming::id_type const& id)
        {
            return Component::is_target_valid(id);
        }
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17 , typename Arg18 , typename Arg19>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13 , Arg14 && arg14 , Arg15 && arg15 , Arg16 && arg16 , Arg17 && arg17 , Arg18 && arg18 , Arg19 && arg19) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component const>::call(lva)) << ")";
                    (get_lva<Component const>::call(lva)->*F)(
                        std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ) , std::forward<Arg14>( arg14 ) , std::forward<Arg15>( arg15 ) , std::forward<Arg16>( arg16 ) , std::forward<Arg17>( arg17 ) , std::forward<Arg18>( arg18 ) , std::forward<Arg19>( arg19 ));
                }
                catch (hpx::thread_interrupted const&) {
                     
                }
                catch (hpx::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << "): " << e.what();
                    
                    hpx::report_error(boost::current_exception());
                }
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << ")";
                    
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
                    lva, util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)) , util::get< 5>(std::forward<Arguments>( args)) , util::get< 6>(std::forward<Arguments>( args)) , util::get< 7>(std::forward<Arguments>( args)) , util::get< 8>(std::forward<Arguments>( args)) , util::get< 9>(std::forward<Arguments>( args)) , util::get< 10>(std::forward<Arguments>( args)) , util::get< 11>(std::forward<Arguments>( args)) , util::get< 12>(std::forward<Arguments>( args)) , util::get< 13>(std::forward<Arguments>( args)) , util::get< 14>(std::forward<Arguments>( args)) , util::get< 15>(std::forward<Arguments>( args)) , util::get< 16>(std::forward<Arguments>( args)) , util::get< 17>(std::forward<Arguments>( args)) , util::get< 18>(std::forward<Arguments>( args)) , util::get< 19>(std::forward<Arguments>( args))));
        }
        
        
        
        
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, Arguments && args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                base_type::construct_continuation_thread_object_function(
                    cont, F, get_lva<Component const>::call(lva),
                    std::forward<Arguments>(args)));
        }
        
        template <typename Arguments>
        BOOST_FORCEINLINE static R
        execute_function(naming::address::address_type lva,
            Arguments && args)
        {
            LTM_(debug)
                << "component_base_action" << 20
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component const>::call(lva)) << ")";
            return (get_lva<Component const>::call(lva)->*F)(
                util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)) , util::get< 5>(std::forward<Arguments>( args)) , util::get< 6>(std::forward<Arguments>( args)) , util::get< 7>(std::forward<Arguments>( args)) , util::get< 8>(std::forward<Arguments>( args)) , util::get< 9>(std::forward<Arguments>( args)) , util::get< 10>(std::forward<Arguments>( args)) , util::get< 11>(std::forward<Arguments>( args)) , util::get< 12>(std::forward<Arguments>( args)) , util::get< 13>(std::forward<Arguments>( args)) , util::get< 14>(std::forward<Arguments>( args)) , util::get< 15>(std::forward<Arguments>( args)) , util::get< 16>(std::forward<Arguments>( args)) , util::get< 17>(std::forward<Arguments>( args)) , util::get< 18>(std::forward<Arguments>( args)) , util::get< 19>(std::forward<Arguments>( args)));
        }
    };
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19) const,
        typename Derived>
    struct component_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19) const, F, Derived>
      : component_base_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19) const, F,
            typename detail::action_type<
                component_action<
                    R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19) const, F, Derived>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            component_action, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    
    template <
        typename Component, typename R, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19) const, typename Derived>
    struct make_action<R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19) const,
            F, Derived, boost::mpl::false_>
      : component_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19) const, F, Derived>
    {
        typedef component_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19) const, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19) const,
        typename Derived>
    struct component_direct_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19) const, F, Derived>
      : component_base_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19) const, F,
            typename detail::action_type<
                component_direct_action<
                    R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19) const, F, Derived>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            component_direct_action, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    
    template <
        typename Component, typename R, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19) const, typename Derived>
    struct make_action<R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19) const,
            F, Derived, boost::mpl::true_>
      : component_direct_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19) const, F, Derived>
    {
        typedef component_direct_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19) const, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19) const, typename Derived>
    class component_base_action<
            void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19) const, F, Derived>
      : public basic_action<
            Component const, util::unused_type(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type , typename util::decay<T18>::type , typename util::decay<T19>::type),
            Derived>
    {
    public:
        typedef basic_action<Component const,
            util::unused_type(typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type , typename util::decay<T18>::type , typename util::decay<T19>::type), Derived>
            base_type;
        
        static bool is_target_valid(naming::id_type const& id)
        {
            return Component::is_target_valid(id);
        }
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17 , typename Arg18 , typename Arg19>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13 , Arg14 && arg14 , Arg15 && arg15 , Arg16 && arg16 , Arg17 && arg17 , Arg18 && arg18 , Arg19 && arg19) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component const>::call(lva)) << ")";
                    (get_lva<Component const>::call(lva)->*F)(
                        std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ) , std::forward<Arg14>( arg14 ) , std::forward<Arg15>( arg15 ) , std::forward<Arg16>( arg16 ) , std::forward<Arg17>( arg17 ) , std::forward<Arg18>( arg18 ) , std::forward<Arg19>( arg19 ));
                }
                catch (hpx::thread_interrupted const&) {
                     
                }
                catch (hpx::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << "): " << e.what();
                    
                    hpx::report_error(boost::current_exception());
                }
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << ")";
                    
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
                    util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)) , util::get< 5>(std::forward<Arguments>( args)) , util::get< 6>(std::forward<Arguments>( args)) , util::get< 7>(std::forward<Arguments>( args)) , util::get< 8>(std::forward<Arguments>( args)) , util::get< 9>(std::forward<Arguments>( args)) , util::get< 10>(std::forward<Arguments>( args)) , util::get< 11>(std::forward<Arguments>( args)) , util::get< 12>(std::forward<Arguments>( args)) , util::get< 13>(std::forward<Arguments>( args)) , util::get< 14>(std::forward<Arguments>( args)) , util::get< 15>(std::forward<Arguments>( args)) , util::get< 16>(std::forward<Arguments>( args)) , util::get< 17>(std::forward<Arguments>( args)) , util::get< 18>(std::forward<Arguments>( args)) , util::get< 19>(std::forward<Arguments>( args))));
        }
        
        
        
        
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, Arguments && args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                base_type::construct_continuation_thread_object_function_void(
                    cont, F, get_lva<Component const>::call(lva),
                    std::forward<Arguments>(args)));
        }
        
        template <typename Arguments>
        BOOST_FORCEINLINE static util::unused_type
        execute_function(naming::address::address_type lva,
            Arguments && args)
        {
            LTM_(debug)
                << "component_base_action" << 20
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component const>::call(lva)) << ")";
            (get_lva<Component const>::call(lva)->*F)(
                util::get< 0>(std::forward<Arguments>( args)) , util::get< 1>(std::forward<Arguments>( args)) , util::get< 2>(std::forward<Arguments>( args)) , util::get< 3>(std::forward<Arguments>( args)) , util::get< 4>(std::forward<Arguments>( args)) , util::get< 5>(std::forward<Arguments>( args)) , util::get< 6>(std::forward<Arguments>( args)) , util::get< 7>(std::forward<Arguments>( args)) , util::get< 8>(std::forward<Arguments>( args)) , util::get< 9>(std::forward<Arguments>( args)) , util::get< 10>(std::forward<Arguments>( args)) , util::get< 11>(std::forward<Arguments>( args)) , util::get< 12>(std::forward<Arguments>( args)) , util::get< 13>(std::forward<Arguments>( args)) , util::get< 14>(std::forward<Arguments>( args)) , util::get< 15>(std::forward<Arguments>( args)) , util::get< 16>(std::forward<Arguments>( args)) , util::get< 17>(std::forward<Arguments>( args)) , util::get< 18>(std::forward<Arguments>( args)) , util::get< 19>(std::forward<Arguments>( args)));
            return util::unused;
        }
    };
}}
