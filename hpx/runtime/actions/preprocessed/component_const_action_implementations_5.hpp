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
    class basic_action_impl<
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
                << "basic_action_impl" << 1
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
    struct action<
            R (Component::*)(T0) const, F, Derived>
      : basic_action_impl<
            R (Component::*)(T0) const, F,
            typename detail::action_type<
                action<
                    R (Component::*)(T0) const, F, Derived>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            action, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    
    
    template <
        typename Component, typename R,
        typename T0,
        R (Component::*F)(T0) const,
        typename Derived>
    struct direct_action<
            R (Component::*)(T0) const, F, Derived>
      : basic_action_impl<
            R (Component::*)(T0) const, F,
            typename detail::action_type<
                direct_action<
                    R (Component::*)(T0) const, F, Derived>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_action, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    
    
    template <
        typename Component, typename T0,
        void (Component::*F)(T0) const, typename Derived>
    class basic_action_impl<
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
                << "basic_action_impl" << 1
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
    class basic_action_impl<
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
                << "basic_action_impl" << 2
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
    struct action<
            R (Component::*)(T0 , T1) const, F, Derived>
      : basic_action_impl<
            R (Component::*)(T0 , T1) const, F,
            typename detail::action_type<
                action<
                    R (Component::*)(T0 , T1) const, F, Derived>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            action, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1,
        R (Component::*F)(T0 , T1) const,
        typename Derived>
    struct direct_action<
            R (Component::*)(T0 , T1) const, F, Derived>
      : basic_action_impl<
            R (Component::*)(T0 , T1) const, F,
            typename detail::action_type<
                direct_action<
                    R (Component::*)(T0 , T1) const, F, Derived>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_action, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    
    
    template <
        typename Component, typename T0 , typename T1,
        void (Component::*F)(T0 , T1) const, typename Derived>
    class basic_action_impl<
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
                << "basic_action_impl" << 2
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
    class basic_action_impl<
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
                << "basic_action_impl" << 3
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
    struct action<
            R (Component::*)(T0 , T1 , T2) const, F, Derived>
      : basic_action_impl<
            R (Component::*)(T0 , T1 , T2) const, F,
            typename detail::action_type<
                action<
                    R (Component::*)(T0 , T1 , T2) const, F, Derived>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            action, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1 , typename T2,
        R (Component::*F)(T0 , T1 , T2) const,
        typename Derived>
    struct direct_action<
            R (Component::*)(T0 , T1 , T2) const, F, Derived>
      : basic_action_impl<
            R (Component::*)(T0 , T1 , T2) const, F,
            typename detail::action_type<
                direct_action<
                    R (Component::*)(T0 , T1 , T2) const, F, Derived>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_action, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    
    
    template <
        typename Component, typename T0 , typename T1 , typename T2,
        void (Component::*F)(T0 , T1 , T2) const, typename Derived>
    class basic_action_impl<
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
                << "basic_action_impl" << 3
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
    class basic_action_impl<
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
                << "basic_action_impl" << 4
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
    struct action<
            R (Component::*)(T0 , T1 , T2 , T3) const, F, Derived>
      : basic_action_impl<
            R (Component::*)(T0 , T1 , T2 , T3) const, F,
            typename detail::action_type<
                action<
                    R (Component::*)(T0 , T1 , T2 , T3) const, F, Derived>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            action, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1 , typename T2 , typename T3,
        R (Component::*F)(T0 , T1 , T2 , T3) const,
        typename Derived>
    struct direct_action<
            R (Component::*)(T0 , T1 , T2 , T3) const, F, Derived>
      : basic_action_impl<
            R (Component::*)(T0 , T1 , T2 , T3) const, F,
            typename detail::action_type<
                direct_action<
                    R (Component::*)(T0 , T1 , T2 , T3) const, F, Derived>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_action, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3,
        void (Component::*F)(T0 , T1 , T2 , T3) const, typename Derived>
    class basic_action_impl<
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
                << "basic_action_impl" << 4
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
    class basic_action_impl<
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
                << "basic_action_impl" << 5
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
    struct action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4) const, F, Derived>
      : basic_action_impl<
            R (Component::*)(T0 , T1 , T2 , T3 , T4) const, F,
            typename detail::action_type<
                action<
                    R (Component::*)(T0 , T1 , T2 , T3 , T4) const, F, Derived>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            action, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    
    
    template <
        typename Component, typename R,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        R (Component::*F)(T0 , T1 , T2 , T3 , T4) const,
        typename Derived>
    struct direct_action<
            R (Component::*)(T0 , T1 , T2 , T3 , T4) const, F, Derived>
      : basic_action_impl<
            R (Component::*)(T0 , T1 , T2 , T3 , T4) const, F,
            typename detail::action_type<
                direct_action<
                    R (Component::*)(T0 , T1 , T2 , T3 , T4) const, F, Derived>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_action, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4) const, typename Derived>
    class basic_action_impl<
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
                << "basic_action_impl" << 5
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
