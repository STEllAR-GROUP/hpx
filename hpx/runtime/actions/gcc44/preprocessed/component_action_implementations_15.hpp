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
        typename Component, typename Result,
        typename T0,
        Result (Component::*F)(T0), typename Derived>
    class base_result_action1
      : public action<
            Component, Result,
            hpx::util::tuple<typename util::decay<T0>::type>,
            Derived>
    {
    public:
        typedef Result result_type;
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
                BOOST_FWD_REF(Arg0) arg0) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component>::call(lva)) << ")";
                    
                    
                    
                    
                    (get_lva<Component>::call(lva)->*F)(
                        boost::move(arg0));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing component action("
                            << detail::get_action_name<Derived>()
                            << ") lva(" << reinterpret_cast<void const*>
                                (get_lva<Component>::call(lva)) << "): " << e.what();
                        
                        hpx::report_error(boost::current_exception());
                    }
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
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            return Derived::decorate_action(
                HPX_STD_BIND(typename Derived::thread_function(), lva,
                    util::get< 0>(boost::forward<Arguments>( args))), lva);
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return Derived::decorate_action(
                    base_type::construct_continuation_thread_object_function(
                        cont, F, get_lva<Component>::call(lva),
                        boost::forward<Arguments>(args)), lva);
        }
        template <typename Arguments>
        BOOST_FORCEINLINE static Result
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "base_result_action" << 1
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            return (get_lva<Component>::call(lva)->*F)(
                util::get< 0>(boost::forward<Arguments>( args)));
        }
    };
    
    
    template <
        typename Component, typename Result,
        typename T0,
        Result (Component::*F)(T0),
        typename Derived = detail::this_type>
    struct result_action1
      : base_result_action1<
            Component, Result,
            T0, F,
            typename detail::action_type<
                result_action1<
                    Component, Result, T0, F>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            result_action1, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Component, typename Result,
        typename T0,
        Result (Component::*F)(T0), typename Derived>
    struct make_action<Result (Component::*)(T0),
            F, Derived, boost::mpl::false_>
      : result_action1<
            Component, Result,
            T0, F, Derived>
    {
        typedef result_action1<
            Component, Result,
            T0, F, Derived
        > type;
    };
    template <typename Component, typename Result,
        typename T0,
        Result (Component::*F)(T0) const, typename Derived>
    struct make_action<Result (Component::*)(T0) const,
            F, Derived, boost::mpl::false_>
      : result_action1<
            Component const, Result,
            T0, F, Derived>
    {
        typedef result_action1<
            Component const, Result,
            T0, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename Result,
        typename T0,
        Result (Component::*F)(T0),
        typename Derived = detail::this_type>
    struct direct_result_action1
      : base_result_action1<
            Component, Result,
            T0, F,
            typename detail::action_type<
                direct_result_action1<
                    Component, Result, T0, F>,
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
    template <typename Component, typename Result,
        typename T0,
        Result (Component::*F)(T0), typename Derived>
    struct make_action<Result (Component::*)(T0),
            F, Derived, boost::mpl::true_>
      : direct_result_action1<
            Component, Result,
            T0, F, Derived>
    {
        typedef direct_result_action1<
            Component, Result,
            T0, F, Derived
        > type;
    };
    template <typename Component, typename Result,
        typename T0,
        Result (Component::*F)(T0) const, typename Derived>
    struct make_action<Result (Component::*)(T0) const,
            F, Derived, boost::mpl::true_>
      : direct_result_action1<
            Component const, Result,
            T0, F, Derived>
    {
        typedef direct_result_action1<
            Component const, Result,
            T0, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename T0,
        void (Component::*F)(T0), typename Derived>
    class base_action1
      : public action<
            Component, util::unused_type,
            hpx::util::tuple<typename util::decay<T0>::type>,
            Derived>
    {
    public:
        typedef util::unused_type result_type;
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
                BOOST_FWD_REF(Arg0) arg0) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component>::call(lva)) << ")";
                    
                    
                    
                    
                    (get_lva<Component>::call(lva)->*F)(
                        boost::move(arg0));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing component action("
                            << detail::get_action_name<Derived>()
                            << ") lva(" << reinterpret_cast<void const*>
                                (get_lva<Component>::call(lva)) << "): " << e.what();
                        
                        hpx::report_error(boost::current_exception());
                    }
                }
                
                
                
                util::force_error_on_lock();
                return threads::terminated;
            }
        };
    public:
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            
            
            return Derived::decorate_action(
                HPX_STD_BIND(typename Derived::thread_function(), lva,
                    util::get< 0>(boost::forward<Arguments>( args))), lva);
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return Derived::decorate_action(
                    base_type::construct_continuation_thread_object_function_void(
                        cont, F, get_lva<Component>::call(lva),
                        boost::forward<Arguments>(args)), lva);
        }
        template <typename Arguments>
        BOOST_FORCEINLINE static util::unused_type
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "base_action" << 1
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            (get_lva<Component>::call(lva)->*F)(
                util::get< 0>(boost::forward<Arguments>( args)));
            return util::unused;
        }
    };
    
    template <
        typename Component, typename T0,
        void (Component::*F)(T0),
        typename Derived = detail::this_type>
    struct action1
      : base_action1<
            Component, T0, F,
            typename detail::action_type<
                action1<
                    Component, T0, F>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            action1, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Component, typename T0,
        void (Component::*F)(T0), typename Derived>
    struct make_action<void (Component::*)(T0),
            F, Derived, boost::mpl::false_>
      : action1<
            Component,
            T0, F, Derived>
    {
        typedef action1<
            Component,
            T0, F, Derived
        > type;
    };
    template <typename Component, typename T0,
        void (Component::*F)(T0) const, typename Derived>
    struct make_action<void (Component::*)(T0) const,
            F, Derived, boost::mpl::false_>
      : action1<
            Component const,
            T0, F, Derived>
    {
        typedef action1<
            Component const,
            T0, F, Derived
        > type;
    };
    
    template <
        typename Component, typename T0,
        void (Component::*F)(T0),
        typename Derived = detail::this_type>
    struct direct_action1
      : base_action1<
            Component, T0, F,
            typename detail::action_type<
                direct_action1<
                    Component, T0, F>,
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
    template <typename Component, typename T0,
        void (Component::*F)(T0), typename Derived>
    struct make_action<void (Component::*)(T0),
            F, Derived, boost::mpl::true_>
      : direct_action1<
            Component,
            T0, F, Derived>
    {
        typedef direct_action1<
            Component,
            T0, F, Derived
        > type;
    };
    template <typename Component, typename T0,
        void (Component::*F)(T0) const, typename Derived>
    struct make_action<void (Component::*)(T0) const,
            F, Derived, boost::mpl::true_>
      : direct_action1<
            Component const,
            T0, F, Derived>
    {
        typedef direct_action1<
            Component const,
            T0, F, Derived
        > type;
    };
    
    
    template <
        typename Component,
        typename T0,
        void (Component::*F)(T0),
        typename Derived>
    struct result_action1<Component, void,
            T0, F, Derived>
      : action1<Component,
            T0, F, Derived>
    {};
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Component, typename Result,
        typename T0 , typename T1,
        Result (Component::*F)(T0 , T1), typename Derived>
    class base_result_action2
      : public action<
            Component, Result,
            hpx::util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type>,
            Derived>
    {
    public:
        typedef Result result_type;
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
                BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component>::call(lva)) << ")";
                    
                    
                    
                    
                    (get_lva<Component>::call(lva)->*F)(
                        boost::move(arg0) , boost::move(arg1));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing component action("
                            << detail::get_action_name<Derived>()
                            << ") lva(" << reinterpret_cast<void const*>
                                (get_lva<Component>::call(lva)) << "): " << e.what();
                        
                        hpx::report_error(boost::current_exception());
                    }
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
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            return Derived::decorate_action(
                HPX_STD_BIND(typename Derived::thread_function(), lva,
                    util::get< 0>(boost::forward<Arguments>( args)) , util::get< 1>(boost::forward<Arguments>( args))), lva);
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return Derived::decorate_action(
                    base_type::construct_continuation_thread_object_function(
                        cont, F, get_lva<Component>::call(lva),
                        boost::forward<Arguments>(args)), lva);
        }
        template <typename Arguments>
        BOOST_FORCEINLINE static Result
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "base_result_action" << 2
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            return (get_lva<Component>::call(lva)->*F)(
                util::get< 0>(boost::forward<Arguments>( args)) , util::get< 1>(boost::forward<Arguments>( args)));
        }
    };
    
    
    template <
        typename Component, typename Result,
        typename T0 , typename T1,
        Result (Component::*F)(T0 , T1),
        typename Derived = detail::this_type>
    struct result_action2
      : base_result_action2<
            Component, Result,
            T0 , T1, F,
            typename detail::action_type<
                result_action2<
                    Component, Result, T0 , T1, F>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            result_action2, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Component, typename Result,
        typename T0 , typename T1,
        Result (Component::*F)(T0 , T1), typename Derived>
    struct make_action<Result (Component::*)(T0 , T1),
            F, Derived, boost::mpl::false_>
      : result_action2<
            Component, Result,
            T0 , T1, F, Derived>
    {
        typedef result_action2<
            Component, Result,
            T0 , T1, F, Derived
        > type;
    };
    template <typename Component, typename Result,
        typename T0 , typename T1,
        Result (Component::*F)(T0 , T1) const, typename Derived>
    struct make_action<Result (Component::*)(T0 , T1) const,
            F, Derived, boost::mpl::false_>
      : result_action2<
            Component const, Result,
            T0 , T1, F, Derived>
    {
        typedef result_action2<
            Component const, Result,
            T0 , T1, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename Result,
        typename T0 , typename T1,
        Result (Component::*F)(T0 , T1),
        typename Derived = detail::this_type>
    struct direct_result_action2
      : base_result_action2<
            Component, Result,
            T0 , T1, F,
            typename detail::action_type<
                direct_result_action2<
                    Component, Result, T0 , T1, F>,
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
    template <typename Component, typename Result,
        typename T0 , typename T1,
        Result (Component::*F)(T0 , T1), typename Derived>
    struct make_action<Result (Component::*)(T0 , T1),
            F, Derived, boost::mpl::true_>
      : direct_result_action2<
            Component, Result,
            T0 , T1, F, Derived>
    {
        typedef direct_result_action2<
            Component, Result,
            T0 , T1, F, Derived
        > type;
    };
    template <typename Component, typename Result,
        typename T0 , typename T1,
        Result (Component::*F)(T0 , T1) const, typename Derived>
    struct make_action<Result (Component::*)(T0 , T1) const,
            F, Derived, boost::mpl::true_>
      : direct_result_action2<
            Component const, Result,
            T0 , T1, F, Derived>
    {
        typedef direct_result_action2<
            Component const, Result,
            T0 , T1, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename T0 , typename T1,
        void (Component::*F)(T0 , T1), typename Derived>
    class base_action2
      : public action<
            Component, util::unused_type,
            hpx::util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type>,
            Derived>
    {
    public:
        typedef util::unused_type result_type;
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
                BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component>::call(lva)) << ")";
                    
                    
                    
                    
                    (get_lva<Component>::call(lva)->*F)(
                        boost::move(arg0) , boost::move(arg1));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing component action("
                            << detail::get_action_name<Derived>()
                            << ") lva(" << reinterpret_cast<void const*>
                                (get_lva<Component>::call(lva)) << "): " << e.what();
                        
                        hpx::report_error(boost::current_exception());
                    }
                }
                
                
                
                util::force_error_on_lock();
                return threads::terminated;
            }
        };
    public:
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            
            
            return Derived::decorate_action(
                HPX_STD_BIND(typename Derived::thread_function(), lva,
                    util::get< 0>(boost::forward<Arguments>( args)) , util::get< 1>(boost::forward<Arguments>( args))), lva);
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return Derived::decorate_action(
                    base_type::construct_continuation_thread_object_function_void(
                        cont, F, get_lva<Component>::call(lva),
                        boost::forward<Arguments>(args)), lva);
        }
        template <typename Arguments>
        BOOST_FORCEINLINE static util::unused_type
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "base_action" << 2
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            (get_lva<Component>::call(lva)->*F)(
                util::get< 0>(boost::forward<Arguments>( args)) , util::get< 1>(boost::forward<Arguments>( args)));
            return util::unused;
        }
    };
    
    template <
        typename Component, typename T0 , typename T1,
        void (Component::*F)(T0 , T1),
        typename Derived = detail::this_type>
    struct action2
      : base_action2<
            Component, T0 , T1, F,
            typename detail::action_type<
                action2<
                    Component, T0 , T1, F>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            action2, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Component, typename T0 , typename T1,
        void (Component::*F)(T0 , T1), typename Derived>
    struct make_action<void (Component::*)(T0 , T1),
            F, Derived, boost::mpl::false_>
      : action2<
            Component,
            T0 , T1, F, Derived>
    {
        typedef action2<
            Component,
            T0 , T1, F, Derived
        > type;
    };
    template <typename Component, typename T0 , typename T1,
        void (Component::*F)(T0 , T1) const, typename Derived>
    struct make_action<void (Component::*)(T0 , T1) const,
            F, Derived, boost::mpl::false_>
      : action2<
            Component const,
            T0 , T1, F, Derived>
    {
        typedef action2<
            Component const,
            T0 , T1, F, Derived
        > type;
    };
    
    template <
        typename Component, typename T0 , typename T1,
        void (Component::*F)(T0 , T1),
        typename Derived = detail::this_type>
    struct direct_action2
      : base_action2<
            Component, T0 , T1, F,
            typename detail::action_type<
                direct_action2<
                    Component, T0 , T1, F>,
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
    template <typename Component, typename T0 , typename T1,
        void (Component::*F)(T0 , T1), typename Derived>
    struct make_action<void (Component::*)(T0 , T1),
            F, Derived, boost::mpl::true_>
      : direct_action2<
            Component,
            T0 , T1, F, Derived>
    {
        typedef direct_action2<
            Component,
            T0 , T1, F, Derived
        > type;
    };
    template <typename Component, typename T0 , typename T1,
        void (Component::*F)(T0 , T1) const, typename Derived>
    struct make_action<void (Component::*)(T0 , T1) const,
            F, Derived, boost::mpl::true_>
      : direct_action2<
            Component const,
            T0 , T1, F, Derived>
    {
        typedef direct_action2<
            Component const,
            T0 , T1, F, Derived
        > type;
    };
    
    
    template <
        typename Component,
        typename T0 , typename T1,
        void (Component::*F)(T0 , T1),
        typename Derived>
    struct result_action2<Component, void,
            T0 , T1, F, Derived>
      : action2<Component,
            T0 , T1, F, Derived>
    {};
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Component, typename Result,
        typename T0 , typename T1 , typename T2,
        Result (Component::*F)(T0 , T1 , T2), typename Derived>
    class base_result_action3
      : public action<
            Component, Result,
            hpx::util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type>,
            Derived>
    {
    public:
        typedef Result result_type;
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
                BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component>::call(lva)) << ")";
                    
                    
                    
                    
                    (get_lva<Component>::call(lva)->*F)(
                        boost::move(arg0) , boost::move(arg1) , boost::move(arg2));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing component action("
                            << detail::get_action_name<Derived>()
                            << ") lva(" << reinterpret_cast<void const*>
                                (get_lva<Component>::call(lva)) << "): " << e.what();
                        
                        hpx::report_error(boost::current_exception());
                    }
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
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            return Derived::decorate_action(
                HPX_STD_BIND(typename Derived::thread_function(), lva,
                    util::get< 0>(boost::forward<Arguments>( args)) , util::get< 1>(boost::forward<Arguments>( args)) , util::get< 2>(boost::forward<Arguments>( args))), lva);
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return Derived::decorate_action(
                    base_type::construct_continuation_thread_object_function(
                        cont, F, get_lva<Component>::call(lva),
                        boost::forward<Arguments>(args)), lva);
        }
        template <typename Arguments>
        BOOST_FORCEINLINE static Result
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "base_result_action" << 3
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            return (get_lva<Component>::call(lva)->*F)(
                util::get< 0>(boost::forward<Arguments>( args)) , util::get< 1>(boost::forward<Arguments>( args)) , util::get< 2>(boost::forward<Arguments>( args)));
        }
    };
    
    
    template <
        typename Component, typename Result,
        typename T0 , typename T1 , typename T2,
        Result (Component::*F)(T0 , T1 , T2),
        typename Derived = detail::this_type>
    struct result_action3
      : base_result_action3<
            Component, Result,
            T0 , T1 , T2, F,
            typename detail::action_type<
                result_action3<
                    Component, Result, T0 , T1 , T2, F>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            result_action3, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2,
        Result (Component::*F)(T0 , T1 , T2), typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2),
            F, Derived, boost::mpl::false_>
      : result_action3<
            Component, Result,
            T0 , T1 , T2, F, Derived>
    {
        typedef result_action3<
            Component, Result,
            T0 , T1 , T2, F, Derived
        > type;
    };
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2,
        Result (Component::*F)(T0 , T1 , T2) const, typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2) const,
            F, Derived, boost::mpl::false_>
      : result_action3<
            Component const, Result,
            T0 , T1 , T2, F, Derived>
    {
        typedef result_action3<
            Component const, Result,
            T0 , T1 , T2, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename Result,
        typename T0 , typename T1 , typename T2,
        Result (Component::*F)(T0 , T1 , T2),
        typename Derived = detail::this_type>
    struct direct_result_action3
      : base_result_action3<
            Component, Result,
            T0 , T1 , T2, F,
            typename detail::action_type<
                direct_result_action3<
                    Component, Result, T0 , T1 , T2, F>,
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
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2,
        Result (Component::*F)(T0 , T1 , T2), typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2),
            F, Derived, boost::mpl::true_>
      : direct_result_action3<
            Component, Result,
            T0 , T1 , T2, F, Derived>
    {
        typedef direct_result_action3<
            Component, Result,
            T0 , T1 , T2, F, Derived
        > type;
    };
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2,
        Result (Component::*F)(T0 , T1 , T2) const, typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2) const,
            F, Derived, boost::mpl::true_>
      : direct_result_action3<
            Component const, Result,
            T0 , T1 , T2, F, Derived>
    {
        typedef direct_result_action3<
            Component const, Result,
            T0 , T1 , T2, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename T0 , typename T1 , typename T2,
        void (Component::*F)(T0 , T1 , T2), typename Derived>
    class base_action3
      : public action<
            Component, util::unused_type,
            hpx::util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type>,
            Derived>
    {
    public:
        typedef util::unused_type result_type;
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
                BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component>::call(lva)) << ")";
                    
                    
                    
                    
                    (get_lva<Component>::call(lva)->*F)(
                        boost::move(arg0) , boost::move(arg1) , boost::move(arg2));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing component action("
                            << detail::get_action_name<Derived>()
                            << ") lva(" << reinterpret_cast<void const*>
                                (get_lva<Component>::call(lva)) << "): " << e.what();
                        
                        hpx::report_error(boost::current_exception());
                    }
                }
                
                
                
                util::force_error_on_lock();
                return threads::terminated;
            }
        };
    public:
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            
            
            return Derived::decorate_action(
                HPX_STD_BIND(typename Derived::thread_function(), lva,
                    util::get< 0>(boost::forward<Arguments>( args)) , util::get< 1>(boost::forward<Arguments>( args)) , util::get< 2>(boost::forward<Arguments>( args))), lva);
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return Derived::decorate_action(
                    base_type::construct_continuation_thread_object_function_void(
                        cont, F, get_lva<Component>::call(lva),
                        boost::forward<Arguments>(args)), lva);
        }
        template <typename Arguments>
        BOOST_FORCEINLINE static util::unused_type
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "base_action" << 3
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            (get_lva<Component>::call(lva)->*F)(
                util::get< 0>(boost::forward<Arguments>( args)) , util::get< 1>(boost::forward<Arguments>( args)) , util::get< 2>(boost::forward<Arguments>( args)));
            return util::unused;
        }
    };
    
    template <
        typename Component, typename T0 , typename T1 , typename T2,
        void (Component::*F)(T0 , T1 , T2),
        typename Derived = detail::this_type>
    struct action3
      : base_action3<
            Component, T0 , T1 , T2, F,
            typename detail::action_type<
                action3<
                    Component, T0 , T1 , T2, F>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            action3, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Component, typename T0 , typename T1 , typename T2,
        void (Component::*F)(T0 , T1 , T2), typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2),
            F, Derived, boost::mpl::false_>
      : action3<
            Component,
            T0 , T1 , T2, F, Derived>
    {
        typedef action3<
            Component,
            T0 , T1 , T2, F, Derived
        > type;
    };
    template <typename Component, typename T0 , typename T1 , typename T2,
        void (Component::*F)(T0 , T1 , T2) const, typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2) const,
            F, Derived, boost::mpl::false_>
      : action3<
            Component const,
            T0 , T1 , T2, F, Derived>
    {
        typedef action3<
            Component const,
            T0 , T1 , T2, F, Derived
        > type;
    };
    
    template <
        typename Component, typename T0 , typename T1 , typename T2,
        void (Component::*F)(T0 , T1 , T2),
        typename Derived = detail::this_type>
    struct direct_action3
      : base_action3<
            Component, T0 , T1 , T2, F,
            typename detail::action_type<
                direct_action3<
                    Component, T0 , T1 , T2, F>,
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
    template <typename Component, typename T0 , typename T1 , typename T2,
        void (Component::*F)(T0 , T1 , T2), typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2),
            F, Derived, boost::mpl::true_>
      : direct_action3<
            Component,
            T0 , T1 , T2, F, Derived>
    {
        typedef direct_action3<
            Component,
            T0 , T1 , T2, F, Derived
        > type;
    };
    template <typename Component, typename T0 , typename T1 , typename T2,
        void (Component::*F)(T0 , T1 , T2) const, typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2) const,
            F, Derived, boost::mpl::true_>
      : direct_action3<
            Component const,
            T0 , T1 , T2, F, Derived>
    {
        typedef direct_action3<
            Component const,
            T0 , T1 , T2, F, Derived
        > type;
    };
    
    
    template <
        typename Component,
        typename T0 , typename T1 , typename T2,
        void (Component::*F)(T0 , T1 , T2),
        typename Derived>
    struct result_action3<Component, void,
            T0 , T1 , T2, F, Derived>
      : action3<Component,
            T0 , T1 , T2, F, Derived>
    {};
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3,
        Result (Component::*F)(T0 , T1 , T2 , T3), typename Derived>
    class base_result_action4
      : public action<
            Component, Result,
            hpx::util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type>,
            Derived>
    {
    public:
        typedef Result result_type;
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
                BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component>::call(lva)) << ")";
                    
                    
                    
                    
                    (get_lva<Component>::call(lva)->*F)(
                        boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing component action("
                            << detail::get_action_name<Derived>()
                            << ") lva(" << reinterpret_cast<void const*>
                                (get_lva<Component>::call(lva)) << "): " << e.what();
                        
                        hpx::report_error(boost::current_exception());
                    }
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
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            return Derived::decorate_action(
                HPX_STD_BIND(typename Derived::thread_function(), lva,
                    util::get< 0>(boost::forward<Arguments>( args)) , util::get< 1>(boost::forward<Arguments>( args)) , util::get< 2>(boost::forward<Arguments>( args)) , util::get< 3>(boost::forward<Arguments>( args))), lva);
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return Derived::decorate_action(
                    base_type::construct_continuation_thread_object_function(
                        cont, F, get_lva<Component>::call(lva),
                        boost::forward<Arguments>(args)), lva);
        }
        template <typename Arguments>
        BOOST_FORCEINLINE static Result
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "base_result_action" << 4
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            return (get_lva<Component>::call(lva)->*F)(
                util::get< 0>(boost::forward<Arguments>( args)) , util::get< 1>(boost::forward<Arguments>( args)) , util::get< 2>(boost::forward<Arguments>( args)) , util::get< 3>(boost::forward<Arguments>( args)));
        }
    };
    
    
    template <
        typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3,
        Result (Component::*F)(T0 , T1 , T2 , T3),
        typename Derived = detail::this_type>
    struct result_action4
      : base_result_action4<
            Component, Result,
            T0 , T1 , T2 , T3, F,
            typename detail::action_type<
                result_action4<
                    Component, Result, T0 , T1 , T2 , T3, F>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            result_action4, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3,
        Result (Component::*F)(T0 , T1 , T2 , T3), typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3),
            F, Derived, boost::mpl::false_>
      : result_action4<
            Component, Result,
            T0 , T1 , T2 , T3, F, Derived>
    {
        typedef result_action4<
            Component, Result,
            T0 , T1 , T2 , T3, F, Derived
        > type;
    };
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3,
        Result (Component::*F)(T0 , T1 , T2 , T3) const, typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3) const,
            F, Derived, boost::mpl::false_>
      : result_action4<
            Component const, Result,
            T0 , T1 , T2 , T3, F, Derived>
    {
        typedef result_action4<
            Component const, Result,
            T0 , T1 , T2 , T3, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3,
        Result (Component::*F)(T0 , T1 , T2 , T3),
        typename Derived = detail::this_type>
    struct direct_result_action4
      : base_result_action4<
            Component, Result,
            T0 , T1 , T2 , T3, F,
            typename detail::action_type<
                direct_result_action4<
                    Component, Result, T0 , T1 , T2 , T3, F>,
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
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3,
        Result (Component::*F)(T0 , T1 , T2 , T3), typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3),
            F, Derived, boost::mpl::true_>
      : direct_result_action4<
            Component, Result,
            T0 , T1 , T2 , T3, F, Derived>
    {
        typedef direct_result_action4<
            Component, Result,
            T0 , T1 , T2 , T3, F, Derived
        > type;
    };
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3,
        Result (Component::*F)(T0 , T1 , T2 , T3) const, typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3) const,
            F, Derived, boost::mpl::true_>
      : direct_result_action4<
            Component const, Result,
            T0 , T1 , T2 , T3, F, Derived>
    {
        typedef direct_result_action4<
            Component const, Result,
            T0 , T1 , T2 , T3, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3,
        void (Component::*F)(T0 , T1 , T2 , T3), typename Derived>
    class base_action4
      : public action<
            Component, util::unused_type,
            hpx::util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type>,
            Derived>
    {
    public:
        typedef util::unused_type result_type;
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
                BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component>::call(lva)) << ")";
                    
                    
                    
                    
                    (get_lva<Component>::call(lva)->*F)(
                        boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing component action("
                            << detail::get_action_name<Derived>()
                            << ") lva(" << reinterpret_cast<void const*>
                                (get_lva<Component>::call(lva)) << "): " << e.what();
                        
                        hpx::report_error(boost::current_exception());
                    }
                }
                
                
                
                util::force_error_on_lock();
                return threads::terminated;
            }
        };
    public:
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            
            
            return Derived::decorate_action(
                HPX_STD_BIND(typename Derived::thread_function(), lva,
                    util::get< 0>(boost::forward<Arguments>( args)) , util::get< 1>(boost::forward<Arguments>( args)) , util::get< 2>(boost::forward<Arguments>( args)) , util::get< 3>(boost::forward<Arguments>( args))), lva);
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return Derived::decorate_action(
                    base_type::construct_continuation_thread_object_function_void(
                        cont, F, get_lva<Component>::call(lva),
                        boost::forward<Arguments>(args)), lva);
        }
        template <typename Arguments>
        BOOST_FORCEINLINE static util::unused_type
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "base_action" << 4
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            (get_lva<Component>::call(lva)->*F)(
                util::get< 0>(boost::forward<Arguments>( args)) , util::get< 1>(boost::forward<Arguments>( args)) , util::get< 2>(boost::forward<Arguments>( args)) , util::get< 3>(boost::forward<Arguments>( args)));
            return util::unused;
        }
    };
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3,
        void (Component::*F)(T0 , T1 , T2 , T3),
        typename Derived = detail::this_type>
    struct action4
      : base_action4<
            Component, T0 , T1 , T2 , T3, F,
            typename detail::action_type<
                action4<
                    Component, T0 , T1 , T2 , T3, F>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            action4, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3,
        void (Component::*F)(T0 , T1 , T2 , T3), typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3),
            F, Derived, boost::mpl::false_>
      : action4<
            Component,
            T0 , T1 , T2 , T3, F, Derived>
    {
        typedef action4<
            Component,
            T0 , T1 , T2 , T3, F, Derived
        > type;
    };
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3,
        void (Component::*F)(T0 , T1 , T2 , T3) const, typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3) const,
            F, Derived, boost::mpl::false_>
      : action4<
            Component const,
            T0 , T1 , T2 , T3, F, Derived>
    {
        typedef action4<
            Component const,
            T0 , T1 , T2 , T3, F, Derived
        > type;
    };
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3,
        void (Component::*F)(T0 , T1 , T2 , T3),
        typename Derived = detail::this_type>
    struct direct_action4
      : base_action4<
            Component, T0 , T1 , T2 , T3, F,
            typename detail::action_type<
                direct_action4<
                    Component, T0 , T1 , T2 , T3, F>,
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
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3,
        void (Component::*F)(T0 , T1 , T2 , T3), typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3),
            F, Derived, boost::mpl::true_>
      : direct_action4<
            Component,
            T0 , T1 , T2 , T3, F, Derived>
    {
        typedef direct_action4<
            Component,
            T0 , T1 , T2 , T3, F, Derived
        > type;
    };
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3,
        void (Component::*F)(T0 , T1 , T2 , T3) const, typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3) const,
            F, Derived, boost::mpl::true_>
      : direct_action4<
            Component const,
            T0 , T1 , T2 , T3, F, Derived>
    {
        typedef direct_action4<
            Component const,
            T0 , T1 , T2 , T3, F, Derived
        > type;
    };
    
    
    template <
        typename Component,
        typename T0 , typename T1 , typename T2 , typename T3,
        void (Component::*F)(T0 , T1 , T2 , T3),
        typename Derived>
    struct result_action4<Component, void,
            T0 , T1 , T2 , T3, F, Derived>
      : action4<Component,
            T0 , T1 , T2 , T3, F, Derived>
    {};
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4), typename Derived>
    class base_result_action5
      : public action<
            Component, Result,
            hpx::util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type>,
            Derived>
    {
    public:
        typedef Result result_type;
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
                BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component>::call(lva)) << ")";
                    
                    
                    
                    
                    (get_lva<Component>::call(lva)->*F)(
                        boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing component action("
                            << detail::get_action_name<Derived>()
                            << ") lva(" << reinterpret_cast<void const*>
                                (get_lva<Component>::call(lva)) << "): " << e.what();
                        
                        hpx::report_error(boost::current_exception());
                    }
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
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            return Derived::decorate_action(
                HPX_STD_BIND(typename Derived::thread_function(), lva,
                    util::get< 0>(boost::forward<Arguments>( args)) , util::get< 1>(boost::forward<Arguments>( args)) , util::get< 2>(boost::forward<Arguments>( args)) , util::get< 3>(boost::forward<Arguments>( args)) , util::get< 4>(boost::forward<Arguments>( args))), lva);
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return Derived::decorate_action(
                    base_type::construct_continuation_thread_object_function(
                        cont, F, get_lva<Component>::call(lva),
                        boost::forward<Arguments>(args)), lva);
        }
        template <typename Arguments>
        BOOST_FORCEINLINE static Result
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "base_result_action" << 5
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            return (get_lva<Component>::call(lva)->*F)(
                util::get< 0>(boost::forward<Arguments>( args)) , util::get< 1>(boost::forward<Arguments>( args)) , util::get< 2>(boost::forward<Arguments>( args)) , util::get< 3>(boost::forward<Arguments>( args)) , util::get< 4>(boost::forward<Arguments>( args)));
        }
    };
    
    
    template <
        typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4),
        typename Derived = detail::this_type>
    struct result_action5
      : base_result_action5<
            Component, Result,
            T0 , T1 , T2 , T3 , T4, F,
            typename detail::action_type<
                result_action5<
                    Component, Result, T0 , T1 , T2 , T3 , T4, F>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            result_action5, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4), typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4),
            F, Derived, boost::mpl::false_>
      : result_action5<
            Component, Result,
            T0 , T1 , T2 , T3 , T4, F, Derived>
    {
        typedef result_action5<
            Component, Result,
            T0 , T1 , T2 , T3 , T4, F, Derived
        > type;
    };
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4) const, typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4) const,
            F, Derived, boost::mpl::false_>
      : result_action5<
            Component const, Result,
            T0 , T1 , T2 , T3 , T4, F, Derived>
    {
        typedef result_action5<
            Component const, Result,
            T0 , T1 , T2 , T3 , T4, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4),
        typename Derived = detail::this_type>
    struct direct_result_action5
      : base_result_action5<
            Component, Result,
            T0 , T1 , T2 , T3 , T4, F,
            typename detail::action_type<
                direct_result_action5<
                    Component, Result, T0 , T1 , T2 , T3 , T4, F>,
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
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4), typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4),
            F, Derived, boost::mpl::true_>
      : direct_result_action5<
            Component, Result,
            T0 , T1 , T2 , T3 , T4, F, Derived>
    {
        typedef direct_result_action5<
            Component, Result,
            T0 , T1 , T2 , T3 , T4, F, Derived
        > type;
    };
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4) const, typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4) const,
            F, Derived, boost::mpl::true_>
      : direct_result_action5<
            Component const, Result,
            T0 , T1 , T2 , T3 , T4, F, Derived>
    {
        typedef direct_result_action5<
            Component const, Result,
            T0 , T1 , T2 , T3 , T4, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4), typename Derived>
    class base_action5
      : public action<
            Component, util::unused_type,
            hpx::util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type>,
            Derived>
    {
    public:
        typedef util::unused_type result_type;
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
                BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component>::call(lva)) << ")";
                    
                    
                    
                    
                    (get_lva<Component>::call(lva)->*F)(
                        boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing component action("
                            << detail::get_action_name<Derived>()
                            << ") lva(" << reinterpret_cast<void const*>
                                (get_lva<Component>::call(lva)) << "): " << e.what();
                        
                        hpx::report_error(boost::current_exception());
                    }
                }
                
                
                
                util::force_error_on_lock();
                return threads::terminated;
            }
        };
    public:
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            
            
            return Derived::decorate_action(
                HPX_STD_BIND(typename Derived::thread_function(), lva,
                    util::get< 0>(boost::forward<Arguments>( args)) , util::get< 1>(boost::forward<Arguments>( args)) , util::get< 2>(boost::forward<Arguments>( args)) , util::get< 3>(boost::forward<Arguments>( args)) , util::get< 4>(boost::forward<Arguments>( args))), lva);
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return Derived::decorate_action(
                    base_type::construct_continuation_thread_object_function_void(
                        cont, F, get_lva<Component>::call(lva),
                        boost::forward<Arguments>(args)), lva);
        }
        template <typename Arguments>
        BOOST_FORCEINLINE static util::unused_type
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "base_action" << 5
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            (get_lva<Component>::call(lva)->*F)(
                util::get< 0>(boost::forward<Arguments>( args)) , util::get< 1>(boost::forward<Arguments>( args)) , util::get< 2>(boost::forward<Arguments>( args)) , util::get< 3>(boost::forward<Arguments>( args)) , util::get< 4>(boost::forward<Arguments>( args)));
            return util::unused;
        }
    };
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4),
        typename Derived = detail::this_type>
    struct action5
      : base_action5<
            Component, T0 , T1 , T2 , T3 , T4, F,
            typename detail::action_type<
                action5<
                    Component, T0 , T1 , T2 , T3 , T4, F>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            action5, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4), typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4),
            F, Derived, boost::mpl::false_>
      : action5<
            Component,
            T0 , T1 , T2 , T3 , T4, F, Derived>
    {
        typedef action5<
            Component,
            T0 , T1 , T2 , T3 , T4, F, Derived
        > type;
    };
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4) const, typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4) const,
            F, Derived, boost::mpl::false_>
      : action5<
            Component const,
            T0 , T1 , T2 , T3 , T4, F, Derived>
    {
        typedef action5<
            Component const,
            T0 , T1 , T2 , T3 , T4, F, Derived
        > type;
    };
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4),
        typename Derived = detail::this_type>
    struct direct_action5
      : base_action5<
            Component, T0 , T1 , T2 , T3 , T4, F,
            typename detail::action_type<
                direct_action5<
                    Component, T0 , T1 , T2 , T3 , T4, F>,
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
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4), typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4),
            F, Derived, boost::mpl::true_>
      : direct_action5<
            Component,
            T0 , T1 , T2 , T3 , T4, F, Derived>
    {
        typedef direct_action5<
            Component,
            T0 , T1 , T2 , T3 , T4, F, Derived
        > type;
    };
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4) const, typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4) const,
            F, Derived, boost::mpl::true_>
      : direct_action5<
            Component const,
            T0 , T1 , T2 , T3 , T4, F, Derived>
    {
        typedef direct_action5<
            Component const,
            T0 , T1 , T2 , T3 , T4, F, Derived
        > type;
    };
    
    
    template <
        typename Component,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4),
        typename Derived>
    struct result_action5<Component, void,
            T0 , T1 , T2 , T3 , T4, F, Derived>
      : action5<Component,
            T0 , T1 , T2 , T3 , T4, F, Derived>
    {};
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5), typename Derived>
    class base_result_action6
      : public action<
            Component, Result,
            hpx::util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type>,
            Derived>
    {
    public:
        typedef Result result_type;
        typedef hpx::util::tuple<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type> arguments_type;
        typedef action<Component, result_type, arguments_type, Derived>
            base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component>::call(lva)) << ")";
                    
                    
                    
                    
                    (get_lva<Component>::call(lva)->*F)(
                        boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing component action("
                            << detail::get_action_name<Derived>()
                            << ") lva(" << reinterpret_cast<void const*>
                                (get_lva<Component>::call(lva)) << "): " << e.what();
                        
                        hpx::report_error(boost::current_exception());
                    }
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
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            return Derived::decorate_action(
                HPX_STD_BIND(typename Derived::thread_function(), lva,
                    util::get< 0>(boost::forward<Arguments>( args)) , util::get< 1>(boost::forward<Arguments>( args)) , util::get< 2>(boost::forward<Arguments>( args)) , util::get< 3>(boost::forward<Arguments>( args)) , util::get< 4>(boost::forward<Arguments>( args)) , util::get< 5>(boost::forward<Arguments>( args))), lva);
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return Derived::decorate_action(
                    base_type::construct_continuation_thread_object_function(
                        cont, F, get_lva<Component>::call(lva),
                        boost::forward<Arguments>(args)), lva);
        }
        template <typename Arguments>
        BOOST_FORCEINLINE static Result
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "base_result_action" << 6
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            return (get_lva<Component>::call(lva)->*F)(
                util::get< 0>(boost::forward<Arguments>( args)) , util::get< 1>(boost::forward<Arguments>( args)) , util::get< 2>(boost::forward<Arguments>( args)) , util::get< 3>(boost::forward<Arguments>( args)) , util::get< 4>(boost::forward<Arguments>( args)) , util::get< 5>(boost::forward<Arguments>( args)));
        }
    };
    
    
    template <
        typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5),
        typename Derived = detail::this_type>
    struct result_action6
      : base_result_action6<
            Component, Result,
            T0 , T1 , T2 , T3 , T4 , T5, F,
            typename detail::action_type<
                result_action6<
                    Component, Result, T0 , T1 , T2 , T3 , T4 , T5, F>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            result_action6, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5), typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4 , T5),
            F, Derived, boost::mpl::false_>
      : result_action6<
            Component, Result,
            T0 , T1 , T2 , T3 , T4 , T5, F, Derived>
    {
        typedef result_action6<
            Component, Result,
            T0 , T1 , T2 , T3 , T4 , T5, F, Derived
        > type;
    };
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5) const, typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4 , T5) const,
            F, Derived, boost::mpl::false_>
      : result_action6<
            Component const, Result,
            T0 , T1 , T2 , T3 , T4 , T5, F, Derived>
    {
        typedef result_action6<
            Component const, Result,
            T0 , T1 , T2 , T3 , T4 , T5, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5),
        typename Derived = detail::this_type>
    struct direct_result_action6
      : base_result_action6<
            Component, Result,
            T0 , T1 , T2 , T3 , T4 , T5, F,
            typename detail::action_type<
                direct_result_action6<
                    Component, Result, T0 , T1 , T2 , T3 , T4 , T5, F>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_result_action6, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5), typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4 , T5),
            F, Derived, boost::mpl::true_>
      : direct_result_action6<
            Component, Result,
            T0 , T1 , T2 , T3 , T4 , T5, F, Derived>
    {
        typedef direct_result_action6<
            Component, Result,
            T0 , T1 , T2 , T3 , T4 , T5, F, Derived
        > type;
    };
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5) const, typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4 , T5) const,
            F, Derived, boost::mpl::true_>
      : direct_result_action6<
            Component const, Result,
            T0 , T1 , T2 , T3 , T4 , T5, F, Derived>
    {
        typedef direct_result_action6<
            Component const, Result,
            T0 , T1 , T2 , T3 , T4 , T5, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5), typename Derived>
    class base_action6
      : public action<
            Component, util::unused_type,
            hpx::util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type>,
            Derived>
    {
    public:
        typedef util::unused_type result_type;
        typedef hpx::util::tuple<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type> arguments_type;
        typedef action<Component, result_type, arguments_type, Derived> 
            base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component>::call(lva)) << ")";
                    
                    
                    
                    
                    (get_lva<Component>::call(lva)->*F)(
                        boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing component action("
                            << detail::get_action_name<Derived>()
                            << ") lva(" << reinterpret_cast<void const*>
                                (get_lva<Component>::call(lva)) << "): " << e.what();
                        
                        hpx::report_error(boost::current_exception());
                    }
                }
                
                
                
                util::force_error_on_lock();
                return threads::terminated;
            }
        };
    public:
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            
            
            return Derived::decorate_action(
                HPX_STD_BIND(typename Derived::thread_function(), lva,
                    util::get< 0>(boost::forward<Arguments>( args)) , util::get< 1>(boost::forward<Arguments>( args)) , util::get< 2>(boost::forward<Arguments>( args)) , util::get< 3>(boost::forward<Arguments>( args)) , util::get< 4>(boost::forward<Arguments>( args)) , util::get< 5>(boost::forward<Arguments>( args))), lva);
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return Derived::decorate_action(
                    base_type::construct_continuation_thread_object_function_void(
                        cont, F, get_lva<Component>::call(lva),
                        boost::forward<Arguments>(args)), lva);
        }
        template <typename Arguments>
        BOOST_FORCEINLINE static util::unused_type
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "base_action" << 6
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            (get_lva<Component>::call(lva)->*F)(
                util::get< 0>(boost::forward<Arguments>( args)) , util::get< 1>(boost::forward<Arguments>( args)) , util::get< 2>(boost::forward<Arguments>( args)) , util::get< 3>(boost::forward<Arguments>( args)) , util::get< 4>(boost::forward<Arguments>( args)) , util::get< 5>(boost::forward<Arguments>( args)));
            return util::unused;
        }
    };
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5),
        typename Derived = detail::this_type>
    struct action6
      : base_action6<
            Component, T0 , T1 , T2 , T3 , T4 , T5, F,
            typename detail::action_type<
                action6<
                    Component, T0 , T1 , T2 , T3 , T4 , T5, F>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            action6, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5), typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5),
            F, Derived, boost::mpl::false_>
      : action6<
            Component,
            T0 , T1 , T2 , T3 , T4 , T5, F, Derived>
    {
        typedef action6<
            Component,
            T0 , T1 , T2 , T3 , T4 , T5, F, Derived
        > type;
    };
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5) const, typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5) const,
            F, Derived, boost::mpl::false_>
      : action6<
            Component const,
            T0 , T1 , T2 , T3 , T4 , T5, F, Derived>
    {
        typedef action6<
            Component const,
            T0 , T1 , T2 , T3 , T4 , T5, F, Derived
        > type;
    };
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5),
        typename Derived = detail::this_type>
    struct direct_action6
      : base_action6<
            Component, T0 , T1 , T2 , T3 , T4 , T5, F,
            typename detail::action_type<
                direct_action6<
                    Component, T0 , T1 , T2 , T3 , T4 , T5, F>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_action6, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5), typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5),
            F, Derived, boost::mpl::true_>
      : direct_action6<
            Component,
            T0 , T1 , T2 , T3 , T4 , T5, F, Derived>
    {
        typedef direct_action6<
            Component,
            T0 , T1 , T2 , T3 , T4 , T5, F, Derived
        > type;
    };
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5) const, typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5) const,
            F, Derived, boost::mpl::true_>
      : direct_action6<
            Component const,
            T0 , T1 , T2 , T3 , T4 , T5, F, Derived>
    {
        typedef direct_action6<
            Component const,
            T0 , T1 , T2 , T3 , T4 , T5, F, Derived
        > type;
    };
    
    
    template <
        typename Component,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5),
        typename Derived>
    struct result_action6<Component, void,
            T0 , T1 , T2 , T3 , T4 , T5, F, Derived>
      : action6<Component,
            T0 , T1 , T2 , T3 , T4 , T5, F, Derived>
    {};
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6), typename Derived>
    class base_result_action7
      : public action<
            Component, Result,
            hpx::util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type>,
            Derived>
    {
    public:
        typedef Result result_type;
        typedef hpx::util::tuple<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type> arguments_type;
        typedef action<Component, result_type, arguments_type, Derived>
            base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component>::call(lva)) << ")";
                    
                    
                    
                    
                    (get_lva<Component>::call(lva)->*F)(
                        boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing component action("
                            << detail::get_action_name<Derived>()
                            << ") lva(" << reinterpret_cast<void const*>
                                (get_lva<Component>::call(lva)) << "): " << e.what();
                        
                        hpx::report_error(boost::current_exception());
                    }
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
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            return Derived::decorate_action(
                HPX_STD_BIND(typename Derived::thread_function(), lva,
                    util::get< 0>(boost::forward<Arguments>( args)) , util::get< 1>(boost::forward<Arguments>( args)) , util::get< 2>(boost::forward<Arguments>( args)) , util::get< 3>(boost::forward<Arguments>( args)) , util::get< 4>(boost::forward<Arguments>( args)) , util::get< 5>(boost::forward<Arguments>( args)) , util::get< 6>(boost::forward<Arguments>( args))), lva);
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return Derived::decorate_action(
                    base_type::construct_continuation_thread_object_function(
                        cont, F, get_lva<Component>::call(lva),
                        boost::forward<Arguments>(args)), lva);
        }
        template <typename Arguments>
        BOOST_FORCEINLINE static Result
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "base_result_action" << 7
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            return (get_lva<Component>::call(lva)->*F)(
                util::get< 0>(boost::forward<Arguments>( args)) , util::get< 1>(boost::forward<Arguments>( args)) , util::get< 2>(boost::forward<Arguments>( args)) , util::get< 3>(boost::forward<Arguments>( args)) , util::get< 4>(boost::forward<Arguments>( args)) , util::get< 5>(boost::forward<Arguments>( args)) , util::get< 6>(boost::forward<Arguments>( args)));
        }
    };
    
    
    template <
        typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6),
        typename Derived = detail::this_type>
    struct result_action7
      : base_result_action7<
            Component, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6, F,
            typename detail::action_type<
                result_action7<
                    Component, Result, T0 , T1 , T2 , T3 , T4 , T5 , T6, F>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            result_action7, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6), typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6),
            F, Derived, boost::mpl::false_>
      : result_action7<
            Component, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6, F, Derived>
    {
        typedef result_action7<
            Component, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6, F, Derived
        > type;
    };
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6) const, typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6) const,
            F, Derived, boost::mpl::false_>
      : result_action7<
            Component const, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6, F, Derived>
    {
        typedef result_action7<
            Component const, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6),
        typename Derived = detail::this_type>
    struct direct_result_action7
      : base_result_action7<
            Component, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6, F,
            typename detail::action_type<
                direct_result_action7<
                    Component, Result, T0 , T1 , T2 , T3 , T4 , T5 , T6, F>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_result_action7, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6), typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6),
            F, Derived, boost::mpl::true_>
      : direct_result_action7<
            Component, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6, F, Derived>
    {
        typedef direct_result_action7<
            Component, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6, F, Derived
        > type;
    };
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6) const, typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6) const,
            F, Derived, boost::mpl::true_>
      : direct_result_action7<
            Component const, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6, F, Derived>
    {
        typedef direct_result_action7<
            Component const, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6), typename Derived>
    class base_action7
      : public action<
            Component, util::unused_type,
            hpx::util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type>,
            Derived>
    {
    public:
        typedef util::unused_type result_type;
        typedef hpx::util::tuple<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type> arguments_type;
        typedef action<Component, result_type, arguments_type, Derived> 
            base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component>::call(lva)) << ")";
                    
                    
                    
                    
                    (get_lva<Component>::call(lva)->*F)(
                        boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing component action("
                            << detail::get_action_name<Derived>()
                            << ") lva(" << reinterpret_cast<void const*>
                                (get_lva<Component>::call(lva)) << "): " << e.what();
                        
                        hpx::report_error(boost::current_exception());
                    }
                }
                
                
                
                util::force_error_on_lock();
                return threads::terminated;
            }
        };
    public:
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            
            
            return Derived::decorate_action(
                HPX_STD_BIND(typename Derived::thread_function(), lva,
                    util::get< 0>(boost::forward<Arguments>( args)) , util::get< 1>(boost::forward<Arguments>( args)) , util::get< 2>(boost::forward<Arguments>( args)) , util::get< 3>(boost::forward<Arguments>( args)) , util::get< 4>(boost::forward<Arguments>( args)) , util::get< 5>(boost::forward<Arguments>( args)) , util::get< 6>(boost::forward<Arguments>( args))), lva);
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return Derived::decorate_action(
                    base_type::construct_continuation_thread_object_function_void(
                        cont, F, get_lva<Component>::call(lva),
                        boost::forward<Arguments>(args)), lva);
        }
        template <typename Arguments>
        BOOST_FORCEINLINE static util::unused_type
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "base_action" << 7
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            (get_lva<Component>::call(lva)->*F)(
                util::get< 0>(boost::forward<Arguments>( args)) , util::get< 1>(boost::forward<Arguments>( args)) , util::get< 2>(boost::forward<Arguments>( args)) , util::get< 3>(boost::forward<Arguments>( args)) , util::get< 4>(boost::forward<Arguments>( args)) , util::get< 5>(boost::forward<Arguments>( args)) , util::get< 6>(boost::forward<Arguments>( args)));
            return util::unused;
        }
    };
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6),
        typename Derived = detail::this_type>
    struct action7
      : base_action7<
            Component, T0 , T1 , T2 , T3 , T4 , T5 , T6, F,
            typename detail::action_type<
                action7<
                    Component, T0 , T1 , T2 , T3 , T4 , T5 , T6, F>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            action7, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6), typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6),
            F, Derived, boost::mpl::false_>
      : action7<
            Component,
            T0 , T1 , T2 , T3 , T4 , T5 , T6, F, Derived>
    {
        typedef action7<
            Component,
            T0 , T1 , T2 , T3 , T4 , T5 , T6, F, Derived
        > type;
    };
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6) const, typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6) const,
            F, Derived, boost::mpl::false_>
      : action7<
            Component const,
            T0 , T1 , T2 , T3 , T4 , T5 , T6, F, Derived>
    {
        typedef action7<
            Component const,
            T0 , T1 , T2 , T3 , T4 , T5 , T6, F, Derived
        > type;
    };
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6),
        typename Derived = detail::this_type>
    struct direct_action7
      : base_action7<
            Component, T0 , T1 , T2 , T3 , T4 , T5 , T6, F,
            typename detail::action_type<
                direct_action7<
                    Component, T0 , T1 , T2 , T3 , T4 , T5 , T6, F>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_action7, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6), typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6),
            F, Derived, boost::mpl::true_>
      : direct_action7<
            Component,
            T0 , T1 , T2 , T3 , T4 , T5 , T6, F, Derived>
    {
        typedef direct_action7<
            Component,
            T0 , T1 , T2 , T3 , T4 , T5 , T6, F, Derived
        > type;
    };
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6) const, typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6) const,
            F, Derived, boost::mpl::true_>
      : direct_action7<
            Component const,
            T0 , T1 , T2 , T3 , T4 , T5 , T6, F, Derived>
    {
        typedef direct_action7<
            Component const,
            T0 , T1 , T2 , T3 , T4 , T5 , T6, F, Derived
        > type;
    };
    
    
    template <
        typename Component,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6),
        typename Derived>
    struct result_action7<Component, void,
            T0 , T1 , T2 , T3 , T4 , T5 , T6, F, Derived>
      : action7<Component,
            T0 , T1 , T2 , T3 , T4 , T5 , T6, F, Derived>
    {};
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7), typename Derived>
    class base_result_action8
      : public action<
            Component, Result,
            hpx::util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type>,
            Derived>
    {
    public:
        typedef Result result_type;
        typedef hpx::util::tuple<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type> arguments_type;
        typedef action<Component, result_type, arguments_type, Derived>
            base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component>::call(lva)) << ")";
                    
                    
                    
                    
                    (get_lva<Component>::call(lva)->*F)(
                        boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing component action("
                            << detail::get_action_name<Derived>()
                            << ") lva(" << reinterpret_cast<void const*>
                                (get_lva<Component>::call(lva)) << "): " << e.what();
                        
                        hpx::report_error(boost::current_exception());
                    }
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
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            return Derived::decorate_action(
                HPX_STD_BIND(typename Derived::thread_function(), lva,
                    util::get< 0>(boost::forward<Arguments>( args)) , util::get< 1>(boost::forward<Arguments>( args)) , util::get< 2>(boost::forward<Arguments>( args)) , util::get< 3>(boost::forward<Arguments>( args)) , util::get< 4>(boost::forward<Arguments>( args)) , util::get< 5>(boost::forward<Arguments>( args)) , util::get< 6>(boost::forward<Arguments>( args)) , util::get< 7>(boost::forward<Arguments>( args))), lva);
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return Derived::decorate_action(
                    base_type::construct_continuation_thread_object_function(
                        cont, F, get_lva<Component>::call(lva),
                        boost::forward<Arguments>(args)), lva);
        }
        template <typename Arguments>
        BOOST_FORCEINLINE static Result
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "base_result_action" << 8
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            return (get_lva<Component>::call(lva)->*F)(
                util::get< 0>(boost::forward<Arguments>( args)) , util::get< 1>(boost::forward<Arguments>( args)) , util::get< 2>(boost::forward<Arguments>( args)) , util::get< 3>(boost::forward<Arguments>( args)) , util::get< 4>(boost::forward<Arguments>( args)) , util::get< 5>(boost::forward<Arguments>( args)) , util::get< 6>(boost::forward<Arguments>( args)) , util::get< 7>(boost::forward<Arguments>( args)));
        }
    };
    
    
    template <
        typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7),
        typename Derived = detail::this_type>
    struct result_action8
      : base_result_action8<
            Component, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F,
            typename detail::action_type<
                result_action8<
                    Component, Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            result_action8, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7), typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7),
            F, Derived, boost::mpl::false_>
      : result_action8<
            Component, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F, Derived>
    {
        typedef result_action8<
            Component, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F, Derived
        > type;
    };
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7) const, typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7) const,
            F, Derived, boost::mpl::false_>
      : result_action8<
            Component const, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F, Derived>
    {
        typedef result_action8<
            Component const, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7),
        typename Derived = detail::this_type>
    struct direct_result_action8
      : base_result_action8<
            Component, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F,
            typename detail::action_type<
                direct_result_action8<
                    Component, Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_result_action8, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7), typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7),
            F, Derived, boost::mpl::true_>
      : direct_result_action8<
            Component, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F, Derived>
    {
        typedef direct_result_action8<
            Component, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F, Derived
        > type;
    };
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7) const, typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7) const,
            F, Derived, boost::mpl::true_>
      : direct_result_action8<
            Component const, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F, Derived>
    {
        typedef direct_result_action8<
            Component const, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7), typename Derived>
    class base_action8
      : public action<
            Component, util::unused_type,
            hpx::util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type>,
            Derived>
    {
    public:
        typedef util::unused_type result_type;
        typedef hpx::util::tuple<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type> arguments_type;
        typedef action<Component, result_type, arguments_type, Derived> 
            base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component>::call(lva)) << ")";
                    
                    
                    
                    
                    (get_lva<Component>::call(lva)->*F)(
                        boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing component action("
                            << detail::get_action_name<Derived>()
                            << ") lva(" << reinterpret_cast<void const*>
                                (get_lva<Component>::call(lva)) << "): " << e.what();
                        
                        hpx::report_error(boost::current_exception());
                    }
                }
                
                
                
                util::force_error_on_lock();
                return threads::terminated;
            }
        };
    public:
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            
            
            return Derived::decorate_action(
                HPX_STD_BIND(typename Derived::thread_function(), lva,
                    util::get< 0>(boost::forward<Arguments>( args)) , util::get< 1>(boost::forward<Arguments>( args)) , util::get< 2>(boost::forward<Arguments>( args)) , util::get< 3>(boost::forward<Arguments>( args)) , util::get< 4>(boost::forward<Arguments>( args)) , util::get< 5>(boost::forward<Arguments>( args)) , util::get< 6>(boost::forward<Arguments>( args)) , util::get< 7>(boost::forward<Arguments>( args))), lva);
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return Derived::decorate_action(
                    base_type::construct_continuation_thread_object_function_void(
                        cont, F, get_lva<Component>::call(lva),
                        boost::forward<Arguments>(args)), lva);
        }
        template <typename Arguments>
        BOOST_FORCEINLINE static util::unused_type
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "base_action" << 8
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            (get_lva<Component>::call(lva)->*F)(
                util::get< 0>(boost::forward<Arguments>( args)) , util::get< 1>(boost::forward<Arguments>( args)) , util::get< 2>(boost::forward<Arguments>( args)) , util::get< 3>(boost::forward<Arguments>( args)) , util::get< 4>(boost::forward<Arguments>( args)) , util::get< 5>(boost::forward<Arguments>( args)) , util::get< 6>(boost::forward<Arguments>( args)) , util::get< 7>(boost::forward<Arguments>( args)));
            return util::unused;
        }
    };
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7),
        typename Derived = detail::this_type>
    struct action8
      : base_action8<
            Component, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F,
            typename detail::action_type<
                action8<
                    Component, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            action8, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7), typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7),
            F, Derived, boost::mpl::false_>
      : action8<
            Component,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F, Derived>
    {
        typedef action8<
            Component,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F, Derived
        > type;
    };
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7) const, typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7) const,
            F, Derived, boost::mpl::false_>
      : action8<
            Component const,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F, Derived>
    {
        typedef action8<
            Component const,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F, Derived
        > type;
    };
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7),
        typename Derived = detail::this_type>
    struct direct_action8
      : base_action8<
            Component, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F,
            typename detail::action_type<
                direct_action8<
                    Component, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_action8, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7), typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7),
            F, Derived, boost::mpl::true_>
      : direct_action8<
            Component,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F, Derived>
    {
        typedef direct_action8<
            Component,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F, Derived
        > type;
    };
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7) const, typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7) const,
            F, Derived, boost::mpl::true_>
      : direct_action8<
            Component const,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F, Derived>
    {
        typedef direct_action8<
            Component const,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F, Derived
        > type;
    };
    
    
    template <
        typename Component,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7),
        typename Derived>
    struct result_action8<Component, void,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F, Derived>
      : action8<Component,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F, Derived>
    {};
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8), typename Derived>
    class base_result_action9
      : public action<
            Component, Result,
            hpx::util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type>,
            Derived>
    {
    public:
        typedef Result result_type;
        typedef hpx::util::tuple<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type> arguments_type;
        typedef action<Component, result_type, arguments_type, Derived>
            base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component>::call(lva)) << ")";
                    
                    
                    
                    
                    (get_lva<Component>::call(lva)->*F)(
                        boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing component action("
                            << detail::get_action_name<Derived>()
                            << ") lva(" << reinterpret_cast<void const*>
                                (get_lva<Component>::call(lva)) << "): " << e.what();
                        
                        hpx::report_error(boost::current_exception());
                    }
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
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            return Derived::decorate_action(
                HPX_STD_BIND(typename Derived::thread_function(), lva,
                    util::get< 0>(boost::forward<Arguments>( args)) , util::get< 1>(boost::forward<Arguments>( args)) , util::get< 2>(boost::forward<Arguments>( args)) , util::get< 3>(boost::forward<Arguments>( args)) , util::get< 4>(boost::forward<Arguments>( args)) , util::get< 5>(boost::forward<Arguments>( args)) , util::get< 6>(boost::forward<Arguments>( args)) , util::get< 7>(boost::forward<Arguments>( args)) , util::get< 8>(boost::forward<Arguments>( args))), lva);
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return Derived::decorate_action(
                    base_type::construct_continuation_thread_object_function(
                        cont, F, get_lva<Component>::call(lva),
                        boost::forward<Arguments>(args)), lva);
        }
        template <typename Arguments>
        BOOST_FORCEINLINE static Result
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "base_result_action" << 9
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            return (get_lva<Component>::call(lva)->*F)(
                util::get< 0>(boost::forward<Arguments>( args)) , util::get< 1>(boost::forward<Arguments>( args)) , util::get< 2>(boost::forward<Arguments>( args)) , util::get< 3>(boost::forward<Arguments>( args)) , util::get< 4>(boost::forward<Arguments>( args)) , util::get< 5>(boost::forward<Arguments>( args)) , util::get< 6>(boost::forward<Arguments>( args)) , util::get< 7>(boost::forward<Arguments>( args)) , util::get< 8>(boost::forward<Arguments>( args)));
        }
    };
    
    
    template <
        typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8),
        typename Derived = detail::this_type>
    struct result_action9
      : base_result_action9<
            Component, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F,
            typename detail::action_type<
                result_action9<
                    Component, Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            result_action9, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8), typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8),
            F, Derived, boost::mpl::false_>
      : result_action9<
            Component, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F, Derived>
    {
        typedef result_action9<
            Component, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F, Derived
        > type;
    };
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8) const, typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8) const,
            F, Derived, boost::mpl::false_>
      : result_action9<
            Component const, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F, Derived>
    {
        typedef result_action9<
            Component const, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8),
        typename Derived = detail::this_type>
    struct direct_result_action9
      : base_result_action9<
            Component, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F,
            typename detail::action_type<
                direct_result_action9<
                    Component, Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_result_action9, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8), typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8),
            F, Derived, boost::mpl::true_>
      : direct_result_action9<
            Component, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F, Derived>
    {
        typedef direct_result_action9<
            Component, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F, Derived
        > type;
    };
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8) const, typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8) const,
            F, Derived, boost::mpl::true_>
      : direct_result_action9<
            Component const, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F, Derived>
    {
        typedef direct_result_action9<
            Component const, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8), typename Derived>
    class base_action9
      : public action<
            Component, util::unused_type,
            hpx::util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type>,
            Derived>
    {
    public:
        typedef util::unused_type result_type;
        typedef hpx::util::tuple<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type> arguments_type;
        typedef action<Component, result_type, arguments_type, Derived> 
            base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component>::call(lva)) << ")";
                    
                    
                    
                    
                    (get_lva<Component>::call(lva)->*F)(
                        boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing component action("
                            << detail::get_action_name<Derived>()
                            << ") lva(" << reinterpret_cast<void const*>
                                (get_lva<Component>::call(lva)) << "): " << e.what();
                        
                        hpx::report_error(boost::current_exception());
                    }
                }
                
                
                
                util::force_error_on_lock();
                return threads::terminated;
            }
        };
    public:
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            
            
            return Derived::decorate_action(
                HPX_STD_BIND(typename Derived::thread_function(), lva,
                    util::get< 0>(boost::forward<Arguments>( args)) , util::get< 1>(boost::forward<Arguments>( args)) , util::get< 2>(boost::forward<Arguments>( args)) , util::get< 3>(boost::forward<Arguments>( args)) , util::get< 4>(boost::forward<Arguments>( args)) , util::get< 5>(boost::forward<Arguments>( args)) , util::get< 6>(boost::forward<Arguments>( args)) , util::get< 7>(boost::forward<Arguments>( args)) , util::get< 8>(boost::forward<Arguments>( args))), lva);
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return Derived::decorate_action(
                    base_type::construct_continuation_thread_object_function_void(
                        cont, F, get_lva<Component>::call(lva),
                        boost::forward<Arguments>(args)), lva);
        }
        template <typename Arguments>
        BOOST_FORCEINLINE static util::unused_type
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "base_action" << 9
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            (get_lva<Component>::call(lva)->*F)(
                util::get< 0>(boost::forward<Arguments>( args)) , util::get< 1>(boost::forward<Arguments>( args)) , util::get< 2>(boost::forward<Arguments>( args)) , util::get< 3>(boost::forward<Arguments>( args)) , util::get< 4>(boost::forward<Arguments>( args)) , util::get< 5>(boost::forward<Arguments>( args)) , util::get< 6>(boost::forward<Arguments>( args)) , util::get< 7>(boost::forward<Arguments>( args)) , util::get< 8>(boost::forward<Arguments>( args)));
            return util::unused;
        }
    };
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8),
        typename Derived = detail::this_type>
    struct action9
      : base_action9<
            Component, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F,
            typename detail::action_type<
                action9<
                    Component, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            action9, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8), typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8),
            F, Derived, boost::mpl::false_>
      : action9<
            Component,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F, Derived>
    {
        typedef action9<
            Component,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F, Derived
        > type;
    };
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8) const, typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8) const,
            F, Derived, boost::mpl::false_>
      : action9<
            Component const,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F, Derived>
    {
        typedef action9<
            Component const,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F, Derived
        > type;
    };
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8),
        typename Derived = detail::this_type>
    struct direct_action9
      : base_action9<
            Component, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F,
            typename detail::action_type<
                direct_action9<
                    Component, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_action9, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8), typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8),
            F, Derived, boost::mpl::true_>
      : direct_action9<
            Component,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F, Derived>
    {
        typedef direct_action9<
            Component,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F, Derived
        > type;
    };
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8) const, typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8) const,
            F, Derived, boost::mpl::true_>
      : direct_action9<
            Component const,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F, Derived>
    {
        typedef direct_action9<
            Component const,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F, Derived
        > type;
    };
    
    
    template <
        typename Component,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8),
        typename Derived>
    struct result_action9<Component, void,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F, Derived>
      : action9<Component,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F, Derived>
    {};
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9), typename Derived>
    class base_result_action10
      : public action<
            Component, Result,
            hpx::util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type>,
            Derived>
    {
    public:
        typedef Result result_type;
        typedef hpx::util::tuple<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type> arguments_type;
        typedef action<Component, result_type, arguments_type, Derived>
            base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component>::call(lva)) << ")";
                    
                    
                    
                    
                    (get_lva<Component>::call(lva)->*F)(
                        boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8) , boost::move(arg9));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing component action("
                            << detail::get_action_name<Derived>()
                            << ") lva(" << reinterpret_cast<void const*>
                                (get_lva<Component>::call(lva)) << "): " << e.what();
                        
                        hpx::report_error(boost::current_exception());
                    }
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
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            return Derived::decorate_action(
                HPX_STD_BIND(typename Derived::thread_function(), lva,
                    util::get< 0>(boost::forward<Arguments>( args)) , util::get< 1>(boost::forward<Arguments>( args)) , util::get< 2>(boost::forward<Arguments>( args)) , util::get< 3>(boost::forward<Arguments>( args)) , util::get< 4>(boost::forward<Arguments>( args)) , util::get< 5>(boost::forward<Arguments>( args)) , util::get< 6>(boost::forward<Arguments>( args)) , util::get< 7>(boost::forward<Arguments>( args)) , util::get< 8>(boost::forward<Arguments>( args)) , util::get< 9>(boost::forward<Arguments>( args))), lva);
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return Derived::decorate_action(
                    base_type::construct_continuation_thread_object_function(
                        cont, F, get_lva<Component>::call(lva),
                        boost::forward<Arguments>(args)), lva);
        }
        template <typename Arguments>
        BOOST_FORCEINLINE static Result
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "base_result_action" << 10
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            return (get_lva<Component>::call(lva)->*F)(
                util::get< 0>(boost::forward<Arguments>( args)) , util::get< 1>(boost::forward<Arguments>( args)) , util::get< 2>(boost::forward<Arguments>( args)) , util::get< 3>(boost::forward<Arguments>( args)) , util::get< 4>(boost::forward<Arguments>( args)) , util::get< 5>(boost::forward<Arguments>( args)) , util::get< 6>(boost::forward<Arguments>( args)) , util::get< 7>(boost::forward<Arguments>( args)) , util::get< 8>(boost::forward<Arguments>( args)) , util::get< 9>(boost::forward<Arguments>( args)));
        }
    };
    
    
    template <
        typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9),
        typename Derived = detail::this_type>
    struct result_action10
      : base_result_action10<
            Component, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F,
            typename detail::action_type<
                result_action10<
                    Component, Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            result_action10, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9), typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9),
            F, Derived, boost::mpl::false_>
      : result_action10<
            Component, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F, Derived>
    {
        typedef result_action10<
            Component, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F, Derived
        > type;
    };
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9) const, typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9) const,
            F, Derived, boost::mpl::false_>
      : result_action10<
            Component const, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F, Derived>
    {
        typedef result_action10<
            Component const, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9),
        typename Derived = detail::this_type>
    struct direct_result_action10
      : base_result_action10<
            Component, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F,
            typename detail::action_type<
                direct_result_action10<
                    Component, Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_result_action10, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9), typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9),
            F, Derived, boost::mpl::true_>
      : direct_result_action10<
            Component, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F, Derived>
    {
        typedef direct_result_action10<
            Component, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F, Derived
        > type;
    };
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9) const, typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9) const,
            F, Derived, boost::mpl::true_>
      : direct_result_action10<
            Component const, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F, Derived>
    {
        typedef direct_result_action10<
            Component const, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9), typename Derived>
    class base_action10
      : public action<
            Component, util::unused_type,
            hpx::util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type>,
            Derived>
    {
    public:
        typedef util::unused_type result_type;
        typedef hpx::util::tuple<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type> arguments_type;
        typedef action<Component, result_type, arguments_type, Derived> 
            base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component>::call(lva)) << ")";
                    
                    
                    
                    
                    (get_lva<Component>::call(lva)->*F)(
                        boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8) , boost::move(arg9));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing component action("
                            << detail::get_action_name<Derived>()
                            << ") lva(" << reinterpret_cast<void const*>
                                (get_lva<Component>::call(lva)) << "): " << e.what();
                        
                        hpx::report_error(boost::current_exception());
                    }
                }
                
                
                
                util::force_error_on_lock();
                return threads::terminated;
            }
        };
    public:
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            
            
            return Derived::decorate_action(
                HPX_STD_BIND(typename Derived::thread_function(), lva,
                    util::get< 0>(boost::forward<Arguments>( args)) , util::get< 1>(boost::forward<Arguments>( args)) , util::get< 2>(boost::forward<Arguments>( args)) , util::get< 3>(boost::forward<Arguments>( args)) , util::get< 4>(boost::forward<Arguments>( args)) , util::get< 5>(boost::forward<Arguments>( args)) , util::get< 6>(boost::forward<Arguments>( args)) , util::get< 7>(boost::forward<Arguments>( args)) , util::get< 8>(boost::forward<Arguments>( args)) , util::get< 9>(boost::forward<Arguments>( args))), lva);
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return Derived::decorate_action(
                    base_type::construct_continuation_thread_object_function_void(
                        cont, F, get_lva<Component>::call(lva),
                        boost::forward<Arguments>(args)), lva);
        }
        template <typename Arguments>
        BOOST_FORCEINLINE static util::unused_type
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "base_action" << 10
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            (get_lva<Component>::call(lva)->*F)(
                util::get< 0>(boost::forward<Arguments>( args)) , util::get< 1>(boost::forward<Arguments>( args)) , util::get< 2>(boost::forward<Arguments>( args)) , util::get< 3>(boost::forward<Arguments>( args)) , util::get< 4>(boost::forward<Arguments>( args)) , util::get< 5>(boost::forward<Arguments>( args)) , util::get< 6>(boost::forward<Arguments>( args)) , util::get< 7>(boost::forward<Arguments>( args)) , util::get< 8>(boost::forward<Arguments>( args)) , util::get< 9>(boost::forward<Arguments>( args)));
            return util::unused;
        }
    };
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9),
        typename Derived = detail::this_type>
    struct action10
      : base_action10<
            Component, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F,
            typename detail::action_type<
                action10<
                    Component, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            action10, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9), typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9),
            F, Derived, boost::mpl::false_>
      : action10<
            Component,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F, Derived>
    {
        typedef action10<
            Component,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F, Derived
        > type;
    };
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9) const, typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9) const,
            F, Derived, boost::mpl::false_>
      : action10<
            Component const,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F, Derived>
    {
        typedef action10<
            Component const,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F, Derived
        > type;
    };
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9),
        typename Derived = detail::this_type>
    struct direct_action10
      : base_action10<
            Component, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F,
            typename detail::action_type<
                direct_action10<
                    Component, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_action10, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9), typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9),
            F, Derived, boost::mpl::true_>
      : direct_action10<
            Component,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F, Derived>
    {
        typedef direct_action10<
            Component,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F, Derived
        > type;
    };
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9) const, typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9) const,
            F, Derived, boost::mpl::true_>
      : direct_action10<
            Component const,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F, Derived>
    {
        typedef direct_action10<
            Component const,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F, Derived
        > type;
    };
    
    
    template <
        typename Component,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9),
        typename Derived>
    struct result_action10<Component, void,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F, Derived>
      : action10<Component,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F, Derived>
    {};
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10), typename Derived>
    class base_result_action11
      : public action<
            Component, Result,
            hpx::util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type>,
            Derived>
    {
    public:
        typedef Result result_type;
        typedef hpx::util::tuple<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type> arguments_type;
        typedef action<Component, result_type, arguments_type, Derived>
            base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component>::call(lva)) << ")";
                    
                    
                    
                    
                    (get_lva<Component>::call(lva)->*F)(
                        boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8) , boost::move(arg9) , boost::move(arg10));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing component action("
                            << detail::get_action_name<Derived>()
                            << ") lva(" << reinterpret_cast<void const*>
                                (get_lva<Component>::call(lva)) << "): " << e.what();
                        
                        hpx::report_error(boost::current_exception());
                    }
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
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            return Derived::decorate_action(
                HPX_STD_BIND(typename Derived::thread_function(), lva,
                    util::get< 0>(boost::forward<Arguments>( args)) , util::get< 1>(boost::forward<Arguments>( args)) , util::get< 2>(boost::forward<Arguments>( args)) , util::get< 3>(boost::forward<Arguments>( args)) , util::get< 4>(boost::forward<Arguments>( args)) , util::get< 5>(boost::forward<Arguments>( args)) , util::get< 6>(boost::forward<Arguments>( args)) , util::get< 7>(boost::forward<Arguments>( args)) , util::get< 8>(boost::forward<Arguments>( args)) , util::get< 9>(boost::forward<Arguments>( args)) , util::get< 10>(boost::forward<Arguments>( args))), lva);
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return Derived::decorate_action(
                    base_type::construct_continuation_thread_object_function(
                        cont, F, get_lva<Component>::call(lva),
                        boost::forward<Arguments>(args)), lva);
        }
        template <typename Arguments>
        BOOST_FORCEINLINE static Result
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "base_result_action" << 11
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            return (get_lva<Component>::call(lva)->*F)(
                util::get< 0>(boost::forward<Arguments>( args)) , util::get< 1>(boost::forward<Arguments>( args)) , util::get< 2>(boost::forward<Arguments>( args)) , util::get< 3>(boost::forward<Arguments>( args)) , util::get< 4>(boost::forward<Arguments>( args)) , util::get< 5>(boost::forward<Arguments>( args)) , util::get< 6>(boost::forward<Arguments>( args)) , util::get< 7>(boost::forward<Arguments>( args)) , util::get< 8>(boost::forward<Arguments>( args)) , util::get< 9>(boost::forward<Arguments>( args)) , util::get< 10>(boost::forward<Arguments>( args)));
        }
    };
    
    
    template <
        typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10),
        typename Derived = detail::this_type>
    struct result_action11
      : base_result_action11<
            Component, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10, F,
            typename detail::action_type<
                result_action11<
                    Component, Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10, F>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            result_action11, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10), typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10),
            F, Derived, boost::mpl::false_>
      : result_action11<
            Component, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10, F, Derived>
    {
        typedef result_action11<
            Component, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10, F, Derived
        > type;
    };
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10) const, typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10) const,
            F, Derived, boost::mpl::false_>
      : result_action11<
            Component const, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10, F, Derived>
    {
        typedef result_action11<
            Component const, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10),
        typename Derived = detail::this_type>
    struct direct_result_action11
      : base_result_action11<
            Component, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10, F,
            typename detail::action_type<
                direct_result_action11<
                    Component, Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10, F>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_result_action11, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10), typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10),
            F, Derived, boost::mpl::true_>
      : direct_result_action11<
            Component, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10, F, Derived>
    {
        typedef direct_result_action11<
            Component, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10, F, Derived
        > type;
    };
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10) const, typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10) const,
            F, Derived, boost::mpl::true_>
      : direct_result_action11<
            Component const, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10, F, Derived>
    {
        typedef direct_result_action11<
            Component const, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10), typename Derived>
    class base_action11
      : public action<
            Component, util::unused_type,
            hpx::util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type>,
            Derived>
    {
    public:
        typedef util::unused_type result_type;
        typedef hpx::util::tuple<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type> arguments_type;
        typedef action<Component, result_type, arguments_type, Derived> 
            base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component>::call(lva)) << ")";
                    
                    
                    
                    
                    (get_lva<Component>::call(lva)->*F)(
                        boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8) , boost::move(arg9) , boost::move(arg10));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing component action("
                            << detail::get_action_name<Derived>()
                            << ") lva(" << reinterpret_cast<void const*>
                                (get_lva<Component>::call(lva)) << "): " << e.what();
                        
                        hpx::report_error(boost::current_exception());
                    }
                }
                
                
                
                util::force_error_on_lock();
                return threads::terminated;
            }
        };
    public:
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            
            
            return Derived::decorate_action(
                HPX_STD_BIND(typename Derived::thread_function(), lva,
                    util::get< 0>(boost::forward<Arguments>( args)) , util::get< 1>(boost::forward<Arguments>( args)) , util::get< 2>(boost::forward<Arguments>( args)) , util::get< 3>(boost::forward<Arguments>( args)) , util::get< 4>(boost::forward<Arguments>( args)) , util::get< 5>(boost::forward<Arguments>( args)) , util::get< 6>(boost::forward<Arguments>( args)) , util::get< 7>(boost::forward<Arguments>( args)) , util::get< 8>(boost::forward<Arguments>( args)) , util::get< 9>(boost::forward<Arguments>( args)) , util::get< 10>(boost::forward<Arguments>( args))), lva);
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return Derived::decorate_action(
                    base_type::construct_continuation_thread_object_function_void(
                        cont, F, get_lva<Component>::call(lva),
                        boost::forward<Arguments>(args)), lva);
        }
        template <typename Arguments>
        BOOST_FORCEINLINE static util::unused_type
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "base_action" << 11
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            (get_lva<Component>::call(lva)->*F)(
                util::get< 0>(boost::forward<Arguments>( args)) , util::get< 1>(boost::forward<Arguments>( args)) , util::get< 2>(boost::forward<Arguments>( args)) , util::get< 3>(boost::forward<Arguments>( args)) , util::get< 4>(boost::forward<Arguments>( args)) , util::get< 5>(boost::forward<Arguments>( args)) , util::get< 6>(boost::forward<Arguments>( args)) , util::get< 7>(boost::forward<Arguments>( args)) , util::get< 8>(boost::forward<Arguments>( args)) , util::get< 9>(boost::forward<Arguments>( args)) , util::get< 10>(boost::forward<Arguments>( args)));
            return util::unused;
        }
    };
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10),
        typename Derived = detail::this_type>
    struct action11
      : base_action11<
            Component, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10, F,
            typename detail::action_type<
                action11<
                    Component, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10, F>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            action11, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10), typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10),
            F, Derived, boost::mpl::false_>
      : action11<
            Component,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10, F, Derived>
    {
        typedef action11<
            Component,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10, F, Derived
        > type;
    };
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10) const, typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10) const,
            F, Derived, boost::mpl::false_>
      : action11<
            Component const,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10, F, Derived>
    {
        typedef action11<
            Component const,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10, F, Derived
        > type;
    };
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10),
        typename Derived = detail::this_type>
    struct direct_action11
      : base_action11<
            Component, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10, F,
            typename detail::action_type<
                direct_action11<
                    Component, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10, F>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_action11, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10), typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10),
            F, Derived, boost::mpl::true_>
      : direct_action11<
            Component,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10, F, Derived>
    {
        typedef direct_action11<
            Component,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10, F, Derived
        > type;
    };
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10) const, typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10) const,
            F, Derived, boost::mpl::true_>
      : direct_action11<
            Component const,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10, F, Derived>
    {
        typedef direct_action11<
            Component const,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10, F, Derived
        > type;
    };
    
    
    template <
        typename Component,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10),
        typename Derived>
    struct result_action11<Component, void,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10, F, Derived>
      : action11<Component,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10, F, Derived>
    {};
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11), typename Derived>
    class base_result_action12
      : public action<
            Component, Result,
            hpx::util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type>,
            Derived>
    {
    public:
        typedef Result result_type;
        typedef hpx::util::tuple<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type> arguments_type;
        typedef action<Component, result_type, arguments_type, Derived>
            base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component>::call(lva)) << ")";
                    
                    
                    
                    
                    (get_lva<Component>::call(lva)->*F)(
                        boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8) , boost::move(arg9) , boost::move(arg10) , boost::move(arg11));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing component action("
                            << detail::get_action_name<Derived>()
                            << ") lva(" << reinterpret_cast<void const*>
                                (get_lva<Component>::call(lva)) << "): " << e.what();
                        
                        hpx::report_error(boost::current_exception());
                    }
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
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            return Derived::decorate_action(
                HPX_STD_BIND(typename Derived::thread_function(), lva,
                    util::get< 0>(boost::forward<Arguments>( args)) , util::get< 1>(boost::forward<Arguments>( args)) , util::get< 2>(boost::forward<Arguments>( args)) , util::get< 3>(boost::forward<Arguments>( args)) , util::get< 4>(boost::forward<Arguments>( args)) , util::get< 5>(boost::forward<Arguments>( args)) , util::get< 6>(boost::forward<Arguments>( args)) , util::get< 7>(boost::forward<Arguments>( args)) , util::get< 8>(boost::forward<Arguments>( args)) , util::get< 9>(boost::forward<Arguments>( args)) , util::get< 10>(boost::forward<Arguments>( args)) , util::get< 11>(boost::forward<Arguments>( args))), lva);
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return Derived::decorate_action(
                    base_type::construct_continuation_thread_object_function(
                        cont, F, get_lva<Component>::call(lva),
                        boost::forward<Arguments>(args)), lva);
        }
        template <typename Arguments>
        BOOST_FORCEINLINE static Result
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "base_result_action" << 12
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            return (get_lva<Component>::call(lva)->*F)(
                util::get< 0>(boost::forward<Arguments>( args)) , util::get< 1>(boost::forward<Arguments>( args)) , util::get< 2>(boost::forward<Arguments>( args)) , util::get< 3>(boost::forward<Arguments>( args)) , util::get< 4>(boost::forward<Arguments>( args)) , util::get< 5>(boost::forward<Arguments>( args)) , util::get< 6>(boost::forward<Arguments>( args)) , util::get< 7>(boost::forward<Arguments>( args)) , util::get< 8>(boost::forward<Arguments>( args)) , util::get< 9>(boost::forward<Arguments>( args)) , util::get< 10>(boost::forward<Arguments>( args)) , util::get< 11>(boost::forward<Arguments>( args)));
        }
    };
    
    
    template <
        typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11),
        typename Derived = detail::this_type>
    struct result_action12
      : base_result_action12<
            Component, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11, F,
            typename detail::action_type<
                result_action12<
                    Component, Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11, F>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            result_action12, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11), typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11),
            F, Derived, boost::mpl::false_>
      : result_action12<
            Component, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11, F, Derived>
    {
        typedef result_action12<
            Component, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11, F, Derived
        > type;
    };
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11) const, typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11) const,
            F, Derived, boost::mpl::false_>
      : result_action12<
            Component const, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11, F, Derived>
    {
        typedef result_action12<
            Component const, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11),
        typename Derived = detail::this_type>
    struct direct_result_action12
      : base_result_action12<
            Component, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11, F,
            typename detail::action_type<
                direct_result_action12<
                    Component, Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11, F>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_result_action12, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11), typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11),
            F, Derived, boost::mpl::true_>
      : direct_result_action12<
            Component, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11, F, Derived>
    {
        typedef direct_result_action12<
            Component, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11, F, Derived
        > type;
    };
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11) const, typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11) const,
            F, Derived, boost::mpl::true_>
      : direct_result_action12<
            Component const, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11, F, Derived>
    {
        typedef direct_result_action12<
            Component const, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11), typename Derived>
    class base_action12
      : public action<
            Component, util::unused_type,
            hpx::util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type>,
            Derived>
    {
    public:
        typedef util::unused_type result_type;
        typedef hpx::util::tuple<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type> arguments_type;
        typedef action<Component, result_type, arguments_type, Derived> 
            base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component>::call(lva)) << ")";
                    
                    
                    
                    
                    (get_lva<Component>::call(lva)->*F)(
                        boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8) , boost::move(arg9) , boost::move(arg10) , boost::move(arg11));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing component action("
                            << detail::get_action_name<Derived>()
                            << ") lva(" << reinterpret_cast<void const*>
                                (get_lva<Component>::call(lva)) << "): " << e.what();
                        
                        hpx::report_error(boost::current_exception());
                    }
                }
                
                
                
                util::force_error_on_lock();
                return threads::terminated;
            }
        };
    public:
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            
            
            return Derived::decorate_action(
                HPX_STD_BIND(typename Derived::thread_function(), lva,
                    util::get< 0>(boost::forward<Arguments>( args)) , util::get< 1>(boost::forward<Arguments>( args)) , util::get< 2>(boost::forward<Arguments>( args)) , util::get< 3>(boost::forward<Arguments>( args)) , util::get< 4>(boost::forward<Arguments>( args)) , util::get< 5>(boost::forward<Arguments>( args)) , util::get< 6>(boost::forward<Arguments>( args)) , util::get< 7>(boost::forward<Arguments>( args)) , util::get< 8>(boost::forward<Arguments>( args)) , util::get< 9>(boost::forward<Arguments>( args)) , util::get< 10>(boost::forward<Arguments>( args)) , util::get< 11>(boost::forward<Arguments>( args))), lva);
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return Derived::decorate_action(
                    base_type::construct_continuation_thread_object_function_void(
                        cont, F, get_lva<Component>::call(lva),
                        boost::forward<Arguments>(args)), lva);
        }
        template <typename Arguments>
        BOOST_FORCEINLINE static util::unused_type
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "base_action" << 12
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            (get_lva<Component>::call(lva)->*F)(
                util::get< 0>(boost::forward<Arguments>( args)) , util::get< 1>(boost::forward<Arguments>( args)) , util::get< 2>(boost::forward<Arguments>( args)) , util::get< 3>(boost::forward<Arguments>( args)) , util::get< 4>(boost::forward<Arguments>( args)) , util::get< 5>(boost::forward<Arguments>( args)) , util::get< 6>(boost::forward<Arguments>( args)) , util::get< 7>(boost::forward<Arguments>( args)) , util::get< 8>(boost::forward<Arguments>( args)) , util::get< 9>(boost::forward<Arguments>( args)) , util::get< 10>(boost::forward<Arguments>( args)) , util::get< 11>(boost::forward<Arguments>( args)));
            return util::unused;
        }
    };
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11),
        typename Derived = detail::this_type>
    struct action12
      : base_action12<
            Component, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11, F,
            typename detail::action_type<
                action12<
                    Component, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11, F>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            action12, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11), typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11),
            F, Derived, boost::mpl::false_>
      : action12<
            Component,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11, F, Derived>
    {
        typedef action12<
            Component,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11, F, Derived
        > type;
    };
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11) const, typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11) const,
            F, Derived, boost::mpl::false_>
      : action12<
            Component const,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11, F, Derived>
    {
        typedef action12<
            Component const,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11, F, Derived
        > type;
    };
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11),
        typename Derived = detail::this_type>
    struct direct_action12
      : base_action12<
            Component, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11, F,
            typename detail::action_type<
                direct_action12<
                    Component, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11, F>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_action12, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11), typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11),
            F, Derived, boost::mpl::true_>
      : direct_action12<
            Component,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11, F, Derived>
    {
        typedef direct_action12<
            Component,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11, F, Derived
        > type;
    };
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11) const, typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11) const,
            F, Derived, boost::mpl::true_>
      : direct_action12<
            Component const,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11, F, Derived>
    {
        typedef direct_action12<
            Component const,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11, F, Derived
        > type;
    };
    
    
    template <
        typename Component,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11),
        typename Derived>
    struct result_action12<Component, void,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11, F, Derived>
      : action12<Component,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11, F, Derived>
    {};
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12), typename Derived>
    class base_result_action13
      : public action<
            Component, Result,
            hpx::util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type>,
            Derived>
    {
    public:
        typedef Result result_type;
        typedef hpx::util::tuple<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type> arguments_type;
        typedef action<Component, result_type, arguments_type, Derived>
            base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component>::call(lva)) << ")";
                    
                    
                    
                    
                    (get_lva<Component>::call(lva)->*F)(
                        boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8) , boost::move(arg9) , boost::move(arg10) , boost::move(arg11) , boost::move(arg12));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing component action("
                            << detail::get_action_name<Derived>()
                            << ") lva(" << reinterpret_cast<void const*>
                                (get_lva<Component>::call(lva)) << "): " << e.what();
                        
                        hpx::report_error(boost::current_exception());
                    }
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
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            return Derived::decorate_action(
                HPX_STD_BIND(typename Derived::thread_function(), lva,
                    util::get< 0>(boost::forward<Arguments>( args)) , util::get< 1>(boost::forward<Arguments>( args)) , util::get< 2>(boost::forward<Arguments>( args)) , util::get< 3>(boost::forward<Arguments>( args)) , util::get< 4>(boost::forward<Arguments>( args)) , util::get< 5>(boost::forward<Arguments>( args)) , util::get< 6>(boost::forward<Arguments>( args)) , util::get< 7>(boost::forward<Arguments>( args)) , util::get< 8>(boost::forward<Arguments>( args)) , util::get< 9>(boost::forward<Arguments>( args)) , util::get< 10>(boost::forward<Arguments>( args)) , util::get< 11>(boost::forward<Arguments>( args)) , util::get< 12>(boost::forward<Arguments>( args))), lva);
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return Derived::decorate_action(
                    base_type::construct_continuation_thread_object_function(
                        cont, F, get_lva<Component>::call(lva),
                        boost::forward<Arguments>(args)), lva);
        }
        template <typename Arguments>
        BOOST_FORCEINLINE static Result
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "base_result_action" << 13
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            return (get_lva<Component>::call(lva)->*F)(
                util::get< 0>(boost::forward<Arguments>( args)) , util::get< 1>(boost::forward<Arguments>( args)) , util::get< 2>(boost::forward<Arguments>( args)) , util::get< 3>(boost::forward<Arguments>( args)) , util::get< 4>(boost::forward<Arguments>( args)) , util::get< 5>(boost::forward<Arguments>( args)) , util::get< 6>(boost::forward<Arguments>( args)) , util::get< 7>(boost::forward<Arguments>( args)) , util::get< 8>(boost::forward<Arguments>( args)) , util::get< 9>(boost::forward<Arguments>( args)) , util::get< 10>(boost::forward<Arguments>( args)) , util::get< 11>(boost::forward<Arguments>( args)) , util::get< 12>(boost::forward<Arguments>( args)));
        }
    };
    
    
    template <
        typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12),
        typename Derived = detail::this_type>
    struct result_action13
      : base_result_action13<
            Component, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12, F,
            typename detail::action_type<
                result_action13<
                    Component, Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12, F>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            result_action13, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12), typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12),
            F, Derived, boost::mpl::false_>
      : result_action13<
            Component, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12, F, Derived>
    {
        typedef result_action13<
            Component, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12, F, Derived
        > type;
    };
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12) const, typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12) const,
            F, Derived, boost::mpl::false_>
      : result_action13<
            Component const, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12, F, Derived>
    {
        typedef result_action13<
            Component const, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12),
        typename Derived = detail::this_type>
    struct direct_result_action13
      : base_result_action13<
            Component, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12, F,
            typename detail::action_type<
                direct_result_action13<
                    Component, Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12, F>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_result_action13, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12), typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12),
            F, Derived, boost::mpl::true_>
      : direct_result_action13<
            Component, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12, F, Derived>
    {
        typedef direct_result_action13<
            Component, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12, F, Derived
        > type;
    };
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12) const, typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12) const,
            F, Derived, boost::mpl::true_>
      : direct_result_action13<
            Component const, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12, F, Derived>
    {
        typedef direct_result_action13<
            Component const, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12), typename Derived>
    class base_action13
      : public action<
            Component, util::unused_type,
            hpx::util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type>,
            Derived>
    {
    public:
        typedef util::unused_type result_type;
        typedef hpx::util::tuple<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type> arguments_type;
        typedef action<Component, result_type, arguments_type, Derived> 
            base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component>::call(lva)) << ")";
                    
                    
                    
                    
                    (get_lva<Component>::call(lva)->*F)(
                        boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8) , boost::move(arg9) , boost::move(arg10) , boost::move(arg11) , boost::move(arg12));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing component action("
                            << detail::get_action_name<Derived>()
                            << ") lva(" << reinterpret_cast<void const*>
                                (get_lva<Component>::call(lva)) << "): " << e.what();
                        
                        hpx::report_error(boost::current_exception());
                    }
                }
                
                
                
                util::force_error_on_lock();
                return threads::terminated;
            }
        };
    public:
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            
            
            return Derived::decorate_action(
                HPX_STD_BIND(typename Derived::thread_function(), lva,
                    util::get< 0>(boost::forward<Arguments>( args)) , util::get< 1>(boost::forward<Arguments>( args)) , util::get< 2>(boost::forward<Arguments>( args)) , util::get< 3>(boost::forward<Arguments>( args)) , util::get< 4>(boost::forward<Arguments>( args)) , util::get< 5>(boost::forward<Arguments>( args)) , util::get< 6>(boost::forward<Arguments>( args)) , util::get< 7>(boost::forward<Arguments>( args)) , util::get< 8>(boost::forward<Arguments>( args)) , util::get< 9>(boost::forward<Arguments>( args)) , util::get< 10>(boost::forward<Arguments>( args)) , util::get< 11>(boost::forward<Arguments>( args)) , util::get< 12>(boost::forward<Arguments>( args))), lva);
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return Derived::decorate_action(
                    base_type::construct_continuation_thread_object_function_void(
                        cont, F, get_lva<Component>::call(lva),
                        boost::forward<Arguments>(args)), lva);
        }
        template <typename Arguments>
        BOOST_FORCEINLINE static util::unused_type
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "base_action" << 13
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            (get_lva<Component>::call(lva)->*F)(
                util::get< 0>(boost::forward<Arguments>( args)) , util::get< 1>(boost::forward<Arguments>( args)) , util::get< 2>(boost::forward<Arguments>( args)) , util::get< 3>(boost::forward<Arguments>( args)) , util::get< 4>(boost::forward<Arguments>( args)) , util::get< 5>(boost::forward<Arguments>( args)) , util::get< 6>(boost::forward<Arguments>( args)) , util::get< 7>(boost::forward<Arguments>( args)) , util::get< 8>(boost::forward<Arguments>( args)) , util::get< 9>(boost::forward<Arguments>( args)) , util::get< 10>(boost::forward<Arguments>( args)) , util::get< 11>(boost::forward<Arguments>( args)) , util::get< 12>(boost::forward<Arguments>( args)));
            return util::unused;
        }
    };
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12),
        typename Derived = detail::this_type>
    struct action13
      : base_action13<
            Component, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12, F,
            typename detail::action_type<
                action13<
                    Component, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12, F>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            action13, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12), typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12),
            F, Derived, boost::mpl::false_>
      : action13<
            Component,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12, F, Derived>
    {
        typedef action13<
            Component,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12, F, Derived
        > type;
    };
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12) const, typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12) const,
            F, Derived, boost::mpl::false_>
      : action13<
            Component const,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12, F, Derived>
    {
        typedef action13<
            Component const,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12, F, Derived
        > type;
    };
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12),
        typename Derived = detail::this_type>
    struct direct_action13
      : base_action13<
            Component, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12, F,
            typename detail::action_type<
                direct_action13<
                    Component, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12, F>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_action13, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12), typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12),
            F, Derived, boost::mpl::true_>
      : direct_action13<
            Component,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12, F, Derived>
    {
        typedef direct_action13<
            Component,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12, F, Derived
        > type;
    };
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12) const, typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12) const,
            F, Derived, boost::mpl::true_>
      : direct_action13<
            Component const,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12, F, Derived>
    {
        typedef direct_action13<
            Component const,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12, F, Derived
        > type;
    };
    
    
    template <
        typename Component,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12),
        typename Derived>
    struct result_action13<Component, void,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12, F, Derived>
      : action13<Component,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12, F, Derived>
    {};
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13), typename Derived>
    class base_result_action14
      : public action<
            Component, Result,
            hpx::util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type>,
            Derived>
    {
    public:
        typedef Result result_type;
        typedef hpx::util::tuple<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type> arguments_type;
        typedef action<Component, result_type, arguments_type, Derived>
            base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component>::call(lva)) << ")";
                    
                    
                    
                    
                    (get_lva<Component>::call(lva)->*F)(
                        boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8) , boost::move(arg9) , boost::move(arg10) , boost::move(arg11) , boost::move(arg12) , boost::move(arg13));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing component action("
                            << detail::get_action_name<Derived>()
                            << ") lva(" << reinterpret_cast<void const*>
                                (get_lva<Component>::call(lva)) << "): " << e.what();
                        
                        hpx::report_error(boost::current_exception());
                    }
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
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            return Derived::decorate_action(
                HPX_STD_BIND(typename Derived::thread_function(), lva,
                    util::get< 0>(boost::forward<Arguments>( args)) , util::get< 1>(boost::forward<Arguments>( args)) , util::get< 2>(boost::forward<Arguments>( args)) , util::get< 3>(boost::forward<Arguments>( args)) , util::get< 4>(boost::forward<Arguments>( args)) , util::get< 5>(boost::forward<Arguments>( args)) , util::get< 6>(boost::forward<Arguments>( args)) , util::get< 7>(boost::forward<Arguments>( args)) , util::get< 8>(boost::forward<Arguments>( args)) , util::get< 9>(boost::forward<Arguments>( args)) , util::get< 10>(boost::forward<Arguments>( args)) , util::get< 11>(boost::forward<Arguments>( args)) , util::get< 12>(boost::forward<Arguments>( args)) , util::get< 13>(boost::forward<Arguments>( args))), lva);
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return Derived::decorate_action(
                    base_type::construct_continuation_thread_object_function(
                        cont, F, get_lva<Component>::call(lva),
                        boost::forward<Arguments>(args)), lva);
        }
        template <typename Arguments>
        BOOST_FORCEINLINE static Result
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "base_result_action" << 14
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            return (get_lva<Component>::call(lva)->*F)(
                util::get< 0>(boost::forward<Arguments>( args)) , util::get< 1>(boost::forward<Arguments>( args)) , util::get< 2>(boost::forward<Arguments>( args)) , util::get< 3>(boost::forward<Arguments>( args)) , util::get< 4>(boost::forward<Arguments>( args)) , util::get< 5>(boost::forward<Arguments>( args)) , util::get< 6>(boost::forward<Arguments>( args)) , util::get< 7>(boost::forward<Arguments>( args)) , util::get< 8>(boost::forward<Arguments>( args)) , util::get< 9>(boost::forward<Arguments>( args)) , util::get< 10>(boost::forward<Arguments>( args)) , util::get< 11>(boost::forward<Arguments>( args)) , util::get< 12>(boost::forward<Arguments>( args)) , util::get< 13>(boost::forward<Arguments>( args)));
        }
    };
    
    
    template <
        typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13),
        typename Derived = detail::this_type>
    struct result_action14
      : base_result_action14<
            Component, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13, F,
            typename detail::action_type<
                result_action14<
                    Component, Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13, F>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            result_action14, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13), typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13),
            F, Derived, boost::mpl::false_>
      : result_action14<
            Component, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13, F, Derived>
    {
        typedef result_action14<
            Component, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13, F, Derived
        > type;
    };
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13) const, typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13) const,
            F, Derived, boost::mpl::false_>
      : result_action14<
            Component const, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13, F, Derived>
    {
        typedef result_action14<
            Component const, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13),
        typename Derived = detail::this_type>
    struct direct_result_action14
      : base_result_action14<
            Component, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13, F,
            typename detail::action_type<
                direct_result_action14<
                    Component, Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13, F>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_result_action14, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13), typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13),
            F, Derived, boost::mpl::true_>
      : direct_result_action14<
            Component, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13, F, Derived>
    {
        typedef direct_result_action14<
            Component, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13, F, Derived
        > type;
    };
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13) const, typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13) const,
            F, Derived, boost::mpl::true_>
      : direct_result_action14<
            Component const, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13, F, Derived>
    {
        typedef direct_result_action14<
            Component const, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13), typename Derived>
    class base_action14
      : public action<
            Component, util::unused_type,
            hpx::util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type>,
            Derived>
    {
    public:
        typedef util::unused_type result_type;
        typedef hpx::util::tuple<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type> arguments_type;
        typedef action<Component, result_type, arguments_type, Derived> 
            base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component>::call(lva)) << ")";
                    
                    
                    
                    
                    (get_lva<Component>::call(lva)->*F)(
                        boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8) , boost::move(arg9) , boost::move(arg10) , boost::move(arg11) , boost::move(arg12) , boost::move(arg13));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing component action("
                            << detail::get_action_name<Derived>()
                            << ") lva(" << reinterpret_cast<void const*>
                                (get_lva<Component>::call(lva)) << "): " << e.what();
                        
                        hpx::report_error(boost::current_exception());
                    }
                }
                
                
                
                util::force_error_on_lock();
                return threads::terminated;
            }
        };
    public:
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            
            
            return Derived::decorate_action(
                HPX_STD_BIND(typename Derived::thread_function(), lva,
                    util::get< 0>(boost::forward<Arguments>( args)) , util::get< 1>(boost::forward<Arguments>( args)) , util::get< 2>(boost::forward<Arguments>( args)) , util::get< 3>(boost::forward<Arguments>( args)) , util::get< 4>(boost::forward<Arguments>( args)) , util::get< 5>(boost::forward<Arguments>( args)) , util::get< 6>(boost::forward<Arguments>( args)) , util::get< 7>(boost::forward<Arguments>( args)) , util::get< 8>(boost::forward<Arguments>( args)) , util::get< 9>(boost::forward<Arguments>( args)) , util::get< 10>(boost::forward<Arguments>( args)) , util::get< 11>(boost::forward<Arguments>( args)) , util::get< 12>(boost::forward<Arguments>( args)) , util::get< 13>(boost::forward<Arguments>( args))), lva);
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return Derived::decorate_action(
                    base_type::construct_continuation_thread_object_function_void(
                        cont, F, get_lva<Component>::call(lva),
                        boost::forward<Arguments>(args)), lva);
        }
        template <typename Arguments>
        BOOST_FORCEINLINE static util::unused_type
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "base_action" << 14
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            (get_lva<Component>::call(lva)->*F)(
                util::get< 0>(boost::forward<Arguments>( args)) , util::get< 1>(boost::forward<Arguments>( args)) , util::get< 2>(boost::forward<Arguments>( args)) , util::get< 3>(boost::forward<Arguments>( args)) , util::get< 4>(boost::forward<Arguments>( args)) , util::get< 5>(boost::forward<Arguments>( args)) , util::get< 6>(boost::forward<Arguments>( args)) , util::get< 7>(boost::forward<Arguments>( args)) , util::get< 8>(boost::forward<Arguments>( args)) , util::get< 9>(boost::forward<Arguments>( args)) , util::get< 10>(boost::forward<Arguments>( args)) , util::get< 11>(boost::forward<Arguments>( args)) , util::get< 12>(boost::forward<Arguments>( args)) , util::get< 13>(boost::forward<Arguments>( args)));
            return util::unused;
        }
    };
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13),
        typename Derived = detail::this_type>
    struct action14
      : base_action14<
            Component, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13, F,
            typename detail::action_type<
                action14<
                    Component, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13, F>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            action14, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13), typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13),
            F, Derived, boost::mpl::false_>
      : action14<
            Component,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13, F, Derived>
    {
        typedef action14<
            Component,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13, F, Derived
        > type;
    };
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13) const, typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13) const,
            F, Derived, boost::mpl::false_>
      : action14<
            Component const,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13, F, Derived>
    {
        typedef action14<
            Component const,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13, F, Derived
        > type;
    };
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13),
        typename Derived = detail::this_type>
    struct direct_action14
      : base_action14<
            Component, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13, F,
            typename detail::action_type<
                direct_action14<
                    Component, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13, F>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_action14, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13), typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13),
            F, Derived, boost::mpl::true_>
      : direct_action14<
            Component,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13, F, Derived>
    {
        typedef direct_action14<
            Component,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13, F, Derived
        > type;
    };
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13) const, typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13) const,
            F, Derived, boost::mpl::true_>
      : direct_action14<
            Component const,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13, F, Derived>
    {
        typedef direct_action14<
            Component const,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13, F, Derived
        > type;
    };
    
    
    template <
        typename Component,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13),
        typename Derived>
    struct result_action14<Component, void,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13, F, Derived>
      : action14<Component,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13, F, Derived>
    {};
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14), typename Derived>
    class base_result_action15
      : public action<
            Component, Result,
            hpx::util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type>,
            Derived>
    {
    public:
        typedef Result result_type;
        typedef hpx::util::tuple<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type> arguments_type;
        typedef action<Component, result_type, arguments_type, Derived>
            base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component>::call(lva)) << ")";
                    
                    
                    
                    
                    (get_lva<Component>::call(lva)->*F)(
                        boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8) , boost::move(arg9) , boost::move(arg10) , boost::move(arg11) , boost::move(arg12) , boost::move(arg13) , boost::move(arg14));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing component action("
                            << detail::get_action_name<Derived>()
                            << ") lva(" << reinterpret_cast<void const*>
                                (get_lva<Component>::call(lva)) << "): " << e.what();
                        
                        hpx::report_error(boost::current_exception());
                    }
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
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            return Derived::decorate_action(
                HPX_STD_BIND(typename Derived::thread_function(), lva,
                    util::get< 0>(boost::forward<Arguments>( args)) , util::get< 1>(boost::forward<Arguments>( args)) , util::get< 2>(boost::forward<Arguments>( args)) , util::get< 3>(boost::forward<Arguments>( args)) , util::get< 4>(boost::forward<Arguments>( args)) , util::get< 5>(boost::forward<Arguments>( args)) , util::get< 6>(boost::forward<Arguments>( args)) , util::get< 7>(boost::forward<Arguments>( args)) , util::get< 8>(boost::forward<Arguments>( args)) , util::get< 9>(boost::forward<Arguments>( args)) , util::get< 10>(boost::forward<Arguments>( args)) , util::get< 11>(boost::forward<Arguments>( args)) , util::get< 12>(boost::forward<Arguments>( args)) , util::get< 13>(boost::forward<Arguments>( args)) , util::get< 14>(boost::forward<Arguments>( args))), lva);
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return Derived::decorate_action(
                    base_type::construct_continuation_thread_object_function(
                        cont, F, get_lva<Component>::call(lva),
                        boost::forward<Arguments>(args)), lva);
        }
        template <typename Arguments>
        BOOST_FORCEINLINE static Result
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "base_result_action" << 15
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            return (get_lva<Component>::call(lva)->*F)(
                util::get< 0>(boost::forward<Arguments>( args)) , util::get< 1>(boost::forward<Arguments>( args)) , util::get< 2>(boost::forward<Arguments>( args)) , util::get< 3>(boost::forward<Arguments>( args)) , util::get< 4>(boost::forward<Arguments>( args)) , util::get< 5>(boost::forward<Arguments>( args)) , util::get< 6>(boost::forward<Arguments>( args)) , util::get< 7>(boost::forward<Arguments>( args)) , util::get< 8>(boost::forward<Arguments>( args)) , util::get< 9>(boost::forward<Arguments>( args)) , util::get< 10>(boost::forward<Arguments>( args)) , util::get< 11>(boost::forward<Arguments>( args)) , util::get< 12>(boost::forward<Arguments>( args)) , util::get< 13>(boost::forward<Arguments>( args)) , util::get< 14>(boost::forward<Arguments>( args)));
        }
    };
    
    
    template <
        typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14),
        typename Derived = detail::this_type>
    struct result_action15
      : base_result_action15<
            Component, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14, F,
            typename detail::action_type<
                result_action15<
                    Component, Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14, F>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            result_action15, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14), typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14),
            F, Derived, boost::mpl::false_>
      : result_action15<
            Component, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14, F, Derived>
    {
        typedef result_action15<
            Component, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14, F, Derived
        > type;
    };
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14) const, typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14) const,
            F, Derived, boost::mpl::false_>
      : result_action15<
            Component const, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14, F, Derived>
    {
        typedef result_action15<
            Component const, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14),
        typename Derived = detail::this_type>
    struct direct_result_action15
      : base_result_action15<
            Component, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14, F,
            typename detail::action_type<
                direct_result_action15<
                    Component, Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14, F>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_result_action15, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14), typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14),
            F, Derived, boost::mpl::true_>
      : direct_result_action15<
            Component, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14, F, Derived>
    {
        typedef direct_result_action15<
            Component, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14, F, Derived
        > type;
    };
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14) const, typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14) const,
            F, Derived, boost::mpl::true_>
      : direct_result_action15<
            Component const, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14, F, Derived>
    {
        typedef direct_result_action15<
            Component const, Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14, F, Derived
        > type;
    };
    
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14), typename Derived>
    class base_action15
      : public action<
            Component, util::unused_type,
            hpx::util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type>,
            Derived>
    {
    public:
        typedef util::unused_type result_type;
        typedef hpx::util::tuple<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type> arguments_type;
        typedef action<Component, result_type, arguments_type, Derived> 
            base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva,
                BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component>::call(lva)) << ")";
                    
                    
                    
                    
                    (get_lva<Component>::call(lva)->*F)(
                        boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4) , boost::move(arg5) , boost::move(arg6) , boost::move(arg7) , boost::move(arg8) , boost::move(arg9) , boost::move(arg10) , boost::move(arg11) , boost::move(arg12) , boost::move(arg13) , boost::move(arg14));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing component action("
                            << detail::get_action_name<Derived>()
                            << ") lva(" << reinterpret_cast<void const*>
                                (get_lva<Component>::call(lva)) << "): " << e.what();
                        
                        hpx::report_error(boost::current_exception());
                    }
                }
                
                
                
                util::force_error_on_lock();
                return threads::terminated;
            }
        };
    public:
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            
            
            return Derived::decorate_action(
                HPX_STD_BIND(typename Derived::thread_function(), lva,
                    util::get< 0>(boost::forward<Arguments>( args)) , util::get< 1>(boost::forward<Arguments>( args)) , util::get< 2>(boost::forward<Arguments>( args)) , util::get< 3>(boost::forward<Arguments>( args)) , util::get< 4>(boost::forward<Arguments>( args)) , util::get< 5>(boost::forward<Arguments>( args)) , util::get< 6>(boost::forward<Arguments>( args)) , util::get< 7>(boost::forward<Arguments>( args)) , util::get< 8>(boost::forward<Arguments>( args)) , util::get< 9>(boost::forward<Arguments>( args)) , util::get< 10>(boost::forward<Arguments>( args)) , util::get< 11>(boost::forward<Arguments>( args)) , util::get< 12>(boost::forward<Arguments>( args)) , util::get< 13>(boost::forward<Arguments>( args)) , util::get< 14>(boost::forward<Arguments>( args))), lva);
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return Derived::decorate_action(
                    base_type::construct_continuation_thread_object_function_void(
                        cont, F, get_lva<Component>::call(lva),
                        boost::forward<Arguments>(args)), lva);
        }
        template <typename Arguments>
        BOOST_FORCEINLINE static util::unused_type
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "base_action" << 15
                << "::execute_function name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            (get_lva<Component>::call(lva)->*F)(
                util::get< 0>(boost::forward<Arguments>( args)) , util::get< 1>(boost::forward<Arguments>( args)) , util::get< 2>(boost::forward<Arguments>( args)) , util::get< 3>(boost::forward<Arguments>( args)) , util::get< 4>(boost::forward<Arguments>( args)) , util::get< 5>(boost::forward<Arguments>( args)) , util::get< 6>(boost::forward<Arguments>( args)) , util::get< 7>(boost::forward<Arguments>( args)) , util::get< 8>(boost::forward<Arguments>( args)) , util::get< 9>(boost::forward<Arguments>( args)) , util::get< 10>(boost::forward<Arguments>( args)) , util::get< 11>(boost::forward<Arguments>( args)) , util::get< 12>(boost::forward<Arguments>( args)) , util::get< 13>(boost::forward<Arguments>( args)) , util::get< 14>(boost::forward<Arguments>( args)));
            return util::unused;
        }
    };
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14),
        typename Derived = detail::this_type>
    struct action15
      : base_action15<
            Component, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14, F,
            typename detail::action_type<
                action15<
                    Component, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14, F>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            action15, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14), typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14),
            F, Derived, boost::mpl::false_>
      : action15<
            Component,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14, F, Derived>
    {
        typedef action15<
            Component,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14, F, Derived
        > type;
    };
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14) const, typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14) const,
            F, Derived, boost::mpl::false_>
      : action15<
            Component const,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14, F, Derived>
    {
        typedef action15<
            Component const,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14, F, Derived
        > type;
    };
    
    template <
        typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14),
        typename Derived = detail::this_type>
    struct direct_action15
      : base_action15<
            Component, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14, F,
            typename detail::action_type<
                direct_action15<
                    Component, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14, F>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_action15, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14), typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14),
            F, Derived, boost::mpl::true_>
      : direct_action15<
            Component,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14, F, Derived>
    {
        typedef direct_action15<
            Component,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14, F, Derived
        > type;
    };
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14) const, typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14) const,
            F, Derived, boost::mpl::true_>
      : direct_action15<
            Component const,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14, F, Derived>
    {
        typedef direct_action15<
            Component const,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14, F, Derived
        > type;
    };
    
    
    template <
        typename Component,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14),
        typename Derived>
    struct result_action15<Component, void,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14, F, Derived>
      : action15<Component,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14, F, Derived>
    {};
}}
