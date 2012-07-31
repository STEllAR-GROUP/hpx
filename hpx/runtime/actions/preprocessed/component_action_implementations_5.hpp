// Copyright (c) 2007-2012 Hartmut Kaiser
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


namespace hpx { namespace actions
{
    
    
    template <
        typename Component, typename Result, int Action,
        typename T0,
        Result (Component::*F)(T0), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default>
    class base_result_action1
      : public action<
            Component, Action, Result,
            hpx::util::tuple1<typename detail::remove_qualifiers<T0>::type>,
            Derived, Priority>
    {
    public:
        typedef Result result_type;
        typedef hpx::util::tuple1<
            typename detail::remove_qualifiers<T0>::type> arguments_type;
        typedef action<Component, Action, result_type, arguments_type,
                       Derived, Priority>
            base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0>
            result_type operator()(
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
        typedef boost::mpl::false_ direct_execution;
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            return HPX_STD_BIND(
                typename Derived::thread_function(),
                lva, util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0));
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return boost::move(
                base_type::construct_continuation_thread_object_function(
                    cont, F, get_lva<Component>::call(lva),
                    boost::forward<Arguments>(args)));
        }
    };
    
    
    template <
        typename Component, typename Result, int Action,
        typename T0,
        Result (Component::*F)(T0),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct result_action1
      : base_result_action1<
            Component, Result, Action,
            T0, F,
            typename detail::action_type<
                result_action1<
                    Component, Result, Action, T0, F,
                        Priority>,
                Derived
            >::type, Priority>
    {
        typedef typename detail::action_type<
            result_action1<
                Component, Result, Action, T0, F,
                    Priority>,
            Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Component, typename Result,
        typename T0,
        Result (Component::*F)(T0), typename Derived>
    struct make_action<Result (Component::*)(T0),
            F, Derived, boost::mpl::false_>
      : detail::make_base_action<result_action1<
            Component, Result, component_result_action_arg1,
            T0, F, threads::thread_priority_default,
            Derived> >
    {};
    template <typename Component, typename Result,
        typename T0,
        Result (Component::*F)(T0) const, typename Derived>
    struct make_action<Result (Component::*)(T0) const,
            F, Derived, boost::mpl::false_>
      : detail::make_base_action<result_action1<
            Component const, Result,
            component_result_action_arg1,
            T0, F, threads::thread_priority_default,
            Derived> >
    {};
    
    
    template <
        typename Component, typename Result, int Action,
        typename T0,
        Result (Component::*F)(T0),
        typename Derived = detail::this_type>
    struct direct_result_action1
      : base_result_action1<
            Component, Result, Action,
            T0, F,
            typename detail::action_type<
                direct_result_action1<
                    Component, Result, Action, T0, F>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_result_action1<
                Component, Result, Action, T0, F>,
                Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        template <typename Arguments>
        static Result
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "direct_result_action" << 1
                << "::execute_function name("
                << detail::get_action_name<derived_type>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            return (get_lva<Component>::call(lva)->*F)(
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0));
        }
        
        
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
      : detail::make_base_action<direct_result_action1<
            Component, Result, component_result_action_arg1,
            T0, F, Derived> >
    {};
    template <typename Component, typename Result,
        typename T0,
        Result (Component::*F)(T0) const, typename Derived>
    struct make_action<Result (Component::*)(T0) const,
            F, Derived, boost::mpl::true_>
      : detail::make_base_action<direct_result_action1<
            Component const, Result,
            component_result_action_arg1,
            T0, F, Derived> >
    {};
    
    
    template <
        typename Component, int Action, typename T0,
        void (Component::*F)(T0), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default>
    class base_action1
      : public action<
            Component, Action, util::unused_type,
            hpx::util::tuple1<typename detail::remove_qualifiers<T0>::type>,
            Derived, Priority>
    {
    public:
        typedef util::unused_type result_type;
        typedef hpx::util::tuple1<
            typename detail::remove_qualifiers<T0>::type> arguments_type;
        typedef action<Component, Action, result_type, arguments_type,
                       Derived, Priority> base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0>
            result_type operator()(
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
            
            
            return HPX_STD_BIND(
                typename Derived::thread_function(), lva,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0));
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return boost::move(
                base_type::construct_continuation_thread_object_function_void(
                    cont, F, get_lva<Component>::call(lva),
                    boost::forward<Arguments>(args)));
        }
    };
    
    template <
        typename Component, int Action, typename T0,
        void (Component::*F)(T0),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct action1
      : base_action1<
            Component, Action, T0, F,
            typename detail::action_type<
                action1<
                    Component, Action, T0, F, Priority>,
                Derived
            >::type, Priority>
    {
        typedef typename detail::action_type<
            action1<
                Component, Action, T0, F, Priority>,
            Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Component, typename T0,
        void (Component::*F)(T0), typename Derived>
    struct make_action<void (Component::*)(T0),
            F, Derived, boost::mpl::false_>
      : detail::make_base_action<action1<
            Component, component_action_arg1,
            T0, F, threads::thread_priority_default,
            Derived> >
    {};
    template <typename Component, typename T0,
        void (Component::*F)(T0) const, typename Derived>
    struct make_action<void (Component::*)(T0) const,
            F, Derived, boost::mpl::false_>
      : detail::make_base_action<action1<
            Component const, component_action_arg1,
            T0, F, threads::thread_priority_default,
            Derived> >
    {};
    
    template <
        typename Component, int Action, typename T0,
        void (Component::*F)(T0),
        typename Derived = detail::this_type>
    struct direct_action1
      : base_action1<
            Component, Action, T0, F,
            typename detail::action_type<
                direct_action1<
                    Component, Action, T0, F>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_action1<
                Component, Action, T0, F>,
                Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        template <typename Arguments>
        static util::unused_type
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "direct_action" << 1
                << "::execute_function name("
                << detail::get_action_name<derived_type>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            (get_lva<Component>::call(lva)->*F)(
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0));
            return util::unused;
        }
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename Component, typename T0,
        void (Component::*F)(T0), typename Derived>
    struct make_action<void (Component::*)(T0),
            F, Derived, boost::mpl::true_>
      : detail::make_base_action<direct_action1<
            Component, component_action_arg1,
            T0, F, Derived> >
    {};
    template <typename Component, typename T0,
        void (Component::*F)(T0) const, typename Derived>
    struct make_action<void (Component::*)(T0) const,
            F, Derived, boost::mpl::true_>
      : detail::make_base_action<direct_action1<
            Component const, component_action_arg1,
            T0, F, Derived> >
    {};
    
    
    template <
        typename Component, int Action,
        typename T0,
        void (Component::*F)(T0),
        threads::thread_priority Priority,
        typename Derived>
    struct result_action1<Component, void, Action,
            T0, F, Priority, Derived>
      : action1<Component, Action,
            T0, F, Priority, Derived>
    {};
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Component, typename Result, int Action,
        typename T0 , typename T1,
        Result (Component::*F)(T0 , T1), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default>
    class base_result_action2
      : public action<
            Component, Action, Result,
            hpx::util::tuple2<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type>,
            Derived, Priority>
    {
    public:
        typedef Result result_type;
        typedef hpx::util::tuple2<
            typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type> arguments_type;
        typedef action<Component, Action, result_type, arguments_type,
                       Derived, Priority>
            base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1>
            result_type operator()(
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
        typedef boost::mpl::false_ direct_execution;
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            return HPX_STD_BIND(
                typename Derived::thread_function(),
                lva, util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1));
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return boost::move(
                base_type::construct_continuation_thread_object_function(
                    cont, F, get_lva<Component>::call(lva),
                    boost::forward<Arguments>(args)));
        }
    };
    
    
    template <
        typename Component, typename Result, int Action,
        typename T0 , typename T1,
        Result (Component::*F)(T0 , T1),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct result_action2
      : base_result_action2<
            Component, Result, Action,
            T0 , T1, F,
            typename detail::action_type<
                result_action2<
                    Component, Result, Action, T0 , T1, F,
                        Priority>,
                Derived
            >::type, Priority>
    {
        typedef typename detail::action_type<
            result_action2<
                Component, Result, Action, T0 , T1, F,
                    Priority>,
            Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Component, typename Result,
        typename T0 , typename T1,
        Result (Component::*F)(T0 , T1), typename Derived>
    struct make_action<Result (Component::*)(T0 , T1),
            F, Derived, boost::mpl::false_>
      : detail::make_base_action<result_action2<
            Component, Result, component_result_action_arg2,
            T0 , T1, F, threads::thread_priority_default,
            Derived> >
    {};
    template <typename Component, typename Result,
        typename T0 , typename T1,
        Result (Component::*F)(T0 , T1) const, typename Derived>
    struct make_action<Result (Component::*)(T0 , T1) const,
            F, Derived, boost::mpl::false_>
      : detail::make_base_action<result_action2<
            Component const, Result,
            component_result_action_arg2,
            T0 , T1, F, threads::thread_priority_default,
            Derived> >
    {};
    
    
    template <
        typename Component, typename Result, int Action,
        typename T0 , typename T1,
        Result (Component::*F)(T0 , T1),
        typename Derived = detail::this_type>
    struct direct_result_action2
      : base_result_action2<
            Component, Result, Action,
            T0 , T1, F,
            typename detail::action_type<
                direct_result_action2<
                    Component, Result, Action, T0 , T1, F>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_result_action2<
                Component, Result, Action, T0 , T1, F>,
                Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        template <typename Arguments>
        static Result
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "direct_result_action" << 2
                << "::execute_function name("
                << detail::get_action_name<derived_type>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            return (get_lva<Component>::call(lva)->*F)(
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1));
        }
        
        
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
      : detail::make_base_action<direct_result_action2<
            Component, Result, component_result_action_arg2,
            T0 , T1, F, Derived> >
    {};
    template <typename Component, typename Result,
        typename T0 , typename T1,
        Result (Component::*F)(T0 , T1) const, typename Derived>
    struct make_action<Result (Component::*)(T0 , T1) const,
            F, Derived, boost::mpl::true_>
      : detail::make_base_action<direct_result_action2<
            Component const, Result,
            component_result_action_arg2,
            T0 , T1, F, Derived> >
    {};
    
    
    template <
        typename Component, int Action, typename T0 , typename T1,
        void (Component::*F)(T0 , T1), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default>
    class base_action2
      : public action<
            Component, Action, util::unused_type,
            hpx::util::tuple2<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type>,
            Derived, Priority>
    {
    public:
        typedef util::unused_type result_type;
        typedef hpx::util::tuple2<
            typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type> arguments_type;
        typedef action<Component, Action, result_type, arguments_type,
                       Derived, Priority> base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1>
            result_type operator()(
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
            
            
            return HPX_STD_BIND(
                typename Derived::thread_function(), lva,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1));
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return boost::move(
                base_type::construct_continuation_thread_object_function_void(
                    cont, F, get_lva<Component>::call(lva),
                    boost::forward<Arguments>(args)));
        }
    };
    
    template <
        typename Component, int Action, typename T0 , typename T1,
        void (Component::*F)(T0 , T1),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct action2
      : base_action2<
            Component, Action, T0 , T1, F,
            typename detail::action_type<
                action2<
                    Component, Action, T0 , T1, F, Priority>,
                Derived
            >::type, Priority>
    {
        typedef typename detail::action_type<
            action2<
                Component, Action, T0 , T1, F, Priority>,
            Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Component, typename T0 , typename T1,
        void (Component::*F)(T0 , T1), typename Derived>
    struct make_action<void (Component::*)(T0 , T1),
            F, Derived, boost::mpl::false_>
      : detail::make_base_action<action2<
            Component, component_action_arg2,
            T0 , T1, F, threads::thread_priority_default,
            Derived> >
    {};
    template <typename Component, typename T0 , typename T1,
        void (Component::*F)(T0 , T1) const, typename Derived>
    struct make_action<void (Component::*)(T0 , T1) const,
            F, Derived, boost::mpl::false_>
      : detail::make_base_action<action2<
            Component const, component_action_arg2,
            T0 , T1, F, threads::thread_priority_default,
            Derived> >
    {};
    
    template <
        typename Component, int Action, typename T0 , typename T1,
        void (Component::*F)(T0 , T1),
        typename Derived = detail::this_type>
    struct direct_action2
      : base_action2<
            Component, Action, T0 , T1, F,
            typename detail::action_type<
                direct_action2<
                    Component, Action, T0 , T1, F>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_action2<
                Component, Action, T0 , T1, F>,
                Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        template <typename Arguments>
        static util::unused_type
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "direct_action" << 2
                << "::execute_function name("
                << detail::get_action_name<derived_type>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            (get_lva<Component>::call(lva)->*F)(
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1));
            return util::unused;
        }
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename Component, typename T0 , typename T1,
        void (Component::*F)(T0 , T1), typename Derived>
    struct make_action<void (Component::*)(T0 , T1),
            F, Derived, boost::mpl::true_>
      : detail::make_base_action<direct_action2<
            Component, component_action_arg2,
            T0 , T1, F, Derived> >
    {};
    template <typename Component, typename T0 , typename T1,
        void (Component::*F)(T0 , T1) const, typename Derived>
    struct make_action<void (Component::*)(T0 , T1) const,
            F, Derived, boost::mpl::true_>
      : detail::make_base_action<direct_action2<
            Component const, component_action_arg2,
            T0 , T1, F, Derived> >
    {};
    
    
    template <
        typename Component, int Action,
        typename T0 , typename T1,
        void (Component::*F)(T0 , T1),
        threads::thread_priority Priority,
        typename Derived>
    struct result_action2<Component, void, Action,
            T0 , T1, F, Priority, Derived>
      : action2<Component, Action,
            T0 , T1, F, Priority, Derived>
    {};
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Component, typename Result, int Action,
        typename T0 , typename T1 , typename T2,
        Result (Component::*F)(T0 , T1 , T2), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default>
    class base_result_action3
      : public action<
            Component, Action, Result,
            hpx::util::tuple3<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type>,
            Derived, Priority>
    {
    public:
        typedef Result result_type;
        typedef hpx::util::tuple3<
            typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type> arguments_type;
        typedef action<Component, Action, result_type, arguments_type,
                       Derived, Priority>
            base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2>
            result_type operator()(
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
        typedef boost::mpl::false_ direct_execution;
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            return HPX_STD_BIND(
                typename Derived::thread_function(),
                lva, util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2));
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return boost::move(
                base_type::construct_continuation_thread_object_function(
                    cont, F, get_lva<Component>::call(lva),
                    boost::forward<Arguments>(args)));
        }
    };
    
    
    template <
        typename Component, typename Result, int Action,
        typename T0 , typename T1 , typename T2,
        Result (Component::*F)(T0 , T1 , T2),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct result_action3
      : base_result_action3<
            Component, Result, Action,
            T0 , T1 , T2, F,
            typename detail::action_type<
                result_action3<
                    Component, Result, Action, T0 , T1 , T2, F,
                        Priority>,
                Derived
            >::type, Priority>
    {
        typedef typename detail::action_type<
            result_action3<
                Component, Result, Action, T0 , T1 , T2, F,
                    Priority>,
            Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2,
        Result (Component::*F)(T0 , T1 , T2), typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2),
            F, Derived, boost::mpl::false_>
      : detail::make_base_action<result_action3<
            Component, Result, component_result_action_arg3,
            T0 , T1 , T2, F, threads::thread_priority_default,
            Derived> >
    {};
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2,
        Result (Component::*F)(T0 , T1 , T2) const, typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2) const,
            F, Derived, boost::mpl::false_>
      : detail::make_base_action<result_action3<
            Component const, Result,
            component_result_action_arg3,
            T0 , T1 , T2, F, threads::thread_priority_default,
            Derived> >
    {};
    
    
    template <
        typename Component, typename Result, int Action,
        typename T0 , typename T1 , typename T2,
        Result (Component::*F)(T0 , T1 , T2),
        typename Derived = detail::this_type>
    struct direct_result_action3
      : base_result_action3<
            Component, Result, Action,
            T0 , T1 , T2, F,
            typename detail::action_type<
                direct_result_action3<
                    Component, Result, Action, T0 , T1 , T2, F>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_result_action3<
                Component, Result, Action, T0 , T1 , T2, F>,
                Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        template <typename Arguments>
        static Result
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "direct_result_action" << 3
                << "::execute_function name("
                << detail::get_action_name<derived_type>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            return (get_lva<Component>::call(lva)->*F)(
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2));
        }
        
        
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
      : detail::make_base_action<direct_result_action3<
            Component, Result, component_result_action_arg3,
            T0 , T1 , T2, F, Derived> >
    {};
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2,
        Result (Component::*F)(T0 , T1 , T2) const, typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2) const,
            F, Derived, boost::mpl::true_>
      : detail::make_base_action<direct_result_action3<
            Component const, Result,
            component_result_action_arg3,
            T0 , T1 , T2, F, Derived> >
    {};
    
    
    template <
        typename Component, int Action, typename T0 , typename T1 , typename T2,
        void (Component::*F)(T0 , T1 , T2), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default>
    class base_action3
      : public action<
            Component, Action, util::unused_type,
            hpx::util::tuple3<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type>,
            Derived, Priority>
    {
    public:
        typedef util::unused_type result_type;
        typedef hpx::util::tuple3<
            typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type> arguments_type;
        typedef action<Component, Action, result_type, arguments_type,
                       Derived, Priority> base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2>
            result_type operator()(
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
            
            
            return HPX_STD_BIND(
                typename Derived::thread_function(), lva,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2));
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return boost::move(
                base_type::construct_continuation_thread_object_function_void(
                    cont, F, get_lva<Component>::call(lva),
                    boost::forward<Arguments>(args)));
        }
    };
    
    template <
        typename Component, int Action, typename T0 , typename T1 , typename T2,
        void (Component::*F)(T0 , T1 , T2),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct action3
      : base_action3<
            Component, Action, T0 , T1 , T2, F,
            typename detail::action_type<
                action3<
                    Component, Action, T0 , T1 , T2, F, Priority>,
                Derived
            >::type, Priority>
    {
        typedef typename detail::action_type<
            action3<
                Component, Action, T0 , T1 , T2, F, Priority>,
            Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Component, typename T0 , typename T1 , typename T2,
        void (Component::*F)(T0 , T1 , T2), typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2),
            F, Derived, boost::mpl::false_>
      : detail::make_base_action<action3<
            Component, component_action_arg3,
            T0 , T1 , T2, F, threads::thread_priority_default,
            Derived> >
    {};
    template <typename Component, typename T0 , typename T1 , typename T2,
        void (Component::*F)(T0 , T1 , T2) const, typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2) const,
            F, Derived, boost::mpl::false_>
      : detail::make_base_action<action3<
            Component const, component_action_arg3,
            T0 , T1 , T2, F, threads::thread_priority_default,
            Derived> >
    {};
    
    template <
        typename Component, int Action, typename T0 , typename T1 , typename T2,
        void (Component::*F)(T0 , T1 , T2),
        typename Derived = detail::this_type>
    struct direct_action3
      : base_action3<
            Component, Action, T0 , T1 , T2, F,
            typename detail::action_type<
                direct_action3<
                    Component, Action, T0 , T1 , T2, F>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_action3<
                Component, Action, T0 , T1 , T2, F>,
                Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        template <typename Arguments>
        static util::unused_type
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "direct_action" << 3
                << "::execute_function name("
                << detail::get_action_name<derived_type>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            (get_lva<Component>::call(lva)->*F)(
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2));
            return util::unused;
        }
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename Component, typename T0 , typename T1 , typename T2,
        void (Component::*F)(T0 , T1 , T2), typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2),
            F, Derived, boost::mpl::true_>
      : detail::make_base_action<direct_action3<
            Component, component_action_arg3,
            T0 , T1 , T2, F, Derived> >
    {};
    template <typename Component, typename T0 , typename T1 , typename T2,
        void (Component::*F)(T0 , T1 , T2) const, typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2) const,
            F, Derived, boost::mpl::true_>
      : detail::make_base_action<direct_action3<
            Component const, component_action_arg3,
            T0 , T1 , T2, F, Derived> >
    {};
    
    
    template <
        typename Component, int Action,
        typename T0 , typename T1 , typename T2,
        void (Component::*F)(T0 , T1 , T2),
        threads::thread_priority Priority,
        typename Derived>
    struct result_action3<Component, void, Action,
            T0 , T1 , T2, F, Priority, Derived>
      : action3<Component, Action,
            T0 , T1 , T2, F, Priority, Derived>
    {};
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Component, typename Result, int Action,
        typename T0 , typename T1 , typename T2 , typename T3,
        Result (Component::*F)(T0 , T1 , T2 , T3), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default>
    class base_result_action4
      : public action<
            Component, Action, Result,
            hpx::util::tuple4<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type>,
            Derived, Priority>
    {
    public:
        typedef Result result_type;
        typedef hpx::util::tuple4<
            typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type> arguments_type;
        typedef action<Component, Action, result_type, arguments_type,
                       Derived, Priority>
            base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
            result_type operator()(
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
        typedef boost::mpl::false_ direct_execution;
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            return HPX_STD_BIND(
                typename Derived::thread_function(),
                lva, util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3));
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return boost::move(
                base_type::construct_continuation_thread_object_function(
                    cont, F, get_lva<Component>::call(lva),
                    boost::forward<Arguments>(args)));
        }
    };
    
    
    template <
        typename Component, typename Result, int Action,
        typename T0 , typename T1 , typename T2 , typename T3,
        Result (Component::*F)(T0 , T1 , T2 , T3),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct result_action4
      : base_result_action4<
            Component, Result, Action,
            T0 , T1 , T2 , T3, F,
            typename detail::action_type<
                result_action4<
                    Component, Result, Action, T0 , T1 , T2 , T3, F,
                        Priority>,
                Derived
            >::type, Priority>
    {
        typedef typename detail::action_type<
            result_action4<
                Component, Result, Action, T0 , T1 , T2 , T3, F,
                    Priority>,
            Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3,
        Result (Component::*F)(T0 , T1 , T2 , T3), typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3),
            F, Derived, boost::mpl::false_>
      : detail::make_base_action<result_action4<
            Component, Result, component_result_action_arg4,
            T0 , T1 , T2 , T3, F, threads::thread_priority_default,
            Derived> >
    {};
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3,
        Result (Component::*F)(T0 , T1 , T2 , T3) const, typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3) const,
            F, Derived, boost::mpl::false_>
      : detail::make_base_action<result_action4<
            Component const, Result,
            component_result_action_arg4,
            T0 , T1 , T2 , T3, F, threads::thread_priority_default,
            Derived> >
    {};
    
    
    template <
        typename Component, typename Result, int Action,
        typename T0 , typename T1 , typename T2 , typename T3,
        Result (Component::*F)(T0 , T1 , T2 , T3),
        typename Derived = detail::this_type>
    struct direct_result_action4
      : base_result_action4<
            Component, Result, Action,
            T0 , T1 , T2 , T3, F,
            typename detail::action_type<
                direct_result_action4<
                    Component, Result, Action, T0 , T1 , T2 , T3, F>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_result_action4<
                Component, Result, Action, T0 , T1 , T2 , T3, F>,
                Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        template <typename Arguments>
        static Result
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "direct_result_action" << 4
                << "::execute_function name("
                << detail::get_action_name<derived_type>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            return (get_lva<Component>::call(lva)->*F)(
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3));
        }
        
        
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
      : detail::make_base_action<direct_result_action4<
            Component, Result, component_result_action_arg4,
            T0 , T1 , T2 , T3, F, Derived> >
    {};
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3,
        Result (Component::*F)(T0 , T1 , T2 , T3) const, typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3) const,
            F, Derived, boost::mpl::true_>
      : detail::make_base_action<direct_result_action4<
            Component const, Result,
            component_result_action_arg4,
            T0 , T1 , T2 , T3, F, Derived> >
    {};
    
    
    template <
        typename Component, int Action, typename T0 , typename T1 , typename T2 , typename T3,
        void (Component::*F)(T0 , T1 , T2 , T3), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default>
    class base_action4
      : public action<
            Component, Action, util::unused_type,
            hpx::util::tuple4<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type>,
            Derived, Priority>
    {
    public:
        typedef util::unused_type result_type;
        typedef hpx::util::tuple4<
            typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type> arguments_type;
        typedef action<Component, Action, result_type, arguments_type,
                       Derived, Priority> base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
            result_type operator()(
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
            
            
            return HPX_STD_BIND(
                typename Derived::thread_function(), lva,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3));
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return boost::move(
                base_type::construct_continuation_thread_object_function_void(
                    cont, F, get_lva<Component>::call(lva),
                    boost::forward<Arguments>(args)));
        }
    };
    
    template <
        typename Component, int Action, typename T0 , typename T1 , typename T2 , typename T3,
        void (Component::*F)(T0 , T1 , T2 , T3),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct action4
      : base_action4<
            Component, Action, T0 , T1 , T2 , T3, F,
            typename detail::action_type<
                action4<
                    Component, Action, T0 , T1 , T2 , T3, F, Priority>,
                Derived
            >::type, Priority>
    {
        typedef typename detail::action_type<
            action4<
                Component, Action, T0 , T1 , T2 , T3, F, Priority>,
            Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3,
        void (Component::*F)(T0 , T1 , T2 , T3), typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3),
            F, Derived, boost::mpl::false_>
      : detail::make_base_action<action4<
            Component, component_action_arg4,
            T0 , T1 , T2 , T3, F, threads::thread_priority_default,
            Derived> >
    {};
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3,
        void (Component::*F)(T0 , T1 , T2 , T3) const, typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3) const,
            F, Derived, boost::mpl::false_>
      : detail::make_base_action<action4<
            Component const, component_action_arg4,
            T0 , T1 , T2 , T3, F, threads::thread_priority_default,
            Derived> >
    {};
    
    template <
        typename Component, int Action, typename T0 , typename T1 , typename T2 , typename T3,
        void (Component::*F)(T0 , T1 , T2 , T3),
        typename Derived = detail::this_type>
    struct direct_action4
      : base_action4<
            Component, Action, T0 , T1 , T2 , T3, F,
            typename detail::action_type<
                direct_action4<
                    Component, Action, T0 , T1 , T2 , T3, F>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_action4<
                Component, Action, T0 , T1 , T2 , T3, F>,
                Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        template <typename Arguments>
        static util::unused_type
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "direct_action" << 4
                << "::execute_function name("
                << detail::get_action_name<derived_type>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            (get_lva<Component>::call(lva)->*F)(
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3));
            return util::unused;
        }
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3,
        void (Component::*F)(T0 , T1 , T2 , T3), typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3),
            F, Derived, boost::mpl::true_>
      : detail::make_base_action<direct_action4<
            Component, component_action_arg4,
            T0 , T1 , T2 , T3, F, Derived> >
    {};
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3,
        void (Component::*F)(T0 , T1 , T2 , T3) const, typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3) const,
            F, Derived, boost::mpl::true_>
      : detail::make_base_action<direct_action4<
            Component const, component_action_arg4,
            T0 , T1 , T2 , T3, F, Derived> >
    {};
    
    
    template <
        typename Component, int Action,
        typename T0 , typename T1 , typename T2 , typename T3,
        void (Component::*F)(T0 , T1 , T2 , T3),
        threads::thread_priority Priority,
        typename Derived>
    struct result_action4<Component, void, Action,
            T0 , T1 , T2 , T3, F, Priority, Derived>
      : action4<Component, Action,
            T0 , T1 , T2 , T3, F, Priority, Derived>
    {};
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Component, typename Result, int Action,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default>
    class base_result_action5
      : public action<
            Component, Action, Result,
            hpx::util::tuple5<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type>,
            Derived, Priority>
    {
    public:
        typedef Result result_type;
        typedef hpx::util::tuple5<
            typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type> arguments_type;
        typedef action<Component, Action, result_type, arguments_type,
                       Derived, Priority>
            base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
            result_type operator()(
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
        typedef boost::mpl::false_ direct_execution;
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            return HPX_STD_BIND(
                typename Derived::thread_function(),
                lva, util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4));
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return boost::move(
                base_type::construct_continuation_thread_object_function(
                    cont, F, get_lva<Component>::call(lva),
                    boost::forward<Arguments>(args)));
        }
    };
    
    
    template <
        typename Component, typename Result, int Action,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct result_action5
      : base_result_action5<
            Component, Result, Action,
            T0 , T1 , T2 , T3 , T4, F,
            typename detail::action_type<
                result_action5<
                    Component, Result, Action, T0 , T1 , T2 , T3 , T4, F,
                        Priority>,
                Derived
            >::type, Priority>
    {
        typedef typename detail::action_type<
            result_action5<
                Component, Result, Action, T0 , T1 , T2 , T3 , T4, F,
                    Priority>,
            Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4), typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4),
            F, Derived, boost::mpl::false_>
      : detail::make_base_action<result_action5<
            Component, Result, component_result_action_arg5,
            T0 , T1 , T2 , T3 , T4, F, threads::thread_priority_default,
            Derived> >
    {};
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4) const, typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4) const,
            F, Derived, boost::mpl::false_>
      : detail::make_base_action<result_action5<
            Component const, Result,
            component_result_action_arg5,
            T0 , T1 , T2 , T3 , T4, F, threads::thread_priority_default,
            Derived> >
    {};
    
    
    template <
        typename Component, typename Result, int Action,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4),
        typename Derived = detail::this_type>
    struct direct_result_action5
      : base_result_action5<
            Component, Result, Action,
            T0 , T1 , T2 , T3 , T4, F,
            typename detail::action_type<
                direct_result_action5<
                    Component, Result, Action, T0 , T1 , T2 , T3 , T4, F>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_result_action5<
                Component, Result, Action, T0 , T1 , T2 , T3 , T4, F>,
                Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        template <typename Arguments>
        static Result
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "direct_result_action" << 5
                << "::execute_function name("
                << detail::get_action_name<derived_type>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            return (get_lva<Component>::call(lva)->*F)(
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4));
        }
        
        
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
      : detail::make_base_action<direct_result_action5<
            Component, Result, component_result_action_arg5,
            T0 , T1 , T2 , T3 , T4, F, Derived> >
    {};
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4) const, typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4) const,
            F, Derived, boost::mpl::true_>
      : detail::make_base_action<direct_result_action5<
            Component const, Result,
            component_result_action_arg5,
            T0 , T1 , T2 , T3 , T4, F, Derived> >
    {};
    
    
    template <
        typename Component, int Action, typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default>
    class base_action5
      : public action<
            Component, Action, util::unused_type,
            hpx::util::tuple5<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type>,
            Derived, Priority>
    {
    public:
        typedef util::unused_type result_type;
        typedef hpx::util::tuple5<
            typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type> arguments_type;
        typedef action<Component, Action, result_type, arguments_type,
                       Derived, Priority> base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
            result_type operator()(
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
            
            
            return HPX_STD_BIND(
                typename Derived::thread_function(), lva,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4));
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return boost::move(
                base_type::construct_continuation_thread_object_function_void(
                    cont, F, get_lva<Component>::call(lva),
                    boost::forward<Arguments>(args)));
        }
    };
    
    template <
        typename Component, int Action, typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct action5
      : base_action5<
            Component, Action, T0 , T1 , T2 , T3 , T4, F,
            typename detail::action_type<
                action5<
                    Component, Action, T0 , T1 , T2 , T3 , T4, F, Priority>,
                Derived
            >::type, Priority>
    {
        typedef typename detail::action_type<
            action5<
                Component, Action, T0 , T1 , T2 , T3 , T4, F, Priority>,
            Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4), typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4),
            F, Derived, boost::mpl::false_>
      : detail::make_base_action<action5<
            Component, component_action_arg5,
            T0 , T1 , T2 , T3 , T4, F, threads::thread_priority_default,
            Derived> >
    {};
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4) const, typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4) const,
            F, Derived, boost::mpl::false_>
      : detail::make_base_action<action5<
            Component const, component_action_arg5,
            T0 , T1 , T2 , T3 , T4, F, threads::thread_priority_default,
            Derived> >
    {};
    
    template <
        typename Component, int Action, typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4),
        typename Derived = detail::this_type>
    struct direct_action5
      : base_action5<
            Component, Action, T0 , T1 , T2 , T3 , T4, F,
            typename detail::action_type<
                direct_action5<
                    Component, Action, T0 , T1 , T2 , T3 , T4, F>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_action5<
                Component, Action, T0 , T1 , T2 , T3 , T4, F>,
                Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        template <typename Arguments>
        static util::unused_type
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "direct_action" << 5
                << "::execute_function name("
                << detail::get_action_name<derived_type>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            (get_lva<Component>::call(lva)->*F)(
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4));
            return util::unused;
        }
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4), typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4),
            F, Derived, boost::mpl::true_>
      : detail::make_base_action<direct_action5<
            Component, component_action_arg5,
            T0 , T1 , T2 , T3 , T4, F, Derived> >
    {};
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4) const, typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4) const,
            F, Derived, boost::mpl::true_>
      : detail::make_base_action<direct_action5<
            Component const, component_action_arg5,
            T0 , T1 , T2 , T3 , T4, F, Derived> >
    {};
    
    
    template <
        typename Component, int Action,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4),
        threads::thread_priority Priority,
        typename Derived>
    struct result_action5<Component, void, Action,
            T0 , T1 , T2 , T3 , T4, F, Priority, Derived>
      : action5<Component, Action,
            T0 , T1 , T2 , T3 , T4, F, Priority, Derived>
    {};
}}
