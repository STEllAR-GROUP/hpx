// Copyright (c) 2007-2012 Hartmut Kaiser
// Copyright (c)      2012 Thomas Heller
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
namespace hpx { namespace actions
{
    
    
    template <
        typename Component, typename Result, int Action,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default>
    class base_result_action6
      : public action<
            Component, Action, Result,
            hpx::util::tuple6<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type>,
            Derived, Priority>
    {
    public:
        typedef Result result_type;
        typedef hpx::util::tuple6<
            typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type> arguments_type;
        typedef action<Component, Action, result_type, arguments_type,
                       Derived, Priority>
            base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
            result_type operator()(
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
        typedef boost::mpl::false_ direct_execution;
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            return HPX_STD_BIND(
                typename Derived::thread_function(),
                lva, util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5));
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
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct result_action6
      : base_result_action6<
            Component, Result, Action,
            T0 , T1 , T2 , T3 , T4 , T5, F,
            typename detail::action_type<
                result_action6<
                    Component, Result, Action, T0 , T1 , T2 , T3 , T4 , T5, F,
                        Priority>,
                Derived
            >::type, Priority>
    {
        typedef typename detail::action_type<
            result_action6<
                Component, Result, Action, T0 , T1 , T2 , T3 , T4 , T5, F,
                    Priority>,
            Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5), typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4 , T5),
            F, Derived, boost::mpl::false_>
      : detail::make_base_action<result_action6<
            Component, Result, component_result_action_arg6,
            T0 , T1 , T2 , T3 , T4 , T5, F, threads::thread_priority_default,
            Derived> >
    {};
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5) const, typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4 , T5) const,
            F, Derived, boost::mpl::false_>
      : detail::make_base_action<result_action6<
            Component const, Result,
            component_result_action_arg6,
            T0 , T1 , T2 , T3 , T4 , T5, F, threads::thread_priority_default,
            Derived> >
    {};
    
    
    template <
        typename Component, typename Result, int Action,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5),
        typename Derived = detail::this_type>
    struct direct_result_action6
      : base_result_action6<
            Component, Result, Action,
            T0 , T1 , T2 , T3 , T4 , T5, F,
            typename detail::action_type<
                direct_result_action6<
                    Component, Result, Action, T0 , T1 , T2 , T3 , T4 , T5, F>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_result_action6<
                Component, Result, Action, T0 , T1 , T2 , T3 , T4 , T5, F>,
                Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        template <typename Arguments>
        static Result
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "direct_result_action" << 6
                << "::execute_function name("
                << detail::get_action_name<derived_type>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            return (get_lva<Component>::call(lva)->*F)(
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5));
        }
        
        
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
      : detail::make_base_action<direct_result_action6<
            Component, Result, component_result_action_arg6,
            T0 , T1 , T2 , T3 , T4 , T5, F, Derived> >
    {};
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5) const, typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4 , T5) const,
            F, Derived, boost::mpl::true_>
      : detail::make_base_action<direct_result_action6<
            Component const, Result,
            component_result_action_arg6,
            T0 , T1 , T2 , T3 , T4 , T5, F, Derived> >
    {};
    
    
    template <
        typename Component, int Action, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default>
    class base_action6
      : public action<
            Component, Action, util::unused_type,
            hpx::util::tuple6<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type>,
            Derived, Priority>
    {
    public:
        typedef util::unused_type result_type;
        typedef hpx::util::tuple6<
            typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type> arguments_type;
        typedef action<Component, Action, result_type, arguments_type,
                       Derived, Priority> base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
            result_type operator()(
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
            
            
            return HPX_STD_BIND(
                typename Derived::thread_function(), lva,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5));
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
        typename Component, int Action, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct action6
      : base_action6<
            Component, Action, T0 , T1 , T2 , T3 , T4 , T5, F,
            typename detail::action_type<
                action6<
                    Component, Action, T0 , T1 , T2 , T3 , T4 , T5, F, Priority>,
                Derived
            >::type, Priority>
    {
        typedef typename detail::action_type<
            action6<
                Component, Action, T0 , T1 , T2 , T3 , T4 , T5, F, Priority>,
            Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5), typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5),
            F, Derived, boost::mpl::false_>
      : detail::make_base_action<action6<
            Component, component_action_arg6,
            T0 , T1 , T2 , T3 , T4 , T5, F, threads::thread_priority_default,
            Derived> >
    {};
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5) const, typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5) const,
            F, Derived, boost::mpl::false_>
      : detail::make_base_action<action6<
            Component const, component_action_arg6,
            T0 , T1 , T2 , T3 , T4 , T5, F, threads::thread_priority_default,
            Derived> >
    {};
    
    template <
        typename Component, int Action, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5),
        typename Derived = detail::this_type>
    struct direct_action6
      : base_action6<
            Component, Action, T0 , T1 , T2 , T3 , T4 , T5, F,
            typename detail::action_type<
                direct_action6<
                    Component, Action, T0 , T1 , T2 , T3 , T4 , T5, F>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_action6<
                Component, Action, T0 , T1 , T2 , T3 , T4 , T5, F>,
                Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        template <typename Arguments>
        static util::unused_type
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "direct_action" << 6
                << "::execute_function name("
                << detail::get_action_name<derived_type>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            (get_lva<Component>::call(lva)->*F)(
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5));
            return util::unused;
        }
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5), typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5),
            F, Derived, boost::mpl::true_>
      : detail::make_base_action<direct_action6<
            Component, component_action_arg6,
            T0 , T1 , T2 , T3 , T4 , T5, F, Derived> >
    {};
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5) const, typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5) const,
            F, Derived, boost::mpl::true_>
      : detail::make_base_action<direct_action6<
            Component const, component_action_arg6,
            T0 , T1 , T2 , T3 , T4 , T5, F, Derived> >
    {};
    
    
    template <
        typename Component, int Action,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5),
        threads::thread_priority Priority,
        typename Derived>
    struct result_action6<Component, void, Action,
            T0 , T1 , T2 , T3 , T4 , T5, F, Priority, Derived>
      : action6<Component, Action,
            T0 , T1 , T2 , T3 , T4 , T5, F, Priority, Derived>
    {};
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Component, typename Result, int Action,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default>
    class base_result_action7
      : public action<
            Component, Action, Result,
            hpx::util::tuple7<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type>,
            Derived, Priority>
    {
    public:
        typedef Result result_type;
        typedef hpx::util::tuple7<
            typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type> arguments_type;
        typedef action<Component, Action, result_type, arguments_type,
                       Derived, Priority>
            base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
            result_type operator()(
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
        typedef boost::mpl::false_ direct_execution;
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            return HPX_STD_BIND(
                typename Derived::thread_function(),
                lva, util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6));
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
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct result_action7
      : base_result_action7<
            Component, Result, Action,
            T0 , T1 , T2 , T3 , T4 , T5 , T6, F,
            typename detail::action_type<
                result_action7<
                    Component, Result, Action, T0 , T1 , T2 , T3 , T4 , T5 , T6, F,
                        Priority>,
                Derived
            >::type, Priority>
    {
        typedef typename detail::action_type<
            result_action7<
                Component, Result, Action, T0 , T1 , T2 , T3 , T4 , T5 , T6, F,
                    Priority>,
            Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6), typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6),
            F, Derived, boost::mpl::false_>
      : detail::make_base_action<result_action7<
            Component, Result, component_result_action_arg7,
            T0 , T1 , T2 , T3 , T4 , T5 , T6, F, threads::thread_priority_default,
            Derived> >
    {};
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6) const, typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6) const,
            F, Derived, boost::mpl::false_>
      : detail::make_base_action<result_action7<
            Component const, Result,
            component_result_action_arg7,
            T0 , T1 , T2 , T3 , T4 , T5 , T6, F, threads::thread_priority_default,
            Derived> >
    {};
    
    
    template <
        typename Component, typename Result, int Action,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6),
        typename Derived = detail::this_type>
    struct direct_result_action7
      : base_result_action7<
            Component, Result, Action,
            T0 , T1 , T2 , T3 , T4 , T5 , T6, F,
            typename detail::action_type<
                direct_result_action7<
                    Component, Result, Action, T0 , T1 , T2 , T3 , T4 , T5 , T6, F>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_result_action7<
                Component, Result, Action, T0 , T1 , T2 , T3 , T4 , T5 , T6, F>,
                Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        template <typename Arguments>
        static Result
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "direct_result_action" << 7
                << "::execute_function name("
                << detail::get_action_name<derived_type>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            return (get_lva<Component>::call(lva)->*F)(
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6));
        }
        
        
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
      : detail::make_base_action<direct_result_action7<
            Component, Result, component_result_action_arg7,
            T0 , T1 , T2 , T3 , T4 , T5 , T6, F, Derived> >
    {};
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6) const, typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6) const,
            F, Derived, boost::mpl::true_>
      : detail::make_base_action<direct_result_action7<
            Component const, Result,
            component_result_action_arg7,
            T0 , T1 , T2 , T3 , T4 , T5 , T6, F, Derived> >
    {};
    
    
    template <
        typename Component, int Action, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default>
    class base_action7
      : public action<
            Component, Action, util::unused_type,
            hpx::util::tuple7<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type>,
            Derived, Priority>
    {
    public:
        typedef util::unused_type result_type;
        typedef hpx::util::tuple7<
            typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type> arguments_type;
        typedef action<Component, Action, result_type, arguments_type,
                       Derived, Priority> base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
            result_type operator()(
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
            
            
            return HPX_STD_BIND(
                typename Derived::thread_function(), lva,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6));
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
        typename Component, int Action, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct action7
      : base_action7<
            Component, Action, T0 , T1 , T2 , T3 , T4 , T5 , T6, F,
            typename detail::action_type<
                action7<
                    Component, Action, T0 , T1 , T2 , T3 , T4 , T5 , T6, F, Priority>,
                Derived
            >::type, Priority>
    {
        typedef typename detail::action_type<
            action7<
                Component, Action, T0 , T1 , T2 , T3 , T4 , T5 , T6, F, Priority>,
            Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6), typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6),
            F, Derived, boost::mpl::false_>
      : detail::make_base_action<action7<
            Component, component_action_arg7,
            T0 , T1 , T2 , T3 , T4 , T5 , T6, F, threads::thread_priority_default,
            Derived> >
    {};
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6) const, typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6) const,
            F, Derived, boost::mpl::false_>
      : detail::make_base_action<action7<
            Component const, component_action_arg7,
            T0 , T1 , T2 , T3 , T4 , T5 , T6, F, threads::thread_priority_default,
            Derived> >
    {};
    
    template <
        typename Component, int Action, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6),
        typename Derived = detail::this_type>
    struct direct_action7
      : base_action7<
            Component, Action, T0 , T1 , T2 , T3 , T4 , T5 , T6, F,
            typename detail::action_type<
                direct_action7<
                    Component, Action, T0 , T1 , T2 , T3 , T4 , T5 , T6, F>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_action7<
                Component, Action, T0 , T1 , T2 , T3 , T4 , T5 , T6, F>,
                Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        template <typename Arguments>
        static util::unused_type
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "direct_action" << 7
                << "::execute_function name("
                << detail::get_action_name<derived_type>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            (get_lva<Component>::call(lva)->*F)(
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6));
            return util::unused;
        }
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6), typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6),
            F, Derived, boost::mpl::true_>
      : detail::make_base_action<direct_action7<
            Component, component_action_arg7,
            T0 , T1 , T2 , T3 , T4 , T5 , T6, F, Derived> >
    {};
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6) const, typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6) const,
            F, Derived, boost::mpl::true_>
      : detail::make_base_action<direct_action7<
            Component const, component_action_arg7,
            T0 , T1 , T2 , T3 , T4 , T5 , T6, F, Derived> >
    {};
    
    
    template <
        typename Component, int Action,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6),
        threads::thread_priority Priority,
        typename Derived>
    struct result_action7<Component, void, Action,
            T0 , T1 , T2 , T3 , T4 , T5 , T6, F, Priority, Derived>
      : action7<Component, Action,
            T0 , T1 , T2 , T3 , T4 , T5 , T6, F, Priority, Derived>
    {};
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Component, typename Result, int Action,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default>
    class base_result_action8
      : public action<
            Component, Action, Result,
            hpx::util::tuple8<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type>,
            Derived, Priority>
    {
    public:
        typedef Result result_type;
        typedef hpx::util::tuple8<
            typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type> arguments_type;
        typedef action<Component, Action, result_type, arguments_type,
                       Derived, Priority>
            base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
            result_type operator()(
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
        typedef boost::mpl::false_ direct_execution;
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            return HPX_STD_BIND(
                typename Derived::thread_function(),
                lva, util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7));
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
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct result_action8
      : base_result_action8<
            Component, Result, Action,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F,
            typename detail::action_type<
                result_action8<
                    Component, Result, Action, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F,
                        Priority>,
                Derived
            >::type, Priority>
    {
        typedef typename detail::action_type<
            result_action8<
                Component, Result, Action, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F,
                    Priority>,
            Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7), typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7),
            F, Derived, boost::mpl::false_>
      : detail::make_base_action<result_action8<
            Component, Result, component_result_action_arg8,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F, threads::thread_priority_default,
            Derived> >
    {};
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7) const, typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7) const,
            F, Derived, boost::mpl::false_>
      : detail::make_base_action<result_action8<
            Component const, Result,
            component_result_action_arg8,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F, threads::thread_priority_default,
            Derived> >
    {};
    
    
    template <
        typename Component, typename Result, int Action,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7),
        typename Derived = detail::this_type>
    struct direct_result_action8
      : base_result_action8<
            Component, Result, Action,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F,
            typename detail::action_type<
                direct_result_action8<
                    Component, Result, Action, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_result_action8<
                Component, Result, Action, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F>,
                Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        template <typename Arguments>
        static Result
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "direct_result_action" << 8
                << "::execute_function name("
                << detail::get_action_name<derived_type>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            return (get_lva<Component>::call(lva)->*F)(
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7));
        }
        
        
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
      : detail::make_base_action<direct_result_action8<
            Component, Result, component_result_action_arg8,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F, Derived> >
    {};
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7) const, typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7) const,
            F, Derived, boost::mpl::true_>
      : detail::make_base_action<direct_result_action8<
            Component const, Result,
            component_result_action_arg8,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F, Derived> >
    {};
    
    
    template <
        typename Component, int Action, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default>
    class base_action8
      : public action<
            Component, Action, util::unused_type,
            hpx::util::tuple8<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type>,
            Derived, Priority>
    {
    public:
        typedef util::unused_type result_type;
        typedef hpx::util::tuple8<
            typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type> arguments_type;
        typedef action<Component, Action, result_type, arguments_type,
                       Derived, Priority> base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
            result_type operator()(
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
            
            
            return HPX_STD_BIND(
                typename Derived::thread_function(), lva,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7));
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
        typename Component, int Action, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct action8
      : base_action8<
            Component, Action, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F,
            typename detail::action_type<
                action8<
                    Component, Action, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F, Priority>,
                Derived
            >::type, Priority>
    {
        typedef typename detail::action_type<
            action8<
                Component, Action, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F, Priority>,
            Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7), typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7),
            F, Derived, boost::mpl::false_>
      : detail::make_base_action<action8<
            Component, component_action_arg8,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F, threads::thread_priority_default,
            Derived> >
    {};
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7) const, typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7) const,
            F, Derived, boost::mpl::false_>
      : detail::make_base_action<action8<
            Component const, component_action_arg8,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F, threads::thread_priority_default,
            Derived> >
    {};
    
    template <
        typename Component, int Action, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7),
        typename Derived = detail::this_type>
    struct direct_action8
      : base_action8<
            Component, Action, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F,
            typename detail::action_type<
                direct_action8<
                    Component, Action, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_action8<
                Component, Action, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F>,
                Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        template <typename Arguments>
        static util::unused_type
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "direct_action" << 8
                << "::execute_function name("
                << detail::get_action_name<derived_type>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            (get_lva<Component>::call(lva)->*F)(
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7));
            return util::unused;
        }
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7), typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7),
            F, Derived, boost::mpl::true_>
      : detail::make_base_action<direct_action8<
            Component, component_action_arg8,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F, Derived> >
    {};
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7) const, typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7) const,
            F, Derived, boost::mpl::true_>
      : detail::make_base_action<direct_action8<
            Component const, component_action_arg8,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F, Derived> >
    {};
    
    
    template <
        typename Component, int Action,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7),
        threads::thread_priority Priority,
        typename Derived>
    struct result_action8<Component, void, Action,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F, Priority, Derived>
      : action8<Component, Action,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F, Priority, Derived>
    {};
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Component, typename Result, int Action,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default>
    class base_result_action9
      : public action<
            Component, Action, Result,
            hpx::util::tuple9<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type , typename detail::remove_qualifiers<T8>::type>,
            Derived, Priority>
    {
    public:
        typedef Result result_type;
        typedef hpx::util::tuple9<
            typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type , typename detail::remove_qualifiers<T8>::type> arguments_type;
        typedef action<Component, Action, result_type, arguments_type,
                       Derived, Priority>
            base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
            result_type operator()(
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
        typedef boost::mpl::false_ direct_execution;
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            return HPX_STD_BIND(
                typename Derived::thread_function(),
                lva, util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type8>::call( args. a8));
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
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct result_action9
      : base_result_action9<
            Component, Result, Action,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F,
            typename detail::action_type<
                result_action9<
                    Component, Result, Action, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F,
                        Priority>,
                Derived
            >::type, Priority>
    {
        typedef typename detail::action_type<
            result_action9<
                Component, Result, Action, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F,
                    Priority>,
            Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8), typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8),
            F, Derived, boost::mpl::false_>
      : detail::make_base_action<result_action9<
            Component, Result, component_result_action_arg9,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F, threads::thread_priority_default,
            Derived> >
    {};
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8) const, typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8) const,
            F, Derived, boost::mpl::false_>
      : detail::make_base_action<result_action9<
            Component const, Result,
            component_result_action_arg9,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F, threads::thread_priority_default,
            Derived> >
    {};
    
    
    template <
        typename Component, typename Result, int Action,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8),
        typename Derived = detail::this_type>
    struct direct_result_action9
      : base_result_action9<
            Component, Result, Action,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F,
            typename detail::action_type<
                direct_result_action9<
                    Component, Result, Action, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_result_action9<
                Component, Result, Action, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F>,
                Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        template <typename Arguments>
        static Result
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "direct_result_action" << 9
                << "::execute_function name("
                << detail::get_action_name<derived_type>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            return (get_lva<Component>::call(lva)->*F)(
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type8>::call( args. a8));
        }
        
        
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
      : detail::make_base_action<direct_result_action9<
            Component, Result, component_result_action_arg9,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F, Derived> >
    {};
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8) const, typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8) const,
            F, Derived, boost::mpl::true_>
      : detail::make_base_action<direct_result_action9<
            Component const, Result,
            component_result_action_arg9,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F, Derived> >
    {};
    
    
    template <
        typename Component, int Action, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default>
    class base_action9
      : public action<
            Component, Action, util::unused_type,
            hpx::util::tuple9<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type , typename detail::remove_qualifiers<T8>::type>,
            Derived, Priority>
    {
    public:
        typedef util::unused_type result_type;
        typedef hpx::util::tuple9<
            typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type , typename detail::remove_qualifiers<T8>::type> arguments_type;
        typedef action<Component, Action, result_type, arguments_type,
                       Derived, Priority> base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
            result_type operator()(
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
            
            
            return HPX_STD_BIND(
                typename Derived::thread_function(), lva,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type8>::call( args. a8));
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
        typename Component, int Action, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct action9
      : base_action9<
            Component, Action, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F,
            typename detail::action_type<
                action9<
                    Component, Action, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F, Priority>,
                Derived
            >::type, Priority>
    {
        typedef typename detail::action_type<
            action9<
                Component, Action, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F, Priority>,
            Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8), typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8),
            F, Derived, boost::mpl::false_>
      : detail::make_base_action<action9<
            Component, component_action_arg9,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F, threads::thread_priority_default,
            Derived> >
    {};
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8) const, typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8) const,
            F, Derived, boost::mpl::false_>
      : detail::make_base_action<action9<
            Component const, component_action_arg9,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F, threads::thread_priority_default,
            Derived> >
    {};
    
    template <
        typename Component, int Action, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8),
        typename Derived = detail::this_type>
    struct direct_action9
      : base_action9<
            Component, Action, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F,
            typename detail::action_type<
                direct_action9<
                    Component, Action, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_action9<
                Component, Action, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F>,
                Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        template <typename Arguments>
        static util::unused_type
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "direct_action" << 9
                << "::execute_function name("
                << detail::get_action_name<derived_type>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            (get_lva<Component>::call(lva)->*F)(
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type8>::call( args. a8));
            return util::unused;
        }
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8), typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8),
            F, Derived, boost::mpl::true_>
      : detail::make_base_action<direct_action9<
            Component, component_action_arg9,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F, Derived> >
    {};
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8) const, typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8) const,
            F, Derived, boost::mpl::true_>
      : detail::make_base_action<direct_action9<
            Component const, component_action_arg9,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F, Derived> >
    {};
    
    
    template <
        typename Component, int Action,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8),
        threads::thread_priority Priority,
        typename Derived>
    struct result_action9<Component, void, Action,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F, Priority, Derived>
      : action9<Component, Action,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F, Priority, Derived>
    {};
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Component, typename Result, int Action,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default>
    class base_result_action10
      : public action<
            Component, Action, Result,
            hpx::util::tuple10<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type , typename detail::remove_qualifiers<T8>::type , typename detail::remove_qualifiers<T9>::type>,
            Derived, Priority>
    {
    public:
        typedef Result result_type;
        typedef hpx::util::tuple10<
            typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type , typename detail::remove_qualifiers<T8>::type , typename detail::remove_qualifiers<T9>::type> arguments_type;
        typedef action<Component, Action, result_type, arguments_type,
                       Derived, Priority>
            base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
            result_type operator()(
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
        typedef boost::mpl::false_ direct_execution;
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            return HPX_STD_BIND(
                typename Derived::thread_function(),
                lva, util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type9>::call( args. a9));
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
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct result_action10
      : base_result_action10<
            Component, Result, Action,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F,
            typename detail::action_type<
                result_action10<
                    Component, Result, Action, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F,
                        Priority>,
                Derived
            >::type, Priority>
    {
        typedef typename detail::action_type<
            result_action10<
                Component, Result, Action, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F,
                    Priority>,
            Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9), typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9),
            F, Derived, boost::mpl::false_>
      : detail::make_base_action<result_action10<
            Component, Result, component_result_action_arg10,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F, threads::thread_priority_default,
            Derived> >
    {};
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9) const, typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9) const,
            F, Derived, boost::mpl::false_>
      : detail::make_base_action<result_action10<
            Component const, Result,
            component_result_action_arg10,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F, threads::thread_priority_default,
            Derived> >
    {};
    
    
    template <
        typename Component, typename Result, int Action,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9),
        typename Derived = detail::this_type>
    struct direct_result_action10
      : base_result_action10<
            Component, Result, Action,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F,
            typename detail::action_type<
                direct_result_action10<
                    Component, Result, Action, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_result_action10<
                Component, Result, Action, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F>,
                Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        template <typename Arguments>
        static Result
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "direct_result_action" << 10
                << "::execute_function name("
                << detail::get_action_name<derived_type>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            return (get_lva<Component>::call(lva)->*F)(
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type9>::call( args. a9));
        }
        
        
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
      : detail::make_base_action<direct_result_action10<
            Component, Result, component_result_action_arg10,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F, Derived> >
    {};
    template <typename Component, typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9,
        Result (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9) const, typename Derived>
    struct make_action<Result (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9) const,
            F, Derived, boost::mpl::true_>
      : detail::make_base_action<direct_result_action10<
            Component const, Result,
            component_result_action_arg10,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F, Derived> >
    {};
    
    
    template <
        typename Component, int Action, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default>
    class base_action10
      : public action<
            Component, Action, util::unused_type,
            hpx::util::tuple10<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type , typename detail::remove_qualifiers<T8>::type , typename detail::remove_qualifiers<T9>::type>,
            Derived, Priority>
    {
    public:
        typedef util::unused_type result_type;
        typedef hpx::util::tuple10<
            typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type , typename detail::remove_qualifiers<T8>::type , typename detail::remove_qualifiers<T9>::type> arguments_type;
        typedef action<Component, Action, result_type, arguments_type,
                       Derived, Priority> base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
            result_type operator()(
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
            
            
            return HPX_STD_BIND(
                typename Derived::thread_function(), lva,
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type9>::call( args. a9));
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
        typename Component, int Action, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct action10
      : base_action10<
            Component, Action, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F,
            typename detail::action_type<
                action10<
                    Component, Action, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F, Priority>,
                Derived
            >::type, Priority>
    {
        typedef typename detail::action_type<
            action10<
                Component, Action, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F, Priority>,
            Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9), typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9),
            F, Derived, boost::mpl::false_>
      : detail::make_base_action<action10<
            Component, component_action_arg10,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F, threads::thread_priority_default,
            Derived> >
    {};
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9) const, typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9) const,
            F, Derived, boost::mpl::false_>
      : detail::make_base_action<action10<
            Component const, component_action_arg10,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F, threads::thread_priority_default,
            Derived> >
    {};
    
    template <
        typename Component, int Action, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9),
        typename Derived = detail::this_type>
    struct direct_action10
      : base_action10<
            Component, Action, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F,
            typename detail::action_type<
                direct_action10<
                    Component, Action, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F>,
                    Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_action10<
                Component, Action, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F>,
                Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        template <typename Arguments>
        static util::unused_type
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "direct_action" << 10
                << "::execute_function name("
                << detail::get_action_name<derived_type>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            (get_lva<Component>::call(lva)->*F)(
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type9>::call( args. a9));
            return util::unused;
        }
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9), typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9),
            F, Derived, boost::mpl::true_>
      : detail::make_base_action<direct_action10<
            Component, component_action_arg10,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F, Derived> >
    {};
    template <typename Component, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9) const, typename Derived>
    struct make_action<void (Component::*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9) const,
            F, Derived, boost::mpl::true_>
      : detail::make_base_action<direct_action10<
            Component const, component_action_arg10,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F, Derived> >
    {};
    
    
    template <
        typename Component, int Action,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9,
        void (Component::*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9),
        threads::thread_priority Priority,
        typename Derived>
    struct result_action10<Component, void, Action,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F, Priority, Derived>
      : action10<Component, Action,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F, Priority, Derived>
    {};
}}
