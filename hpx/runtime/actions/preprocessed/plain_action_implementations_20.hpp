// Copyright (c) 2007-2012 Hartmut Kaiser
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


namespace hpx { namespace actions
{
    
    
    template <
        typename Result,
        typename T0,
        Result (*F)(T0), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default
    >
    class plain_base_result_action1
      : public action<
            components::server::plain_function<Derived>,
            function_result_action_arg1,
            Result,
            hpx::util::tuple1<typename detail::remove_qualifiers<T0>::type>,
            Derived, Priority>
    {
    public:
        typedef Result result_type;
        typedef hpx::util::tuple1<
            typename detail::remove_qualifiers<T0>::type> arguments_type;
        typedef action<
            components::server::plain_function<Derived>,
            function_result_action_arg1, result_type,
            arguments_type, Derived, Priority> base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0>
            result_type operator()(
                HPX_FWD_ARGS(2, 0, ( Arg, arg))) const
            {
                try {
                    LTM_(debug) << "Executing plain action("
                                << detail::get_action_name<Derived>()
                                << ").";
                    
                    
                    
                    
                    
                    F(HPX_MOVE_ARGS(2, 0, arg));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing plain action("
                            << detail::get_action_name<Derived>()
                            << "): " << e.what();
                        
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
            return HPX_STD_BIND(typename Derived::thread_function(),
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0));
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return boost::move(
                base_type::construct_continuation_thread_function(
                    cont, F, boost::forward<Arguments>(args)));
        }
    };
    
    
    template <
        typename Result, typename T0,
        Result (*F)(T0),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct plain_result_action1
      : plain_base_result_action1<Result,
          T0, F,
          typename detail::action_type<
              plain_result_action1<
                  Result, T0, F, Priority>, Derived
          >::type, Priority>
    {
        typedef typename detail::action_type<
            plain_result_action1<
                Result, T0, F, Priority>, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Result, typename T0,
        Result (*F)(T0), typename Derived>
    struct make_action<Result (*)(T0), F, Derived, boost::mpl::false_>
      : boost::mpl::identity<plain_result_action1<Result,
            T0, F, threads::thread_priority_default,
            Derived> >
    {};
    
    
    template <
        typename Result, typename T0,
        Result (*F)(T0),
        typename Derived = detail::this_type>
    struct plain_direct_result_action1
      : plain_base_result_action1<Result,
          T0, F,
          typename detail::action_type<
              plain_direct_result_action1<
                  Result, T0, F>, Derived
          >::type>
    {
        typedef typename detail::action_type<
            plain_direct_result_action1<
                Result, T0, F>, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        template <typename Arguments>
        static Result
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "plain_direct_result_action" << 1
                << "::execute_function name("
                << detail::get_action_name<derived_type>() << ")";
            return F(util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0));
        }
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename Result, typename T0,
        Result (*F)(T0), typename Derived>
    struct make_action<Result (*)(T0), F, Derived, boost::mpl::true_>
      : boost::mpl::identity<plain_direct_result_action1<
            Result, T0, F, Derived> >
    {};
    
    
    template <
        typename T0,
        void (*F)(T0), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default>
    class plain_base_action1
      : public action<
            components::server::plain_function<Derived>,
            function_action_arg1,
            util::unused_type,
            hpx::util::tuple1<typename detail::remove_qualifiers<T0>::type>,
            Derived, Priority>
    {
    public:
        typedef util::unused_type result_type;
        typedef
            hpx::util::tuple1<typename detail::remove_qualifiers<T0>::type>
        arguments_type;
        typedef action<
            components::server::plain_function<Derived>,
            function_action_arg1, result_type,
            arguments_type, Derived, Priority> base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0>
            result_type operator()(
                HPX_FWD_ARGS(2, 0, ( Arg, arg))) const
            {
                try {
                    LTM_(debug) << "Executing plain action("
                                << detail::get_action_name<Derived>()
                                << ").";
                    
                    
                    
                    
                    
                    F(HPX_MOVE_ARGS(2, 0, arg));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing plain action("
                            << detail::get_action_name<Derived>()
                            << "): " << e.what();
                        
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
            return HPX_STD_BIND(typename Derived::thread_function(),
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0));
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return
                base_type::construct_continuation_thread_function_void(
                    cont, F, boost::forward<Arguments>(args));
        }
    };
    
    template <
        typename T0,
        void (*F)(T0),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct plain_action1
      : plain_base_action1<
            T0, F,
            typename detail::action_type<
                plain_action1<
                    T0, F, Priority>, Derived
            >::type, Priority>
    {
        typedef typename detail::action_type<
            plain_action1<
                T0, F, Priority>, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename T0,
        void (*F)(T0), typename Derived>
    struct make_action<void (*)(T0), F, Derived, boost::mpl::false_>
      : boost::mpl::identity<plain_action1<
            T0, F, threads::thread_priority_default,
            Derived> >
    {};
    
    template <
        typename T0,
        void (*F)(T0),
        typename Derived = detail::this_type>
    struct plain_direct_action1
      : plain_base_action1<
            T0, F,
            typename detail::action_type<
                plain_direct_action1<
                    T0, F>, Derived
            >::type>
    {
        typedef typename detail::action_type<
            plain_direct_action1<
                T0, F>, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        template <typename Arguments>
        static util::unused_type
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "plain_direct_action" << 1
                << "::execute_function name("
                << detail::get_action_name<derived_type>() << ")";
            F(util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0));
            return util::unused;
        }
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename T0,
        void (*F)(T0), typename Derived>
    struct make_action<void (*)(T0), F, Derived, boost::mpl::true_>
      : boost::mpl::identity<plain_direct_action1<
            T0, F, Derived> >
    {};
    
    
    template <
        typename T0,
        void (*F)(T0),
        threads::thread_priority Priority, typename Derived>
    struct plain_result_action1<
                void, T0, F, Priority, Derived>
      : plain_action1<
            T0, F, Priority, Derived>
    {};
}}
namespace hpx { namespace traits
{
    template <typename Arg0,
        void (*F)(Arg0),
        hpx::threads::thread_priority Priority, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_action1<
                    Arg0, F, Priority> >, Enable>
      : boost::mpl::false_
    {};
    template <typename Arg0,
        void (*F)(Arg0), typename Derived,
        typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_direct_action1<
                    Arg0, F, Derived> >, Enable>
      : boost::mpl::false_
    {};
    template <typename R, typename Arg0,
        R(*F)(Arg0),
        hpx::threads::thread_priority Priority, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_result_action1<
                    R, Arg0, F, Priority> >, Enable>
      : boost::mpl::false_
    {};
    template <typename R, typename Arg0,
        R(*F)(Arg0), typename Derived, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_direct_result_action1<
                    R, Arg0, F, Derived> >, Enable>
      : boost::mpl::false_
    {};
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Result,
        typename T0 , typename T1,
        Result (*F)(T0 , T1), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default
    >
    class plain_base_result_action2
      : public action<
            components::server::plain_function<Derived>,
            function_result_action_arg2,
            Result,
            hpx::util::tuple2<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type>,
            Derived, Priority>
    {
    public:
        typedef Result result_type;
        typedef hpx::util::tuple2<
            typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type> arguments_type;
        typedef action<
            components::server::plain_function<Derived>,
            function_result_action_arg2, result_type,
            arguments_type, Derived, Priority> base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1>
            result_type operator()(
                HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg))) const
            {
                try {
                    LTM_(debug) << "Executing plain action("
                                << detail::get_action_name<Derived>()
                                << ").";
                    
                    
                    
                    
                    
                    F(HPX_MOVE_ARGS(2, 0, arg) , HPX_MOVE_ARGS(2, 1, arg));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing plain action("
                            << detail::get_action_name<Derived>()
                            << "): " << e.what();
                        
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
            return HPX_STD_BIND(typename Derived::thread_function(),
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1));
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return boost::move(
                base_type::construct_continuation_thread_function(
                    cont, F, boost::forward<Arguments>(args)));
        }
    };
    
    
    template <
        typename Result, typename T0 , typename T1,
        Result (*F)(T0 , T1),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct plain_result_action2
      : plain_base_result_action2<Result,
          T0 , T1, F,
          typename detail::action_type<
              plain_result_action2<
                  Result, T0 , T1, F, Priority>, Derived
          >::type, Priority>
    {
        typedef typename detail::action_type<
            plain_result_action2<
                Result, T0 , T1, F, Priority>, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Result, typename T0 , typename T1,
        Result (*F)(T0 , T1), typename Derived>
    struct make_action<Result (*)(T0 , T1), F, Derived, boost::mpl::false_>
      : boost::mpl::identity<plain_result_action2<Result,
            T0 , T1, F, threads::thread_priority_default,
            Derived> >
    {};
    
    
    template <
        typename Result, typename T0 , typename T1,
        Result (*F)(T0 , T1),
        typename Derived = detail::this_type>
    struct plain_direct_result_action2
      : plain_base_result_action2<Result,
          T0 , T1, F,
          typename detail::action_type<
              plain_direct_result_action2<
                  Result, T0 , T1, F>, Derived
          >::type>
    {
        typedef typename detail::action_type<
            plain_direct_result_action2<
                Result, T0 , T1, F>, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        template <typename Arguments>
        static Result
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "plain_direct_result_action" << 2
                << "::execute_function name("
                << detail::get_action_name<derived_type>() << ")";
            return F(util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1));
        }
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename Result, typename T0 , typename T1,
        Result (*F)(T0 , T1), typename Derived>
    struct make_action<Result (*)(T0 , T1), F, Derived, boost::mpl::true_>
      : boost::mpl::identity<plain_direct_result_action2<
            Result, T0 , T1, F, Derived> >
    {};
    
    
    template <
        typename T0 , typename T1,
        void (*F)(T0 , T1), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default>
    class plain_base_action2
      : public action<
            components::server::plain_function<Derived>,
            function_action_arg2,
            util::unused_type,
            hpx::util::tuple2<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type>,
            Derived, Priority>
    {
    public:
        typedef util::unused_type result_type;
        typedef
            hpx::util::tuple2<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type>
        arguments_type;
        typedef action<
            components::server::plain_function<Derived>,
            function_action_arg2, result_type,
            arguments_type, Derived, Priority> base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1>
            result_type operator()(
                HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg))) const
            {
                try {
                    LTM_(debug) << "Executing plain action("
                                << detail::get_action_name<Derived>()
                                << ").";
                    
                    
                    
                    
                    
                    F(HPX_MOVE_ARGS(2, 0, arg) , HPX_MOVE_ARGS(2, 1, arg));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing plain action("
                            << detail::get_action_name<Derived>()
                            << "): " << e.what();
                        
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
            return HPX_STD_BIND(typename Derived::thread_function(),
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1));
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return
                base_type::construct_continuation_thread_function_void(
                    cont, F, boost::forward<Arguments>(args));
        }
    };
    
    template <
        typename T0 , typename T1,
        void (*F)(T0 , T1),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct plain_action2
      : plain_base_action2<
            T0 , T1, F,
            typename detail::action_type<
                plain_action2<
                    T0 , T1, F, Priority>, Derived
            >::type, Priority>
    {
        typedef typename detail::action_type<
            plain_action2<
                T0 , T1, F, Priority>, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename T0 , typename T1,
        void (*F)(T0 , T1), typename Derived>
    struct make_action<void (*)(T0 , T1), F, Derived, boost::mpl::false_>
      : boost::mpl::identity<plain_action2<
            T0 , T1, F, threads::thread_priority_default,
            Derived> >
    {};
    
    template <
        typename T0 , typename T1,
        void (*F)(T0 , T1),
        typename Derived = detail::this_type>
    struct plain_direct_action2
      : plain_base_action2<
            T0 , T1, F,
            typename detail::action_type<
                plain_direct_action2<
                    T0 , T1, F>, Derived
            >::type>
    {
        typedef typename detail::action_type<
            plain_direct_action2<
                T0 , T1, F>, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        template <typename Arguments>
        static util::unused_type
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "plain_direct_action" << 2
                << "::execute_function name("
                << detail::get_action_name<derived_type>() << ")";
            F(util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1));
            return util::unused;
        }
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename T0 , typename T1,
        void (*F)(T0 , T1), typename Derived>
    struct make_action<void (*)(T0 , T1), F, Derived, boost::mpl::true_>
      : boost::mpl::identity<plain_direct_action2<
            T0 , T1, F, Derived> >
    {};
    
    
    template <
        typename T0 , typename T1,
        void (*F)(T0 , T1),
        threads::thread_priority Priority, typename Derived>
    struct plain_result_action2<
                void, T0 , T1, F, Priority, Derived>
      : plain_action2<
            T0 , T1, F, Priority, Derived>
    {};
}}
namespace hpx { namespace traits
{
    template <typename Arg0 , typename Arg1,
        void (*F)(Arg0 , Arg1),
        hpx::threads::thread_priority Priority, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_action2<
                    Arg0 , Arg1, F, Priority> >, Enable>
      : boost::mpl::false_
    {};
    template <typename Arg0 , typename Arg1,
        void (*F)(Arg0 , Arg1), typename Derived,
        typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_direct_action2<
                    Arg0 , Arg1, F, Derived> >, Enable>
      : boost::mpl::false_
    {};
    template <typename R, typename Arg0 , typename Arg1,
        R(*F)(Arg0 , Arg1),
        hpx::threads::thread_priority Priority, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_result_action2<
                    R, Arg0 , Arg1, F, Priority> >, Enable>
      : boost::mpl::false_
    {};
    template <typename R, typename Arg0 , typename Arg1,
        R(*F)(Arg0 , Arg1), typename Derived, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_direct_result_action2<
                    R, Arg0 , Arg1, F, Derived> >, Enable>
      : boost::mpl::false_
    {};
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Result,
        typename T0 , typename T1 , typename T2,
        Result (*F)(T0 , T1 , T2), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default
    >
    class plain_base_result_action3
      : public action<
            components::server::plain_function<Derived>,
            function_result_action_arg3,
            Result,
            hpx::util::tuple3<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type>,
            Derived, Priority>
    {
    public:
        typedef Result result_type;
        typedef hpx::util::tuple3<
            typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type> arguments_type;
        typedef action<
            components::server::plain_function<Derived>,
            function_result_action_arg3, result_type,
            arguments_type, Derived, Priority> base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2>
            result_type operator()(
                HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg))) const
            {
                try {
                    LTM_(debug) << "Executing plain action("
                                << detail::get_action_name<Derived>()
                                << ").";
                    
                    
                    
                    
                    
                    F(HPX_MOVE_ARGS(2, 0, arg) , HPX_MOVE_ARGS(2, 1, arg) , HPX_MOVE_ARGS(2, 2, arg));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing plain action("
                            << detail::get_action_name<Derived>()
                            << "): " << e.what();
                        
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
            return HPX_STD_BIND(typename Derived::thread_function(),
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2));
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return boost::move(
                base_type::construct_continuation_thread_function(
                    cont, F, boost::forward<Arguments>(args)));
        }
    };
    
    
    template <
        typename Result, typename T0 , typename T1 , typename T2,
        Result (*F)(T0 , T1 , T2),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct plain_result_action3
      : plain_base_result_action3<Result,
          T0 , T1 , T2, F,
          typename detail::action_type<
              plain_result_action3<
                  Result, T0 , T1 , T2, F, Priority>, Derived
          >::type, Priority>
    {
        typedef typename detail::action_type<
            plain_result_action3<
                Result, T0 , T1 , T2, F, Priority>, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Result, typename T0 , typename T1 , typename T2,
        Result (*F)(T0 , T1 , T2), typename Derived>
    struct make_action<Result (*)(T0 , T1 , T2), F, Derived, boost::mpl::false_>
      : boost::mpl::identity<plain_result_action3<Result,
            T0 , T1 , T2, F, threads::thread_priority_default,
            Derived> >
    {};
    
    
    template <
        typename Result, typename T0 , typename T1 , typename T2,
        Result (*F)(T0 , T1 , T2),
        typename Derived = detail::this_type>
    struct plain_direct_result_action3
      : plain_base_result_action3<Result,
          T0 , T1 , T2, F,
          typename detail::action_type<
              plain_direct_result_action3<
                  Result, T0 , T1 , T2, F>, Derived
          >::type>
    {
        typedef typename detail::action_type<
            plain_direct_result_action3<
                Result, T0 , T1 , T2, F>, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        template <typename Arguments>
        static Result
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "plain_direct_result_action" << 3
                << "::execute_function name("
                << detail::get_action_name<derived_type>() << ")";
            return F(util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2));
        }
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename Result, typename T0 , typename T1 , typename T2,
        Result (*F)(T0 , T1 , T2), typename Derived>
    struct make_action<Result (*)(T0 , T1 , T2), F, Derived, boost::mpl::true_>
      : boost::mpl::identity<plain_direct_result_action3<
            Result, T0 , T1 , T2, F, Derived> >
    {};
    
    
    template <
        typename T0 , typename T1 , typename T2,
        void (*F)(T0 , T1 , T2), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default>
    class plain_base_action3
      : public action<
            components::server::plain_function<Derived>,
            function_action_arg3,
            util::unused_type,
            hpx::util::tuple3<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type>,
            Derived, Priority>
    {
    public:
        typedef util::unused_type result_type;
        typedef
            hpx::util::tuple3<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type>
        arguments_type;
        typedef action<
            components::server::plain_function<Derived>,
            function_action_arg3, result_type,
            arguments_type, Derived, Priority> base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2>
            result_type operator()(
                HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg))) const
            {
                try {
                    LTM_(debug) << "Executing plain action("
                                << detail::get_action_name<Derived>()
                                << ").";
                    
                    
                    
                    
                    
                    F(HPX_MOVE_ARGS(2, 0, arg) , HPX_MOVE_ARGS(2, 1, arg) , HPX_MOVE_ARGS(2, 2, arg));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing plain action("
                            << detail::get_action_name<Derived>()
                            << "): " << e.what();
                        
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
            return HPX_STD_BIND(typename Derived::thread_function(),
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2));
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return
                base_type::construct_continuation_thread_function_void(
                    cont, F, boost::forward<Arguments>(args));
        }
    };
    
    template <
        typename T0 , typename T1 , typename T2,
        void (*F)(T0 , T1 , T2),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct plain_action3
      : plain_base_action3<
            T0 , T1 , T2, F,
            typename detail::action_type<
                plain_action3<
                    T0 , T1 , T2, F, Priority>, Derived
            >::type, Priority>
    {
        typedef typename detail::action_type<
            plain_action3<
                T0 , T1 , T2, F, Priority>, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename T0 , typename T1 , typename T2,
        void (*F)(T0 , T1 , T2), typename Derived>
    struct make_action<void (*)(T0 , T1 , T2), F, Derived, boost::mpl::false_>
      : boost::mpl::identity<plain_action3<
            T0 , T1 , T2, F, threads::thread_priority_default,
            Derived> >
    {};
    
    template <
        typename T0 , typename T1 , typename T2,
        void (*F)(T0 , T1 , T2),
        typename Derived = detail::this_type>
    struct plain_direct_action3
      : plain_base_action3<
            T0 , T1 , T2, F,
            typename detail::action_type<
                plain_direct_action3<
                    T0 , T1 , T2, F>, Derived
            >::type>
    {
        typedef typename detail::action_type<
            plain_direct_action3<
                T0 , T1 , T2, F>, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        template <typename Arguments>
        static util::unused_type
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "plain_direct_action" << 3
                << "::execute_function name("
                << detail::get_action_name<derived_type>() << ")";
            F(util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2));
            return util::unused;
        }
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename T0 , typename T1 , typename T2,
        void (*F)(T0 , T1 , T2), typename Derived>
    struct make_action<void (*)(T0 , T1 , T2), F, Derived, boost::mpl::true_>
      : boost::mpl::identity<plain_direct_action3<
            T0 , T1 , T2, F, Derived> >
    {};
    
    
    template <
        typename T0 , typename T1 , typename T2,
        void (*F)(T0 , T1 , T2),
        threads::thread_priority Priority, typename Derived>
    struct plain_result_action3<
                void, T0 , T1 , T2, F, Priority, Derived>
      : plain_action3<
            T0 , T1 , T2, F, Priority, Derived>
    {};
}}
namespace hpx { namespace traits
{
    template <typename Arg0 , typename Arg1 , typename Arg2,
        void (*F)(Arg0 , Arg1 , Arg2),
        hpx::threads::thread_priority Priority, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_action3<
                    Arg0 , Arg1 , Arg2, F, Priority> >, Enable>
      : boost::mpl::false_
    {};
    template <typename Arg0 , typename Arg1 , typename Arg2,
        void (*F)(Arg0 , Arg1 , Arg2), typename Derived,
        typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_direct_action3<
                    Arg0 , Arg1 , Arg2, F, Derived> >, Enable>
      : boost::mpl::false_
    {};
    template <typename R, typename Arg0 , typename Arg1 , typename Arg2,
        R(*F)(Arg0 , Arg1 , Arg2),
        hpx::threads::thread_priority Priority, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_result_action3<
                    R, Arg0 , Arg1 , Arg2, F, Priority> >, Enable>
      : boost::mpl::false_
    {};
    template <typename R, typename Arg0 , typename Arg1 , typename Arg2,
        R(*F)(Arg0 , Arg1 , Arg2), typename Derived, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_direct_result_action3<
                    R, Arg0 , Arg1 , Arg2, F, Derived> >, Enable>
      : boost::mpl::false_
    {};
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Result,
        typename T0 , typename T1 , typename T2 , typename T3,
        Result (*F)(T0 , T1 , T2 , T3), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default
    >
    class plain_base_result_action4
      : public action<
            components::server::plain_function<Derived>,
            function_result_action_arg4,
            Result,
            hpx::util::tuple4<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type>,
            Derived, Priority>
    {
    public:
        typedef Result result_type;
        typedef hpx::util::tuple4<
            typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type> arguments_type;
        typedef action<
            components::server::plain_function<Derived>,
            function_result_action_arg4, result_type,
            arguments_type, Derived, Priority> base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
            result_type operator()(
                HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg))) const
            {
                try {
                    LTM_(debug) << "Executing plain action("
                                << detail::get_action_name<Derived>()
                                << ").";
                    
                    
                    
                    
                    
                    F(HPX_MOVE_ARGS(2, 0, arg) , HPX_MOVE_ARGS(2, 1, arg) , HPX_MOVE_ARGS(2, 2, arg) , HPX_MOVE_ARGS(2, 3, arg));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing plain action("
                            << detail::get_action_name<Derived>()
                            << "): " << e.what();
                        
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
            return HPX_STD_BIND(typename Derived::thread_function(),
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3));
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return boost::move(
                base_type::construct_continuation_thread_function(
                    cont, F, boost::forward<Arguments>(args)));
        }
    };
    
    
    template <
        typename Result, typename T0 , typename T1 , typename T2 , typename T3,
        Result (*F)(T0 , T1 , T2 , T3),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct plain_result_action4
      : plain_base_result_action4<Result,
          T0 , T1 , T2 , T3, F,
          typename detail::action_type<
              plain_result_action4<
                  Result, T0 , T1 , T2 , T3, F, Priority>, Derived
          >::type, Priority>
    {
        typedef typename detail::action_type<
            plain_result_action4<
                Result, T0 , T1 , T2 , T3, F, Priority>, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Result, typename T0 , typename T1 , typename T2 , typename T3,
        Result (*F)(T0 , T1 , T2 , T3), typename Derived>
    struct make_action<Result (*)(T0 , T1 , T2 , T3), F, Derived, boost::mpl::false_>
      : boost::mpl::identity<plain_result_action4<Result,
            T0 , T1 , T2 , T3, F, threads::thread_priority_default,
            Derived> >
    {};
    
    
    template <
        typename Result, typename T0 , typename T1 , typename T2 , typename T3,
        Result (*F)(T0 , T1 , T2 , T3),
        typename Derived = detail::this_type>
    struct plain_direct_result_action4
      : plain_base_result_action4<Result,
          T0 , T1 , T2 , T3, F,
          typename detail::action_type<
              plain_direct_result_action4<
                  Result, T0 , T1 , T2 , T3, F>, Derived
          >::type>
    {
        typedef typename detail::action_type<
            plain_direct_result_action4<
                Result, T0 , T1 , T2 , T3, F>, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        template <typename Arguments>
        static Result
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "plain_direct_result_action" << 4
                << "::execute_function name("
                << detail::get_action_name<derived_type>() << ")";
            return F(util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3));
        }
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename Result, typename T0 , typename T1 , typename T2 , typename T3,
        Result (*F)(T0 , T1 , T2 , T3), typename Derived>
    struct make_action<Result (*)(T0 , T1 , T2 , T3), F, Derived, boost::mpl::true_>
      : boost::mpl::identity<plain_direct_result_action4<
            Result, T0 , T1 , T2 , T3, F, Derived> >
    {};
    
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3,
        void (*F)(T0 , T1 , T2 , T3), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default>
    class plain_base_action4
      : public action<
            components::server::plain_function<Derived>,
            function_action_arg4,
            util::unused_type,
            hpx::util::tuple4<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type>,
            Derived, Priority>
    {
    public:
        typedef util::unused_type result_type;
        typedef
            hpx::util::tuple4<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type>
        arguments_type;
        typedef action<
            components::server::plain_function<Derived>,
            function_action_arg4, result_type,
            arguments_type, Derived, Priority> base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
            result_type operator()(
                HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg))) const
            {
                try {
                    LTM_(debug) << "Executing plain action("
                                << detail::get_action_name<Derived>()
                                << ").";
                    
                    
                    
                    
                    
                    F(HPX_MOVE_ARGS(2, 0, arg) , HPX_MOVE_ARGS(2, 1, arg) , HPX_MOVE_ARGS(2, 2, arg) , HPX_MOVE_ARGS(2, 3, arg));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing plain action("
                            << detail::get_action_name<Derived>()
                            << "): " << e.what();
                        
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
            return HPX_STD_BIND(typename Derived::thread_function(),
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3));
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return
                base_type::construct_continuation_thread_function_void(
                    cont, F, boost::forward<Arguments>(args));
        }
    };
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3,
        void (*F)(T0 , T1 , T2 , T3),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct plain_action4
      : plain_base_action4<
            T0 , T1 , T2 , T3, F,
            typename detail::action_type<
                plain_action4<
                    T0 , T1 , T2 , T3, F, Priority>, Derived
            >::type, Priority>
    {
        typedef typename detail::action_type<
            plain_action4<
                T0 , T1 , T2 , T3, F, Priority>, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename T0 , typename T1 , typename T2 , typename T3,
        void (*F)(T0 , T1 , T2 , T3), typename Derived>
    struct make_action<void (*)(T0 , T1 , T2 , T3), F, Derived, boost::mpl::false_>
      : boost::mpl::identity<plain_action4<
            T0 , T1 , T2 , T3, F, threads::thread_priority_default,
            Derived> >
    {};
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3,
        void (*F)(T0 , T1 , T2 , T3),
        typename Derived = detail::this_type>
    struct plain_direct_action4
      : plain_base_action4<
            T0 , T1 , T2 , T3, F,
            typename detail::action_type<
                plain_direct_action4<
                    T0 , T1 , T2 , T3, F>, Derived
            >::type>
    {
        typedef typename detail::action_type<
            plain_direct_action4<
                T0 , T1 , T2 , T3, F>, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        template <typename Arguments>
        static util::unused_type
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "plain_direct_action" << 4
                << "::execute_function name("
                << detail::get_action_name<derived_type>() << ")";
            F(util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3));
            return util::unused;
        }
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename T0 , typename T1 , typename T2 , typename T3,
        void (*F)(T0 , T1 , T2 , T3), typename Derived>
    struct make_action<void (*)(T0 , T1 , T2 , T3), F, Derived, boost::mpl::true_>
      : boost::mpl::identity<plain_direct_action4<
            T0 , T1 , T2 , T3, F, Derived> >
    {};
    
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3,
        void (*F)(T0 , T1 , T2 , T3),
        threads::thread_priority Priority, typename Derived>
    struct plain_result_action4<
                void, T0 , T1 , T2 , T3, F, Priority, Derived>
      : plain_action4<
            T0 , T1 , T2 , T3, F, Priority, Derived>
    {};
}}
namespace hpx { namespace traits
{
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3,
        void (*F)(Arg0 , Arg1 , Arg2 , Arg3),
        hpx::threads::thread_priority Priority, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_action4<
                    Arg0 , Arg1 , Arg2 , Arg3, F, Priority> >, Enable>
      : boost::mpl::false_
    {};
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3,
        void (*F)(Arg0 , Arg1 , Arg2 , Arg3), typename Derived,
        typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_direct_action4<
                    Arg0 , Arg1 , Arg2 , Arg3, F, Derived> >, Enable>
      : boost::mpl::false_
    {};
    template <typename R, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3,
        R(*F)(Arg0 , Arg1 , Arg2 , Arg3),
        hpx::threads::thread_priority Priority, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_result_action4<
                    R, Arg0 , Arg1 , Arg2 , Arg3, F, Priority> >, Enable>
      : boost::mpl::false_
    {};
    template <typename R, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3,
        R(*F)(Arg0 , Arg1 , Arg2 , Arg3), typename Derived, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_direct_result_action4<
                    R, Arg0 , Arg1 , Arg2 , Arg3, F, Derived> >, Enable>
      : boost::mpl::false_
    {};
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        Result (*F)(T0 , T1 , T2 , T3 , T4), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default
    >
    class plain_base_result_action5
      : public action<
            components::server::plain_function<Derived>,
            function_result_action_arg5,
            Result,
            hpx::util::tuple5<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type>,
            Derived, Priority>
    {
    public:
        typedef Result result_type;
        typedef hpx::util::tuple5<
            typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type> arguments_type;
        typedef action<
            components::server::plain_function<Derived>,
            function_result_action_arg5, result_type,
            arguments_type, Derived, Priority> base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
            result_type operator()(
                HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg))) const
            {
                try {
                    LTM_(debug) << "Executing plain action("
                                << detail::get_action_name<Derived>()
                                << ").";
                    
                    
                    
                    
                    
                    F(HPX_MOVE_ARGS(2, 0, arg) , HPX_MOVE_ARGS(2, 1, arg) , HPX_MOVE_ARGS(2, 2, arg) , HPX_MOVE_ARGS(2, 3, arg) , HPX_MOVE_ARGS(2, 4, arg));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing plain action("
                            << detail::get_action_name<Derived>()
                            << "): " << e.what();
                        
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
            return HPX_STD_BIND(typename Derived::thread_function(),
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4));
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return boost::move(
                base_type::construct_continuation_thread_function(
                    cont, F, boost::forward<Arguments>(args)));
        }
    };
    
    
    template <
        typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        Result (*F)(T0 , T1 , T2 , T3 , T4),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct plain_result_action5
      : plain_base_result_action5<Result,
          T0 , T1 , T2 , T3 , T4, F,
          typename detail::action_type<
              plain_result_action5<
                  Result, T0 , T1 , T2 , T3 , T4, F, Priority>, Derived
          >::type, Priority>
    {
        typedef typename detail::action_type<
            plain_result_action5<
                Result, T0 , T1 , T2 , T3 , T4, F, Priority>, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        Result (*F)(T0 , T1 , T2 , T3 , T4), typename Derived>
    struct make_action<Result (*)(T0 , T1 , T2 , T3 , T4), F, Derived, boost::mpl::false_>
      : boost::mpl::identity<plain_result_action5<Result,
            T0 , T1 , T2 , T3 , T4, F, threads::thread_priority_default,
            Derived> >
    {};
    
    
    template <
        typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        Result (*F)(T0 , T1 , T2 , T3 , T4),
        typename Derived = detail::this_type>
    struct plain_direct_result_action5
      : plain_base_result_action5<Result,
          T0 , T1 , T2 , T3 , T4, F,
          typename detail::action_type<
              plain_direct_result_action5<
                  Result, T0 , T1 , T2 , T3 , T4, F>, Derived
          >::type>
    {
        typedef typename detail::action_type<
            plain_direct_result_action5<
                Result, T0 , T1 , T2 , T3 , T4, F>, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        template <typename Arguments>
        static Result
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "plain_direct_result_action" << 5
                << "::execute_function name("
                << detail::get_action_name<derived_type>() << ")";
            return F(util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4));
        }
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        Result (*F)(T0 , T1 , T2 , T3 , T4), typename Derived>
    struct make_action<Result (*)(T0 , T1 , T2 , T3 , T4), F, Derived, boost::mpl::true_>
      : boost::mpl::identity<plain_direct_result_action5<
            Result, T0 , T1 , T2 , T3 , T4, F, Derived> >
    {};
    
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        void (*F)(T0 , T1 , T2 , T3 , T4), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default>
    class plain_base_action5
      : public action<
            components::server::plain_function<Derived>,
            function_action_arg5,
            util::unused_type,
            hpx::util::tuple5<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type>,
            Derived, Priority>
    {
    public:
        typedef util::unused_type result_type;
        typedef
            hpx::util::tuple5<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type>
        arguments_type;
        typedef action<
            components::server::plain_function<Derived>,
            function_action_arg5, result_type,
            arguments_type, Derived, Priority> base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
            result_type operator()(
                HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg))) const
            {
                try {
                    LTM_(debug) << "Executing plain action("
                                << detail::get_action_name<Derived>()
                                << ").";
                    
                    
                    
                    
                    
                    F(HPX_MOVE_ARGS(2, 0, arg) , HPX_MOVE_ARGS(2, 1, arg) , HPX_MOVE_ARGS(2, 2, arg) , HPX_MOVE_ARGS(2, 3, arg) , HPX_MOVE_ARGS(2, 4, arg));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing plain action("
                            << detail::get_action_name<Derived>()
                            << "): " << e.what();
                        
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
            return HPX_STD_BIND(typename Derived::thread_function(),
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4));
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return
                base_type::construct_continuation_thread_function_void(
                    cont, F, boost::forward<Arguments>(args));
        }
    };
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        void (*F)(T0 , T1 , T2 , T3 , T4),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct plain_action5
      : plain_base_action5<
            T0 , T1 , T2 , T3 , T4, F,
            typename detail::action_type<
                plain_action5<
                    T0 , T1 , T2 , T3 , T4, F, Priority>, Derived
            >::type, Priority>
    {
        typedef typename detail::action_type<
            plain_action5<
                T0 , T1 , T2 , T3 , T4, F, Priority>, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        void (*F)(T0 , T1 , T2 , T3 , T4), typename Derived>
    struct make_action<void (*)(T0 , T1 , T2 , T3 , T4), F, Derived, boost::mpl::false_>
      : boost::mpl::identity<plain_action5<
            T0 , T1 , T2 , T3 , T4, F, threads::thread_priority_default,
            Derived> >
    {};
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        void (*F)(T0 , T1 , T2 , T3 , T4),
        typename Derived = detail::this_type>
    struct plain_direct_action5
      : plain_base_action5<
            T0 , T1 , T2 , T3 , T4, F,
            typename detail::action_type<
                plain_direct_action5<
                    T0 , T1 , T2 , T3 , T4, F>, Derived
            >::type>
    {
        typedef typename detail::action_type<
            plain_direct_action5<
                T0 , T1 , T2 , T3 , T4, F>, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        template <typename Arguments>
        static util::unused_type
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "plain_direct_action" << 5
                << "::execute_function name("
                << detail::get_action_name<derived_type>() << ")";
            F(util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4));
            return util::unused;
        }
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        void (*F)(T0 , T1 , T2 , T3 , T4), typename Derived>
    struct make_action<void (*)(T0 , T1 , T2 , T3 , T4), F, Derived, boost::mpl::true_>
      : boost::mpl::identity<plain_direct_action5<
            T0 , T1 , T2 , T3 , T4, F, Derived> >
    {};
    
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        void (*F)(T0 , T1 , T2 , T3 , T4),
        threads::thread_priority Priority, typename Derived>
    struct plain_result_action5<
                void, T0 , T1 , T2 , T3 , T4, F, Priority, Derived>
      : plain_action5<
            T0 , T1 , T2 , T3 , T4, F, Priority, Derived>
    {};
}}
namespace hpx { namespace traits
{
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4,
        void (*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4),
        hpx::threads::thread_priority Priority, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_action5<
                    Arg0 , Arg1 , Arg2 , Arg3 , Arg4, F, Priority> >, Enable>
      : boost::mpl::false_
    {};
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4,
        void (*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4), typename Derived,
        typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_direct_action5<
                    Arg0 , Arg1 , Arg2 , Arg3 , Arg4, F, Derived> >, Enable>
      : boost::mpl::false_
    {};
    template <typename R, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4,
        R(*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4),
        hpx::threads::thread_priority Priority, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_result_action5<
                    R, Arg0 , Arg1 , Arg2 , Arg3 , Arg4, F, Priority> >, Enable>
      : boost::mpl::false_
    {};
    template <typename R, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4,
        R(*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4), typename Derived, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_direct_result_action5<
                    R, Arg0 , Arg1 , Arg2 , Arg3 , Arg4, F, Derived> >, Enable>
      : boost::mpl::false_
    {};
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default
    >
    class plain_base_result_action6
      : public action<
            components::server::plain_function<Derived>,
            function_result_action_arg6,
            Result,
            hpx::util::tuple6<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type>,
            Derived, Priority>
    {
    public:
        typedef Result result_type;
        typedef hpx::util::tuple6<
            typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type> arguments_type;
        typedef action<
            components::server::plain_function<Derived>,
            function_result_action_arg6, result_type,
            arguments_type, Derived, Priority> base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
            result_type operator()(
                HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg))) const
            {
                try {
                    LTM_(debug) << "Executing plain action("
                                << detail::get_action_name<Derived>()
                                << ").";
                    
                    
                    
                    
                    
                    F(HPX_MOVE_ARGS(2, 0, arg) , HPX_MOVE_ARGS(2, 1, arg) , HPX_MOVE_ARGS(2, 2, arg) , HPX_MOVE_ARGS(2, 3, arg) , HPX_MOVE_ARGS(2, 4, arg) , HPX_MOVE_ARGS(2, 5, arg));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing plain action("
                            << detail::get_action_name<Derived>()
                            << "): " << e.what();
                        
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
            return HPX_STD_BIND(typename Derived::thread_function(),
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5));
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return boost::move(
                base_type::construct_continuation_thread_function(
                    cont, F, boost::forward<Arguments>(args)));
        }
    };
    
    
    template <
        typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct plain_result_action6
      : plain_base_result_action6<Result,
          T0 , T1 , T2 , T3 , T4 , T5, F,
          typename detail::action_type<
              plain_result_action6<
                  Result, T0 , T1 , T2 , T3 , T4 , T5, F, Priority>, Derived
          >::type, Priority>
    {
        typedef typename detail::action_type<
            plain_result_action6<
                Result, T0 , T1 , T2 , T3 , T4 , T5, F, Priority>, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5), typename Derived>
    struct make_action<Result (*)(T0 , T1 , T2 , T3 , T4 , T5), F, Derived, boost::mpl::false_>
      : boost::mpl::identity<plain_result_action6<Result,
            T0 , T1 , T2 , T3 , T4 , T5, F, threads::thread_priority_default,
            Derived> >
    {};
    
    
    template <
        typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5),
        typename Derived = detail::this_type>
    struct plain_direct_result_action6
      : plain_base_result_action6<Result,
          T0 , T1 , T2 , T3 , T4 , T5, F,
          typename detail::action_type<
              plain_direct_result_action6<
                  Result, T0 , T1 , T2 , T3 , T4 , T5, F>, Derived
          >::type>
    {
        typedef typename detail::action_type<
            plain_direct_result_action6<
                Result, T0 , T1 , T2 , T3 , T4 , T5, F>, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        template <typename Arguments>
        static Result
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "plain_direct_result_action" << 6
                << "::execute_function name("
                << detail::get_action_name<derived_type>() << ")";
            return F(util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5));
        }
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5), typename Derived>
    struct make_action<Result (*)(T0 , T1 , T2 , T3 , T4 , T5), F, Derived, boost::mpl::true_>
      : boost::mpl::identity<plain_direct_result_action6<
            Result, T0 , T1 , T2 , T3 , T4 , T5, F, Derived> >
    {};
    
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default>
    class plain_base_action6
      : public action<
            components::server::plain_function<Derived>,
            function_action_arg6,
            util::unused_type,
            hpx::util::tuple6<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type>,
            Derived, Priority>
    {
    public:
        typedef util::unused_type result_type;
        typedef
            hpx::util::tuple6<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type>
        arguments_type;
        typedef action<
            components::server::plain_function<Derived>,
            function_action_arg6, result_type,
            arguments_type, Derived, Priority> base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
            result_type operator()(
                HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg))) const
            {
                try {
                    LTM_(debug) << "Executing plain action("
                                << detail::get_action_name<Derived>()
                                << ").";
                    
                    
                    
                    
                    
                    F(HPX_MOVE_ARGS(2, 0, arg) , HPX_MOVE_ARGS(2, 1, arg) , HPX_MOVE_ARGS(2, 2, arg) , HPX_MOVE_ARGS(2, 3, arg) , HPX_MOVE_ARGS(2, 4, arg) , HPX_MOVE_ARGS(2, 5, arg));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing plain action("
                            << detail::get_action_name<Derived>()
                            << "): " << e.what();
                        
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
            return HPX_STD_BIND(typename Derived::thread_function(),
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5));
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return
                base_type::construct_continuation_thread_function_void(
                    cont, F, boost::forward<Arguments>(args));
        }
    };
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct plain_action6
      : plain_base_action6<
            T0 , T1 , T2 , T3 , T4 , T5, F,
            typename detail::action_type<
                plain_action6<
                    T0 , T1 , T2 , T3 , T4 , T5, F, Priority>, Derived
            >::type, Priority>
    {
        typedef typename detail::action_type<
            plain_action6<
                T0 , T1 , T2 , T3 , T4 , T5, F, Priority>, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5), typename Derived>
    struct make_action<void (*)(T0 , T1 , T2 , T3 , T4 , T5), F, Derived, boost::mpl::false_>
      : boost::mpl::identity<plain_action6<
            T0 , T1 , T2 , T3 , T4 , T5, F, threads::thread_priority_default,
            Derived> >
    {};
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5),
        typename Derived = detail::this_type>
    struct plain_direct_action6
      : plain_base_action6<
            T0 , T1 , T2 , T3 , T4 , T5, F,
            typename detail::action_type<
                plain_direct_action6<
                    T0 , T1 , T2 , T3 , T4 , T5, F>, Derived
            >::type>
    {
        typedef typename detail::action_type<
            plain_direct_action6<
                T0 , T1 , T2 , T3 , T4 , T5, F>, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        template <typename Arguments>
        static util::unused_type
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "plain_direct_action" << 6
                << "::execute_function name("
                << detail::get_action_name<derived_type>() << ")";
            F(util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5));
            return util::unused;
        }
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5), typename Derived>
    struct make_action<void (*)(T0 , T1 , T2 , T3 , T4 , T5), F, Derived, boost::mpl::true_>
      : boost::mpl::identity<plain_direct_action6<
            T0 , T1 , T2 , T3 , T4 , T5, F, Derived> >
    {};
    
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5),
        threads::thread_priority Priority, typename Derived>
    struct plain_result_action6<
                void, T0 , T1 , T2 , T3 , T4 , T5, F, Priority, Derived>
      : plain_action6<
            T0 , T1 , T2 , T3 , T4 , T5, F, Priority, Derived>
    {};
}}
namespace hpx { namespace traits
{
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5,
        void (*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5),
        hpx::threads::thread_priority Priority, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_action6<
                    Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5, F, Priority> >, Enable>
      : boost::mpl::false_
    {};
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5,
        void (*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5), typename Derived,
        typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_direct_action6<
                    Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5, F, Derived> >, Enable>
      : boost::mpl::false_
    {};
    template <typename R, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5,
        R(*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5),
        hpx::threads::thread_priority Priority, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_result_action6<
                    R, Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5, F, Priority> >, Enable>
      : boost::mpl::false_
    {};
    template <typename R, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5,
        R(*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5), typename Derived, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_direct_result_action6<
                    R, Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5, F, Derived> >, Enable>
      : boost::mpl::false_
    {};
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default
    >
    class plain_base_result_action7
      : public action<
            components::server::plain_function<Derived>,
            function_result_action_arg7,
            Result,
            hpx::util::tuple7<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type>,
            Derived, Priority>
    {
    public:
        typedef Result result_type;
        typedef hpx::util::tuple7<
            typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type> arguments_type;
        typedef action<
            components::server::plain_function<Derived>,
            function_result_action_arg7, result_type,
            arguments_type, Derived, Priority> base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
            result_type operator()(
                HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg))) const
            {
                try {
                    LTM_(debug) << "Executing plain action("
                                << detail::get_action_name<Derived>()
                                << ").";
                    
                    
                    
                    
                    
                    F(HPX_MOVE_ARGS(2, 0, arg) , HPX_MOVE_ARGS(2, 1, arg) , HPX_MOVE_ARGS(2, 2, arg) , HPX_MOVE_ARGS(2, 3, arg) , HPX_MOVE_ARGS(2, 4, arg) , HPX_MOVE_ARGS(2, 5, arg) , HPX_MOVE_ARGS(2, 6, arg));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing plain action("
                            << detail::get_action_name<Derived>()
                            << "): " << e.what();
                        
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
            return HPX_STD_BIND(typename Derived::thread_function(),
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6));
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return boost::move(
                base_type::construct_continuation_thread_function(
                    cont, F, boost::forward<Arguments>(args)));
        }
    };
    
    
    template <
        typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct plain_result_action7
      : plain_base_result_action7<Result,
          T0 , T1 , T2 , T3 , T4 , T5 , T6, F,
          typename detail::action_type<
              plain_result_action7<
                  Result, T0 , T1 , T2 , T3 , T4 , T5 , T6, F, Priority>, Derived
          >::type, Priority>
    {
        typedef typename detail::action_type<
            plain_result_action7<
                Result, T0 , T1 , T2 , T3 , T4 , T5 , T6, F, Priority>, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6), typename Derived>
    struct make_action<Result (*)(T0 , T1 , T2 , T3 , T4 , T5 , T6), F, Derived, boost::mpl::false_>
      : boost::mpl::identity<plain_result_action7<Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6, F, threads::thread_priority_default,
            Derived> >
    {};
    
    
    template <
        typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6),
        typename Derived = detail::this_type>
    struct plain_direct_result_action7
      : plain_base_result_action7<Result,
          T0 , T1 , T2 , T3 , T4 , T5 , T6, F,
          typename detail::action_type<
              plain_direct_result_action7<
                  Result, T0 , T1 , T2 , T3 , T4 , T5 , T6, F>, Derived
          >::type>
    {
        typedef typename detail::action_type<
            plain_direct_result_action7<
                Result, T0 , T1 , T2 , T3 , T4 , T5 , T6, F>, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        template <typename Arguments>
        static Result
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "plain_direct_result_action" << 7
                << "::execute_function name("
                << detail::get_action_name<derived_type>() << ")";
            return F(util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6));
        }
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6), typename Derived>
    struct make_action<Result (*)(T0 , T1 , T2 , T3 , T4 , T5 , T6), F, Derived, boost::mpl::true_>
      : boost::mpl::identity<plain_direct_result_action7<
            Result, T0 , T1 , T2 , T3 , T4 , T5 , T6, F, Derived> >
    {};
    
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default>
    class plain_base_action7
      : public action<
            components::server::plain_function<Derived>,
            function_action_arg7,
            util::unused_type,
            hpx::util::tuple7<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type>,
            Derived, Priority>
    {
    public:
        typedef util::unused_type result_type;
        typedef
            hpx::util::tuple7<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type>
        arguments_type;
        typedef action<
            components::server::plain_function<Derived>,
            function_action_arg7, result_type,
            arguments_type, Derived, Priority> base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
            result_type operator()(
                HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg))) const
            {
                try {
                    LTM_(debug) << "Executing plain action("
                                << detail::get_action_name<Derived>()
                                << ").";
                    
                    
                    
                    
                    
                    F(HPX_MOVE_ARGS(2, 0, arg) , HPX_MOVE_ARGS(2, 1, arg) , HPX_MOVE_ARGS(2, 2, arg) , HPX_MOVE_ARGS(2, 3, arg) , HPX_MOVE_ARGS(2, 4, arg) , HPX_MOVE_ARGS(2, 5, arg) , HPX_MOVE_ARGS(2, 6, arg));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing plain action("
                            << detail::get_action_name<Derived>()
                            << "): " << e.what();
                        
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
            return HPX_STD_BIND(typename Derived::thread_function(),
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6));
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return
                base_type::construct_continuation_thread_function_void(
                    cont, F, boost::forward<Arguments>(args));
        }
    };
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct plain_action7
      : plain_base_action7<
            T0 , T1 , T2 , T3 , T4 , T5 , T6, F,
            typename detail::action_type<
                plain_action7<
                    T0 , T1 , T2 , T3 , T4 , T5 , T6, F, Priority>, Derived
            >::type, Priority>
    {
        typedef typename detail::action_type<
            plain_action7<
                T0 , T1 , T2 , T3 , T4 , T5 , T6, F, Priority>, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6), typename Derived>
    struct make_action<void (*)(T0 , T1 , T2 , T3 , T4 , T5 , T6), F, Derived, boost::mpl::false_>
      : boost::mpl::identity<plain_action7<
            T0 , T1 , T2 , T3 , T4 , T5 , T6, F, threads::thread_priority_default,
            Derived> >
    {};
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6),
        typename Derived = detail::this_type>
    struct plain_direct_action7
      : plain_base_action7<
            T0 , T1 , T2 , T3 , T4 , T5 , T6, F,
            typename detail::action_type<
                plain_direct_action7<
                    T0 , T1 , T2 , T3 , T4 , T5 , T6, F>, Derived
            >::type>
    {
        typedef typename detail::action_type<
            plain_direct_action7<
                T0 , T1 , T2 , T3 , T4 , T5 , T6, F>, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        template <typename Arguments>
        static util::unused_type
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "plain_direct_action" << 7
                << "::execute_function name("
                << detail::get_action_name<derived_type>() << ")";
            F(util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6));
            return util::unused;
        }
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6), typename Derived>
    struct make_action<void (*)(T0 , T1 , T2 , T3 , T4 , T5 , T6), F, Derived, boost::mpl::true_>
      : boost::mpl::identity<plain_direct_action7<
            T0 , T1 , T2 , T3 , T4 , T5 , T6, F, Derived> >
    {};
    
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6),
        threads::thread_priority Priority, typename Derived>
    struct plain_result_action7<
                void, T0 , T1 , T2 , T3 , T4 , T5 , T6, F, Priority, Derived>
      : plain_action7<
            T0 , T1 , T2 , T3 , T4 , T5 , T6, F, Priority, Derived>
    {};
}}
namespace hpx { namespace traits
{
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6,
        void (*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6),
        hpx::threads::thread_priority Priority, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_action7<
                    Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6, F, Priority> >, Enable>
      : boost::mpl::false_
    {};
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6,
        void (*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6), typename Derived,
        typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_direct_action7<
                    Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6, F, Derived> >, Enable>
      : boost::mpl::false_
    {};
    template <typename R, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6,
        R(*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6),
        hpx::threads::thread_priority Priority, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_result_action7<
                    R, Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6, F, Priority> >, Enable>
      : boost::mpl::false_
    {};
    template <typename R, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6,
        R(*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6), typename Derived, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_direct_result_action7<
                    R, Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6, F, Derived> >, Enable>
      : boost::mpl::false_
    {};
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default
    >
    class plain_base_result_action8
      : public action<
            components::server::plain_function<Derived>,
            function_result_action_arg8,
            Result,
            hpx::util::tuple8<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type>,
            Derived, Priority>
    {
    public:
        typedef Result result_type;
        typedef hpx::util::tuple8<
            typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type> arguments_type;
        typedef action<
            components::server::plain_function<Derived>,
            function_result_action_arg8, result_type,
            arguments_type, Derived, Priority> base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
            result_type operator()(
                HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg))) const
            {
                try {
                    LTM_(debug) << "Executing plain action("
                                << detail::get_action_name<Derived>()
                                << ").";
                    
                    
                    
                    
                    
                    F(HPX_MOVE_ARGS(2, 0, arg) , HPX_MOVE_ARGS(2, 1, arg) , HPX_MOVE_ARGS(2, 2, arg) , HPX_MOVE_ARGS(2, 3, arg) , HPX_MOVE_ARGS(2, 4, arg) , HPX_MOVE_ARGS(2, 5, arg) , HPX_MOVE_ARGS(2, 6, arg) , HPX_MOVE_ARGS(2, 7, arg));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing plain action("
                            << detail::get_action_name<Derived>()
                            << "): " << e.what();
                        
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
            return HPX_STD_BIND(typename Derived::thread_function(),
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7));
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return boost::move(
                base_type::construct_continuation_thread_function(
                    cont, F, boost::forward<Arguments>(args)));
        }
    };
    
    
    template <
        typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct plain_result_action8
      : plain_base_result_action8<Result,
          T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F,
          typename detail::action_type<
              plain_result_action8<
                  Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F, Priority>, Derived
          >::type, Priority>
    {
        typedef typename detail::action_type<
            plain_result_action8<
                Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F, Priority>, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7), typename Derived>
    struct make_action<Result (*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7), F, Derived, boost::mpl::false_>
      : boost::mpl::identity<plain_result_action8<Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F, threads::thread_priority_default,
            Derived> >
    {};
    
    
    template <
        typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7),
        typename Derived = detail::this_type>
    struct plain_direct_result_action8
      : plain_base_result_action8<Result,
          T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F,
          typename detail::action_type<
              plain_direct_result_action8<
                  Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F>, Derived
          >::type>
    {
        typedef typename detail::action_type<
            plain_direct_result_action8<
                Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F>, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        template <typename Arguments>
        static Result
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "plain_direct_result_action" << 8
                << "::execute_function name("
                << detail::get_action_name<derived_type>() << ")";
            return F(util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7));
        }
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7), typename Derived>
    struct make_action<Result (*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7), F, Derived, boost::mpl::true_>
      : boost::mpl::identity<plain_direct_result_action8<
            Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F, Derived> >
    {};
    
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default>
    class plain_base_action8
      : public action<
            components::server::plain_function<Derived>,
            function_action_arg8,
            util::unused_type,
            hpx::util::tuple8<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type>,
            Derived, Priority>
    {
    public:
        typedef util::unused_type result_type;
        typedef
            hpx::util::tuple8<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type>
        arguments_type;
        typedef action<
            components::server::plain_function<Derived>,
            function_action_arg8, result_type,
            arguments_type, Derived, Priority> base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
            result_type operator()(
                HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg))) const
            {
                try {
                    LTM_(debug) << "Executing plain action("
                                << detail::get_action_name<Derived>()
                                << ").";
                    
                    
                    
                    
                    
                    F(HPX_MOVE_ARGS(2, 0, arg) , HPX_MOVE_ARGS(2, 1, arg) , HPX_MOVE_ARGS(2, 2, arg) , HPX_MOVE_ARGS(2, 3, arg) , HPX_MOVE_ARGS(2, 4, arg) , HPX_MOVE_ARGS(2, 5, arg) , HPX_MOVE_ARGS(2, 6, arg) , HPX_MOVE_ARGS(2, 7, arg));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing plain action("
                            << detail::get_action_name<Derived>()
                            << "): " << e.what();
                        
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
            return HPX_STD_BIND(typename Derived::thread_function(),
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7));
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return
                base_type::construct_continuation_thread_function_void(
                    cont, F, boost::forward<Arguments>(args));
        }
    };
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct plain_action8
      : plain_base_action8<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F,
            typename detail::action_type<
                plain_action8<
                    T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F, Priority>, Derived
            >::type, Priority>
    {
        typedef typename detail::action_type<
            plain_action8<
                T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F, Priority>, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7), typename Derived>
    struct make_action<void (*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7), F, Derived, boost::mpl::false_>
      : boost::mpl::identity<plain_action8<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F, threads::thread_priority_default,
            Derived> >
    {};
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7),
        typename Derived = detail::this_type>
    struct plain_direct_action8
      : plain_base_action8<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F,
            typename detail::action_type<
                plain_direct_action8<
                    T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F>, Derived
            >::type>
    {
        typedef typename detail::action_type<
            plain_direct_action8<
                T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F>, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        template <typename Arguments>
        static util::unused_type
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "plain_direct_action" << 8
                << "::execute_function name("
                << detail::get_action_name<derived_type>() << ")";
            F(util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7));
            return util::unused;
        }
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7), typename Derived>
    struct make_action<void (*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7), F, Derived, boost::mpl::true_>
      : boost::mpl::identity<plain_direct_action8<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F, Derived> >
    {};
    
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7),
        threads::thread_priority Priority, typename Derived>
    struct plain_result_action8<
                void, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F, Priority, Derived>
      : plain_action8<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, F, Priority, Derived>
    {};
}}
namespace hpx { namespace traits
{
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7,
        void (*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7),
        hpx::threads::thread_priority Priority, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_action8<
                    Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7, F, Priority> >, Enable>
      : boost::mpl::false_
    {};
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7,
        void (*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7), typename Derived,
        typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_direct_action8<
                    Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7, F, Derived> >, Enable>
      : boost::mpl::false_
    {};
    template <typename R, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7,
        R(*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7),
        hpx::threads::thread_priority Priority, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_result_action8<
                    R, Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7, F, Priority> >, Enable>
      : boost::mpl::false_
    {};
    template <typename R, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7,
        R(*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7), typename Derived, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_direct_result_action8<
                    R, Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7, F, Derived> >, Enable>
      : boost::mpl::false_
    {};
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default
    >
    class plain_base_result_action9
      : public action<
            components::server::plain_function<Derived>,
            function_result_action_arg9,
            Result,
            hpx::util::tuple9<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type , typename detail::remove_qualifiers<T8>::type>,
            Derived, Priority>
    {
    public:
        typedef Result result_type;
        typedef hpx::util::tuple9<
            typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type , typename detail::remove_qualifiers<T8>::type> arguments_type;
        typedef action<
            components::server::plain_function<Derived>,
            function_result_action_arg9, result_type,
            arguments_type, Derived, Priority> base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
            result_type operator()(
                HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg))) const
            {
                try {
                    LTM_(debug) << "Executing plain action("
                                << detail::get_action_name<Derived>()
                                << ").";
                    
                    
                    
                    
                    
                    F(HPX_MOVE_ARGS(2, 0, arg) , HPX_MOVE_ARGS(2, 1, arg) , HPX_MOVE_ARGS(2, 2, arg) , HPX_MOVE_ARGS(2, 3, arg) , HPX_MOVE_ARGS(2, 4, arg) , HPX_MOVE_ARGS(2, 5, arg) , HPX_MOVE_ARGS(2, 6, arg) , HPX_MOVE_ARGS(2, 7, arg) , HPX_MOVE_ARGS(2, 8, arg));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing plain action("
                            << detail::get_action_name<Derived>()
                            << "): " << e.what();
                        
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
            return HPX_STD_BIND(typename Derived::thread_function(),
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type8>::call( args. a8));
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return boost::move(
                base_type::construct_continuation_thread_function(
                    cont, F, boost::forward<Arguments>(args)));
        }
    };
    
    
    template <
        typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct plain_result_action9
      : plain_base_result_action9<Result,
          T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F,
          typename detail::action_type<
              plain_result_action9<
                  Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F, Priority>, Derived
          >::type, Priority>
    {
        typedef typename detail::action_type<
            plain_result_action9<
                Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F, Priority>, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8), typename Derived>
    struct make_action<Result (*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8), F, Derived, boost::mpl::false_>
      : boost::mpl::identity<plain_result_action9<Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F, threads::thread_priority_default,
            Derived> >
    {};
    
    
    template <
        typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8),
        typename Derived = detail::this_type>
    struct plain_direct_result_action9
      : plain_base_result_action9<Result,
          T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F,
          typename detail::action_type<
              plain_direct_result_action9<
                  Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F>, Derived
          >::type>
    {
        typedef typename detail::action_type<
            plain_direct_result_action9<
                Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F>, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        template <typename Arguments>
        static Result
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "plain_direct_result_action" << 9
                << "::execute_function name("
                << detail::get_action_name<derived_type>() << ")";
            return F(util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type8>::call( args. a8));
        }
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8), typename Derived>
    struct make_action<Result (*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8), F, Derived, boost::mpl::true_>
      : boost::mpl::identity<plain_direct_result_action9<
            Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F, Derived> >
    {};
    
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default>
    class plain_base_action9
      : public action<
            components::server::plain_function<Derived>,
            function_action_arg9,
            util::unused_type,
            hpx::util::tuple9<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type , typename detail::remove_qualifiers<T8>::type>,
            Derived, Priority>
    {
    public:
        typedef util::unused_type result_type;
        typedef
            hpx::util::tuple9<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type , typename detail::remove_qualifiers<T8>::type>
        arguments_type;
        typedef action<
            components::server::plain_function<Derived>,
            function_action_arg9, result_type,
            arguments_type, Derived, Priority> base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
            result_type operator()(
                HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg))) const
            {
                try {
                    LTM_(debug) << "Executing plain action("
                                << detail::get_action_name<Derived>()
                                << ").";
                    
                    
                    
                    
                    
                    F(HPX_MOVE_ARGS(2, 0, arg) , HPX_MOVE_ARGS(2, 1, arg) , HPX_MOVE_ARGS(2, 2, arg) , HPX_MOVE_ARGS(2, 3, arg) , HPX_MOVE_ARGS(2, 4, arg) , HPX_MOVE_ARGS(2, 5, arg) , HPX_MOVE_ARGS(2, 6, arg) , HPX_MOVE_ARGS(2, 7, arg) , HPX_MOVE_ARGS(2, 8, arg));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing plain action("
                            << detail::get_action_name<Derived>()
                            << "): " << e.what();
                        
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
            return HPX_STD_BIND(typename Derived::thread_function(),
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type8>::call( args. a8));
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return
                base_type::construct_continuation_thread_function_void(
                    cont, F, boost::forward<Arguments>(args));
        }
    };
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct plain_action9
      : plain_base_action9<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F,
            typename detail::action_type<
                plain_action9<
                    T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F, Priority>, Derived
            >::type, Priority>
    {
        typedef typename detail::action_type<
            plain_action9<
                T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F, Priority>, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8), typename Derived>
    struct make_action<void (*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8), F, Derived, boost::mpl::false_>
      : boost::mpl::identity<plain_action9<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F, threads::thread_priority_default,
            Derived> >
    {};
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8),
        typename Derived = detail::this_type>
    struct plain_direct_action9
      : plain_base_action9<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F,
            typename detail::action_type<
                plain_direct_action9<
                    T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F>, Derived
            >::type>
    {
        typedef typename detail::action_type<
            plain_direct_action9<
                T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F>, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        template <typename Arguments>
        static util::unused_type
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "plain_direct_action" << 9
                << "::execute_function name("
                << detail::get_action_name<derived_type>() << ")";
            F(util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type8>::call( args. a8));
            return util::unused;
        }
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8), typename Derived>
    struct make_action<void (*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8), F, Derived, boost::mpl::true_>
      : boost::mpl::identity<plain_direct_action9<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F, Derived> >
    {};
    
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8),
        threads::thread_priority Priority, typename Derived>
    struct plain_result_action9<
                void, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F, Priority, Derived>
      : plain_action9<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, F, Priority, Derived>
    {};
}}
namespace hpx { namespace traits
{
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8,
        void (*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8),
        hpx::threads::thread_priority Priority, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_action9<
                    Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8, F, Priority> >, Enable>
      : boost::mpl::false_
    {};
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8,
        void (*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8), typename Derived,
        typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_direct_action9<
                    Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8, F, Derived> >, Enable>
      : boost::mpl::false_
    {};
    template <typename R, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8,
        R(*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8),
        hpx::threads::thread_priority Priority, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_result_action9<
                    R, Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8, F, Priority> >, Enable>
      : boost::mpl::false_
    {};
    template <typename R, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8,
        R(*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8), typename Derived, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_direct_result_action9<
                    R, Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8, F, Derived> >, Enable>
      : boost::mpl::false_
    {};
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default
    >
    class plain_base_result_action10
      : public action<
            components::server::plain_function<Derived>,
            function_result_action_arg10,
            Result,
            hpx::util::tuple10<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type , typename detail::remove_qualifiers<T8>::type , typename detail::remove_qualifiers<T9>::type>,
            Derived, Priority>
    {
    public:
        typedef Result result_type;
        typedef hpx::util::tuple10<
            typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type , typename detail::remove_qualifiers<T8>::type , typename detail::remove_qualifiers<T9>::type> arguments_type;
        typedef action<
            components::server::plain_function<Derived>,
            function_result_action_arg10, result_type,
            arguments_type, Derived, Priority> base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
            result_type operator()(
                HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg))) const
            {
                try {
                    LTM_(debug) << "Executing plain action("
                                << detail::get_action_name<Derived>()
                                << ").";
                    
                    
                    
                    
                    
                    F(HPX_MOVE_ARGS(2, 0, arg) , HPX_MOVE_ARGS(2, 1, arg) , HPX_MOVE_ARGS(2, 2, arg) , HPX_MOVE_ARGS(2, 3, arg) , HPX_MOVE_ARGS(2, 4, arg) , HPX_MOVE_ARGS(2, 5, arg) , HPX_MOVE_ARGS(2, 6, arg) , HPX_MOVE_ARGS(2, 7, arg) , HPX_MOVE_ARGS(2, 8, arg) , HPX_MOVE_ARGS(2, 9, arg));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing plain action("
                            << detail::get_action_name<Derived>()
                            << "): " << e.what();
                        
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
            return HPX_STD_BIND(typename Derived::thread_function(),
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type9>::call( args. a9));
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return boost::move(
                base_type::construct_continuation_thread_function(
                    cont, F, boost::forward<Arguments>(args)));
        }
    };
    
    
    template <
        typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct plain_result_action10
      : plain_base_result_action10<Result,
          T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F,
          typename detail::action_type<
              plain_result_action10<
                  Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F, Priority>, Derived
          >::type, Priority>
    {
        typedef typename detail::action_type<
            plain_result_action10<
                Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F, Priority>, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9), typename Derived>
    struct make_action<Result (*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9), F, Derived, boost::mpl::false_>
      : boost::mpl::identity<plain_result_action10<Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F, threads::thread_priority_default,
            Derived> >
    {};
    
    
    template <
        typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9),
        typename Derived = detail::this_type>
    struct plain_direct_result_action10
      : plain_base_result_action10<Result,
          T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F,
          typename detail::action_type<
              plain_direct_result_action10<
                  Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F>, Derived
          >::type>
    {
        typedef typename detail::action_type<
            plain_direct_result_action10<
                Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F>, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        template <typename Arguments>
        static Result
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "plain_direct_result_action" << 10
                << "::execute_function name("
                << detail::get_action_name<derived_type>() << ")";
            return F(util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type9>::call( args. a9));
        }
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9), typename Derived>
    struct make_action<Result (*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9), F, Derived, boost::mpl::true_>
      : boost::mpl::identity<plain_direct_result_action10<
            Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F, Derived> >
    {};
    
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default>
    class plain_base_action10
      : public action<
            components::server::plain_function<Derived>,
            function_action_arg10,
            util::unused_type,
            hpx::util::tuple10<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type , typename detail::remove_qualifiers<T8>::type , typename detail::remove_qualifiers<T9>::type>,
            Derived, Priority>
    {
    public:
        typedef util::unused_type result_type;
        typedef
            hpx::util::tuple10<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type , typename detail::remove_qualifiers<T8>::type , typename detail::remove_qualifiers<T9>::type>
        arguments_type;
        typedef action<
            components::server::plain_function<Derived>,
            function_action_arg10, result_type,
            arguments_type, Derived, Priority> base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
            result_type operator()(
                HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg))) const
            {
                try {
                    LTM_(debug) << "Executing plain action("
                                << detail::get_action_name<Derived>()
                                << ").";
                    
                    
                    
                    
                    
                    F(HPX_MOVE_ARGS(2, 0, arg) , HPX_MOVE_ARGS(2, 1, arg) , HPX_MOVE_ARGS(2, 2, arg) , HPX_MOVE_ARGS(2, 3, arg) , HPX_MOVE_ARGS(2, 4, arg) , HPX_MOVE_ARGS(2, 5, arg) , HPX_MOVE_ARGS(2, 6, arg) , HPX_MOVE_ARGS(2, 7, arg) , HPX_MOVE_ARGS(2, 8, arg) , HPX_MOVE_ARGS(2, 9, arg));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing plain action("
                            << detail::get_action_name<Derived>()
                            << "): " << e.what();
                        
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
            return HPX_STD_BIND(typename Derived::thread_function(),
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type9>::call( args. a9));
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return
                base_type::construct_continuation_thread_function_void(
                    cont, F, boost::forward<Arguments>(args));
        }
    };
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct plain_action10
      : plain_base_action10<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F,
            typename detail::action_type<
                plain_action10<
                    T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F, Priority>, Derived
            >::type, Priority>
    {
        typedef typename detail::action_type<
            plain_action10<
                T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F, Priority>, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9), typename Derived>
    struct make_action<void (*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9), F, Derived, boost::mpl::false_>
      : boost::mpl::identity<plain_action10<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F, threads::thread_priority_default,
            Derived> >
    {};
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9),
        typename Derived = detail::this_type>
    struct plain_direct_action10
      : plain_base_action10<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F,
            typename detail::action_type<
                plain_direct_action10<
                    T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F>, Derived
            >::type>
    {
        typedef typename detail::action_type<
            plain_direct_action10<
                T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F>, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        template <typename Arguments>
        static util::unused_type
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "plain_direct_action" << 10
                << "::execute_function name("
                << detail::get_action_name<derived_type>() << ")";
            F(util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type9>::call( args. a9));
            return util::unused;
        }
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9), typename Derived>
    struct make_action<void (*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9), F, Derived, boost::mpl::true_>
      : boost::mpl::identity<plain_direct_action10<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F, Derived> >
    {};
    
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9),
        threads::thread_priority Priority, typename Derived>
    struct plain_result_action10<
                void, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F, Priority, Derived>
      : plain_action10<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, F, Priority, Derived>
    {};
}}
namespace hpx { namespace traits
{
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9,
        void (*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9),
        hpx::threads::thread_priority Priority, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_action10<
                    Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9, F, Priority> >, Enable>
      : boost::mpl::false_
    {};
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9,
        void (*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9), typename Derived,
        typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_direct_action10<
                    Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9, F, Derived> >, Enable>
      : boost::mpl::false_
    {};
    template <typename R, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9,
        R(*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9),
        hpx::threads::thread_priority Priority, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_result_action10<
                    R, Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9, F, Priority> >, Enable>
      : boost::mpl::false_
    {};
    template <typename R, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9,
        R(*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9), typename Derived, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_direct_result_action10<
                    R, Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9, F, Derived> >, Enable>
      : boost::mpl::false_
    {};
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default
    >
    class plain_base_result_action11
      : public action<
            components::server::plain_function<Derived>,
            function_result_action_arg11,
            Result,
            hpx::util::tuple11<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type , typename detail::remove_qualifiers<T8>::type , typename detail::remove_qualifiers<T9>::type , typename detail::remove_qualifiers<T10>::type>,
            Derived, Priority>
    {
    public:
        typedef Result result_type;
        typedef hpx::util::tuple11<
            typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type , typename detail::remove_qualifiers<T8>::type , typename detail::remove_qualifiers<T9>::type , typename detail::remove_qualifiers<T10>::type> arguments_type;
        typedef action<
            components::server::plain_function<Derived>,
            function_result_action_arg11, result_type,
            arguments_type, Derived, Priority> base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10>
            result_type operator()(
                HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)) , HPX_FWD_ARGS(2, 10, ( Arg, arg))) const
            {
                try {
                    LTM_(debug) << "Executing plain action("
                                << detail::get_action_name<Derived>()
                                << ").";
                    
                    
                    
                    
                    
                    F(HPX_MOVE_ARGS(2, 0, arg) , HPX_MOVE_ARGS(2, 1, arg) , HPX_MOVE_ARGS(2, 2, arg) , HPX_MOVE_ARGS(2, 3, arg) , HPX_MOVE_ARGS(2, 4, arg) , HPX_MOVE_ARGS(2, 5, arg) , HPX_MOVE_ARGS(2, 6, arg) , HPX_MOVE_ARGS(2, 7, arg) , HPX_MOVE_ARGS(2, 8, arg) , HPX_MOVE_ARGS(2, 9, arg) , HPX_MOVE_ARGS(2, 10, arg));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing plain action("
                            << detail::get_action_name<Derived>()
                            << "): " << e.what();
                        
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
            return HPX_STD_BIND(typename Derived::thread_function(),
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type10>::call( args. a10));
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return boost::move(
                base_type::construct_continuation_thread_function(
                    cont, F, boost::forward<Arguments>(args)));
        }
    };
    
    
    template <
        typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct plain_result_action11
      : plain_base_result_action11<Result,
          T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10, F,
          typename detail::action_type<
              plain_result_action11<
                  Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10, F, Priority>, Derived
          >::type, Priority>
    {
        typedef typename detail::action_type<
            plain_result_action11<
                Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10, F, Priority>, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10), typename Derived>
    struct make_action<Result (*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10), F, Derived, boost::mpl::false_>
      : boost::mpl::identity<plain_result_action11<Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10, F, threads::thread_priority_default,
            Derived> >
    {};
    
    
    template <
        typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10),
        typename Derived = detail::this_type>
    struct plain_direct_result_action11
      : plain_base_result_action11<Result,
          T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10, F,
          typename detail::action_type<
              plain_direct_result_action11<
                  Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10, F>, Derived
          >::type>
    {
        typedef typename detail::action_type<
            plain_direct_result_action11<
                Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10, F>, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        template <typename Arguments>
        static Result
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "plain_direct_result_action" << 11
                << "::execute_function name("
                << detail::get_action_name<derived_type>() << ")";
            return F(util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type10>::call( args. a10));
        }
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10), typename Derived>
    struct make_action<Result (*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10), F, Derived, boost::mpl::true_>
      : boost::mpl::identity<plain_direct_result_action11<
            Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10, F, Derived> >
    {};
    
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default>
    class plain_base_action11
      : public action<
            components::server::plain_function<Derived>,
            function_action_arg11,
            util::unused_type,
            hpx::util::tuple11<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type , typename detail::remove_qualifiers<T8>::type , typename detail::remove_qualifiers<T9>::type , typename detail::remove_qualifiers<T10>::type>,
            Derived, Priority>
    {
    public:
        typedef util::unused_type result_type;
        typedef
            hpx::util::tuple11<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type , typename detail::remove_qualifiers<T8>::type , typename detail::remove_qualifiers<T9>::type , typename detail::remove_qualifiers<T10>::type>
        arguments_type;
        typedef action<
            components::server::plain_function<Derived>,
            function_action_arg11, result_type,
            arguments_type, Derived, Priority> base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10>
            result_type operator()(
                HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)) , HPX_FWD_ARGS(2, 10, ( Arg, arg))) const
            {
                try {
                    LTM_(debug) << "Executing plain action("
                                << detail::get_action_name<Derived>()
                                << ").";
                    
                    
                    
                    
                    
                    F(HPX_MOVE_ARGS(2, 0, arg) , HPX_MOVE_ARGS(2, 1, arg) , HPX_MOVE_ARGS(2, 2, arg) , HPX_MOVE_ARGS(2, 3, arg) , HPX_MOVE_ARGS(2, 4, arg) , HPX_MOVE_ARGS(2, 5, arg) , HPX_MOVE_ARGS(2, 6, arg) , HPX_MOVE_ARGS(2, 7, arg) , HPX_MOVE_ARGS(2, 8, arg) , HPX_MOVE_ARGS(2, 9, arg) , HPX_MOVE_ARGS(2, 10, arg));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing plain action("
                            << detail::get_action_name<Derived>()
                            << "): " << e.what();
                        
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
            return HPX_STD_BIND(typename Derived::thread_function(),
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type10>::call( args. a10));
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return
                base_type::construct_continuation_thread_function_void(
                    cont, F, boost::forward<Arguments>(args));
        }
    };
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct plain_action11
      : plain_base_action11<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10, F,
            typename detail::action_type<
                plain_action11<
                    T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10, F, Priority>, Derived
            >::type, Priority>
    {
        typedef typename detail::action_type<
            plain_action11<
                T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10, F, Priority>, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10), typename Derived>
    struct make_action<void (*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10), F, Derived, boost::mpl::false_>
      : boost::mpl::identity<plain_action11<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10, F, threads::thread_priority_default,
            Derived> >
    {};
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10),
        typename Derived = detail::this_type>
    struct plain_direct_action11
      : plain_base_action11<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10, F,
            typename detail::action_type<
                plain_direct_action11<
                    T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10, F>, Derived
            >::type>
    {
        typedef typename detail::action_type<
            plain_direct_action11<
                T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10, F>, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        template <typename Arguments>
        static util::unused_type
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "plain_direct_action" << 11
                << "::execute_function name("
                << detail::get_action_name<derived_type>() << ")";
            F(util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type10>::call( args. a10));
            return util::unused;
        }
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10), typename Derived>
    struct make_action<void (*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10), F, Derived, boost::mpl::true_>
      : boost::mpl::identity<plain_direct_action11<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10, F, Derived> >
    {};
    
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10),
        threads::thread_priority Priority, typename Derived>
    struct plain_result_action11<
                void, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10, F, Priority, Derived>
      : plain_action11<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10, F, Priority, Derived>
    {};
}}
namespace hpx { namespace traits
{
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10,
        void (*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10),
        hpx::threads::thread_priority Priority, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_action11<
                    Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10, F, Priority> >, Enable>
      : boost::mpl::false_
    {};
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10,
        void (*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10), typename Derived,
        typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_direct_action11<
                    Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10, F, Derived> >, Enable>
      : boost::mpl::false_
    {};
    template <typename R, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10,
        R(*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10),
        hpx::threads::thread_priority Priority, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_result_action11<
                    R, Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10, F, Priority> >, Enable>
      : boost::mpl::false_
    {};
    template <typename R, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10,
        R(*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10), typename Derived, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_direct_result_action11<
                    R, Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10, F, Derived> >, Enable>
      : boost::mpl::false_
    {};
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default
    >
    class plain_base_result_action12
      : public action<
            components::server::plain_function<Derived>,
            function_result_action_arg12,
            Result,
            hpx::util::tuple12<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type , typename detail::remove_qualifiers<T8>::type , typename detail::remove_qualifiers<T9>::type , typename detail::remove_qualifiers<T10>::type , typename detail::remove_qualifiers<T11>::type>,
            Derived, Priority>
    {
    public:
        typedef Result result_type;
        typedef hpx::util::tuple12<
            typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type , typename detail::remove_qualifiers<T8>::type , typename detail::remove_qualifiers<T9>::type , typename detail::remove_qualifiers<T10>::type , typename detail::remove_qualifiers<T11>::type> arguments_type;
        typedef action<
            components::server::plain_function<Derived>,
            function_result_action_arg12, result_type,
            arguments_type, Derived, Priority> base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11>
            result_type operator()(
                HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)) , HPX_FWD_ARGS(2, 10, ( Arg, arg)) , HPX_FWD_ARGS(2, 11, ( Arg, arg))) const
            {
                try {
                    LTM_(debug) << "Executing plain action("
                                << detail::get_action_name<Derived>()
                                << ").";
                    
                    
                    
                    
                    
                    F(HPX_MOVE_ARGS(2, 0, arg) , HPX_MOVE_ARGS(2, 1, arg) , HPX_MOVE_ARGS(2, 2, arg) , HPX_MOVE_ARGS(2, 3, arg) , HPX_MOVE_ARGS(2, 4, arg) , HPX_MOVE_ARGS(2, 5, arg) , HPX_MOVE_ARGS(2, 6, arg) , HPX_MOVE_ARGS(2, 7, arg) , HPX_MOVE_ARGS(2, 8, arg) , HPX_MOVE_ARGS(2, 9, arg) , HPX_MOVE_ARGS(2, 10, arg) , HPX_MOVE_ARGS(2, 11, arg));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing plain action("
                            << detail::get_action_name<Derived>()
                            << "): " << e.what();
                        
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
            return HPX_STD_BIND(typename Derived::thread_function(),
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type11>::call( args. a11));
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return boost::move(
                base_type::construct_continuation_thread_function(
                    cont, F, boost::forward<Arguments>(args)));
        }
    };
    
    
    template <
        typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct plain_result_action12
      : plain_base_result_action12<Result,
          T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11, F,
          typename detail::action_type<
              plain_result_action12<
                  Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11, F, Priority>, Derived
          >::type, Priority>
    {
        typedef typename detail::action_type<
            plain_result_action12<
                Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11, F, Priority>, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11), typename Derived>
    struct make_action<Result (*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11), F, Derived, boost::mpl::false_>
      : boost::mpl::identity<plain_result_action12<Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11, F, threads::thread_priority_default,
            Derived> >
    {};
    
    
    template <
        typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11),
        typename Derived = detail::this_type>
    struct plain_direct_result_action12
      : plain_base_result_action12<Result,
          T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11, F,
          typename detail::action_type<
              plain_direct_result_action12<
                  Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11, F>, Derived
          >::type>
    {
        typedef typename detail::action_type<
            plain_direct_result_action12<
                Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11, F>, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        template <typename Arguments>
        static Result
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "plain_direct_result_action" << 12
                << "::execute_function name("
                << detail::get_action_name<derived_type>() << ")";
            return F(util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type11>::call( args. a11));
        }
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11), typename Derived>
    struct make_action<Result (*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11), F, Derived, boost::mpl::true_>
      : boost::mpl::identity<plain_direct_result_action12<
            Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11, F, Derived> >
    {};
    
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default>
    class plain_base_action12
      : public action<
            components::server::plain_function<Derived>,
            function_action_arg12,
            util::unused_type,
            hpx::util::tuple12<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type , typename detail::remove_qualifiers<T8>::type , typename detail::remove_qualifiers<T9>::type , typename detail::remove_qualifiers<T10>::type , typename detail::remove_qualifiers<T11>::type>,
            Derived, Priority>
    {
    public:
        typedef util::unused_type result_type;
        typedef
            hpx::util::tuple12<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type , typename detail::remove_qualifiers<T8>::type , typename detail::remove_qualifiers<T9>::type , typename detail::remove_qualifiers<T10>::type , typename detail::remove_qualifiers<T11>::type>
        arguments_type;
        typedef action<
            components::server::plain_function<Derived>,
            function_action_arg12, result_type,
            arguments_type, Derived, Priority> base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11>
            result_type operator()(
                HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)) , HPX_FWD_ARGS(2, 10, ( Arg, arg)) , HPX_FWD_ARGS(2, 11, ( Arg, arg))) const
            {
                try {
                    LTM_(debug) << "Executing plain action("
                                << detail::get_action_name<Derived>()
                                << ").";
                    
                    
                    
                    
                    
                    F(HPX_MOVE_ARGS(2, 0, arg) , HPX_MOVE_ARGS(2, 1, arg) , HPX_MOVE_ARGS(2, 2, arg) , HPX_MOVE_ARGS(2, 3, arg) , HPX_MOVE_ARGS(2, 4, arg) , HPX_MOVE_ARGS(2, 5, arg) , HPX_MOVE_ARGS(2, 6, arg) , HPX_MOVE_ARGS(2, 7, arg) , HPX_MOVE_ARGS(2, 8, arg) , HPX_MOVE_ARGS(2, 9, arg) , HPX_MOVE_ARGS(2, 10, arg) , HPX_MOVE_ARGS(2, 11, arg));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing plain action("
                            << detail::get_action_name<Derived>()
                            << "): " << e.what();
                        
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
            return HPX_STD_BIND(typename Derived::thread_function(),
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type11>::call( args. a11));
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return
                base_type::construct_continuation_thread_function_void(
                    cont, F, boost::forward<Arguments>(args));
        }
    };
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct plain_action12
      : plain_base_action12<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11, F,
            typename detail::action_type<
                plain_action12<
                    T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11, F, Priority>, Derived
            >::type, Priority>
    {
        typedef typename detail::action_type<
            plain_action12<
                T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11, F, Priority>, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11), typename Derived>
    struct make_action<void (*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11), F, Derived, boost::mpl::false_>
      : boost::mpl::identity<plain_action12<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11, F, threads::thread_priority_default,
            Derived> >
    {};
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11),
        typename Derived = detail::this_type>
    struct plain_direct_action12
      : plain_base_action12<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11, F,
            typename detail::action_type<
                plain_direct_action12<
                    T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11, F>, Derived
            >::type>
    {
        typedef typename detail::action_type<
            plain_direct_action12<
                T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11, F>, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        template <typename Arguments>
        static util::unused_type
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "plain_direct_action" << 12
                << "::execute_function name("
                << detail::get_action_name<derived_type>() << ")";
            F(util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type11>::call( args. a11));
            return util::unused;
        }
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11), typename Derived>
    struct make_action<void (*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11), F, Derived, boost::mpl::true_>
      : boost::mpl::identity<plain_direct_action12<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11, F, Derived> >
    {};
    
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11),
        threads::thread_priority Priority, typename Derived>
    struct plain_result_action12<
                void, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11, F, Priority, Derived>
      : plain_action12<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11, F, Priority, Derived>
    {};
}}
namespace hpx { namespace traits
{
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11,
        void (*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11),
        hpx::threads::thread_priority Priority, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_action12<
                    Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11, F, Priority> >, Enable>
      : boost::mpl::false_
    {};
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11,
        void (*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11), typename Derived,
        typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_direct_action12<
                    Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11, F, Derived> >, Enable>
      : boost::mpl::false_
    {};
    template <typename R, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11,
        R(*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11),
        hpx::threads::thread_priority Priority, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_result_action12<
                    R, Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11, F, Priority> >, Enable>
      : boost::mpl::false_
    {};
    template <typename R, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11,
        R(*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11), typename Derived, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_direct_result_action12<
                    R, Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11, F, Derived> >, Enable>
      : boost::mpl::false_
    {};
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default
    >
    class plain_base_result_action13
      : public action<
            components::server::plain_function<Derived>,
            function_result_action_arg13,
            Result,
            hpx::util::tuple13<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type , typename detail::remove_qualifiers<T8>::type , typename detail::remove_qualifiers<T9>::type , typename detail::remove_qualifiers<T10>::type , typename detail::remove_qualifiers<T11>::type , typename detail::remove_qualifiers<T12>::type>,
            Derived, Priority>
    {
    public:
        typedef Result result_type;
        typedef hpx::util::tuple13<
            typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type , typename detail::remove_qualifiers<T8>::type , typename detail::remove_qualifiers<T9>::type , typename detail::remove_qualifiers<T10>::type , typename detail::remove_qualifiers<T11>::type , typename detail::remove_qualifiers<T12>::type> arguments_type;
        typedef action<
            components::server::plain_function<Derived>,
            function_result_action_arg13, result_type,
            arguments_type, Derived, Priority> base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12>
            result_type operator()(
                HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)) , HPX_FWD_ARGS(2, 10, ( Arg, arg)) , HPX_FWD_ARGS(2, 11, ( Arg, arg)) , HPX_FWD_ARGS(2, 12, ( Arg, arg))) const
            {
                try {
                    LTM_(debug) << "Executing plain action("
                                << detail::get_action_name<Derived>()
                                << ").";
                    
                    
                    
                    
                    
                    F(HPX_MOVE_ARGS(2, 0, arg) , HPX_MOVE_ARGS(2, 1, arg) , HPX_MOVE_ARGS(2, 2, arg) , HPX_MOVE_ARGS(2, 3, arg) , HPX_MOVE_ARGS(2, 4, arg) , HPX_MOVE_ARGS(2, 5, arg) , HPX_MOVE_ARGS(2, 6, arg) , HPX_MOVE_ARGS(2, 7, arg) , HPX_MOVE_ARGS(2, 8, arg) , HPX_MOVE_ARGS(2, 9, arg) , HPX_MOVE_ARGS(2, 10, arg) , HPX_MOVE_ARGS(2, 11, arg) , HPX_MOVE_ARGS(2, 12, arg));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing plain action("
                            << detail::get_action_name<Derived>()
                            << "): " << e.what();
                        
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
            return HPX_STD_BIND(typename Derived::thread_function(),
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type12>::call( args. a12));
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return boost::move(
                base_type::construct_continuation_thread_function(
                    cont, F, boost::forward<Arguments>(args)));
        }
    };
    
    
    template <
        typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct plain_result_action13
      : plain_base_result_action13<Result,
          T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12, F,
          typename detail::action_type<
              plain_result_action13<
                  Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12, F, Priority>, Derived
          >::type, Priority>
    {
        typedef typename detail::action_type<
            plain_result_action13<
                Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12, F, Priority>, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12), typename Derived>
    struct make_action<Result (*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12), F, Derived, boost::mpl::false_>
      : boost::mpl::identity<plain_result_action13<Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12, F, threads::thread_priority_default,
            Derived> >
    {};
    
    
    template <
        typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12),
        typename Derived = detail::this_type>
    struct plain_direct_result_action13
      : plain_base_result_action13<Result,
          T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12, F,
          typename detail::action_type<
              plain_direct_result_action13<
                  Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12, F>, Derived
          >::type>
    {
        typedef typename detail::action_type<
            plain_direct_result_action13<
                Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12, F>, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        template <typename Arguments>
        static Result
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "plain_direct_result_action" << 13
                << "::execute_function name("
                << detail::get_action_name<derived_type>() << ")";
            return F(util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type12>::call( args. a12));
        }
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12), typename Derived>
    struct make_action<Result (*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12), F, Derived, boost::mpl::true_>
      : boost::mpl::identity<plain_direct_result_action13<
            Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12, F, Derived> >
    {};
    
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default>
    class plain_base_action13
      : public action<
            components::server::plain_function<Derived>,
            function_action_arg13,
            util::unused_type,
            hpx::util::tuple13<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type , typename detail::remove_qualifiers<T8>::type , typename detail::remove_qualifiers<T9>::type , typename detail::remove_qualifiers<T10>::type , typename detail::remove_qualifiers<T11>::type , typename detail::remove_qualifiers<T12>::type>,
            Derived, Priority>
    {
    public:
        typedef util::unused_type result_type;
        typedef
            hpx::util::tuple13<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type , typename detail::remove_qualifiers<T8>::type , typename detail::remove_qualifiers<T9>::type , typename detail::remove_qualifiers<T10>::type , typename detail::remove_qualifiers<T11>::type , typename detail::remove_qualifiers<T12>::type>
        arguments_type;
        typedef action<
            components::server::plain_function<Derived>,
            function_action_arg13, result_type,
            arguments_type, Derived, Priority> base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12>
            result_type operator()(
                HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)) , HPX_FWD_ARGS(2, 10, ( Arg, arg)) , HPX_FWD_ARGS(2, 11, ( Arg, arg)) , HPX_FWD_ARGS(2, 12, ( Arg, arg))) const
            {
                try {
                    LTM_(debug) << "Executing plain action("
                                << detail::get_action_name<Derived>()
                                << ").";
                    
                    
                    
                    
                    
                    F(HPX_MOVE_ARGS(2, 0, arg) , HPX_MOVE_ARGS(2, 1, arg) , HPX_MOVE_ARGS(2, 2, arg) , HPX_MOVE_ARGS(2, 3, arg) , HPX_MOVE_ARGS(2, 4, arg) , HPX_MOVE_ARGS(2, 5, arg) , HPX_MOVE_ARGS(2, 6, arg) , HPX_MOVE_ARGS(2, 7, arg) , HPX_MOVE_ARGS(2, 8, arg) , HPX_MOVE_ARGS(2, 9, arg) , HPX_MOVE_ARGS(2, 10, arg) , HPX_MOVE_ARGS(2, 11, arg) , HPX_MOVE_ARGS(2, 12, arg));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing plain action("
                            << detail::get_action_name<Derived>()
                            << "): " << e.what();
                        
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
            return HPX_STD_BIND(typename Derived::thread_function(),
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type12>::call( args. a12));
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return
                base_type::construct_continuation_thread_function_void(
                    cont, F, boost::forward<Arguments>(args));
        }
    };
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct plain_action13
      : plain_base_action13<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12, F,
            typename detail::action_type<
                plain_action13<
                    T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12, F, Priority>, Derived
            >::type, Priority>
    {
        typedef typename detail::action_type<
            plain_action13<
                T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12, F, Priority>, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12), typename Derived>
    struct make_action<void (*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12), F, Derived, boost::mpl::false_>
      : boost::mpl::identity<plain_action13<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12, F, threads::thread_priority_default,
            Derived> >
    {};
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12),
        typename Derived = detail::this_type>
    struct plain_direct_action13
      : plain_base_action13<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12, F,
            typename detail::action_type<
                plain_direct_action13<
                    T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12, F>, Derived
            >::type>
    {
        typedef typename detail::action_type<
            plain_direct_action13<
                T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12, F>, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        template <typename Arguments>
        static util::unused_type
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "plain_direct_action" << 13
                << "::execute_function name("
                << detail::get_action_name<derived_type>() << ")";
            F(util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type12>::call( args. a12));
            return util::unused;
        }
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12), typename Derived>
    struct make_action<void (*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12), F, Derived, boost::mpl::true_>
      : boost::mpl::identity<plain_direct_action13<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12, F, Derived> >
    {};
    
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12),
        threads::thread_priority Priority, typename Derived>
    struct plain_result_action13<
                void, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12, F, Priority, Derived>
      : plain_action13<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12, F, Priority, Derived>
    {};
}}
namespace hpx { namespace traits
{
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12,
        void (*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12),
        hpx::threads::thread_priority Priority, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_action13<
                    Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12, F, Priority> >, Enable>
      : boost::mpl::false_
    {};
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12,
        void (*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12), typename Derived,
        typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_direct_action13<
                    Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12, F, Derived> >, Enable>
      : boost::mpl::false_
    {};
    template <typename R, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12,
        R(*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12),
        hpx::threads::thread_priority Priority, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_result_action13<
                    R, Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12, F, Priority> >, Enable>
      : boost::mpl::false_
    {};
    template <typename R, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12,
        R(*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12), typename Derived, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_direct_result_action13<
                    R, Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12, F, Derived> >, Enable>
      : boost::mpl::false_
    {};
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default
    >
    class plain_base_result_action14
      : public action<
            components::server::plain_function<Derived>,
            function_result_action_arg14,
            Result,
            hpx::util::tuple14<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type , typename detail::remove_qualifiers<T8>::type , typename detail::remove_qualifiers<T9>::type , typename detail::remove_qualifiers<T10>::type , typename detail::remove_qualifiers<T11>::type , typename detail::remove_qualifiers<T12>::type , typename detail::remove_qualifiers<T13>::type>,
            Derived, Priority>
    {
    public:
        typedef Result result_type;
        typedef hpx::util::tuple14<
            typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type , typename detail::remove_qualifiers<T8>::type , typename detail::remove_qualifiers<T9>::type , typename detail::remove_qualifiers<T10>::type , typename detail::remove_qualifiers<T11>::type , typename detail::remove_qualifiers<T12>::type , typename detail::remove_qualifiers<T13>::type> arguments_type;
        typedef action<
            components::server::plain_function<Derived>,
            function_result_action_arg14, result_type,
            arguments_type, Derived, Priority> base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13>
            result_type operator()(
                HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)) , HPX_FWD_ARGS(2, 10, ( Arg, arg)) , HPX_FWD_ARGS(2, 11, ( Arg, arg)) , HPX_FWD_ARGS(2, 12, ( Arg, arg)) , HPX_FWD_ARGS(2, 13, ( Arg, arg))) const
            {
                try {
                    LTM_(debug) << "Executing plain action("
                                << detail::get_action_name<Derived>()
                                << ").";
                    
                    
                    
                    
                    
                    F(HPX_MOVE_ARGS(2, 0, arg) , HPX_MOVE_ARGS(2, 1, arg) , HPX_MOVE_ARGS(2, 2, arg) , HPX_MOVE_ARGS(2, 3, arg) , HPX_MOVE_ARGS(2, 4, arg) , HPX_MOVE_ARGS(2, 5, arg) , HPX_MOVE_ARGS(2, 6, arg) , HPX_MOVE_ARGS(2, 7, arg) , HPX_MOVE_ARGS(2, 8, arg) , HPX_MOVE_ARGS(2, 9, arg) , HPX_MOVE_ARGS(2, 10, arg) , HPX_MOVE_ARGS(2, 11, arg) , HPX_MOVE_ARGS(2, 12, arg) , HPX_MOVE_ARGS(2, 13, arg));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing plain action("
                            << detail::get_action_name<Derived>()
                            << "): " << e.what();
                        
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
            return HPX_STD_BIND(typename Derived::thread_function(),
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type12>::call( args. a12) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type13>::call( args. a13));
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return boost::move(
                base_type::construct_continuation_thread_function(
                    cont, F, boost::forward<Arguments>(args)));
        }
    };
    
    
    template <
        typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct plain_result_action14
      : plain_base_result_action14<Result,
          T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13, F,
          typename detail::action_type<
              plain_result_action14<
                  Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13, F, Priority>, Derived
          >::type, Priority>
    {
        typedef typename detail::action_type<
            plain_result_action14<
                Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13, F, Priority>, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13), typename Derived>
    struct make_action<Result (*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13), F, Derived, boost::mpl::false_>
      : boost::mpl::identity<plain_result_action14<Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13, F, threads::thread_priority_default,
            Derived> >
    {};
    
    
    template <
        typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13),
        typename Derived = detail::this_type>
    struct plain_direct_result_action14
      : plain_base_result_action14<Result,
          T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13, F,
          typename detail::action_type<
              plain_direct_result_action14<
                  Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13, F>, Derived
          >::type>
    {
        typedef typename detail::action_type<
            plain_direct_result_action14<
                Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13, F>, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        template <typename Arguments>
        static Result
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "plain_direct_result_action" << 14
                << "::execute_function name("
                << detail::get_action_name<derived_type>() << ")";
            return F(util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type12>::call( args. a12) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type13>::call( args. a13));
        }
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13), typename Derived>
    struct make_action<Result (*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13), F, Derived, boost::mpl::true_>
      : boost::mpl::identity<plain_direct_result_action14<
            Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13, F, Derived> >
    {};
    
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default>
    class plain_base_action14
      : public action<
            components::server::plain_function<Derived>,
            function_action_arg14,
            util::unused_type,
            hpx::util::tuple14<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type , typename detail::remove_qualifiers<T8>::type , typename detail::remove_qualifiers<T9>::type , typename detail::remove_qualifiers<T10>::type , typename detail::remove_qualifiers<T11>::type , typename detail::remove_qualifiers<T12>::type , typename detail::remove_qualifiers<T13>::type>,
            Derived, Priority>
    {
    public:
        typedef util::unused_type result_type;
        typedef
            hpx::util::tuple14<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type , typename detail::remove_qualifiers<T8>::type , typename detail::remove_qualifiers<T9>::type , typename detail::remove_qualifiers<T10>::type , typename detail::remove_qualifiers<T11>::type , typename detail::remove_qualifiers<T12>::type , typename detail::remove_qualifiers<T13>::type>
        arguments_type;
        typedef action<
            components::server::plain_function<Derived>,
            function_action_arg14, result_type,
            arguments_type, Derived, Priority> base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13>
            result_type operator()(
                HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)) , HPX_FWD_ARGS(2, 10, ( Arg, arg)) , HPX_FWD_ARGS(2, 11, ( Arg, arg)) , HPX_FWD_ARGS(2, 12, ( Arg, arg)) , HPX_FWD_ARGS(2, 13, ( Arg, arg))) const
            {
                try {
                    LTM_(debug) << "Executing plain action("
                                << detail::get_action_name<Derived>()
                                << ").";
                    
                    
                    
                    
                    
                    F(HPX_MOVE_ARGS(2, 0, arg) , HPX_MOVE_ARGS(2, 1, arg) , HPX_MOVE_ARGS(2, 2, arg) , HPX_MOVE_ARGS(2, 3, arg) , HPX_MOVE_ARGS(2, 4, arg) , HPX_MOVE_ARGS(2, 5, arg) , HPX_MOVE_ARGS(2, 6, arg) , HPX_MOVE_ARGS(2, 7, arg) , HPX_MOVE_ARGS(2, 8, arg) , HPX_MOVE_ARGS(2, 9, arg) , HPX_MOVE_ARGS(2, 10, arg) , HPX_MOVE_ARGS(2, 11, arg) , HPX_MOVE_ARGS(2, 12, arg) , HPX_MOVE_ARGS(2, 13, arg));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing plain action("
                            << detail::get_action_name<Derived>()
                            << "): " << e.what();
                        
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
            return HPX_STD_BIND(typename Derived::thread_function(),
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type12>::call( args. a12) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type13>::call( args. a13));
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return
                base_type::construct_continuation_thread_function_void(
                    cont, F, boost::forward<Arguments>(args));
        }
    };
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct plain_action14
      : plain_base_action14<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13, F,
            typename detail::action_type<
                plain_action14<
                    T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13, F, Priority>, Derived
            >::type, Priority>
    {
        typedef typename detail::action_type<
            plain_action14<
                T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13, F, Priority>, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13), typename Derived>
    struct make_action<void (*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13), F, Derived, boost::mpl::false_>
      : boost::mpl::identity<plain_action14<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13, F, threads::thread_priority_default,
            Derived> >
    {};
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13),
        typename Derived = detail::this_type>
    struct plain_direct_action14
      : plain_base_action14<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13, F,
            typename detail::action_type<
                plain_direct_action14<
                    T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13, F>, Derived
            >::type>
    {
        typedef typename detail::action_type<
            plain_direct_action14<
                T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13, F>, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        template <typename Arguments>
        static util::unused_type
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "plain_direct_action" << 14
                << "::execute_function name("
                << detail::get_action_name<derived_type>() << ")";
            F(util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type12>::call( args. a12) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type13>::call( args. a13));
            return util::unused;
        }
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13), typename Derived>
    struct make_action<void (*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13), F, Derived, boost::mpl::true_>
      : boost::mpl::identity<plain_direct_action14<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13, F, Derived> >
    {};
    
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13),
        threads::thread_priority Priority, typename Derived>
    struct plain_result_action14<
                void, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13, F, Priority, Derived>
      : plain_action14<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13, F, Priority, Derived>
    {};
}}
namespace hpx { namespace traits
{
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13,
        void (*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13),
        hpx::threads::thread_priority Priority, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_action14<
                    Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13, F, Priority> >, Enable>
      : boost::mpl::false_
    {};
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13,
        void (*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13), typename Derived,
        typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_direct_action14<
                    Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13, F, Derived> >, Enable>
      : boost::mpl::false_
    {};
    template <typename R, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13,
        R(*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13),
        hpx::threads::thread_priority Priority, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_result_action14<
                    R, Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13, F, Priority> >, Enable>
      : boost::mpl::false_
    {};
    template <typename R, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13,
        R(*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13), typename Derived, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_direct_result_action14<
                    R, Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13, F, Derived> >, Enable>
      : boost::mpl::false_
    {};
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default
    >
    class plain_base_result_action15
      : public action<
            components::server::plain_function<Derived>,
            function_result_action_arg15,
            Result,
            hpx::util::tuple15<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type , typename detail::remove_qualifiers<T8>::type , typename detail::remove_qualifiers<T9>::type , typename detail::remove_qualifiers<T10>::type , typename detail::remove_qualifiers<T11>::type , typename detail::remove_qualifiers<T12>::type , typename detail::remove_qualifiers<T13>::type , typename detail::remove_qualifiers<T14>::type>,
            Derived, Priority>
    {
    public:
        typedef Result result_type;
        typedef hpx::util::tuple15<
            typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type , typename detail::remove_qualifiers<T8>::type , typename detail::remove_qualifiers<T9>::type , typename detail::remove_qualifiers<T10>::type , typename detail::remove_qualifiers<T11>::type , typename detail::remove_qualifiers<T12>::type , typename detail::remove_qualifiers<T13>::type , typename detail::remove_qualifiers<T14>::type> arguments_type;
        typedef action<
            components::server::plain_function<Derived>,
            function_result_action_arg15, result_type,
            arguments_type, Derived, Priority> base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14>
            result_type operator()(
                HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)) , HPX_FWD_ARGS(2, 10, ( Arg, arg)) , HPX_FWD_ARGS(2, 11, ( Arg, arg)) , HPX_FWD_ARGS(2, 12, ( Arg, arg)) , HPX_FWD_ARGS(2, 13, ( Arg, arg)) , HPX_FWD_ARGS(2, 14, ( Arg, arg))) const
            {
                try {
                    LTM_(debug) << "Executing plain action("
                                << detail::get_action_name<Derived>()
                                << ").";
                    
                    
                    
                    
                    
                    F(HPX_MOVE_ARGS(2, 0, arg) , HPX_MOVE_ARGS(2, 1, arg) , HPX_MOVE_ARGS(2, 2, arg) , HPX_MOVE_ARGS(2, 3, arg) , HPX_MOVE_ARGS(2, 4, arg) , HPX_MOVE_ARGS(2, 5, arg) , HPX_MOVE_ARGS(2, 6, arg) , HPX_MOVE_ARGS(2, 7, arg) , HPX_MOVE_ARGS(2, 8, arg) , HPX_MOVE_ARGS(2, 9, arg) , HPX_MOVE_ARGS(2, 10, arg) , HPX_MOVE_ARGS(2, 11, arg) , HPX_MOVE_ARGS(2, 12, arg) , HPX_MOVE_ARGS(2, 13, arg) , HPX_MOVE_ARGS(2, 14, arg));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing plain action("
                            << detail::get_action_name<Derived>()
                            << "): " << e.what();
                        
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
            return HPX_STD_BIND(typename Derived::thread_function(),
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type12>::call( args. a12) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type13>::call( args. a13) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type14>::call( args. a14));
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return boost::move(
                base_type::construct_continuation_thread_function(
                    cont, F, boost::forward<Arguments>(args)));
        }
    };
    
    
    template <
        typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct plain_result_action15
      : plain_base_result_action15<Result,
          T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14, F,
          typename detail::action_type<
              plain_result_action15<
                  Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14, F, Priority>, Derived
          >::type, Priority>
    {
        typedef typename detail::action_type<
            plain_result_action15<
                Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14, F, Priority>, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14), typename Derived>
    struct make_action<Result (*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14), F, Derived, boost::mpl::false_>
      : boost::mpl::identity<plain_result_action15<Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14, F, threads::thread_priority_default,
            Derived> >
    {};
    
    
    template <
        typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14),
        typename Derived = detail::this_type>
    struct plain_direct_result_action15
      : plain_base_result_action15<Result,
          T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14, F,
          typename detail::action_type<
              plain_direct_result_action15<
                  Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14, F>, Derived
          >::type>
    {
        typedef typename detail::action_type<
            plain_direct_result_action15<
                Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14, F>, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        template <typename Arguments>
        static Result
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "plain_direct_result_action" << 15
                << "::execute_function name("
                << detail::get_action_name<derived_type>() << ")";
            return F(util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type12>::call( args. a12) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type13>::call( args. a13) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type14>::call( args. a14));
        }
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14), typename Derived>
    struct make_action<Result (*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14), F, Derived, boost::mpl::true_>
      : boost::mpl::identity<plain_direct_result_action15<
            Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14, F, Derived> >
    {};
    
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default>
    class plain_base_action15
      : public action<
            components::server::plain_function<Derived>,
            function_action_arg15,
            util::unused_type,
            hpx::util::tuple15<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type , typename detail::remove_qualifiers<T8>::type , typename detail::remove_qualifiers<T9>::type , typename detail::remove_qualifiers<T10>::type , typename detail::remove_qualifiers<T11>::type , typename detail::remove_qualifiers<T12>::type , typename detail::remove_qualifiers<T13>::type , typename detail::remove_qualifiers<T14>::type>,
            Derived, Priority>
    {
    public:
        typedef util::unused_type result_type;
        typedef
            hpx::util::tuple15<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type , typename detail::remove_qualifiers<T8>::type , typename detail::remove_qualifiers<T9>::type , typename detail::remove_qualifiers<T10>::type , typename detail::remove_qualifiers<T11>::type , typename detail::remove_qualifiers<T12>::type , typename detail::remove_qualifiers<T13>::type , typename detail::remove_qualifiers<T14>::type>
        arguments_type;
        typedef action<
            components::server::plain_function<Derived>,
            function_action_arg15, result_type,
            arguments_type, Derived, Priority> base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14>
            result_type operator()(
                HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)) , HPX_FWD_ARGS(2, 10, ( Arg, arg)) , HPX_FWD_ARGS(2, 11, ( Arg, arg)) , HPX_FWD_ARGS(2, 12, ( Arg, arg)) , HPX_FWD_ARGS(2, 13, ( Arg, arg)) , HPX_FWD_ARGS(2, 14, ( Arg, arg))) const
            {
                try {
                    LTM_(debug) << "Executing plain action("
                                << detail::get_action_name<Derived>()
                                << ").";
                    
                    
                    
                    
                    
                    F(HPX_MOVE_ARGS(2, 0, arg) , HPX_MOVE_ARGS(2, 1, arg) , HPX_MOVE_ARGS(2, 2, arg) , HPX_MOVE_ARGS(2, 3, arg) , HPX_MOVE_ARGS(2, 4, arg) , HPX_MOVE_ARGS(2, 5, arg) , HPX_MOVE_ARGS(2, 6, arg) , HPX_MOVE_ARGS(2, 7, arg) , HPX_MOVE_ARGS(2, 8, arg) , HPX_MOVE_ARGS(2, 9, arg) , HPX_MOVE_ARGS(2, 10, arg) , HPX_MOVE_ARGS(2, 11, arg) , HPX_MOVE_ARGS(2, 12, arg) , HPX_MOVE_ARGS(2, 13, arg) , HPX_MOVE_ARGS(2, 14, arg));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing plain action("
                            << detail::get_action_name<Derived>()
                            << "): " << e.what();
                        
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
            return HPX_STD_BIND(typename Derived::thread_function(),
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type12>::call( args. a12) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type13>::call( args. a13) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type14>::call( args. a14));
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return
                base_type::construct_continuation_thread_function_void(
                    cont, F, boost::forward<Arguments>(args));
        }
    };
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct plain_action15
      : plain_base_action15<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14, F,
            typename detail::action_type<
                plain_action15<
                    T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14, F, Priority>, Derived
            >::type, Priority>
    {
        typedef typename detail::action_type<
            plain_action15<
                T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14, F, Priority>, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14), typename Derived>
    struct make_action<void (*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14), F, Derived, boost::mpl::false_>
      : boost::mpl::identity<plain_action15<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14, F, threads::thread_priority_default,
            Derived> >
    {};
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14),
        typename Derived = detail::this_type>
    struct plain_direct_action15
      : plain_base_action15<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14, F,
            typename detail::action_type<
                plain_direct_action15<
                    T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14, F>, Derived
            >::type>
    {
        typedef typename detail::action_type<
            plain_direct_action15<
                T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14, F>, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        template <typename Arguments>
        static util::unused_type
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "plain_direct_action" << 15
                << "::execute_function name("
                << detail::get_action_name<derived_type>() << ")";
            F(util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type12>::call( args. a12) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type13>::call( args. a13) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type14>::call( args. a14));
            return util::unused;
        }
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14), typename Derived>
    struct make_action<void (*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14), F, Derived, boost::mpl::true_>
      : boost::mpl::identity<plain_direct_action15<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14, F, Derived> >
    {};
    
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14),
        threads::thread_priority Priority, typename Derived>
    struct plain_result_action15<
                void, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14, F, Priority, Derived>
      : plain_action15<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14, F, Priority, Derived>
    {};
}}
namespace hpx { namespace traits
{
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14,
        void (*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14),
        hpx::threads::thread_priority Priority, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_action15<
                    Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14, F, Priority> >, Enable>
      : boost::mpl::false_
    {};
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14,
        void (*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14), typename Derived,
        typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_direct_action15<
                    Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14, F, Derived> >, Enable>
      : boost::mpl::false_
    {};
    template <typename R, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14,
        R(*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14),
        hpx::threads::thread_priority Priority, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_result_action15<
                    R, Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14, F, Priority> >, Enable>
      : boost::mpl::false_
    {};
    template <typename R, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14,
        R(*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14), typename Derived, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_direct_result_action15<
                    R, Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14, F, Derived> >, Enable>
      : boost::mpl::false_
    {};
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default
    >
    class plain_base_result_action16
      : public action<
            components::server::plain_function<Derived>,
            function_result_action_arg16,
            Result,
            hpx::util::tuple16<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type , typename detail::remove_qualifiers<T8>::type , typename detail::remove_qualifiers<T9>::type , typename detail::remove_qualifiers<T10>::type , typename detail::remove_qualifiers<T11>::type , typename detail::remove_qualifiers<T12>::type , typename detail::remove_qualifiers<T13>::type , typename detail::remove_qualifiers<T14>::type , typename detail::remove_qualifiers<T15>::type>,
            Derived, Priority>
    {
    public:
        typedef Result result_type;
        typedef hpx::util::tuple16<
            typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type , typename detail::remove_qualifiers<T8>::type , typename detail::remove_qualifiers<T9>::type , typename detail::remove_qualifiers<T10>::type , typename detail::remove_qualifiers<T11>::type , typename detail::remove_qualifiers<T12>::type , typename detail::remove_qualifiers<T13>::type , typename detail::remove_qualifiers<T14>::type , typename detail::remove_qualifiers<T15>::type> arguments_type;
        typedef action<
            components::server::plain_function<Derived>,
            function_result_action_arg16, result_type,
            arguments_type, Derived, Priority> base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15>
            result_type operator()(
                HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)) , HPX_FWD_ARGS(2, 10, ( Arg, arg)) , HPX_FWD_ARGS(2, 11, ( Arg, arg)) , HPX_FWD_ARGS(2, 12, ( Arg, arg)) , HPX_FWD_ARGS(2, 13, ( Arg, arg)) , HPX_FWD_ARGS(2, 14, ( Arg, arg)) , HPX_FWD_ARGS(2, 15, ( Arg, arg))) const
            {
                try {
                    LTM_(debug) << "Executing plain action("
                                << detail::get_action_name<Derived>()
                                << ").";
                    
                    
                    
                    
                    
                    F(HPX_MOVE_ARGS(2, 0, arg) , HPX_MOVE_ARGS(2, 1, arg) , HPX_MOVE_ARGS(2, 2, arg) , HPX_MOVE_ARGS(2, 3, arg) , HPX_MOVE_ARGS(2, 4, arg) , HPX_MOVE_ARGS(2, 5, arg) , HPX_MOVE_ARGS(2, 6, arg) , HPX_MOVE_ARGS(2, 7, arg) , HPX_MOVE_ARGS(2, 8, arg) , HPX_MOVE_ARGS(2, 9, arg) , HPX_MOVE_ARGS(2, 10, arg) , HPX_MOVE_ARGS(2, 11, arg) , HPX_MOVE_ARGS(2, 12, arg) , HPX_MOVE_ARGS(2, 13, arg) , HPX_MOVE_ARGS(2, 14, arg) , HPX_MOVE_ARGS(2, 15, arg));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing plain action("
                            << detail::get_action_name<Derived>()
                            << "): " << e.what();
                        
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
            return HPX_STD_BIND(typename Derived::thread_function(),
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type12>::call( args. a12) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type13>::call( args. a13) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type14>::call( args. a14) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type15>::call( args. a15));
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return boost::move(
                base_type::construct_continuation_thread_function(
                    cont, F, boost::forward<Arguments>(args)));
        }
    };
    
    
    template <
        typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct plain_result_action16
      : plain_base_result_action16<Result,
          T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15, F,
          typename detail::action_type<
              plain_result_action16<
                  Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15, F, Priority>, Derived
          >::type, Priority>
    {
        typedef typename detail::action_type<
            plain_result_action16<
                Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15, F, Priority>, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15), typename Derived>
    struct make_action<Result (*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15), F, Derived, boost::mpl::false_>
      : boost::mpl::identity<plain_result_action16<Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15, F, threads::thread_priority_default,
            Derived> >
    {};
    
    
    template <
        typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15),
        typename Derived = detail::this_type>
    struct plain_direct_result_action16
      : plain_base_result_action16<Result,
          T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15, F,
          typename detail::action_type<
              plain_direct_result_action16<
                  Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15, F>, Derived
          >::type>
    {
        typedef typename detail::action_type<
            plain_direct_result_action16<
                Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15, F>, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        template <typename Arguments>
        static Result
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "plain_direct_result_action" << 16
                << "::execute_function name("
                << detail::get_action_name<derived_type>() << ")";
            return F(util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type12>::call( args. a12) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type13>::call( args. a13) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type14>::call( args. a14) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type15>::call( args. a15));
        }
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15), typename Derived>
    struct make_action<Result (*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15), F, Derived, boost::mpl::true_>
      : boost::mpl::identity<plain_direct_result_action16<
            Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15, F, Derived> >
    {};
    
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default>
    class plain_base_action16
      : public action<
            components::server::plain_function<Derived>,
            function_action_arg16,
            util::unused_type,
            hpx::util::tuple16<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type , typename detail::remove_qualifiers<T8>::type , typename detail::remove_qualifiers<T9>::type , typename detail::remove_qualifiers<T10>::type , typename detail::remove_qualifiers<T11>::type , typename detail::remove_qualifiers<T12>::type , typename detail::remove_qualifiers<T13>::type , typename detail::remove_qualifiers<T14>::type , typename detail::remove_qualifiers<T15>::type>,
            Derived, Priority>
    {
    public:
        typedef util::unused_type result_type;
        typedef
            hpx::util::tuple16<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type , typename detail::remove_qualifiers<T8>::type , typename detail::remove_qualifiers<T9>::type , typename detail::remove_qualifiers<T10>::type , typename detail::remove_qualifiers<T11>::type , typename detail::remove_qualifiers<T12>::type , typename detail::remove_qualifiers<T13>::type , typename detail::remove_qualifiers<T14>::type , typename detail::remove_qualifiers<T15>::type>
        arguments_type;
        typedef action<
            components::server::plain_function<Derived>,
            function_action_arg16, result_type,
            arguments_type, Derived, Priority> base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15>
            result_type operator()(
                HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)) , HPX_FWD_ARGS(2, 10, ( Arg, arg)) , HPX_FWD_ARGS(2, 11, ( Arg, arg)) , HPX_FWD_ARGS(2, 12, ( Arg, arg)) , HPX_FWD_ARGS(2, 13, ( Arg, arg)) , HPX_FWD_ARGS(2, 14, ( Arg, arg)) , HPX_FWD_ARGS(2, 15, ( Arg, arg))) const
            {
                try {
                    LTM_(debug) << "Executing plain action("
                                << detail::get_action_name<Derived>()
                                << ").";
                    
                    
                    
                    
                    
                    F(HPX_MOVE_ARGS(2, 0, arg) , HPX_MOVE_ARGS(2, 1, arg) , HPX_MOVE_ARGS(2, 2, arg) , HPX_MOVE_ARGS(2, 3, arg) , HPX_MOVE_ARGS(2, 4, arg) , HPX_MOVE_ARGS(2, 5, arg) , HPX_MOVE_ARGS(2, 6, arg) , HPX_MOVE_ARGS(2, 7, arg) , HPX_MOVE_ARGS(2, 8, arg) , HPX_MOVE_ARGS(2, 9, arg) , HPX_MOVE_ARGS(2, 10, arg) , HPX_MOVE_ARGS(2, 11, arg) , HPX_MOVE_ARGS(2, 12, arg) , HPX_MOVE_ARGS(2, 13, arg) , HPX_MOVE_ARGS(2, 14, arg) , HPX_MOVE_ARGS(2, 15, arg));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing plain action("
                            << detail::get_action_name<Derived>()
                            << "): " << e.what();
                        
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
            return HPX_STD_BIND(typename Derived::thread_function(),
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type12>::call( args. a12) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type13>::call( args. a13) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type14>::call( args. a14) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type15>::call( args. a15));
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return
                base_type::construct_continuation_thread_function_void(
                    cont, F, boost::forward<Arguments>(args));
        }
    };
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct plain_action16
      : plain_base_action16<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15, F,
            typename detail::action_type<
                plain_action16<
                    T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15, F, Priority>, Derived
            >::type, Priority>
    {
        typedef typename detail::action_type<
            plain_action16<
                T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15, F, Priority>, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15), typename Derived>
    struct make_action<void (*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15), F, Derived, boost::mpl::false_>
      : boost::mpl::identity<plain_action16<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15, F, threads::thread_priority_default,
            Derived> >
    {};
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15),
        typename Derived = detail::this_type>
    struct plain_direct_action16
      : plain_base_action16<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15, F,
            typename detail::action_type<
                plain_direct_action16<
                    T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15, F>, Derived
            >::type>
    {
        typedef typename detail::action_type<
            plain_direct_action16<
                T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15, F>, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        template <typename Arguments>
        static util::unused_type
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "plain_direct_action" << 16
                << "::execute_function name("
                << detail::get_action_name<derived_type>() << ")";
            F(util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type12>::call( args. a12) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type13>::call( args. a13) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type14>::call( args. a14) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type15>::call( args. a15));
            return util::unused;
        }
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15), typename Derived>
    struct make_action<void (*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15), F, Derived, boost::mpl::true_>
      : boost::mpl::identity<plain_direct_action16<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15, F, Derived> >
    {};
    
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15),
        threads::thread_priority Priority, typename Derived>
    struct plain_result_action16<
                void, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15, F, Priority, Derived>
      : plain_action16<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15, F, Priority, Derived>
    {};
}}
namespace hpx { namespace traits
{
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15,
        void (*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15),
        hpx::threads::thread_priority Priority, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_action16<
                    Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15, F, Priority> >, Enable>
      : boost::mpl::false_
    {};
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15,
        void (*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15), typename Derived,
        typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_direct_action16<
                    Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15, F, Derived> >, Enable>
      : boost::mpl::false_
    {};
    template <typename R, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15,
        R(*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15),
        hpx::threads::thread_priority Priority, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_result_action16<
                    R, Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15, F, Priority> >, Enable>
      : boost::mpl::false_
    {};
    template <typename R, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15,
        R(*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15), typename Derived, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_direct_result_action16<
                    R, Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15, F, Derived> >, Enable>
      : boost::mpl::false_
    {};
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default
    >
    class plain_base_result_action17
      : public action<
            components::server::plain_function<Derived>,
            function_result_action_arg17,
            Result,
            hpx::util::tuple17<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type , typename detail::remove_qualifiers<T8>::type , typename detail::remove_qualifiers<T9>::type , typename detail::remove_qualifiers<T10>::type , typename detail::remove_qualifiers<T11>::type , typename detail::remove_qualifiers<T12>::type , typename detail::remove_qualifiers<T13>::type , typename detail::remove_qualifiers<T14>::type , typename detail::remove_qualifiers<T15>::type , typename detail::remove_qualifiers<T16>::type>,
            Derived, Priority>
    {
    public:
        typedef Result result_type;
        typedef hpx::util::tuple17<
            typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type , typename detail::remove_qualifiers<T8>::type , typename detail::remove_qualifiers<T9>::type , typename detail::remove_qualifiers<T10>::type , typename detail::remove_qualifiers<T11>::type , typename detail::remove_qualifiers<T12>::type , typename detail::remove_qualifiers<T13>::type , typename detail::remove_qualifiers<T14>::type , typename detail::remove_qualifiers<T15>::type , typename detail::remove_qualifiers<T16>::type> arguments_type;
        typedef action<
            components::server::plain_function<Derived>,
            function_result_action_arg17, result_type,
            arguments_type, Derived, Priority> base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16>
            result_type operator()(
                HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)) , HPX_FWD_ARGS(2, 10, ( Arg, arg)) , HPX_FWD_ARGS(2, 11, ( Arg, arg)) , HPX_FWD_ARGS(2, 12, ( Arg, arg)) , HPX_FWD_ARGS(2, 13, ( Arg, arg)) , HPX_FWD_ARGS(2, 14, ( Arg, arg)) , HPX_FWD_ARGS(2, 15, ( Arg, arg)) , HPX_FWD_ARGS(2, 16, ( Arg, arg))) const
            {
                try {
                    LTM_(debug) << "Executing plain action("
                                << detail::get_action_name<Derived>()
                                << ").";
                    
                    
                    
                    
                    
                    F(HPX_MOVE_ARGS(2, 0, arg) , HPX_MOVE_ARGS(2, 1, arg) , HPX_MOVE_ARGS(2, 2, arg) , HPX_MOVE_ARGS(2, 3, arg) , HPX_MOVE_ARGS(2, 4, arg) , HPX_MOVE_ARGS(2, 5, arg) , HPX_MOVE_ARGS(2, 6, arg) , HPX_MOVE_ARGS(2, 7, arg) , HPX_MOVE_ARGS(2, 8, arg) , HPX_MOVE_ARGS(2, 9, arg) , HPX_MOVE_ARGS(2, 10, arg) , HPX_MOVE_ARGS(2, 11, arg) , HPX_MOVE_ARGS(2, 12, arg) , HPX_MOVE_ARGS(2, 13, arg) , HPX_MOVE_ARGS(2, 14, arg) , HPX_MOVE_ARGS(2, 15, arg) , HPX_MOVE_ARGS(2, 16, arg));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing plain action("
                            << detail::get_action_name<Derived>()
                            << "): " << e.what();
                        
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
            return HPX_STD_BIND(typename Derived::thread_function(),
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type12>::call( args. a12) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type13>::call( args. a13) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type14>::call( args. a14) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type15>::call( args. a15) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type16>::call( args. a16));
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return boost::move(
                base_type::construct_continuation_thread_function(
                    cont, F, boost::forward<Arguments>(args)));
        }
    };
    
    
    template <
        typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct plain_result_action17
      : plain_base_result_action17<Result,
          T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16, F,
          typename detail::action_type<
              plain_result_action17<
                  Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16, F, Priority>, Derived
          >::type, Priority>
    {
        typedef typename detail::action_type<
            plain_result_action17<
                Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16, F, Priority>, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16), typename Derived>
    struct make_action<Result (*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16), F, Derived, boost::mpl::false_>
      : boost::mpl::identity<plain_result_action17<Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16, F, threads::thread_priority_default,
            Derived> >
    {};
    
    
    template <
        typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16),
        typename Derived = detail::this_type>
    struct plain_direct_result_action17
      : plain_base_result_action17<Result,
          T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16, F,
          typename detail::action_type<
              plain_direct_result_action17<
                  Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16, F>, Derived
          >::type>
    {
        typedef typename detail::action_type<
            plain_direct_result_action17<
                Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16, F>, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        template <typename Arguments>
        static Result
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "plain_direct_result_action" << 17
                << "::execute_function name("
                << detail::get_action_name<derived_type>() << ")";
            return F(util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type12>::call( args. a12) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type13>::call( args. a13) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type14>::call( args. a14) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type15>::call( args. a15) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type16>::call( args. a16));
        }
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16), typename Derived>
    struct make_action<Result (*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16), F, Derived, boost::mpl::true_>
      : boost::mpl::identity<plain_direct_result_action17<
            Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16, F, Derived> >
    {};
    
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default>
    class plain_base_action17
      : public action<
            components::server::plain_function<Derived>,
            function_action_arg17,
            util::unused_type,
            hpx::util::tuple17<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type , typename detail::remove_qualifiers<T8>::type , typename detail::remove_qualifiers<T9>::type , typename detail::remove_qualifiers<T10>::type , typename detail::remove_qualifiers<T11>::type , typename detail::remove_qualifiers<T12>::type , typename detail::remove_qualifiers<T13>::type , typename detail::remove_qualifiers<T14>::type , typename detail::remove_qualifiers<T15>::type , typename detail::remove_qualifiers<T16>::type>,
            Derived, Priority>
    {
    public:
        typedef util::unused_type result_type;
        typedef
            hpx::util::tuple17<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type , typename detail::remove_qualifiers<T8>::type , typename detail::remove_qualifiers<T9>::type , typename detail::remove_qualifiers<T10>::type , typename detail::remove_qualifiers<T11>::type , typename detail::remove_qualifiers<T12>::type , typename detail::remove_qualifiers<T13>::type , typename detail::remove_qualifiers<T14>::type , typename detail::remove_qualifiers<T15>::type , typename detail::remove_qualifiers<T16>::type>
        arguments_type;
        typedef action<
            components::server::plain_function<Derived>,
            function_action_arg17, result_type,
            arguments_type, Derived, Priority> base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16>
            result_type operator()(
                HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)) , HPX_FWD_ARGS(2, 10, ( Arg, arg)) , HPX_FWD_ARGS(2, 11, ( Arg, arg)) , HPX_FWD_ARGS(2, 12, ( Arg, arg)) , HPX_FWD_ARGS(2, 13, ( Arg, arg)) , HPX_FWD_ARGS(2, 14, ( Arg, arg)) , HPX_FWD_ARGS(2, 15, ( Arg, arg)) , HPX_FWD_ARGS(2, 16, ( Arg, arg))) const
            {
                try {
                    LTM_(debug) << "Executing plain action("
                                << detail::get_action_name<Derived>()
                                << ").";
                    
                    
                    
                    
                    
                    F(HPX_MOVE_ARGS(2, 0, arg) , HPX_MOVE_ARGS(2, 1, arg) , HPX_MOVE_ARGS(2, 2, arg) , HPX_MOVE_ARGS(2, 3, arg) , HPX_MOVE_ARGS(2, 4, arg) , HPX_MOVE_ARGS(2, 5, arg) , HPX_MOVE_ARGS(2, 6, arg) , HPX_MOVE_ARGS(2, 7, arg) , HPX_MOVE_ARGS(2, 8, arg) , HPX_MOVE_ARGS(2, 9, arg) , HPX_MOVE_ARGS(2, 10, arg) , HPX_MOVE_ARGS(2, 11, arg) , HPX_MOVE_ARGS(2, 12, arg) , HPX_MOVE_ARGS(2, 13, arg) , HPX_MOVE_ARGS(2, 14, arg) , HPX_MOVE_ARGS(2, 15, arg) , HPX_MOVE_ARGS(2, 16, arg));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing plain action("
                            << detail::get_action_name<Derived>()
                            << "): " << e.what();
                        
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
            return HPX_STD_BIND(typename Derived::thread_function(),
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type12>::call( args. a12) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type13>::call( args. a13) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type14>::call( args. a14) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type15>::call( args. a15) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type16>::call( args. a16));
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return
                base_type::construct_continuation_thread_function_void(
                    cont, F, boost::forward<Arguments>(args));
        }
    };
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct plain_action17
      : plain_base_action17<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16, F,
            typename detail::action_type<
                plain_action17<
                    T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16, F, Priority>, Derived
            >::type, Priority>
    {
        typedef typename detail::action_type<
            plain_action17<
                T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16, F, Priority>, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16), typename Derived>
    struct make_action<void (*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16), F, Derived, boost::mpl::false_>
      : boost::mpl::identity<plain_action17<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16, F, threads::thread_priority_default,
            Derived> >
    {};
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16),
        typename Derived = detail::this_type>
    struct plain_direct_action17
      : plain_base_action17<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16, F,
            typename detail::action_type<
                plain_direct_action17<
                    T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16, F>, Derived
            >::type>
    {
        typedef typename detail::action_type<
            plain_direct_action17<
                T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16, F>, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        template <typename Arguments>
        static util::unused_type
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "plain_direct_action" << 17
                << "::execute_function name("
                << detail::get_action_name<derived_type>() << ")";
            F(util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type12>::call( args. a12) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type13>::call( args. a13) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type14>::call( args. a14) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type15>::call( args. a15) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type16>::call( args. a16));
            return util::unused;
        }
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16), typename Derived>
    struct make_action<void (*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16), F, Derived, boost::mpl::true_>
      : boost::mpl::identity<plain_direct_action17<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16, F, Derived> >
    {};
    
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16),
        threads::thread_priority Priority, typename Derived>
    struct plain_result_action17<
                void, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16, F, Priority, Derived>
      : plain_action17<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16, F, Priority, Derived>
    {};
}}
namespace hpx { namespace traits
{
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16,
        void (*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16),
        hpx::threads::thread_priority Priority, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_action17<
                    Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16, F, Priority> >, Enable>
      : boost::mpl::false_
    {};
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16,
        void (*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16), typename Derived,
        typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_direct_action17<
                    Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16, F, Derived> >, Enable>
      : boost::mpl::false_
    {};
    template <typename R, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16,
        R(*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16),
        hpx::threads::thread_priority Priority, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_result_action17<
                    R, Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16, F, Priority> >, Enable>
      : boost::mpl::false_
    {};
    template <typename R, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16,
        R(*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16), typename Derived, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_direct_result_action17<
                    R, Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16, F, Derived> >, Enable>
      : boost::mpl::false_
    {};
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default
    >
    class plain_base_result_action18
      : public action<
            components::server::plain_function<Derived>,
            function_result_action_arg18,
            Result,
            hpx::util::tuple18<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type , typename detail::remove_qualifiers<T8>::type , typename detail::remove_qualifiers<T9>::type , typename detail::remove_qualifiers<T10>::type , typename detail::remove_qualifiers<T11>::type , typename detail::remove_qualifiers<T12>::type , typename detail::remove_qualifiers<T13>::type , typename detail::remove_qualifiers<T14>::type , typename detail::remove_qualifiers<T15>::type , typename detail::remove_qualifiers<T16>::type , typename detail::remove_qualifiers<T17>::type>,
            Derived, Priority>
    {
    public:
        typedef Result result_type;
        typedef hpx::util::tuple18<
            typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type , typename detail::remove_qualifiers<T8>::type , typename detail::remove_qualifiers<T9>::type , typename detail::remove_qualifiers<T10>::type , typename detail::remove_qualifiers<T11>::type , typename detail::remove_qualifiers<T12>::type , typename detail::remove_qualifiers<T13>::type , typename detail::remove_qualifiers<T14>::type , typename detail::remove_qualifiers<T15>::type , typename detail::remove_qualifiers<T16>::type , typename detail::remove_qualifiers<T17>::type> arguments_type;
        typedef action<
            components::server::plain_function<Derived>,
            function_result_action_arg18, result_type,
            arguments_type, Derived, Priority> base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17>
            result_type operator()(
                HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)) , HPX_FWD_ARGS(2, 10, ( Arg, arg)) , HPX_FWD_ARGS(2, 11, ( Arg, arg)) , HPX_FWD_ARGS(2, 12, ( Arg, arg)) , HPX_FWD_ARGS(2, 13, ( Arg, arg)) , HPX_FWD_ARGS(2, 14, ( Arg, arg)) , HPX_FWD_ARGS(2, 15, ( Arg, arg)) , HPX_FWD_ARGS(2, 16, ( Arg, arg)) , HPX_FWD_ARGS(2, 17, ( Arg, arg))) const
            {
                try {
                    LTM_(debug) << "Executing plain action("
                                << detail::get_action_name<Derived>()
                                << ").";
                    
                    
                    
                    
                    
                    F(HPX_MOVE_ARGS(2, 0, arg) , HPX_MOVE_ARGS(2, 1, arg) , HPX_MOVE_ARGS(2, 2, arg) , HPX_MOVE_ARGS(2, 3, arg) , HPX_MOVE_ARGS(2, 4, arg) , HPX_MOVE_ARGS(2, 5, arg) , HPX_MOVE_ARGS(2, 6, arg) , HPX_MOVE_ARGS(2, 7, arg) , HPX_MOVE_ARGS(2, 8, arg) , HPX_MOVE_ARGS(2, 9, arg) , HPX_MOVE_ARGS(2, 10, arg) , HPX_MOVE_ARGS(2, 11, arg) , HPX_MOVE_ARGS(2, 12, arg) , HPX_MOVE_ARGS(2, 13, arg) , HPX_MOVE_ARGS(2, 14, arg) , HPX_MOVE_ARGS(2, 15, arg) , HPX_MOVE_ARGS(2, 16, arg) , HPX_MOVE_ARGS(2, 17, arg));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing plain action("
                            << detail::get_action_name<Derived>()
                            << "): " << e.what();
                        
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
            return HPX_STD_BIND(typename Derived::thread_function(),
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type12>::call( args. a12) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type13>::call( args. a13) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type14>::call( args. a14) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type15>::call( args. a15) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type16>::call( args. a16) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type17>::call( args. a17));
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return boost::move(
                base_type::construct_continuation_thread_function(
                    cont, F, boost::forward<Arguments>(args)));
        }
    };
    
    
    template <
        typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct plain_result_action18
      : plain_base_result_action18<Result,
          T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17, F,
          typename detail::action_type<
              plain_result_action18<
                  Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17, F, Priority>, Derived
          >::type, Priority>
    {
        typedef typename detail::action_type<
            plain_result_action18<
                Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17, F, Priority>, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17), typename Derived>
    struct make_action<Result (*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17), F, Derived, boost::mpl::false_>
      : boost::mpl::identity<plain_result_action18<Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17, F, threads::thread_priority_default,
            Derived> >
    {};
    
    
    template <
        typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17),
        typename Derived = detail::this_type>
    struct plain_direct_result_action18
      : plain_base_result_action18<Result,
          T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17, F,
          typename detail::action_type<
              plain_direct_result_action18<
                  Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17, F>, Derived
          >::type>
    {
        typedef typename detail::action_type<
            plain_direct_result_action18<
                Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17, F>, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        template <typename Arguments>
        static Result
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "plain_direct_result_action" << 18
                << "::execute_function name("
                << detail::get_action_name<derived_type>() << ")";
            return F(util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type12>::call( args. a12) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type13>::call( args. a13) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type14>::call( args. a14) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type15>::call( args. a15) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type16>::call( args. a16) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type17>::call( args. a17));
        }
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17), typename Derived>
    struct make_action<Result (*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17), F, Derived, boost::mpl::true_>
      : boost::mpl::identity<plain_direct_result_action18<
            Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17, F, Derived> >
    {};
    
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default>
    class plain_base_action18
      : public action<
            components::server::plain_function<Derived>,
            function_action_arg18,
            util::unused_type,
            hpx::util::tuple18<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type , typename detail::remove_qualifiers<T8>::type , typename detail::remove_qualifiers<T9>::type , typename detail::remove_qualifiers<T10>::type , typename detail::remove_qualifiers<T11>::type , typename detail::remove_qualifiers<T12>::type , typename detail::remove_qualifiers<T13>::type , typename detail::remove_qualifiers<T14>::type , typename detail::remove_qualifiers<T15>::type , typename detail::remove_qualifiers<T16>::type , typename detail::remove_qualifiers<T17>::type>,
            Derived, Priority>
    {
    public:
        typedef util::unused_type result_type;
        typedef
            hpx::util::tuple18<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type , typename detail::remove_qualifiers<T8>::type , typename detail::remove_qualifiers<T9>::type , typename detail::remove_qualifiers<T10>::type , typename detail::remove_qualifiers<T11>::type , typename detail::remove_qualifiers<T12>::type , typename detail::remove_qualifiers<T13>::type , typename detail::remove_qualifiers<T14>::type , typename detail::remove_qualifiers<T15>::type , typename detail::remove_qualifiers<T16>::type , typename detail::remove_qualifiers<T17>::type>
        arguments_type;
        typedef action<
            components::server::plain_function<Derived>,
            function_action_arg18, result_type,
            arguments_type, Derived, Priority> base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17>
            result_type operator()(
                HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)) , HPX_FWD_ARGS(2, 10, ( Arg, arg)) , HPX_FWD_ARGS(2, 11, ( Arg, arg)) , HPX_FWD_ARGS(2, 12, ( Arg, arg)) , HPX_FWD_ARGS(2, 13, ( Arg, arg)) , HPX_FWD_ARGS(2, 14, ( Arg, arg)) , HPX_FWD_ARGS(2, 15, ( Arg, arg)) , HPX_FWD_ARGS(2, 16, ( Arg, arg)) , HPX_FWD_ARGS(2, 17, ( Arg, arg))) const
            {
                try {
                    LTM_(debug) << "Executing plain action("
                                << detail::get_action_name<Derived>()
                                << ").";
                    
                    
                    
                    
                    
                    F(HPX_MOVE_ARGS(2, 0, arg) , HPX_MOVE_ARGS(2, 1, arg) , HPX_MOVE_ARGS(2, 2, arg) , HPX_MOVE_ARGS(2, 3, arg) , HPX_MOVE_ARGS(2, 4, arg) , HPX_MOVE_ARGS(2, 5, arg) , HPX_MOVE_ARGS(2, 6, arg) , HPX_MOVE_ARGS(2, 7, arg) , HPX_MOVE_ARGS(2, 8, arg) , HPX_MOVE_ARGS(2, 9, arg) , HPX_MOVE_ARGS(2, 10, arg) , HPX_MOVE_ARGS(2, 11, arg) , HPX_MOVE_ARGS(2, 12, arg) , HPX_MOVE_ARGS(2, 13, arg) , HPX_MOVE_ARGS(2, 14, arg) , HPX_MOVE_ARGS(2, 15, arg) , HPX_MOVE_ARGS(2, 16, arg) , HPX_MOVE_ARGS(2, 17, arg));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing plain action("
                            << detail::get_action_name<Derived>()
                            << "): " << e.what();
                        
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
            return HPX_STD_BIND(typename Derived::thread_function(),
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type12>::call( args. a12) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type13>::call( args. a13) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type14>::call( args. a14) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type15>::call( args. a15) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type16>::call( args. a16) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type17>::call( args. a17));
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return
                base_type::construct_continuation_thread_function_void(
                    cont, F, boost::forward<Arguments>(args));
        }
    };
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct plain_action18
      : plain_base_action18<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17, F,
            typename detail::action_type<
                plain_action18<
                    T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17, F, Priority>, Derived
            >::type, Priority>
    {
        typedef typename detail::action_type<
            plain_action18<
                T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17, F, Priority>, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17), typename Derived>
    struct make_action<void (*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17), F, Derived, boost::mpl::false_>
      : boost::mpl::identity<plain_action18<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17, F, threads::thread_priority_default,
            Derived> >
    {};
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17),
        typename Derived = detail::this_type>
    struct plain_direct_action18
      : plain_base_action18<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17, F,
            typename detail::action_type<
                plain_direct_action18<
                    T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17, F>, Derived
            >::type>
    {
        typedef typename detail::action_type<
            plain_direct_action18<
                T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17, F>, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        template <typename Arguments>
        static util::unused_type
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "plain_direct_action" << 18
                << "::execute_function name("
                << detail::get_action_name<derived_type>() << ")";
            F(util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type12>::call( args. a12) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type13>::call( args. a13) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type14>::call( args. a14) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type15>::call( args. a15) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type16>::call( args. a16) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type17>::call( args. a17));
            return util::unused;
        }
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17), typename Derived>
    struct make_action<void (*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17), F, Derived, boost::mpl::true_>
      : boost::mpl::identity<plain_direct_action18<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17, F, Derived> >
    {};
    
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17),
        threads::thread_priority Priority, typename Derived>
    struct plain_result_action18<
                void, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17, F, Priority, Derived>
      : plain_action18<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17, F, Priority, Derived>
    {};
}}
namespace hpx { namespace traits
{
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17,
        void (*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17),
        hpx::threads::thread_priority Priority, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_action18<
                    Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17, F, Priority> >, Enable>
      : boost::mpl::false_
    {};
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17,
        void (*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17), typename Derived,
        typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_direct_action18<
                    Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17, F, Derived> >, Enable>
      : boost::mpl::false_
    {};
    template <typename R, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17,
        R(*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17),
        hpx::threads::thread_priority Priority, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_result_action18<
                    R, Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17, F, Priority> >, Enable>
      : boost::mpl::false_
    {};
    template <typename R, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17,
        R(*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17), typename Derived, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_direct_result_action18<
                    R, Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17, F, Derived> >, Enable>
      : boost::mpl::false_
    {};
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default
    >
    class plain_base_result_action19
      : public action<
            components::server::plain_function<Derived>,
            function_result_action_arg19,
            Result,
            hpx::util::tuple19<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type , typename detail::remove_qualifiers<T8>::type , typename detail::remove_qualifiers<T9>::type , typename detail::remove_qualifiers<T10>::type , typename detail::remove_qualifiers<T11>::type , typename detail::remove_qualifiers<T12>::type , typename detail::remove_qualifiers<T13>::type , typename detail::remove_qualifiers<T14>::type , typename detail::remove_qualifiers<T15>::type , typename detail::remove_qualifiers<T16>::type , typename detail::remove_qualifiers<T17>::type , typename detail::remove_qualifiers<T18>::type>,
            Derived, Priority>
    {
    public:
        typedef Result result_type;
        typedef hpx::util::tuple19<
            typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type , typename detail::remove_qualifiers<T8>::type , typename detail::remove_qualifiers<T9>::type , typename detail::remove_qualifiers<T10>::type , typename detail::remove_qualifiers<T11>::type , typename detail::remove_qualifiers<T12>::type , typename detail::remove_qualifiers<T13>::type , typename detail::remove_qualifiers<T14>::type , typename detail::remove_qualifiers<T15>::type , typename detail::remove_qualifiers<T16>::type , typename detail::remove_qualifiers<T17>::type , typename detail::remove_qualifiers<T18>::type> arguments_type;
        typedef action<
            components::server::plain_function<Derived>,
            function_result_action_arg19, result_type,
            arguments_type, Derived, Priority> base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17 , typename Arg18>
            result_type operator()(
                HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)) , HPX_FWD_ARGS(2, 10, ( Arg, arg)) , HPX_FWD_ARGS(2, 11, ( Arg, arg)) , HPX_FWD_ARGS(2, 12, ( Arg, arg)) , HPX_FWD_ARGS(2, 13, ( Arg, arg)) , HPX_FWD_ARGS(2, 14, ( Arg, arg)) , HPX_FWD_ARGS(2, 15, ( Arg, arg)) , HPX_FWD_ARGS(2, 16, ( Arg, arg)) , HPX_FWD_ARGS(2, 17, ( Arg, arg)) , HPX_FWD_ARGS(2, 18, ( Arg, arg))) const
            {
                try {
                    LTM_(debug) << "Executing plain action("
                                << detail::get_action_name<Derived>()
                                << ").";
                    
                    
                    
                    
                    
                    F(HPX_MOVE_ARGS(2, 0, arg) , HPX_MOVE_ARGS(2, 1, arg) , HPX_MOVE_ARGS(2, 2, arg) , HPX_MOVE_ARGS(2, 3, arg) , HPX_MOVE_ARGS(2, 4, arg) , HPX_MOVE_ARGS(2, 5, arg) , HPX_MOVE_ARGS(2, 6, arg) , HPX_MOVE_ARGS(2, 7, arg) , HPX_MOVE_ARGS(2, 8, arg) , HPX_MOVE_ARGS(2, 9, arg) , HPX_MOVE_ARGS(2, 10, arg) , HPX_MOVE_ARGS(2, 11, arg) , HPX_MOVE_ARGS(2, 12, arg) , HPX_MOVE_ARGS(2, 13, arg) , HPX_MOVE_ARGS(2, 14, arg) , HPX_MOVE_ARGS(2, 15, arg) , HPX_MOVE_ARGS(2, 16, arg) , HPX_MOVE_ARGS(2, 17, arg) , HPX_MOVE_ARGS(2, 18, arg));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing plain action("
                            << detail::get_action_name<Derived>()
                            << "): " << e.what();
                        
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
            return HPX_STD_BIND(typename Derived::thread_function(),
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type12>::call( args. a12) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type13>::call( args. a13) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type14>::call( args. a14) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type15>::call( args. a15) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type16>::call( args. a16) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type17>::call( args. a17) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type18>::call( args. a18));
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return boost::move(
                base_type::construct_continuation_thread_function(
                    cont, F, boost::forward<Arguments>(args)));
        }
    };
    
    
    template <
        typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct plain_result_action19
      : plain_base_result_action19<Result,
          T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18, F,
          typename detail::action_type<
              plain_result_action19<
                  Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18, F, Priority>, Derived
          >::type, Priority>
    {
        typedef typename detail::action_type<
            plain_result_action19<
                Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18, F, Priority>, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18), typename Derived>
    struct make_action<Result (*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18), F, Derived, boost::mpl::false_>
      : boost::mpl::identity<plain_result_action19<Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18, F, threads::thread_priority_default,
            Derived> >
    {};
    
    
    template <
        typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18),
        typename Derived = detail::this_type>
    struct plain_direct_result_action19
      : plain_base_result_action19<Result,
          T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18, F,
          typename detail::action_type<
              plain_direct_result_action19<
                  Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18, F>, Derived
          >::type>
    {
        typedef typename detail::action_type<
            plain_direct_result_action19<
                Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18, F>, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        template <typename Arguments>
        static Result
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "plain_direct_result_action" << 19
                << "::execute_function name("
                << detail::get_action_name<derived_type>() << ")";
            return F(util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type12>::call( args. a12) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type13>::call( args. a13) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type14>::call( args. a14) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type15>::call( args. a15) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type16>::call( args. a16) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type17>::call( args. a17) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type18>::call( args. a18));
        }
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18), typename Derived>
    struct make_action<Result (*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18), F, Derived, boost::mpl::true_>
      : boost::mpl::identity<plain_direct_result_action19<
            Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18, F, Derived> >
    {};
    
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default>
    class plain_base_action19
      : public action<
            components::server::plain_function<Derived>,
            function_action_arg19,
            util::unused_type,
            hpx::util::tuple19<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type , typename detail::remove_qualifiers<T8>::type , typename detail::remove_qualifiers<T9>::type , typename detail::remove_qualifiers<T10>::type , typename detail::remove_qualifiers<T11>::type , typename detail::remove_qualifiers<T12>::type , typename detail::remove_qualifiers<T13>::type , typename detail::remove_qualifiers<T14>::type , typename detail::remove_qualifiers<T15>::type , typename detail::remove_qualifiers<T16>::type , typename detail::remove_qualifiers<T17>::type , typename detail::remove_qualifiers<T18>::type>,
            Derived, Priority>
    {
    public:
        typedef util::unused_type result_type;
        typedef
            hpx::util::tuple19<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type , typename detail::remove_qualifiers<T8>::type , typename detail::remove_qualifiers<T9>::type , typename detail::remove_qualifiers<T10>::type , typename detail::remove_qualifiers<T11>::type , typename detail::remove_qualifiers<T12>::type , typename detail::remove_qualifiers<T13>::type , typename detail::remove_qualifiers<T14>::type , typename detail::remove_qualifiers<T15>::type , typename detail::remove_qualifiers<T16>::type , typename detail::remove_qualifiers<T17>::type , typename detail::remove_qualifiers<T18>::type>
        arguments_type;
        typedef action<
            components::server::plain_function<Derived>,
            function_action_arg19, result_type,
            arguments_type, Derived, Priority> base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17 , typename Arg18>
            result_type operator()(
                HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)) , HPX_FWD_ARGS(2, 10, ( Arg, arg)) , HPX_FWD_ARGS(2, 11, ( Arg, arg)) , HPX_FWD_ARGS(2, 12, ( Arg, arg)) , HPX_FWD_ARGS(2, 13, ( Arg, arg)) , HPX_FWD_ARGS(2, 14, ( Arg, arg)) , HPX_FWD_ARGS(2, 15, ( Arg, arg)) , HPX_FWD_ARGS(2, 16, ( Arg, arg)) , HPX_FWD_ARGS(2, 17, ( Arg, arg)) , HPX_FWD_ARGS(2, 18, ( Arg, arg))) const
            {
                try {
                    LTM_(debug) << "Executing plain action("
                                << detail::get_action_name<Derived>()
                                << ").";
                    
                    
                    
                    
                    
                    F(HPX_MOVE_ARGS(2, 0, arg) , HPX_MOVE_ARGS(2, 1, arg) , HPX_MOVE_ARGS(2, 2, arg) , HPX_MOVE_ARGS(2, 3, arg) , HPX_MOVE_ARGS(2, 4, arg) , HPX_MOVE_ARGS(2, 5, arg) , HPX_MOVE_ARGS(2, 6, arg) , HPX_MOVE_ARGS(2, 7, arg) , HPX_MOVE_ARGS(2, 8, arg) , HPX_MOVE_ARGS(2, 9, arg) , HPX_MOVE_ARGS(2, 10, arg) , HPX_MOVE_ARGS(2, 11, arg) , HPX_MOVE_ARGS(2, 12, arg) , HPX_MOVE_ARGS(2, 13, arg) , HPX_MOVE_ARGS(2, 14, arg) , HPX_MOVE_ARGS(2, 15, arg) , HPX_MOVE_ARGS(2, 16, arg) , HPX_MOVE_ARGS(2, 17, arg) , HPX_MOVE_ARGS(2, 18, arg));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing plain action("
                            << detail::get_action_name<Derived>()
                            << "): " << e.what();
                        
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
            return HPX_STD_BIND(typename Derived::thread_function(),
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type12>::call( args. a12) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type13>::call( args. a13) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type14>::call( args. a14) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type15>::call( args. a15) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type16>::call( args. a16) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type17>::call( args. a17) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type18>::call( args. a18));
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return
                base_type::construct_continuation_thread_function_void(
                    cont, F, boost::forward<Arguments>(args));
        }
    };
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct plain_action19
      : plain_base_action19<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18, F,
            typename detail::action_type<
                plain_action19<
                    T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18, F, Priority>, Derived
            >::type, Priority>
    {
        typedef typename detail::action_type<
            plain_action19<
                T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18, F, Priority>, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18), typename Derived>
    struct make_action<void (*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18), F, Derived, boost::mpl::false_>
      : boost::mpl::identity<plain_action19<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18, F, threads::thread_priority_default,
            Derived> >
    {};
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18),
        typename Derived = detail::this_type>
    struct plain_direct_action19
      : plain_base_action19<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18, F,
            typename detail::action_type<
                plain_direct_action19<
                    T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18, F>, Derived
            >::type>
    {
        typedef typename detail::action_type<
            plain_direct_action19<
                T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18, F>, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        template <typename Arguments>
        static util::unused_type
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "plain_direct_action" << 19
                << "::execute_function name("
                << detail::get_action_name<derived_type>() << ")";
            F(util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type12>::call( args. a12) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type13>::call( args. a13) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type14>::call( args. a14) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type15>::call( args. a15) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type16>::call( args. a16) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type17>::call( args. a17) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type18>::call( args. a18));
            return util::unused;
        }
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18), typename Derived>
    struct make_action<void (*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18), F, Derived, boost::mpl::true_>
      : boost::mpl::identity<plain_direct_action19<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18, F, Derived> >
    {};
    
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18),
        threads::thread_priority Priority, typename Derived>
    struct plain_result_action19<
                void, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18, F, Priority, Derived>
      : plain_action19<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18, F, Priority, Derived>
    {};
}}
namespace hpx { namespace traits
{
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17 , typename Arg18,
        void (*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17 , Arg18),
        hpx::threads::thread_priority Priority, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_action19<
                    Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17 , Arg18, F, Priority> >, Enable>
      : boost::mpl::false_
    {};
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17 , typename Arg18,
        void (*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17 , Arg18), typename Derived,
        typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_direct_action19<
                    Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17 , Arg18, F, Derived> >, Enable>
      : boost::mpl::false_
    {};
    template <typename R, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17 , typename Arg18,
        R(*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17 , Arg18),
        hpx::threads::thread_priority Priority, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_result_action19<
                    R, Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17 , Arg18, F, Priority> >, Enable>
      : boost::mpl::false_
    {};
    template <typename R, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17 , typename Arg18,
        R(*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17 , Arg18), typename Derived, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_direct_result_action19<
                    R, Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17 , Arg18, F, Derived> >, Enable>
      : boost::mpl::false_
    {};
}}
namespace hpx { namespace actions
{
    
    
    template <
        typename Result,
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default
    >
    class plain_base_result_action20
      : public action<
            components::server::plain_function<Derived>,
            function_result_action_arg20,
            Result,
            hpx::util::tuple20<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type , typename detail::remove_qualifiers<T8>::type , typename detail::remove_qualifiers<T9>::type , typename detail::remove_qualifiers<T10>::type , typename detail::remove_qualifiers<T11>::type , typename detail::remove_qualifiers<T12>::type , typename detail::remove_qualifiers<T13>::type , typename detail::remove_qualifiers<T14>::type , typename detail::remove_qualifiers<T15>::type , typename detail::remove_qualifiers<T16>::type , typename detail::remove_qualifiers<T17>::type , typename detail::remove_qualifiers<T18>::type , typename detail::remove_qualifiers<T19>::type>,
            Derived, Priority>
    {
    public:
        typedef Result result_type;
        typedef hpx::util::tuple20<
            typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type , typename detail::remove_qualifiers<T8>::type , typename detail::remove_qualifiers<T9>::type , typename detail::remove_qualifiers<T10>::type , typename detail::remove_qualifiers<T11>::type , typename detail::remove_qualifiers<T12>::type , typename detail::remove_qualifiers<T13>::type , typename detail::remove_qualifiers<T14>::type , typename detail::remove_qualifiers<T15>::type , typename detail::remove_qualifiers<T16>::type , typename detail::remove_qualifiers<T17>::type , typename detail::remove_qualifiers<T18>::type , typename detail::remove_qualifiers<T19>::type> arguments_type;
        typedef action<
            components::server::plain_function<Derived>,
            function_result_action_arg20, result_type,
            arguments_type, Derived, Priority> base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17 , typename Arg18 , typename Arg19>
            result_type operator()(
                HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)) , HPX_FWD_ARGS(2, 10, ( Arg, arg)) , HPX_FWD_ARGS(2, 11, ( Arg, arg)) , HPX_FWD_ARGS(2, 12, ( Arg, arg)) , HPX_FWD_ARGS(2, 13, ( Arg, arg)) , HPX_FWD_ARGS(2, 14, ( Arg, arg)) , HPX_FWD_ARGS(2, 15, ( Arg, arg)) , HPX_FWD_ARGS(2, 16, ( Arg, arg)) , HPX_FWD_ARGS(2, 17, ( Arg, arg)) , HPX_FWD_ARGS(2, 18, ( Arg, arg)) , HPX_FWD_ARGS(2, 19, ( Arg, arg))) const
            {
                try {
                    LTM_(debug) << "Executing plain action("
                                << detail::get_action_name<Derived>()
                                << ").";
                    
                    
                    
                    
                    
                    F(HPX_MOVE_ARGS(2, 0, arg) , HPX_MOVE_ARGS(2, 1, arg) , HPX_MOVE_ARGS(2, 2, arg) , HPX_MOVE_ARGS(2, 3, arg) , HPX_MOVE_ARGS(2, 4, arg) , HPX_MOVE_ARGS(2, 5, arg) , HPX_MOVE_ARGS(2, 6, arg) , HPX_MOVE_ARGS(2, 7, arg) , HPX_MOVE_ARGS(2, 8, arg) , HPX_MOVE_ARGS(2, 9, arg) , HPX_MOVE_ARGS(2, 10, arg) , HPX_MOVE_ARGS(2, 11, arg) , HPX_MOVE_ARGS(2, 12, arg) , HPX_MOVE_ARGS(2, 13, arg) , HPX_MOVE_ARGS(2, 14, arg) , HPX_MOVE_ARGS(2, 15, arg) , HPX_MOVE_ARGS(2, 16, arg) , HPX_MOVE_ARGS(2, 17, arg) , HPX_MOVE_ARGS(2, 18, arg) , HPX_MOVE_ARGS(2, 19, arg));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing plain action("
                            << detail::get_action_name<Derived>()
                            << "): " << e.what();
                        
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
            return HPX_STD_BIND(typename Derived::thread_function(),
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type12>::call( args. a12) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type13>::call( args. a13) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type14>::call( args. a14) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type15>::call( args. a15) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type16>::call( args. a16) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type17>::call( args. a17) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type18>::call( args. a18) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type19>::call( args. a19));
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return boost::move(
                base_type::construct_continuation_thread_function(
                    cont, F, boost::forward<Arguments>(args)));
        }
    };
    
    
    template <
        typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct plain_result_action20
      : plain_base_result_action20<Result,
          T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19, F,
          typename detail::action_type<
              plain_result_action20<
                  Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19, F, Priority>, Derived
          >::type, Priority>
    {
        typedef typename detail::action_type<
            plain_result_action20<
                Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19, F, Priority>, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19), typename Derived>
    struct make_action<Result (*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19), F, Derived, boost::mpl::false_>
      : boost::mpl::identity<plain_result_action20<Result,
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19, F, threads::thread_priority_default,
            Derived> >
    {};
    
    
    template <
        typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19),
        typename Derived = detail::this_type>
    struct plain_direct_result_action20
      : plain_base_result_action20<Result,
          T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19, F,
          typename detail::action_type<
              plain_direct_result_action20<
                  Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19, F>, Derived
          >::type>
    {
        typedef typename detail::action_type<
            plain_direct_result_action20<
                Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19, F>, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        template <typename Arguments>
        static Result
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "plain_direct_result_action" << 20
                << "::execute_function name("
                << detail::get_action_name<derived_type>() << ")";
            return F(util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type12>::call( args. a12) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type13>::call( args. a13) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type14>::call( args. a14) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type15>::call( args. a15) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type16>::call( args. a16) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type17>::call( args. a17) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type18>::call( args. a18) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type19>::call( args. a19));
        }
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19,
        Result (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19), typename Derived>
    struct make_action<Result (*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19), F, Derived, boost::mpl::true_>
      : boost::mpl::identity<plain_direct_result_action20<
            Result, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19, F, Derived> >
    {};
    
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default>
    class plain_base_action20
      : public action<
            components::server::plain_function<Derived>,
            function_action_arg20,
            util::unused_type,
            hpx::util::tuple20<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type , typename detail::remove_qualifiers<T8>::type , typename detail::remove_qualifiers<T9>::type , typename detail::remove_qualifiers<T10>::type , typename detail::remove_qualifiers<T11>::type , typename detail::remove_qualifiers<T12>::type , typename detail::remove_qualifiers<T13>::type , typename detail::remove_qualifiers<T14>::type , typename detail::remove_qualifiers<T15>::type , typename detail::remove_qualifiers<T16>::type , typename detail::remove_qualifiers<T17>::type , typename detail::remove_qualifiers<T18>::type , typename detail::remove_qualifiers<T19>::type>,
            Derived, Priority>
    {
    public:
        typedef util::unused_type result_type;
        typedef
            hpx::util::tuple20<typename detail::remove_qualifiers<T0>::type , typename detail::remove_qualifiers<T1>::type , typename detail::remove_qualifiers<T2>::type , typename detail::remove_qualifiers<T3>::type , typename detail::remove_qualifiers<T4>::type , typename detail::remove_qualifiers<T5>::type , typename detail::remove_qualifiers<T6>::type , typename detail::remove_qualifiers<T7>::type , typename detail::remove_qualifiers<T8>::type , typename detail::remove_qualifiers<T9>::type , typename detail::remove_qualifiers<T10>::type , typename detail::remove_qualifiers<T11>::type , typename detail::remove_qualifiers<T12>::type , typename detail::remove_qualifiers<T13>::type , typename detail::remove_qualifiers<T14>::type , typename detail::remove_qualifiers<T15>::type , typename detail::remove_qualifiers<T16>::type , typename detail::remove_qualifiers<T17>::type , typename detail::remove_qualifiers<T18>::type , typename detail::remove_qualifiers<T19>::type>
        arguments_type;
        typedef action<
            components::server::plain_function<Derived>,
            function_action_arg20, result_type,
            arguments_type, Derived, Priority> base_type;
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17 , typename Arg18 , typename Arg19>
            result_type operator()(
                HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)) , HPX_FWD_ARGS(2, 10, ( Arg, arg)) , HPX_FWD_ARGS(2, 11, ( Arg, arg)) , HPX_FWD_ARGS(2, 12, ( Arg, arg)) , HPX_FWD_ARGS(2, 13, ( Arg, arg)) , HPX_FWD_ARGS(2, 14, ( Arg, arg)) , HPX_FWD_ARGS(2, 15, ( Arg, arg)) , HPX_FWD_ARGS(2, 16, ( Arg, arg)) , HPX_FWD_ARGS(2, 17, ( Arg, arg)) , HPX_FWD_ARGS(2, 18, ( Arg, arg)) , HPX_FWD_ARGS(2, 19, ( Arg, arg))) const
            {
                try {
                    LTM_(debug) << "Executing plain action("
                                << detail::get_action_name<Derived>()
                                << ").";
                    
                    
                    
                    
                    
                    F(HPX_MOVE_ARGS(2, 0, arg) , HPX_MOVE_ARGS(2, 1, arg) , HPX_MOVE_ARGS(2, 2, arg) , HPX_MOVE_ARGS(2, 3, arg) , HPX_MOVE_ARGS(2, 4, arg) , HPX_MOVE_ARGS(2, 5, arg) , HPX_MOVE_ARGS(2, 6, arg) , HPX_MOVE_ARGS(2, 7, arg) , HPX_MOVE_ARGS(2, 8, arg) , HPX_MOVE_ARGS(2, 9, arg) , HPX_MOVE_ARGS(2, 10, arg) , HPX_MOVE_ARGS(2, 11, arg) , HPX_MOVE_ARGS(2, 12, arg) , HPX_MOVE_ARGS(2, 13, arg) , HPX_MOVE_ARGS(2, 14, arg) , HPX_MOVE_ARGS(2, 15, arg) , HPX_MOVE_ARGS(2, 16, arg) , HPX_MOVE_ARGS(2, 17, arg) , HPX_MOVE_ARGS(2, 18, arg) , HPX_MOVE_ARGS(2, 19, arg));
                }
                catch (hpx::exception const& e) {
                    if (e.get_error() != hpx::thread_interrupted) {
                        LTM_(error)
                            << "Unhandled exception while executing plain action("
                            << detail::get_action_name<Derived>()
                            << "): " << e.what();
                        
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
            return HPX_STD_BIND(typename Derived::thread_function(),
                util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type12>::call( args. a12) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type13>::call( args. a13) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type14>::call( args. a14) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type15>::call( args. a15) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type16>::call( args. a16) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type17>::call( args. a17) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type18>::call( args. a18) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type19>::call( args. a19));
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return
                base_type::construct_continuation_thread_function_void(
                    cont, F, boost::forward<Arguments>(args));
        }
    };
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    struct plain_action20
      : plain_base_action20<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19, F,
            typename detail::action_type<
                plain_action20<
                    T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19, F, Priority>, Derived
            >::type, Priority>
    {
        typedef typename detail::action_type<
            plain_action20<
                T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19, F, Priority>, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19), typename Derived>
    struct make_action<void (*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19), F, Derived, boost::mpl::false_>
      : boost::mpl::identity<plain_action20<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19, F, threads::thread_priority_default,
            Derived> >
    {};
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19),
        typename Derived = detail::this_type>
    struct plain_direct_action20
      : plain_base_action20<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19, F,
            typename detail::action_type<
                plain_direct_action20<
                    T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19, F>, Derived
            >::type>
    {
        typedef typename detail::action_type<
            plain_direct_action20<
                T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19, F>, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        template <typename Arguments>
        static util::unused_type
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "plain_direct_action" << 20
                << "::execute_function name("
                << detail::get_action_name<derived_type>() << ")";
            F(util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type0>::call( args. a0) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type1>::call( args. a1) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type2>::call( args. a2) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type3>::call( args. a3) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type4>::call( args. a4) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type5>::call( args. a5) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type6>::call( args. a6) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type7>::call( args. a7) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type8>::call( args. a8) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type9>::call( args. a9) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type10>::call( args. a10) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type11>::call( args. a11) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type12>::call( args. a12) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type13>::call( args. a13) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type14>::call( args. a14) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type15>::call( args. a15) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type16>::call( args. a16) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type17>::call( args. a17) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type18>::call( args. a18) , util::detail::move_if_no_ref< typename util::detail::remove_reference<Arguments>::type:: member_type19>::call( args. a19));
            return util::unused;
        }
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19), typename Derived>
    struct make_action<void (*)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19), F, Derived, boost::mpl::true_>
      : boost::mpl::identity<plain_direct_action20<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19, F, Derived> >
    {};
    
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19,
        void (*F)(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19),
        threads::thread_priority Priority, typename Derived>
    struct plain_result_action20<
                void, T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19, F, Priority, Derived>
      : plain_action20<
            T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17 , T18 , T19, F, Priority, Derived>
    {};
}}
namespace hpx { namespace traits
{
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17 , typename Arg18 , typename Arg19,
        void (*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17 , Arg18 , Arg19),
        hpx::threads::thread_priority Priority, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_action20<
                    Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17 , Arg18 , Arg19, F, Priority> >, Enable>
      : boost::mpl::false_
    {};
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17 , typename Arg18 , typename Arg19,
        void (*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17 , Arg18 , Arg19), typename Derived,
        typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_direct_action20<
                    Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17 , Arg18 , Arg19, F, Derived> >, Enable>
      : boost::mpl::false_
    {};
    template <typename R, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17 , typename Arg18 , typename Arg19,
        R(*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17 , Arg18 , Arg19),
        hpx::threads::thread_priority Priority, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_result_action20<
                    R, Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17 , Arg18 , Arg19, F, Priority> >, Enable>
      : boost::mpl::false_
    {};
    template <typename R, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17 , typename Arg18 , typename Arg19,
        R(*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17 , Arg18 , Arg19), typename Derived, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_direct_result_action20<
                    R, Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17 , Arg18 , Arg19, F, Derived> >, Enable>
      : boost::mpl::false_
    {};
}}
