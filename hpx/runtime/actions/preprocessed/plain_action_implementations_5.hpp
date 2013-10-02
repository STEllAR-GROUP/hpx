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
        typename Result,
        typename T0,
        Result (*F)(T0), typename Derived>
    class plain_base_result_action1
      : public action<
            components::server::plain_function<Derived>,
            Result,
            hpx::util::tuple<typename util::decay<T0>::type>,
            Derived>
    {
    public:
        typedef Result result_type;
        typedef typename detail::remote_action_result<Result>::type
            remote_result_type;
        typedef hpx::util::tuple<
            typename util::decay<T0>::type> arguments_type;
        typedef action<
            components::server::plain_function<Derived>, remote_result_type,
            arguments_type, Derived> base_type;
        
        static bool is_target_valid(naming::id_type const& id)
        {
            return naming::is_locality(id);
        }
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0>
            BOOST_FORCEINLINE result_type operator()(
                BOOST_FWD_REF(Arg0) arg0) const
            {
                try {
                    LTM_(debug) << "Executing plain action("
                                << detail::get_action_name<Derived>()
                                << ").";
                    
                    
                    
                    
                    
                    F(boost::move(arg0));
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
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing plain action("
                        << detail::get_action_name<Derived>() << ")";
                    
                    hpx::report_error(boost::current_exception());
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
            return boost::move(Derived::decorate_action(
                HPX_STD_BIND(typename Derived::thread_function(),
                    boost::forward< typename util::remove_reference<Arguments>::type:: member_type0>( args. a0)), lva));
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return boost::move(Derived::decorate_action(
                base_type::construct_continuation_thread_function(
                    cont, F, boost::forward<Arguments>(args)), lva));
        }
        
        template <typename Arguments>
        BOOST_FORCEINLINE static Result
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "plain_base_result_action" << 1
                << "::execute_function name("
                << detail::get_action_name<Derived>() << ")";
            return F(boost::forward< typename util::remove_reference<Arguments>::type:: member_type0>( args. a0));
        }
    };
    
    
    template <
        typename Result, typename T0,
        Result (*F)(T0),
        typename Derived = detail::this_type>
    struct plain_result_action1
      : plain_base_result_action1<Result,
          T0, F,
          typename detail::action_type<
              plain_result_action1<
                  Result, T0, F>, Derived
          >::type>
    {
        typedef typename detail::action_type<
            plain_result_action1, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Result, typename T0,
        Result (*F)(T0), typename Derived>
    struct make_action<Result (*)(T0), F, Derived, boost::mpl::false_>
      : plain_result_action1<
            Result, T0, F, Derived>
    {
        typedef plain_result_action1<
            Result, T0, F, Derived
        > type;
    };
    
    
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
            plain_direct_result_action1, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename Result, typename T0,
        Result (*F)(T0), typename Derived>
    struct make_action<Result (*)(T0), F, Derived, boost::mpl::true_>
      : plain_direct_result_action1<
            Result, T0, F, Derived>
    {
        typedef plain_direct_result_action1<
            Result, T0, F, Derived
        > type;
    };
    
    
    template <
        typename T0,
        void (*F)(T0), typename Derived>
    class plain_base_action1
      : public action<
            components::server::plain_function<Derived>,
            util::unused_type,
            hpx::util::tuple<typename util::decay<T0>::type>,
            Derived>
    {
    public:
        typedef util::unused_type result_type;
        typedef util::unused_type remote_result_type;
        typedef
            hpx::util::tuple<typename util::decay<T0>::type>
        arguments_type;
        typedef action<
            components::server::plain_function<Derived>, remote_result_type,
            arguments_type, Derived> base_type;
        
        static bool is_target_valid(naming::id_type const& id)
        {
            return naming::is_locality(id);
        }
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0>
            BOOST_FORCEINLINE result_type operator()(
                BOOST_FWD_REF(Arg0) arg0) const
            {
                try {
                    LTM_(debug) << "Executing plain action("
                                << detail::get_action_name<Derived>()
                                << ").";
                    
                    
                    
                    
                    
                    F(boost::move(arg0));
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
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing plain action("
                        << detail::get_action_name<Derived>() << ")";
                    
                    hpx::report_error(boost::current_exception());
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
            return boost::move(Derived::decorate_action(
                HPX_STD_BIND(typename Derived::thread_function(),
                    boost::forward< typename util::remove_reference<Arguments>::type:: member_type0>( args. a0)), lva));
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return boost::move(Derived::decorate_action(
                base_type::construct_continuation_thread_function_void(
                    cont, F, boost::forward<Arguments>(args)), lva));
        }
        
        template <typename Arguments>
        BOOST_FORCEINLINE static util::unused_type
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "plain_base_action" << 1
                << "::execute_function name("
                << detail::get_action_name<Derived>() << ")";
            F(boost::forward< typename util::remove_reference<Arguments>::type:: member_type0>( args. a0));
            return util::unused;
        }
    };
    
    template <
        typename T0,
        void (*F)(T0),
        typename Derived = detail::this_type>
    struct plain_action1
      : plain_base_action1<
            T0, F,
            typename detail::action_type<
                plain_action1<
                    T0, F>, Derived
            >::type>
    {
        typedef typename detail::action_type<
            plain_action1, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename T0,
        void (*F)(T0), typename Derived>
    struct make_action<void (*)(T0), F, Derived, boost::mpl::false_>
      : plain_action1<
            T0, F, Derived>
    {
        typedef plain_action1<
            T0, F, Derived
        > type;
    };
    
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
            plain_direct_action1, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename T0,
        void (*F)(T0), typename Derived>
    struct make_action<void (*)(T0), F, Derived, boost::mpl::true_>
      : plain_direct_action1<
            T0, F, Derived>
    {
        typedef plain_direct_action1<
            T0, F, Derived
        > type;
    };
    
    
    template <
        typename T0,
        void (*F)(T0), typename Derived>
    struct plain_result_action1<
                void, T0, F, Derived>
      : plain_action1<
            T0, F, Derived>
    {};
}}
namespace hpx { namespace traits
{
    template <typename Arg0,
        void (*F)(Arg0), typename Derived, 
        typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_action1<
                    Arg0, F, Derived> >, Enable>
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
        R(*F)(Arg0), typename Derived, 
        typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_result_action1<
                    R, Arg0, F, Derived> >, Enable>
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
        Result (*F)(T0 , T1), typename Derived>
    class plain_base_result_action2
      : public action<
            components::server::plain_function<Derived>,
            Result,
            hpx::util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type>,
            Derived>
    {
    public:
        typedef Result result_type;
        typedef typename detail::remote_action_result<Result>::type
            remote_result_type;
        typedef hpx::util::tuple<
            typename util::decay<T0>::type , typename util::decay<T1>::type> arguments_type;
        typedef action<
            components::server::plain_function<Derived>, remote_result_type,
            arguments_type, Derived> base_type;
        
        static bool is_target_valid(naming::id_type const& id)
        {
            return naming::is_locality(id);
        }
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1>
            BOOST_FORCEINLINE result_type operator()(
                BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1) const
            {
                try {
                    LTM_(debug) << "Executing plain action("
                                << detail::get_action_name<Derived>()
                                << ").";
                    
                    
                    
                    
                    
                    F(boost::move(arg0) , boost::move(arg1));
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
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing plain action("
                        << detail::get_action_name<Derived>() << ")";
                    
                    hpx::report_error(boost::current_exception());
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
            return boost::move(Derived::decorate_action(
                HPX_STD_BIND(typename Derived::thread_function(),
                    boost::forward< typename util::remove_reference<Arguments>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments>::type:: member_type1>( args. a1)), lva));
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return boost::move(Derived::decorate_action(
                base_type::construct_continuation_thread_function(
                    cont, F, boost::forward<Arguments>(args)), lva));
        }
        
        template <typename Arguments>
        BOOST_FORCEINLINE static Result
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "plain_base_result_action" << 2
                << "::execute_function name("
                << detail::get_action_name<Derived>() << ")";
            return F(boost::forward< typename util::remove_reference<Arguments>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments>::type:: member_type1>( args. a1));
        }
    };
    
    
    template <
        typename Result, typename T0 , typename T1,
        Result (*F)(T0 , T1),
        typename Derived = detail::this_type>
    struct plain_result_action2
      : plain_base_result_action2<Result,
          T0 , T1, F,
          typename detail::action_type<
              plain_result_action2<
                  Result, T0 , T1, F>, Derived
          >::type>
    {
        typedef typename detail::action_type<
            plain_result_action2, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Result, typename T0 , typename T1,
        Result (*F)(T0 , T1), typename Derived>
    struct make_action<Result (*)(T0 , T1), F, Derived, boost::mpl::false_>
      : plain_result_action2<
            Result, T0 , T1, F, Derived>
    {
        typedef plain_result_action2<
            Result, T0 , T1, F, Derived
        > type;
    };
    
    
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
            plain_direct_result_action2, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename Result, typename T0 , typename T1,
        Result (*F)(T0 , T1), typename Derived>
    struct make_action<Result (*)(T0 , T1), F, Derived, boost::mpl::true_>
      : plain_direct_result_action2<
            Result, T0 , T1, F, Derived>
    {
        typedef plain_direct_result_action2<
            Result, T0 , T1, F, Derived
        > type;
    };
    
    
    template <
        typename T0 , typename T1,
        void (*F)(T0 , T1), typename Derived>
    class plain_base_action2
      : public action<
            components::server::plain_function<Derived>,
            util::unused_type,
            hpx::util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type>,
            Derived>
    {
    public:
        typedef util::unused_type result_type;
        typedef util::unused_type remote_result_type;
        typedef
            hpx::util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type>
        arguments_type;
        typedef action<
            components::server::plain_function<Derived>, remote_result_type,
            arguments_type, Derived> base_type;
        
        static bool is_target_valid(naming::id_type const& id)
        {
            return naming::is_locality(id);
        }
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1>
            BOOST_FORCEINLINE result_type operator()(
                BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1) const
            {
                try {
                    LTM_(debug) << "Executing plain action("
                                << detail::get_action_name<Derived>()
                                << ").";
                    
                    
                    
                    
                    
                    F(boost::move(arg0) , boost::move(arg1));
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
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing plain action("
                        << detail::get_action_name<Derived>() << ")";
                    
                    hpx::report_error(boost::current_exception());
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
            return boost::move(Derived::decorate_action(
                HPX_STD_BIND(typename Derived::thread_function(),
                    boost::forward< typename util::remove_reference<Arguments>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments>::type:: member_type1>( args. a1)), lva));
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return boost::move(Derived::decorate_action(
                base_type::construct_continuation_thread_function_void(
                    cont, F, boost::forward<Arguments>(args)), lva));
        }
        
        template <typename Arguments>
        BOOST_FORCEINLINE static util::unused_type
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "plain_base_action" << 2
                << "::execute_function name("
                << detail::get_action_name<Derived>() << ")";
            F(boost::forward< typename util::remove_reference<Arguments>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments>::type:: member_type1>( args. a1));
            return util::unused;
        }
    };
    
    template <
        typename T0 , typename T1,
        void (*F)(T0 , T1),
        typename Derived = detail::this_type>
    struct plain_action2
      : plain_base_action2<
            T0 , T1, F,
            typename detail::action_type<
                plain_action2<
                    T0 , T1, F>, Derived
            >::type>
    {
        typedef typename detail::action_type<
            plain_action2, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename T0 , typename T1,
        void (*F)(T0 , T1), typename Derived>
    struct make_action<void (*)(T0 , T1), F, Derived, boost::mpl::false_>
      : plain_action2<
            T0 , T1, F, Derived>
    {
        typedef plain_action2<
            T0 , T1, F, Derived
        > type;
    };
    
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
            plain_direct_action2, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename T0 , typename T1,
        void (*F)(T0 , T1), typename Derived>
    struct make_action<void (*)(T0 , T1), F, Derived, boost::mpl::true_>
      : plain_direct_action2<
            T0 , T1, F, Derived>
    {
        typedef plain_direct_action2<
            T0 , T1, F, Derived
        > type;
    };
    
    
    template <
        typename T0 , typename T1,
        void (*F)(T0 , T1), typename Derived>
    struct plain_result_action2<
                void, T0 , T1, F, Derived>
      : plain_action2<
            T0 , T1, F, Derived>
    {};
}}
namespace hpx { namespace traits
{
    template <typename Arg0 , typename Arg1,
        void (*F)(Arg0 , Arg1), typename Derived, 
        typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_action2<
                    Arg0 , Arg1, F, Derived> >, Enable>
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
        R(*F)(Arg0 , Arg1), typename Derived, 
        typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_result_action2<
                    R, Arg0 , Arg1, F, Derived> >, Enable>
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
        Result (*F)(T0 , T1 , T2), typename Derived>
    class plain_base_result_action3
      : public action<
            components::server::plain_function<Derived>,
            Result,
            hpx::util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type>,
            Derived>
    {
    public:
        typedef Result result_type;
        typedef typename detail::remote_action_result<Result>::type
            remote_result_type;
        typedef hpx::util::tuple<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type> arguments_type;
        typedef action<
            components::server::plain_function<Derived>, remote_result_type,
            arguments_type, Derived> base_type;
        
        static bool is_target_valid(naming::id_type const& id)
        {
            return naming::is_locality(id);
        }
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2>
            BOOST_FORCEINLINE result_type operator()(
                BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2) const
            {
                try {
                    LTM_(debug) << "Executing plain action("
                                << detail::get_action_name<Derived>()
                                << ").";
                    
                    
                    
                    
                    
                    F(boost::move(arg0) , boost::move(arg1) , boost::move(arg2));
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
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing plain action("
                        << detail::get_action_name<Derived>() << ")";
                    
                    hpx::report_error(boost::current_exception());
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
            return boost::move(Derived::decorate_action(
                HPX_STD_BIND(typename Derived::thread_function(),
                    boost::forward< typename util::remove_reference<Arguments>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments>::type:: member_type2>( args. a2)), lva));
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return boost::move(Derived::decorate_action(
                base_type::construct_continuation_thread_function(
                    cont, F, boost::forward<Arguments>(args)), lva));
        }
        
        template <typename Arguments>
        BOOST_FORCEINLINE static Result
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "plain_base_result_action" << 3
                << "::execute_function name("
                << detail::get_action_name<Derived>() << ")";
            return F(boost::forward< typename util::remove_reference<Arguments>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments>::type:: member_type2>( args. a2));
        }
    };
    
    
    template <
        typename Result, typename T0 , typename T1 , typename T2,
        Result (*F)(T0 , T1 , T2),
        typename Derived = detail::this_type>
    struct plain_result_action3
      : plain_base_result_action3<Result,
          T0 , T1 , T2, F,
          typename detail::action_type<
              plain_result_action3<
                  Result, T0 , T1 , T2, F>, Derived
          >::type>
    {
        typedef typename detail::action_type<
            plain_result_action3, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Result, typename T0 , typename T1 , typename T2,
        Result (*F)(T0 , T1 , T2), typename Derived>
    struct make_action<Result (*)(T0 , T1 , T2), F, Derived, boost::mpl::false_>
      : plain_result_action3<
            Result, T0 , T1 , T2, F, Derived>
    {
        typedef plain_result_action3<
            Result, T0 , T1 , T2, F, Derived
        > type;
    };
    
    
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
            plain_direct_result_action3, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename Result, typename T0 , typename T1 , typename T2,
        Result (*F)(T0 , T1 , T2), typename Derived>
    struct make_action<Result (*)(T0 , T1 , T2), F, Derived, boost::mpl::true_>
      : plain_direct_result_action3<
            Result, T0 , T1 , T2, F, Derived>
    {
        typedef plain_direct_result_action3<
            Result, T0 , T1 , T2, F, Derived
        > type;
    };
    
    
    template <
        typename T0 , typename T1 , typename T2,
        void (*F)(T0 , T1 , T2), typename Derived>
    class plain_base_action3
      : public action<
            components::server::plain_function<Derived>,
            util::unused_type,
            hpx::util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type>,
            Derived>
    {
    public:
        typedef util::unused_type result_type;
        typedef util::unused_type remote_result_type;
        typedef
            hpx::util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type>
        arguments_type;
        typedef action<
            components::server::plain_function<Derived>, remote_result_type,
            arguments_type, Derived> base_type;
        
        static bool is_target_valid(naming::id_type const& id)
        {
            return naming::is_locality(id);
        }
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2>
            BOOST_FORCEINLINE result_type operator()(
                BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2) const
            {
                try {
                    LTM_(debug) << "Executing plain action("
                                << detail::get_action_name<Derived>()
                                << ").";
                    
                    
                    
                    
                    
                    F(boost::move(arg0) , boost::move(arg1) , boost::move(arg2));
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
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing plain action("
                        << detail::get_action_name<Derived>() << ")";
                    
                    hpx::report_error(boost::current_exception());
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
            return boost::move(Derived::decorate_action(
                HPX_STD_BIND(typename Derived::thread_function(),
                    boost::forward< typename util::remove_reference<Arguments>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments>::type:: member_type2>( args. a2)), lva));
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return boost::move(Derived::decorate_action(
                base_type::construct_continuation_thread_function_void(
                    cont, F, boost::forward<Arguments>(args)), lva));
        }
        
        template <typename Arguments>
        BOOST_FORCEINLINE static util::unused_type
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "plain_base_action" << 3
                << "::execute_function name("
                << detail::get_action_name<Derived>() << ")";
            F(boost::forward< typename util::remove_reference<Arguments>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments>::type:: member_type2>( args. a2));
            return util::unused;
        }
    };
    
    template <
        typename T0 , typename T1 , typename T2,
        void (*F)(T0 , T1 , T2),
        typename Derived = detail::this_type>
    struct plain_action3
      : plain_base_action3<
            T0 , T1 , T2, F,
            typename detail::action_type<
                plain_action3<
                    T0 , T1 , T2, F>, Derived
            >::type>
    {
        typedef typename detail::action_type<
            plain_action3, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename T0 , typename T1 , typename T2,
        void (*F)(T0 , T1 , T2), typename Derived>
    struct make_action<void (*)(T0 , T1 , T2), F, Derived, boost::mpl::false_>
      : plain_action3<
            T0 , T1 , T2, F, Derived>
    {
        typedef plain_action3<
            T0 , T1 , T2, F, Derived
        > type;
    };
    
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
            plain_direct_action3, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename T0 , typename T1 , typename T2,
        void (*F)(T0 , T1 , T2), typename Derived>
    struct make_action<void (*)(T0 , T1 , T2), F, Derived, boost::mpl::true_>
      : plain_direct_action3<
            T0 , T1 , T2, F, Derived>
    {
        typedef plain_direct_action3<
            T0 , T1 , T2, F, Derived
        > type;
    };
    
    
    template <
        typename T0 , typename T1 , typename T2,
        void (*F)(T0 , T1 , T2), typename Derived>
    struct plain_result_action3<
                void, T0 , T1 , T2, F, Derived>
      : plain_action3<
            T0 , T1 , T2, F, Derived>
    {};
}}
namespace hpx { namespace traits
{
    template <typename Arg0 , typename Arg1 , typename Arg2,
        void (*F)(Arg0 , Arg1 , Arg2), typename Derived, 
        typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_action3<
                    Arg0 , Arg1 , Arg2, F, Derived> >, Enable>
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
        R(*F)(Arg0 , Arg1 , Arg2), typename Derived, 
        typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_result_action3<
                    R, Arg0 , Arg1 , Arg2, F, Derived> >, Enable>
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
        Result (*F)(T0 , T1 , T2 , T3), typename Derived>
    class plain_base_result_action4
      : public action<
            components::server::plain_function<Derived>,
            Result,
            hpx::util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type>,
            Derived>
    {
    public:
        typedef Result result_type;
        typedef typename detail::remote_action_result<Result>::type
            remote_result_type;
        typedef hpx::util::tuple<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type> arguments_type;
        typedef action<
            components::server::plain_function<Derived>, remote_result_type,
            arguments_type, Derived> base_type;
        
        static bool is_target_valid(naming::id_type const& id)
        {
            return naming::is_locality(id);
        }
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
            BOOST_FORCEINLINE result_type operator()(
                BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3) const
            {
                try {
                    LTM_(debug) << "Executing plain action("
                                << detail::get_action_name<Derived>()
                                << ").";
                    
                    
                    
                    
                    
                    F(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3));
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
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing plain action("
                        << detail::get_action_name<Derived>() << ")";
                    
                    hpx::report_error(boost::current_exception());
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
            return boost::move(Derived::decorate_action(
                HPX_STD_BIND(typename Derived::thread_function(),
                    boost::forward< typename util::remove_reference<Arguments>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments>::type:: member_type2>( args. a2) , boost::forward< typename util::remove_reference<Arguments>::type:: member_type3>( args. a3)), lva));
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return boost::move(Derived::decorate_action(
                base_type::construct_continuation_thread_function(
                    cont, F, boost::forward<Arguments>(args)), lva));
        }
        
        template <typename Arguments>
        BOOST_FORCEINLINE static Result
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "plain_base_result_action" << 4
                << "::execute_function name("
                << detail::get_action_name<Derived>() << ")";
            return F(boost::forward< typename util::remove_reference<Arguments>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments>::type:: member_type2>( args. a2) , boost::forward< typename util::remove_reference<Arguments>::type:: member_type3>( args. a3));
        }
    };
    
    
    template <
        typename Result, typename T0 , typename T1 , typename T2 , typename T3,
        Result (*F)(T0 , T1 , T2 , T3),
        typename Derived = detail::this_type>
    struct plain_result_action4
      : plain_base_result_action4<Result,
          T0 , T1 , T2 , T3, F,
          typename detail::action_type<
              plain_result_action4<
                  Result, T0 , T1 , T2 , T3, F>, Derived
          >::type>
    {
        typedef typename detail::action_type<
            plain_result_action4, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Result, typename T0 , typename T1 , typename T2 , typename T3,
        Result (*F)(T0 , T1 , T2 , T3), typename Derived>
    struct make_action<Result (*)(T0 , T1 , T2 , T3), F, Derived, boost::mpl::false_>
      : plain_result_action4<
            Result, T0 , T1 , T2 , T3, F, Derived>
    {
        typedef plain_result_action4<
            Result, T0 , T1 , T2 , T3, F, Derived
        > type;
    };
    
    
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
            plain_direct_result_action4, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename Result, typename T0 , typename T1 , typename T2 , typename T3,
        Result (*F)(T0 , T1 , T2 , T3), typename Derived>
    struct make_action<Result (*)(T0 , T1 , T2 , T3), F, Derived, boost::mpl::true_>
      : plain_direct_result_action4<
            Result, T0 , T1 , T2 , T3, F, Derived>
    {
        typedef plain_direct_result_action4<
            Result, T0 , T1 , T2 , T3, F, Derived
        > type;
    };
    
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3,
        void (*F)(T0 , T1 , T2 , T3), typename Derived>
    class plain_base_action4
      : public action<
            components::server::plain_function<Derived>,
            util::unused_type,
            hpx::util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type>,
            Derived>
    {
    public:
        typedef util::unused_type result_type;
        typedef util::unused_type remote_result_type;
        typedef
            hpx::util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type>
        arguments_type;
        typedef action<
            components::server::plain_function<Derived>, remote_result_type,
            arguments_type, Derived> base_type;
        
        static bool is_target_valid(naming::id_type const& id)
        {
            return naming::is_locality(id);
        }
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
            BOOST_FORCEINLINE result_type operator()(
                BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3) const
            {
                try {
                    LTM_(debug) << "Executing plain action("
                                << detail::get_action_name<Derived>()
                                << ").";
                    
                    
                    
                    
                    
                    F(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3));
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
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing plain action("
                        << detail::get_action_name<Derived>() << ")";
                    
                    hpx::report_error(boost::current_exception());
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
            return boost::move(Derived::decorate_action(
                HPX_STD_BIND(typename Derived::thread_function(),
                    boost::forward< typename util::remove_reference<Arguments>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments>::type:: member_type2>( args. a2) , boost::forward< typename util::remove_reference<Arguments>::type:: member_type3>( args. a3)), lva));
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return boost::move(Derived::decorate_action(
                base_type::construct_continuation_thread_function_void(
                    cont, F, boost::forward<Arguments>(args)), lva));
        }
        
        template <typename Arguments>
        BOOST_FORCEINLINE static util::unused_type
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "plain_base_action" << 4
                << "::execute_function name("
                << detail::get_action_name<Derived>() << ")";
            F(boost::forward< typename util::remove_reference<Arguments>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments>::type:: member_type2>( args. a2) , boost::forward< typename util::remove_reference<Arguments>::type:: member_type3>( args. a3));
            return util::unused;
        }
    };
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3,
        void (*F)(T0 , T1 , T2 , T3),
        typename Derived = detail::this_type>
    struct plain_action4
      : plain_base_action4<
            T0 , T1 , T2 , T3, F,
            typename detail::action_type<
                plain_action4<
                    T0 , T1 , T2 , T3, F>, Derived
            >::type>
    {
        typedef typename detail::action_type<
            plain_action4, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename T0 , typename T1 , typename T2 , typename T3,
        void (*F)(T0 , T1 , T2 , T3), typename Derived>
    struct make_action<void (*)(T0 , T1 , T2 , T3), F, Derived, boost::mpl::false_>
      : plain_action4<
            T0 , T1 , T2 , T3, F, Derived>
    {
        typedef plain_action4<
            T0 , T1 , T2 , T3, F, Derived
        > type;
    };
    
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
            plain_direct_action4, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename T0 , typename T1 , typename T2 , typename T3,
        void (*F)(T0 , T1 , T2 , T3), typename Derived>
    struct make_action<void (*)(T0 , T1 , T2 , T3), F, Derived, boost::mpl::true_>
      : plain_direct_action4<
            T0 , T1 , T2 , T3, F, Derived>
    {
        typedef plain_direct_action4<
            T0 , T1 , T2 , T3, F, Derived
        > type;
    };
    
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3,
        void (*F)(T0 , T1 , T2 , T3), typename Derived>
    struct plain_result_action4<
                void, T0 , T1 , T2 , T3, F, Derived>
      : plain_action4<
            T0 , T1 , T2 , T3, F, Derived>
    {};
}}
namespace hpx { namespace traits
{
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3,
        void (*F)(Arg0 , Arg1 , Arg2 , Arg3), typename Derived, 
        typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_action4<
                    Arg0 , Arg1 , Arg2 , Arg3, F, Derived> >, Enable>
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
        R(*F)(Arg0 , Arg1 , Arg2 , Arg3), typename Derived, 
        typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_result_action4<
                    R, Arg0 , Arg1 , Arg2 , Arg3, F, Derived> >, Enable>
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
        Result (*F)(T0 , T1 , T2 , T3 , T4), typename Derived>
    class plain_base_result_action5
      : public action<
            components::server::plain_function<Derived>,
            Result,
            hpx::util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type>,
            Derived>
    {
    public:
        typedef Result result_type;
        typedef typename detail::remote_action_result<Result>::type
            remote_result_type;
        typedef hpx::util::tuple<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type> arguments_type;
        typedef action<
            components::server::plain_function<Derived>, remote_result_type,
            arguments_type, Derived> base_type;
        
        static bool is_target_valid(naming::id_type const& id)
        {
            return naming::is_locality(id);
        }
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
            BOOST_FORCEINLINE result_type operator()(
                BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4) const
            {
                try {
                    LTM_(debug) << "Executing plain action("
                                << detail::get_action_name<Derived>()
                                << ").";
                    
                    
                    
                    
                    
                    F(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4));
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
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing plain action("
                        << detail::get_action_name<Derived>() << ")";
                    
                    hpx::report_error(boost::current_exception());
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
            return boost::move(Derived::decorate_action(
                HPX_STD_BIND(typename Derived::thread_function(),
                    boost::forward< typename util::remove_reference<Arguments>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments>::type:: member_type2>( args. a2) , boost::forward< typename util::remove_reference<Arguments>::type:: member_type3>( args. a3) , boost::forward< typename util::remove_reference<Arguments>::type:: member_type4>( args. a4)), lva));
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return boost::move(Derived::decorate_action(
                base_type::construct_continuation_thread_function(
                    cont, F, boost::forward<Arguments>(args)), lva));
        }
        
        template <typename Arguments>
        BOOST_FORCEINLINE static Result
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "plain_base_result_action" << 5
                << "::execute_function name("
                << detail::get_action_name<Derived>() << ")";
            return F(boost::forward< typename util::remove_reference<Arguments>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments>::type:: member_type2>( args. a2) , boost::forward< typename util::remove_reference<Arguments>::type:: member_type3>( args. a3) , boost::forward< typename util::remove_reference<Arguments>::type:: member_type4>( args. a4));
        }
    };
    
    
    template <
        typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        Result (*F)(T0 , T1 , T2 , T3 , T4),
        typename Derived = detail::this_type>
    struct plain_result_action5
      : plain_base_result_action5<Result,
          T0 , T1 , T2 , T3 , T4, F,
          typename detail::action_type<
              plain_result_action5<
                  Result, T0 , T1 , T2 , T3 , T4, F>, Derived
          >::type>
    {
        typedef typename detail::action_type<
            plain_result_action5, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        Result (*F)(T0 , T1 , T2 , T3 , T4), typename Derived>
    struct make_action<Result (*)(T0 , T1 , T2 , T3 , T4), F, Derived, boost::mpl::false_>
      : plain_result_action5<
            Result, T0 , T1 , T2 , T3 , T4, F, Derived>
    {
        typedef plain_result_action5<
            Result, T0 , T1 , T2 , T3 , T4, F, Derived
        > type;
    };
    
    
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
            plain_direct_result_action5, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename Result, typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        Result (*F)(T0 , T1 , T2 , T3 , T4), typename Derived>
    struct make_action<Result (*)(T0 , T1 , T2 , T3 , T4), F, Derived, boost::mpl::true_>
      : plain_direct_result_action5<
            Result, T0 , T1 , T2 , T3 , T4, F, Derived>
    {
        typedef plain_direct_result_action5<
            Result, T0 , T1 , T2 , T3 , T4, F, Derived
        > type;
    };
    
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        void (*F)(T0 , T1 , T2 , T3 , T4), typename Derived>
    class plain_base_action5
      : public action<
            components::server::plain_function<Derived>,
            util::unused_type,
            hpx::util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type>,
            Derived>
    {
    public:
        typedef util::unused_type result_type;
        typedef util::unused_type remote_result_type;
        typedef
            hpx::util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type>
        arguments_type;
        typedef action<
            components::server::plain_function<Derived>, remote_result_type,
            arguments_type, Derived> base_type;
        
        static bool is_target_valid(naming::id_type const& id)
        {
            return naming::is_locality(id);
        }
    protected:
        
        
        
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
            BOOST_FORCEINLINE result_type operator()(
                BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4) const
            {
                try {
                    LTM_(debug) << "Executing plain action("
                                << detail::get_action_name<Derived>()
                                << ").";
                    
                    
                    
                    
                    
                    F(boost::move(arg0) , boost::move(arg1) , boost::move(arg2) , boost::move(arg3) , boost::move(arg4));
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
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing plain action("
                        << detail::get_action_name<Derived>() << ")";
                    
                    hpx::report_error(boost::current_exception());
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
            return boost::move(Derived::decorate_action(
                HPX_STD_BIND(typename Derived::thread_function(),
                    boost::forward< typename util::remove_reference<Arguments>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments>::type:: member_type2>( args. a2) , boost::forward< typename util::remove_reference<Arguments>::type:: member_type3>( args. a3) , boost::forward< typename util::remove_reference<Arguments>::type:: member_type4>( args. a4)), lva));
        }
        
        
        
        
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return boost::move(Derived::decorate_action(
                base_type::construct_continuation_thread_function_void(
                    cont, F, boost::forward<Arguments>(args)), lva));
        }
        
        template <typename Arguments>
        BOOST_FORCEINLINE static util::unused_type
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            LTM_(debug)
                << "plain_base_action" << 5
                << "::execute_function name("
                << detail::get_action_name<Derived>() << ")";
            F(boost::forward< typename util::remove_reference<Arguments>::type:: member_type0>( args. a0) , boost::forward< typename util::remove_reference<Arguments>::type:: member_type1>( args. a1) , boost::forward< typename util::remove_reference<Arguments>::type:: member_type2>( args. a2) , boost::forward< typename util::remove_reference<Arguments>::type:: member_type3>( args. a3) , boost::forward< typename util::remove_reference<Arguments>::type:: member_type4>( args. a4));
            return util::unused;
        }
    };
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        void (*F)(T0 , T1 , T2 , T3 , T4),
        typename Derived = detail::this_type>
    struct plain_action5
      : plain_base_action5<
            T0 , T1 , T2 , T3 , T4, F,
            typename detail::action_type<
                plain_action5<
                    T0 , T1 , T2 , T3 , T4, F>, Derived
            >::type>
    {
        typedef typename detail::action_type<
            plain_action5, Derived
        >::type derived_type;
        typedef boost::mpl::false_ direct_execution;
    };
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        void (*F)(T0 , T1 , T2 , T3 , T4), typename Derived>
    struct make_action<void (*)(T0 , T1 , T2 , T3 , T4), F, Derived, boost::mpl::false_>
      : plain_action5<
            T0 , T1 , T2 , T3 , T4, F, Derived>
    {
        typedef plain_action5<
            T0 , T1 , T2 , T3 , T4, F, Derived
        > type;
    };
    
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
            plain_direct_action5, Derived
        >::type derived_type;
        typedef boost::mpl::true_ direct_execution;
        
        
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        void (*F)(T0 , T1 , T2 , T3 , T4), typename Derived>
    struct make_action<void (*)(T0 , T1 , T2 , T3 , T4), F, Derived, boost::mpl::true_>
      : plain_direct_action5<
            T0 , T1 , T2 , T3 , T4, F, Derived>
    {
        typedef plain_direct_action5<
            T0 , T1 , T2 , T3 , T4, F, Derived
        > type;
    };
    
    
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        void (*F)(T0 , T1 , T2 , T3 , T4), typename Derived>
    struct plain_result_action5<
                void, T0 , T1 , T2 , T3 , T4, F, Derived>
      : plain_action5<
            T0 , T1 , T2 , T3 , T4, F, Derived>
    {};
}}
namespace hpx { namespace traits
{
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4,
        void (*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4), typename Derived, 
        typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_action5<
                    Arg0 , Arg1 , Arg2 , Arg3 , Arg4, F, Derived> >, Enable>
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
        R(*F)(Arg0 , Arg1 , Arg2 , Arg3 , Arg4), typename Derived, 
        typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_result_action5<
                    R, Arg0 , Arg1 , Arg2 , Arg3 , Arg4, F, Derived> >, Enable>
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
