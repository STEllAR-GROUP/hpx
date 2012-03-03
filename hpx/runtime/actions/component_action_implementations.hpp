//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_RUNTIME_ACTIONS_ACTION_IMPLEMENTATIONS_MAY_20_2008_1104AM)
#define HPX_RUNTIME_ACTIONS_ACTION_IMPLEMENTATIONS_MAY_20_2008_1104AM

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repeat.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (1, HPX_ACTION_ARGUMENT_LIMIT,                                        \
    "hpx/runtime/actions/component_action_implementations.hpp"))              \
    /**/

#include BOOST_PP_ITERATE()

#endif

///////////////////////////////////////////////////////////////////////////////
//  Preprocessor vertical repetition code
///////////////////////////////////////////////////////////////////////////////
#else // defined(BOOST_PP_IS_ITERATING)

#define N BOOST_PP_ITERATION()
#define HPX_ACTION_ARGUMENT(z, n, data)                                       \
        BOOST_PP_COMMA_IF(n) boost::move(data.get<n>())                       \
    /**/
#define HPX_ACTION_DIRECT_ARGUMENT(z, n, data)                                \
        BOOST_PP_COMMA_IF(n) boost::move(boost::fusion::at_c<n>(data))        \
    /**/
#define HPX_REMOVE_QUALIFIERS(z, n, data)                                     \
        BOOST_PP_COMMA_IF(n)                                                  \
        typename detail::remove_qualifiers<BOOST_PP_CAT(T, n)>::type          \
    /**/

#define HPX_FWD_ARGS(z, n, _)                                                 \
        BOOST_PP_COMMA_IF(n)                                                  \
            BOOST_FWD_REF(BOOST_PP_CAT(Arg, n)) BOOST_PP_CAT(arg, n)          \
    /**/

#define HPX_MOVE_ARGS(z, n, _)                                                \
        BOOST_PP_COMMA_IF(n)                                                  \
            boost::move(BOOST_PP_CAT(arg, n))                                 \
    /**/

#define HPX_FORWARD_ARGS(z, n, _)                                             \
        BOOST_PP_COMMA_IF(n)                                                  \
            boost::forward<BOOST_PP_CAT(Arg, n)>(BOOST_PP_CAT(arg, n))        \
    /**/

namespace hpx { namespace actions
{
    ///////////////////////////////////////////////////////////////////////////
    //  N parameter version, with result
    template <
        typename Component, typename Result, int Action,
        BOOST_PP_ENUM_PARAMS(N, typename T),
        Result (Component::*F)(BOOST_PP_ENUM_PARAMS(N, T)), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default>
    class BOOST_PP_CAT(base_result_action, N)
      : public action<
            Component, Action, Result,
            boost::fusion::vector<BOOST_PP_REPEAT(N, HPX_REMOVE_QUALIFIERS, _)>,
            Derived, Priority>
    {
    public:
        typedef Result result_type;
        typedef boost::fusion::vector<
            BOOST_PP_REPEAT(N, HPX_REMOVE_QUALIFIERS, _)> arguments_type;
        typedef action<Component, Action, result_type, arguments_type,
                       Derived, Priority>
            base_type;

        explicit BOOST_PP_CAT(base_result_action, N)(
                threads::thread_priority priority = Priority)
          : base_type(priority)
        {}

        // construct an action from its arguments
        template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        BOOST_PP_CAT(base_result_action, N)(
                BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
          : base_type(BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _))
        {}

        template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        BOOST_PP_CAT(base_result_action, N)(
                threads::thread_priority priority,
                BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
          : base_type(priority, BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _))
        {}

    protected:
        /// The \a thread_function will be registered as the thread
        /// function of a thread. It encapsulates the execution of the
        /// original function (given by \a func).
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;

            template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
            result_type operator()(
                naming::address::address_type lva,
                BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _)) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component>::call(lva)) << ")";
                    (get_lva<Component>::call(lva)->*F)(
                        BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _));
                }
                catch (hpx::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component>::call(lva)) << "): " << e.what();

                    // report this error to the console in any case
                    hpx::report_error(boost::current_exception());
                }
                return threads::terminated;
            }
        };

    public:
        typedef boost::mpl::false_ direct_execution;

        // This static construct_thread_function allows to construct
        // a proper thread function for a thread without having to
        // instantiate the base_result_actionN type. This is used by the
        // applier in case no continuation has been supplied.
        template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(naming::address::address_type lva,
            BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
        {
            return HPX_STD_BIND(typename Derived::thread_function()
                    , lva, BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _));
        }

        // This static construct_thread_function allows to construct
        // a proper thread function for a thread without having to
        // instantiate the base_result_actionN type. This is used by the
        // applier in case a continuation has been supplied
        template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva,
            BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
        {
            return base_type::construct_continuation_thread_object_function(
                cont, F, get_lva<Component>::call(lva),
                BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _));
        }

        /// serialization support
        static void register_base()
        {
            util::void_cast_register_nonvirt<
                BOOST_PP_CAT(base_result_action, N), base_type>();
            base_type::register_base();
        }

    private:
        // This get_thread_function will be invoked to retrieve the thread
        // function for an action which has to be invoked without continuations.
        HPX_STD_FUNCTION<threads::thread_function_type>
        get_thread_function(naming::address::address_type lva)
        {
            return construct_thread_function(lva,
                BOOST_PP_REPEAT(N, HPX_ACTION_ARGUMENT, (*this)));
        }

        // This get_thread_function will be invoked to retrieve the thread
        // function for an action which has to be invoked with continuations.
        HPX_STD_FUNCTION<threads::thread_function_type>
        get_thread_function(continuation_type& cont,
            naming::address::address_type lva)
        {
            return construct_thread_function(cont, lva,
                BOOST_PP_REPEAT(N, HPX_ACTION_ARGUMENT, (*this)));
        }

    private:
        // serialization support
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int /*version*/)
        {
            ar & util::base_object_nonvirt<base_type>(*this);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    //  N parameter version, direct execution with result
    template <
        typename Component, typename Result, int Action,
        BOOST_PP_ENUM_PARAMS(N, typename T),
        Result (Component::*F)(BOOST_PP_ENUM_PARAMS(N, T)),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    class BOOST_PP_CAT(result_action, N)
      : public BOOST_PP_CAT(base_result_action, N)<
            Component, Result, Action,
            BOOST_PP_ENUM_PARAMS(N, T), F,
            typename detail::action_type<
                BOOST_PP_CAT(result_action, N)<
                    Component, Result, Action, BOOST_PP_ENUM_PARAMS(N, T), F,
                        Priority>,
                Derived
            >::type, Priority>
    {
    private:
        typedef typename detail::action_type<
            BOOST_PP_CAT(result_action, N)<
                Component, Result, Action, BOOST_PP_ENUM_PARAMS(N, T), F,
                    Priority>,
            Derived
        >::type derived_type;

        typedef BOOST_PP_CAT(base_result_action, N)<
            Component, Result, Action, BOOST_PP_ENUM_PARAMS(N, T), F,
            derived_type, Priority> base_type;

    public:
        BOOST_PP_CAT(result_action, N)(
                threads::thread_priority priority = Priority)
          : base_type(priority)
        {}

        template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        static Result execute_function(
            naming::address::address_type lva,
            BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
        {
            LTM_(debug)
                << "base_result_action" << N
                << "::execute_function name("
                << detail::get_action_name<derived_type>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";

            return (get_lva<Component>::call(lva)->*F)(
                BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _));
        }

        // construct an action from its arguments
        template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        BOOST_PP_CAT(result_action, N)(
                BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
          : base_type(BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _))
        {}

        template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        BOOST_PP_CAT(result_action, N)(
                threads::thread_priority priority,
                BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
          : base_type(priority, BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _))
        {}

        /// serialization support
        static void register_base()
        {
            util::void_cast_register_nonvirt<
                BOOST_PP_CAT(result_action, N), base_type>();
            base_type::register_base();
        }

    private:
        // serialization support
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int /*version*/)
        {
            ar & util::base_object_nonvirt<base_type>(*this);
        }

    private:
        threads::thread_init_data&
        get_thread_init_data(naming::address::address_type lva,
            threads::thread_init_data& data)
        {
            data.lva = lva;
            data.func = this->construct_thread_function(lva,
                BOOST_PP_REPEAT(N, HPX_ACTION_ARGUMENT, (*this)));
            data.description = detail::get_action_name<derived_type>();
            data.parent_id =
                reinterpret_cast<threads::thread_id_type>(this->parent_id_);
            data.parent_prefix = this->parent_locality_;
            data.priority = this->priority_;
            return data;
        }

        threads::thread_init_data&
        get_thread_init_data(continuation_type& cont,
            naming::address::address_type lva,
            threads::thread_init_data& data)
        {
            data.lva = lva;
            data.func = this->construct_thread_function(cont, lva,
                BOOST_PP_REPEAT(N, HPX_ACTION_ARGUMENT, (*this)));
            data.description = detail::get_action_name<derived_type>();
            data.parent_id =
                reinterpret_cast<threads::thread_id_type>(this->parent_id_);
            data.parent_prefix = this->parent_locality_;
            data.priority = this->priority_;
            return data;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    //  N parameter version, direct execution with result
    template <
        typename Component, typename Result, int Action,
        BOOST_PP_ENUM_PARAMS(N, typename T),
        Result (Component::*F)(BOOST_PP_ENUM_PARAMS(N, T)),
        typename Derived = detail::this_type>
    class BOOST_PP_CAT(direct_result_action, N)
      : public BOOST_PP_CAT(base_result_action, N)<
            Component, Result, Action,
            BOOST_PP_ENUM_PARAMS(N, T), F,
            typename detail::action_type<
                BOOST_PP_CAT(direct_result_action, N)<
                    Component, Result, Action, BOOST_PP_ENUM_PARAMS(N, T), F>,
                    Derived
            >::type>
    {
    private:
        typedef typename detail::action_type<
            BOOST_PP_CAT(direct_result_action, N)<
                Component, Result, Action, BOOST_PP_ENUM_PARAMS(N, T), F>,
                Derived
        >::type derived_type;

        typedef BOOST_PP_CAT(base_result_action, N)<
            Component, Result, Action, BOOST_PP_ENUM_PARAMS(N, T), F,
            derived_type> base_type;

    public:
        BOOST_PP_CAT(direct_result_action, N)()
        {}

        // construct an action from its arguments
        template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        BOOST_PP_CAT(direct_result_action, N)(
                BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
          : base_type(BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _))
        {}

        template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        BOOST_PP_CAT(direct_result_action, N)(
                threads::thread_priority,
                BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
          : base_type(BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _))
        {}

    public:
        typedef boost::mpl::true_ direct_execution;

        template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        static Result
        execute_function(naming::address::address_type lva,
            BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
        {
            LTM_(debug)
                << "base_result_action" << N
                << "::execute_function name("
                << detail::get_action_name<derived_type>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";

            return (get_lva<Component>::call(lva)->*F)(
                BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _));
        }

        /// serialization support
        static void register_base()
        {
            util::void_cast_register_nonvirt<
                BOOST_PP_CAT(direct_result_action, N), base_type>();
            base_type::register_base();
        }

    private:
        // serialization support
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int /*version*/)
        {
            ar & util::base_object_nonvirt<base_type>(*this);
        }

    private:
        threads::thread_init_data&
        get_thread_init_data(naming::address::address_type lva,
            threads::thread_init_data& data)
        {
            data.lva = lva;
            data.func = this->construct_thread_function(lva,
                BOOST_PP_REPEAT(N, HPX_ACTION_ARGUMENT, (*this)));
            data.description = detail::get_action_name<derived_type>();
            data.parent_id =
                reinterpret_cast<threads::thread_id_type>(this->parent_id_);
            data.parent_prefix = this->parent_locality_;
            data.priority = this->priority_;
            return data;
        }

        threads::thread_init_data&
        get_thread_init_data(continuation_type& cont,
            naming::address::address_type lva,
            threads::thread_init_data& data)
        {
            data.lva = lva;
            data.func = this->construct_thread_function(cont, lva,
                BOOST_PP_REPEAT(N, HPX_ACTION_ARGUMENT, (*this)));
            data.description = detail::get_action_name<derived_type>();
            data.parent_id =
                reinterpret_cast<threads::thread_id_type>(this->parent_id_);
            data.parent_prefix = this->parent_locality_;
            data.priority = this->priority_;
            return data;
        }

        /// The function \a get_action_type returns whether this action needs
        /// to be executed in a new thread or directly.
        base_action::action_type get_action_type() const
        {
            return base_action::direct_action;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    //  N parameter version, no result type
    template <
        typename Component, int Action, BOOST_PP_ENUM_PARAMS(N, typename T),
        void (Component::*F)(BOOST_PP_ENUM_PARAMS(N, T)), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default>
    class BOOST_PP_CAT(base_action, N)
      : public action<
            Component, Action, util::unused_type,
            BOOST_PP_CAT(hpx::util::tuple, N)<BOOST_PP_REPEAT(N, HPX_REMOVE_QUALIFIERS, _)>,
            Derived, Priority>
    {
    public:
        typedef util::unused_type result_type;
        typedef BOOST_PP_CAT(hpx::util::tuple, N)<
            BOOST_PP_REPEAT(N, HPX_REMOVE_QUALIFIERS, _)> arguments_type;
        typedef action<Component, Action, result_type, arguments_type,
                       Derived, Priority> base_type;

        explicit BOOST_PP_CAT(base_action, N)(
                threads::thread_priority priority = Priority)
          : base_type(priority)
        {}

        // construct an action from its arguments
        template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        BOOST_PP_CAT(base_action, N)(
                BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
          : base_type(BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _))
        {}

        template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        BOOST_PP_CAT(base_action, N)(
                threads::thread_priority priority,
                BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
          : base_type(priority, BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _))
        {}

    protected:
        /// The \a thread_function will be registered as the thread
        /// function of a thread. It encapsulates the execution of the
        /// original function (given by \a func).
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;

            template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
            result_type operator()(
                naming::address::address_type lva,
                BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _)) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component>::call(lva)) << ")";
                    (get_lva<Component>::call(lva)->*F)(
                        BOOST_PP_REPEAT(N, HPX_MOVE_ARGS, _));
                }
                catch (hpx::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component>::call(lva)) << "): " << e.what();

                    // report this error to the console in any case
                    hpx::report_error(boost::current_exception());
                }
                return threads::terminated;
            }
        };

    public:
        typedef boost::mpl::false_ direct_execution;

        // This static construct_thread_function allows to construct
        // a proper thread function for a thread without having to
        // instantiate the base_actionN type. This is used by the applier in
        // case no continuation has been supplied.
        template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(naming::address::address_type lva,
            BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
        {
            // we need to assign the address of the thread function to a
            // variable to  help the compiler to deduce the function type
            return boost::move(HPX_STD_BIND(
                typename Derived::thread_function(), lva,
                BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _)));
        }

        // This static construct_thread_function allows to construct
        // a proper thread function for a thread without having to
        // instantiate the base_actionN type. This is used by the applier in
        // case a continuation has been supplied
        template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva,
            BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
        {
            return boost::move(
                base_type::construct_continuation_thread_object_function_void(
                    cont, F, get_lva<Component>::call(lva),
                    BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _)));
        }

        /// serialization support
        static void register_base()
        {
            util::void_cast_register_nonvirt<
                BOOST_PP_CAT(base_action, N), base_type>();
            base_type::register_base();
        }

    private:
        ///
        HPX_STD_FUNCTION<threads::thread_function_type>
        get_thread_function(naming::address::address_type lva)
        {
            return boost::move(construct_thread_function(lva,
                BOOST_PP_REPEAT(N, HPX_ACTION_ARGUMENT, (*this))));
        }

        HPX_STD_FUNCTION<threads::thread_function_type>
        get_thread_function(continuation_type& cont,
            naming::address::address_type lva)
        {
            return boost::move(construct_thread_function(cont, lva,
                BOOST_PP_REPEAT(N, HPX_ACTION_ARGUMENT, (*this))));
        }

    private:
        // serialization support
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int /*version*/)
        {
            ar & util::base_object_nonvirt<base_type>(*this);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <
        typename Component, int Action, BOOST_PP_ENUM_PARAMS(N, typename T),
        void (Component::*F)(BOOST_PP_ENUM_PARAMS(N, T)),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    class BOOST_PP_CAT(action, N)
      : public BOOST_PP_CAT(base_action, N)<
            Component, Action, BOOST_PP_ENUM_PARAMS(N, T), F,
            typename detail::action_type<
                BOOST_PP_CAT(action, N)<
                    Component, Action, BOOST_PP_ENUM_PARAMS(N, T), F, Priority>,
                Derived
            >::type, Priority>
    {
    private:
        typedef typename detail::action_type<
            BOOST_PP_CAT(action, N)<
                Component, Action, BOOST_PP_ENUM_PARAMS(N, T), F, Priority>,
            Derived
        >::type derived_type;

        typedef BOOST_PP_CAT(base_action, N)<
            Component, Action, BOOST_PP_ENUM_PARAMS(N, T), F,
            derived_type, Priority> base_type;

    public:
        BOOST_PP_CAT(action, N)(
                threads::thread_priority priority = Priority)
          : base_type(priority)
        {}

        // construct an action from its arguments
        template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        BOOST_PP_CAT(action, N)(
                BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
          : base_type(BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _))
        {}

        template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        BOOST_PP_CAT(action, N)(
                threads::thread_priority priority,
                BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
          : base_type(priority, BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _))
        {}

        template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        static util::unused_type
        execute_function(naming::address::address_type lva,
            BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
        {
            LTM_(debug)
                << "action" << N
                << "::execute_function name("
                << detail::get_action_name<derived_type>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";

            (get_lva<Component>::call(lva)->*F)(
                BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _));
            return util::unused;
        }

        /// serialization support
        static void register_base()
        {
            util::void_cast_register_nonvirt<
                BOOST_PP_CAT(action, N), base_type>();
            base_type::register_base();
        }

    private:
        // serialization support
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int /*version*/)
        {
            ar & util::base_object_nonvirt<base_type>(*this);
        }

    private:
        threads::thread_init_data&
        get_thread_init_data(naming::address::address_type lva,
            threads::thread_init_data& data)
        {
            data.lva = lva;
            data.func = this->construct_thread_function(lva,
                BOOST_PP_REPEAT(N, HPX_ACTION_ARGUMENT, (*this)));
            data.description = detail::get_action_name<derived_type>();
            data.parent_id =
                reinterpret_cast<threads::thread_id_type>(this->parent_id_);
            data.parent_prefix = this->parent_locality_;
            data.priority = this->priority_;
            return data;
        }

        threads::thread_init_data&
        get_thread_init_data(continuation_type& cont,
            naming::address::address_type lva,
            threads::thread_init_data& data)
        {
            data.lva = lva;
            data.func = this->construct_thread_function(cont, lva,
                BOOST_PP_REPEAT(N, HPX_ACTION_ARGUMENT, (*this)));
            data.description = detail::get_action_name<derived_type>();
            data.parent_id =
                reinterpret_cast<threads::thread_id_type>(this->parent_id_);
            data.parent_prefix = this->parent_locality_;
            data.priority = this->priority_;
            return data;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <
        typename Component, int Action, BOOST_PP_ENUM_PARAMS(N, typename T),
        void (Component::*F)(BOOST_PP_ENUM_PARAMS(N, T)),
        typename Derived = detail::this_type>
    class BOOST_PP_CAT(direct_action, N)
      : public BOOST_PP_CAT(base_action, N)<
            Component, Action, BOOST_PP_ENUM_PARAMS(N, T), F,
            typename detail::action_type<
                BOOST_PP_CAT(direct_action, N)<
                    Component, Action, BOOST_PP_ENUM_PARAMS(N, T), F>,
                    Derived
            >::type>
    {
    private:
        typedef typename detail::action_type<
            BOOST_PP_CAT(direct_action, N)<
                Component, Action, BOOST_PP_ENUM_PARAMS(N, T), F>,
                Derived
        >::type derived_type;

        typedef BOOST_PP_CAT(base_action, N)<
            Component, Action, BOOST_PP_ENUM_PARAMS(N, T), F,
            derived_type> base_type;

    public:
        BOOST_PP_CAT(direct_action, N)()
        {}

        // construct an action from its arguments
        template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        BOOST_PP_CAT(direct_action, N)(
                BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
          : base_type(BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _))
        {}

        template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        BOOST_PP_CAT(direct_action, N)(
                threads::thread_priority,
                BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
          : base_type(BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _))
        {}

    public:
        typedef boost::mpl::true_ direct_execution;

        template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        static util::unused_type
        execute_function(naming::address::address_type lva,
            BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
        {
            LTM_(debug)
                << "direct_action" << N
                << "::execute_function name("
                << detail::get_action_name<derived_type>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";

            (get_lva<Component>::call(lva)->*F)(
                BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _));
            return util::unused;
        }

        /// serialization support
        static void register_base()
        {
            util::void_cast_register_nonvirt<
                BOOST_PP_CAT(direct_action, N), base_type>();
            base_type::register_base();
        }

    private:
        // serialization support
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int /*version*/)
        {
            ar & util::base_object_nonvirt<base_type>(*this);
        }

    private:
        /// The function \a get_action_type returns whether this action needs
        /// to be executed in a new thread or directly.
        base_action::action_type get_action_type() const
        {
            return base_action::direct_action;
        }

        threads::thread_init_data&
        get_thread_init_data(naming::address::address_type lva,
            threads::thread_init_data& data)
        {
            data.lva = lva;
            data.func = this->construct_thread_function(lva,
                BOOST_PP_REPEAT(N, HPX_ACTION_ARGUMENT, (*this)));
            data.description = detail::get_action_name<derived_type>();
            data.parent_id =
                reinterpret_cast<threads::thread_id_type>(this->parent_id_);
            data.parent_prefix = this->parent_locality_;
            data.priority = this->priority_;
            return data;
        }

        threads::thread_init_data&
        get_thread_init_data(continuation_type& cont,
            naming::address::address_type lva,
            threads::thread_init_data& data)
        {
            data.lva = lva;
            data.func = this->construct_thread_function(cont, lva,
                BOOST_PP_REPEAT(N, HPX_ACTION_ARGUMENT, (*this)));
            data.description = detail::get_action_name<derived_type>();
            data.parent_id =
                reinterpret_cast<threads::thread_id_type>(this->parent_id_);
            data.parent_prefix = this->parent_locality_;
            data.priority = this->priority_;
            return data;
        }
    };

    template <
        typename Component, int Action,
        BOOST_PP_ENUM_PARAMS(N, typename T),
        void (Component::*F)(BOOST_PP_ENUM_PARAMS(N, T)),
        threads::thread_priority Priority,
        typename Derived>
    class BOOST_PP_CAT(result_action, N)<Component, void, Action,
            BOOST_PP_ENUM_PARAMS(N, T), F, Priority, Derived>
      : public BOOST_PP_CAT(action, N)<Component, Action,
            BOOST_PP_ENUM_PARAMS(N, T), F, Priority, Derived>
    {
        typedef BOOST_PP_CAT(action, N)<Component, Action,
            BOOST_PP_ENUM_PARAMS(N, T), F, Priority, Derived> base_type;

    public:
        BOOST_PP_CAT(result_action, N)(
                threads::thread_priority priority = Priority)
          : base_type(priority)
        {}

        // construct an action from its arguments
        template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        BOOST_PP_CAT(result_action, N)(
                BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
          : base_type(BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _))
        {}

        template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        BOOST_PP_CAT(result_action, N)(
                threads::thread_priority priority,
                BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
          : base_type(priority, BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _))
        {}

        /// serialization support
        static void register_base()
        {
            util::void_cast_register_nonvirt<
                BOOST_PP_CAT(result_action, N), base_type>();
            base_type::register_base();
        }

    private:
        // serialization support
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int /*version*/)
        {
            ar & util::base_object_nonvirt<base_type>(*this);
        }
    };
}}

///////////////////////////////////////////////////////////////////////////////
#undef HPX_FORWARD_ARGS
#undef HPX_FWD_ARGS
#undef HPX_REMOVE_QUALIFIERS
#undef HPX_ACTION_DIRECT_ARGUMENT
#undef HPX_ACTION_ARGUMENT
#undef N

#endif

