//  Copyright (c) 2007-2011 Hartmut Kaiser
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
#define HPX_ACTION_ARGUMENT(z, n, data) BOOST_PP_COMMA_IF(n) data.get<n>()
#define HPX_ACTION_DIRECT_ARGUMENT(z, n, data)                                \
        BOOST_PP_COMMA_IF(n) boost::fusion::at_c<n>(data)                     \
    /**/
#define HPX_REMOVE_QUALIFIERS(z, n, data)                                     \
        BOOST_PP_COMMA_IF(n)                                                  \
        typename detail::remove_qualifiers<BOOST_PP_CAT(T, n)>::type          \
    /**/
#define HPX_PARAM_TYPES(z, n, data)                                           \
        BOOST_PP_COMMA_IF(n)                                                  \
        BOOST_PP_CAT(data, n) const&                                          \
        BOOST_PP_CAT(BOOST_PP_CAT(data, n), _)                                \
    /**/
#define HPX_PARAM_ARGUMENT(z, n, data)                                        \
        BOOST_PP_COMMA_IF(n) BOOST_PP_CAT(BOOST_PP_CAT(data, n), _)           \
    /**/

#define HPX_FWD_ARGS(z, n, _)                                                 \
        BOOST_PP_COMMA_IF(n)                                                  \
            BOOST_FWD_REF(BOOST_PP_CAT(Arg, n)) BOOST_PP_CAT(arg, n)          \
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
        template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        static threads::thread_state_enum thread_function(
            naming::address::address_type lva,
            BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const& arg))
        {
            try {
                LTM_(debug) << "Executing component action("
                            << detail::get_action_name<Derived>()
                            << ") lva(" << reinterpret_cast<void const*>
                                (get_lva<Component>::call(lva)) << ")";
                (get_lva<Component>::call(lva)->*F)(BOOST_PP_ENUM_PARAMS(N, arg));
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

    public:
        typedef boost::mpl::false_ direct_execution;

        // This static construct_thread_function allows to construct
        // a proper thread function for a thread without having to
        // instantiate the base_result_actionN type. This is used by the
        // applier in case no continuation has been supplied.
        template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(naming::address::address_type lva,
            BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const& arg))
        {
            // we need to assign the address of the thread function to a
            // variable to  help the compiler to deduce the function type
            threads::thread_state_enum (*f)(naming::address::address_type,
                    BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const& arg)) =
                &Derived::template thread_function<BOOST_PP_ENUM_PARAMS(N, Arg)>;

            return HPX_STD_BIND(f, lva, BOOST_PP_ENUM_PARAMS(N, arg));
        }

        // This static construct_thread_function allows to construct
        // a proper thread function for a thread without having to
        // instantiate the base_result_actionN type. This is used by the
        // applier in case a continuation has been supplied
        template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva,
            BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const& arg))
        {
            return base_type::construct_continuation_thread_function(
                HPX_STD_BIND(F, get_lva<Component>::call(lva),
                    BOOST_PP_ENUM_PARAMS(N, arg)), cont);
        }

        /// serialization support
        static void register_base()
        {
            using namespace boost::serialization;
            void_cast_register<BOOST_PP_CAT(base_result_action, N), base_type>();
            base_type::register_base();
        }

    private:
        // This get_thread_function will be invoked to retrieve the thread
        // function for an action which has to be invoked without continuations.
        HPX_STD_FUNCTION<threads::thread_function_type>
        get_thread_function(naming::address::address_type lva) const
        {
            return construct_thread_function(lva,
                BOOST_PP_REPEAT(N, HPX_ACTION_ARGUMENT, (*this)));
        }

        // This get_thread_function will be invoked to retrieve the thread
        // function for an action which has to be invoked with continuations.
        HPX_STD_FUNCTION<threads::thread_function_type>
        get_thread_function(continuation_type& cont,
            naming::address::address_type lva) const
        {
            return construct_thread_function(cont, lva,
                BOOST_PP_REPEAT(N, HPX_ACTION_ARGUMENT, (*this)));
        }

        HPX_STD_FUNCTION<threads::thread_function_type>
        get_thread_function(naming::address::address_type lva,
            arguments_type const& arg) const
        {
            return construct_thread_function(lva,
                BOOST_PP_REPEAT(N, HPX_ACTION_DIRECT_ARGUMENT, arg));
        }

        HPX_STD_FUNCTION<threads::thread_function_type>
        get_thread_function(continuation_type& cont,
            naming::address::address_type lva, arguments_type const& arg) const
        {
            return construct_thread_function(cont, lva,
                BOOST_PP_REPEAT(N, HPX_ACTION_DIRECT_ARGUMENT, arg));
        }

    private:
        // serialization support
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int /*version*/)
        {
            ar & boost::serialization::base_object<base_type>(*this);
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

        Result execute_function(
            naming::address::address_type lva,
            BOOST_PP_REPEAT(N, HPX_PARAM_TYPES, T)
        ) const {
            LTM_(debug)
                << "base_result_action" << N
                << "::execute_function name("
                << detail::get_action_name<derived_type>()
                << ") lva(" << lva << ")";
            return (get_lva<Component>::call(lva)->*F)
                (BOOST_PP_REPEAT(N, HPX_PARAM_ARGUMENT, T));
        }

        static Result execute_function_nonvirt(
            naming::address::address_type lva,
            BOOST_PP_REPEAT(N, HPX_PARAM_TYPES, T)
        ) {
            LTM_(debug)
                << "base_result_action" << N
                << "::execute_function_nonvirt name("
                << detail::get_action_name<derived_type>()
                << ") lva(" << lva << ")";
            return (get_lva<Component>::call(lva)->*F)
                (BOOST_PP_REPEAT(N, HPX_PARAM_ARGUMENT, T));
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
            using namespace boost::serialization;
            void_cast_register<BOOST_PP_CAT(result_action, N), base_type>();
            base_type::register_base();
        }

    private:
        // serialization support
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int /*version*/)
        {
            ar & boost::serialization::base_object<base_type>(*this);
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

        threads::thread_init_data&
        get_thread_init_data(naming::address::address_type lva,
            threads::thread_init_data& data,
            typename base_type::arguments_type const& arg)
        {
            data.lva = lva;
            data.func = this->construct_thread_function(lva,
                BOOST_PP_REPEAT(N, HPX_ACTION_DIRECT_ARGUMENT, arg));
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
            threads::thread_init_data& data,
            typename base_type::arguments_type const& arg)
        {
            data.lva = lva;
            data.func = this->construct_thread_function(cont, lva,
                BOOST_PP_REPEAT(N, HPX_ACTION_DIRECT_ARGUMENT, arg));
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

        Result execute_function(
            naming::address::address_type lva,
            BOOST_PP_REPEAT(N, HPX_PARAM_TYPES, T)
        ) const {
            LTM_(debug)
                << "base_result_action" << N
                << "::execute_function name("
                << detail::get_action_name<derived_type>()
                << ") lva(" << lva << ")";
            return (get_lva<Component>::call(lva)->*F)
                (BOOST_PP_REPEAT(N, HPX_PARAM_ARGUMENT, T));
        }

        static Result execute_function_nonvirt(
            naming::address::address_type lva,
            BOOST_PP_REPEAT(N, HPX_PARAM_TYPES, T)
        ) {
            LTM_(debug)
                << "base_result_action" << N
                << "::execute_function_nonvirt name("
                << detail::get_action_name<derived_type>()
                << ") lva(" << lva << ")";
            return (get_lva<Component>::call(lva)->*F)
                (BOOST_PP_REPEAT(N, HPX_PARAM_ARGUMENT, T));
        }

        /// serialization support
        static void register_base()
        {
            using namespace boost::serialization;
            void_cast_register<BOOST_PP_CAT(direct_result_action, N), base_type>();
            base_type::register_base();
        }

    private:
        // serialization support
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int /*version*/)
        {
            ar & boost::serialization::base_object<base_type>(*this);
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

        threads::thread_init_data&
        get_thread_init_data(naming::address::address_type lva,
            threads::thread_init_data& data,
            typename base_type::arguments_type const& arg)
        {
            data.lva = lva;
            data.func = this->construct_thread_function(lva,
                BOOST_PP_REPEAT(N, HPX_ACTION_DIRECT_ARGUMENT, arg));
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
            threads::thread_init_data& data,
            typename base_type::arguments_type const& arg)
        {
            data.lva = lva;
            data.func = this->construct_thread_function(cont, lva,
                BOOST_PP_REPEAT(N, HPX_ACTION_DIRECT_ARGUMENT, arg));
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
            boost::fusion::vector<BOOST_PP_REPEAT(N, HPX_REMOVE_QUALIFIERS, _)>,
            Derived, Priority>
    {
    public:
        typedef util::unused_type result_type;
        typedef boost::fusion::vector<
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
        template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        static threads::thread_state_enum thread_function(
            naming::address::address_type lva,
            BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const& arg))
        {
            try {
                LTM_(debug) << "Executing component action("
                            << detail::get_action_name<Derived>()
                            << ") lva(" << reinterpret_cast<void const*>
                                (get_lva<Component>::call(lva)) << ")";
                (get_lva<Component>::call(lva)->*F)(BOOST_PP_ENUM_PARAMS(N, arg));
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

    public:
        typedef boost::mpl::false_ direct_execution;

        // This static construct_thread_function allows to construct
        // a proper thread function for a thread without having to
        // instantiate the base_actionN type. This is used by the applier in
        // case no continuation has been supplied.
        template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(naming::address::address_type lva,
            BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const& arg))
        {
            // we need to assign the address of the thread function to a
            // variable to  help the compiler to deduce the function type
            threads::thread_state_enum (*f)(naming::address::address_type,
                    BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const& arg)) =
                &Derived::template thread_function<BOOST_PP_ENUM_PARAMS(N, Arg)>;

            return HPX_STD_BIND(f, lva, BOOST_PP_ENUM_PARAMS(N, arg));
        }

        // This static construct_thread_function allows to construct
        // a proper thread function for a thread without having to
        // instantiate the base_actionN type. This is used by the applier in
        // case a continuation has been supplied
        template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva,
            BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const& arg))
        {
            return base_type::construct_continuation_thread_function_void(
                HPX_STD_BIND(F, get_lva<Component>::call(lva),
                    BOOST_PP_ENUM_PARAMS(N, arg)), cont);
        }

        /// serialization support
        static void register_base()
        {
            using namespace boost::serialization;
            void_cast_register<BOOST_PP_CAT(base_action, N), base_type>();
            base_type::register_base();
        }

    private:
        ///
        HPX_STD_FUNCTION<threads::thread_function_type>
        get_thread_function(naming::address::address_type lva) const
        {
            return construct_thread_function(lva,
                BOOST_PP_REPEAT(N, HPX_ACTION_ARGUMENT, (*this)));
        }

        HPX_STD_FUNCTION<threads::thread_function_type>
        get_thread_function(continuation_type& cont,
            naming::address::address_type lva) const
        {
            return construct_thread_function(cont, lva,
                BOOST_PP_REPEAT(N, HPX_ACTION_ARGUMENT, (*this)));
        }

        ///
        HPX_STD_FUNCTION<threads::thread_function_type>
        get_thread_function(naming::address::address_type lva,
            arguments_type const& arg) const
        {
            return construct_thread_function(lva,
                BOOST_PP_REPEAT(N, HPX_ACTION_DIRECT_ARGUMENT, arg));
        }

        HPX_STD_FUNCTION<threads::thread_function_type>
        get_thread_function(continuation_type& cont,
            naming::address::address_type lva, arguments_type const& arg) const
        {
            return construct_thread_function(cont, lva,
                BOOST_PP_REPEAT(N, HPX_ACTION_DIRECT_ARGUMENT, arg));
        }

    private:
        // serialization support
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int /*version*/)
        {
            ar & boost::serialization::base_object<base_type>(*this);
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

        util::unused_type execute_function(
            naming::address::address_type lva,
            BOOST_PP_REPEAT(N, HPX_PARAM_TYPES, T)) const
        {
            LTM_(debug)
                << "action" << N
                << "::execute_function name("
                << detail::get_action_name<derived_type>()
                << ") lva(" << reinterpret_cast<void const*>
                    (get_lva<Component>::call(lva)) << ")";
            (get_lva<Component>::call(lva)->*F)
                (BOOST_PP_REPEAT(N, HPX_PARAM_ARGUMENT, T));
            return util::unused;
        }

        static util::unused_type execute_function_nonvirt(
            naming::address::address_type lva,
            BOOST_PP_REPEAT(N, HPX_PARAM_TYPES, T))
        {
            LTM_(debug)
                << "action" << N
                << "::execute_function_nonvirt name("
                << detail::get_action_name<derived_type>()
                << ") lva(" << reinterpret_cast<void const*>
                    (get_lva<Component>::call(lva)) << ")";
            (get_lva<Component>::call(lva)->*F)
                (BOOST_PP_REPEAT(N, HPX_PARAM_ARGUMENT, T));
            return util::unused;
        }

        /// serialization support
        static void register_base()
        {
            using namespace boost::serialization;
            void_cast_register<BOOST_PP_CAT(action, N), base_type>();
            base_type::register_base();
        }

    private:
        // serialization support
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int /*version*/)
        {
            ar & boost::serialization::base_object<base_type>(*this);
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

        threads::thread_init_data&
        get_thread_init_data(naming::address::address_type lva,
            threads::thread_init_data& data,
            typename base_type::arguments_type const& arg)
        {
            data.lva = lva;
            data.func = this->construct_thread_function(lva,
                BOOST_PP_REPEAT(N, HPX_ACTION_DIRECT_ARGUMENT, arg));
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
            threads::thread_init_data& data,
            typename base_type::arguments_type const& arg)
        {
            data.lva = lva;
            data.func = this->construct_thread_function(cont, lva,
                BOOST_PP_REPEAT(N, HPX_ACTION_DIRECT_ARGUMENT, arg));
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

        util::unused_type execute_function(
            naming::address::address_type lva,
            BOOST_PP_REPEAT(N, HPX_PARAM_TYPES, T)) const
        {
            LTM_(debug)
                << "direct_action" << N
                << "::execute_function name("
                << detail::get_action_name<derived_type>()
                << ") lva(" << reinterpret_cast<void const*>
                    (get_lva<Component>::call(lva)) << ")";
            (get_lva<Component>::call(lva)->*F)
                (BOOST_PP_REPEAT(N, HPX_PARAM_ARGUMENT, T));
            return util::unused;
        }

        static util::unused_type execute_function_nonvirt(
            naming::address::address_type lva,
            BOOST_PP_REPEAT(N, HPX_PARAM_TYPES, T))
        {
            LTM_(debug)
                << "direct_action" << N
                << "::execute_function_nonvirt name("
                << detail::get_action_name<derived_type>()
                << ") lva(" << reinterpret_cast<void const*>
                    (get_lva<Component>::call(lva)) << ")";
            (get_lva<Component>::call(lva)->*F)
                (BOOST_PP_REPEAT(N, HPX_PARAM_ARGUMENT, T));
            return util::unused;
        }

        /// serialization support
        static void register_base()
        {
            using namespace boost::serialization;
            void_cast_register<BOOST_PP_CAT(direct_action, N), base_type>();
            base_type::register_base();
        }

    private:
        // serialization support
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int /*version*/)
        {
            ar & boost::serialization::base_object<base_type>(*this);
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

        threads::thread_init_data&
        get_thread_init_data(naming::address::address_type lva,
            threads::thread_init_data& data,
            typename base_type::arguments_type const& arg)
        {
            data.lva = lva;
            data.func = this->construct_thread_function(lva,
                BOOST_PP_REPEAT(N, HPX_ACTION_DIRECT_ARGUMENT, arg));
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
            threads::thread_init_data& data,
            typename base_type::arguments_type const& arg)
        {
            data.lva = lva;
            data.func = this->construct_thread_function(cont, lva,
                BOOST_PP_REPEAT(N, HPX_ACTION_DIRECT_ARGUMENT, arg));
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
    class BOOST_PP_CAT(result_action, N) <
        Component
      , void
      , Action
      , BOOST_PP_ENUM_PARAMS(N, T)
      , F
      , Priority
      , Derived
    >
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
            using namespace boost::serialization;
            void_cast_register<BOOST_PP_CAT(result_action, N), base_type>();
            base_type::register_base();
        }

    private:
        // serialization support
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int /*version*/)
        {
            ar & boost::serialization::base_object<base_type>(*this);
        }
    };
}}

///////////////////////////////////////////////////////////////////////////////
#undef HPX_FORWARD_ARGS
#undef HPX_FWD_ARGS
#undef HPX_PARAM_ARGUMENT
#undef HPX_PARAM_TYPES
#undef HPX_REMOVE_QUALIFIERS
#undef HPX_ACTION_DIRECT_ARGUMENT
#undef HPX_ACTION_ARGUMENT
#undef N

#endif

