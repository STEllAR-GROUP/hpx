//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_ACTIONS_PLAIN_ACTION_NOV_14_2008_0706PM)
#define HPX_RUNTIME_ACTIONS_PLAIN_ACTION_NOV_14_2008_0706PM

#include <cstdlib>
#include <stdexcept>

#include <hpx/hpx_fwd.hpp>
#include <hpx/config.hpp>
#include <hpx/config/bind.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/actions/action_support.hpp>
#include <hpx/runtime/components/console_error_sink.hpp>
#include <hpx/runtime/components/server/plain_function.hpp>
#include <hpx/util/unused.hpp>
#include <hpx/util/void_cast.hpp>

#include <boost/version.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repeat.hpp>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace actions
{
    ///////////////////////////////////////////////////////////////////////////
#define HPX_FUNCTION_ARG_ENUM(z, n, data)                                     \
        BOOST_PP_CAT(function_action_arg, BOOST_PP_INC(n)) =                  \
            function_action_base + BOOST_PP_INC(n),                           \
    /**/
#define HPX_FUNCTION_RETARG_ENUM(z, n, data)                                  \
        BOOST_PP_CAT(function_result_action_arg, BOOST_PP_INC(n)) =           \
            function_result_action_base + BOOST_PP_INC(n),                    \
    /**/

    enum function_action
    {
        /// plain (free) remotely callable function identifiers
        function_action_base = 100,
        function_action_arg0 = function_action_base + 0,
        BOOST_PP_REPEAT(HPX_ACTION_ARGUMENT_LIMIT, HPX_FUNCTION_ARG_ENUM, _)

        /// plain (free) remotely callable function identifiers with result
        function_result_action_base = 200,
        function_result_action_arg0 = function_result_action_base + 0,
        BOOST_PP_REPEAT(HPX_ACTION_ARGUMENT_LIMIT, HPX_FUNCTION_RETARG_ENUM, _)
    };

#undef HPX_FUNCTION_RETARG_ENUM
#undef HPX_FUNCTION_ARG_ENUM

    ///////////////////////////////////////////////////////////////////////////
    //  Specialized generic plain (free) action types allowing to hold a
    //  different number of arguments
    ///////////////////////////////////////////////////////////////////////////

    // zero argument version
    template <typename Result, Result (*F)(), typename Derived,
      threads::thread_priority Priority = threads::thread_priority_default>
    class plain_base_result_action0
      : public action<
            components::server::plain_function<Derived>,
            function_result_action_arg0, Result, hpx::util::tuple0<>,
            Derived, Priority>
    {
    public:
        typedef Result result_type;
        typedef hpx::util::tuple0<> arguments_type;
        typedef action<
            components::server::plain_function<Derived>,
            function_result_action_arg0, result_type, arguments_type,
            Derived, Priority
        > base_type;

        explicit plain_base_result_action0(threads::thread_priority priority = Priority)
          : base_type(priority)
        {}

    protected:
        /// The \a thread_function will be registered as the thread
        /// function of a thread. It encapsulates the execution of the
        /// original function (given by \a func), while ignoring the return
        /// value.
        template <typename State>   // dummy template parameter
        static threads::thread_state_enum
        thread_function(State)
        {
            try {
                LTM_(debug) << "Executing plain action("
                            << detail::get_action_name<Derived>()
                            << ").";
                F();      // call the function, ignoring the return value
            }
            catch (hpx::exception const& e) {
                LTM_(error)
                    << "Unhandled exception while executing plain action("
                    << detail::get_action_name<Derived>()
                    << "): " << e.what();

                // report this error to the console in any case
                hpx::report_error(boost::current_exception());
            }
            return threads::terminated;
        }

    public:
        typedef boost::mpl::false_ direct_execution;

        /// \brief This static \a construct_thread_function allows to construct
        /// a proper thread function for a \a thread without having to
        /// instantiate the \a plain_base_result_action0 type. This is used by
        /// the \a applier in case no continuation has been supplied.
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(naming::address::address_type lva)
        {
            // we need to assign the address of the thread function to a
            // variable to  help the compiler to deduce the function type
            threads::thread_state_enum (*f)(threads::thread_state_ex_enum) =
                &Derived::template thread_function<threads::thread_state_ex_enum>;
            return f;
        }

        /// \brief This static \a construct_thread_function allows to construct
        /// a proper thread function for a \a thread without having to
        /// instantiate the \a base_result_action0 type. This is used by the \a
        /// applier in case a continuation has been supplied
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva)
        {
            return base_type::construct_continuation_thread_function(cont, F);
        }

        /// serialization support
        static void register_base()
        {
            util::void_cast_register_nonvirt<plain_base_result_action0, base_type>();
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
        /// This \a get_thread_function will be invoked to retrieve the thread
        /// function for an action which has to be invoked without continuations.
        HPX_STD_FUNCTION<threads::thread_function_type>
        get_thread_function(naming::address::address_type lva)
        {
            return construct_thread_function(lva);
        }

        /// This \a get_thread_function will be invoked to retrieve the thread
        /// function for an action which has to be invoked with continuations.
        HPX_STD_FUNCTION<threads::thread_function_type>
        get_thread_function(continuation_type& cont,
            naming::address::address_type lva)
        {
            return construct_thread_function(cont, lva);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Result, Result (*F)(),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    class plain_result_action0
      : public plain_base_result_action0<Result, F,
            typename detail::action_type<
                plain_result_action0<Result, F, Priority>, Derived
            >::type, Priority>
    {
    private:
        typedef typename detail::action_type<
            plain_result_action0<Result, F, Priority>, Derived
        >::type derived_type;

        typedef plain_base_result_action0<Result, F, derived_type, Priority>
            base_type;

    public:
        explicit plain_result_action0(threads::thread_priority priority = Priority)
          : base_type(priority)
        {}

        static Result 
        execute_function(naming::address::address_type lva)
        {
            LTM_(debug)
                << "plain_result_action0::execute_function: name("
                << detail::get_action_name<derived_type>()
                << ")";
            return F();
        }

        /// serialization support
        static void register_base()
        {
            util::void_cast_register_nonvirt<plain_result_action0, base_type>();
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
            data.func = this->construct_thread_function(lva);
            data.description = detail::get_action_name<derived_type>();
            data.parent_id = reinterpret_cast<threads::thread_id_type>(this->parent_id_);
            data.parent_phase = this->parent_phase_;
            data.parent_locality_id = this->parent_locality_;
            data.priority = this->priority_;
            return data;
        }

        threads::thread_init_data&
        get_thread_init_data(continuation_type& cont,
            naming::address::address_type lva,
            threads::thread_init_data& data)
        {
            data.lva = lva;
            data.func = this->construct_thread_function(cont, lva);
            data.description = detail::get_action_name<derived_type>();
            data.parent_id = reinterpret_cast<threads::thread_id_type>(this->parent_id_);
            data.parent_phase = this->parent_phase_;
            data.parent_locality_id = this->parent_locality_;
            data.priority = this->priority_;
            return data;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Result, Result (*F)(),
        typename Derived = detail::this_type>
    class plain_direct_result_action0
      : public plain_base_result_action0<Result, F,
            typename detail::action_type<
                plain_direct_result_action0<Result, F>, Derived
            >::type>
    {
    private:
        typedef typename detail::action_type<
            plain_direct_result_action0<Result, F>, Derived
        >::type derived_type;

        typedef plain_base_result_action0<Result, F, derived_type> base_type;

    public:
        plain_direct_result_action0()
        {}

        explicit plain_direct_result_action0(threads::thread_priority)
        {}

    public:
        typedef boost::mpl::true_ direct_execution;

        static Result 
        execute_function(naming::address::address_type lva)
        {
            LTM_(debug)
                << "plain_direct_result_action0::execute_function: name("
                << detail::get_action_name<derived_type>()
                << ")";
            return F();
        }

        /// serialization support
        static void register_base()
        {
            util::void_cast_register_nonvirt<plain_direct_result_action0, base_type>();
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
            data.func = this->construct_thread_function(lva);
            data.description = detail::get_action_name<derived_type>();
            data.parent_id =
                reinterpret_cast<threads::thread_id_type>(this->parent_id_);
            data.parent_phase = this->parent_phase_;
            data.parent_locality_id = this->parent_locality_;
            data.priority = this->priority_;
            return data;
        }

        threads::thread_init_data&
        get_thread_init_data(continuation_type& cont,
            naming::address::address_type lva,
            threads::thread_init_data& data)
        {
            data.lva = lva;
            data.func = this->construct_thread_function(cont, lva);
            data.description = detail::get_action_name<derived_type>();
            data.parent_id =
                reinterpret_cast<threads::thread_id_type>(this->parent_id_);
            data.parent_phase = this->parent_phase_;
            data.parent_locality_id = this->parent_locality_;
            data.priority = this->priority_;
            return data;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    //  zero parameter version, no result value
    template <void (*F)(), typename Derived,
      threads::thread_priority Priority = threads::thread_priority_default>
    class plain_base_action0
      : public action<
            components::server::plain_function<Derived>,
            function_action_arg0, util::unused_type, hpx::util::tuple0<>,
            Derived, Priority>
    {
    public:
        typedef util::unused_type result_type;
        typedef hpx::util::tuple0<> arguments_type;
        typedef action<
            components::server::plain_function<Derived>,
            function_action_arg0, result_type, arguments_type,
            Derived, Priority> base_type;

        explicit plain_base_action0(threads::thread_priority priority = Priority)
          : base_type(priority)
        {}

    protected:
        /// The \a continuation_thread_function will be registered as the thread
        /// function of a thread. It encapsulates the execution of the
        /// original function (given by \a func), while ignoring the return
        /// value.
        template <typename State>   // dummy template parameter
        static threads::thread_state_enum
        thread_function(State)
        {
            try {
                LTM_(debug) << "Executing plain action("
                            << detail::get_action_name<Derived>()
                            << ").";
                F();      // call the function, ignoring the return value
            }
            catch (hpx::exception const& e) {
                LTM_(error)
                    << "Unhandled exception while executing plain action("
                    << detail::get_action_name<Derived>()
                    << "): " << e.what();

                // report this error to the console in any case
                hpx::report_error(boost::current_exception());
            }
            return threads::terminated;
        }

    public:
        typedef boost::mpl::false_ direct_execution;

        /// \brief This static \a construct_thread_function allows to construct
        /// a proper thread function for a \a thread without having to
        /// instantiate the base_action0 type. This is used by the \a applier in
        /// case no continuation has been supplied.
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(naming::address::address_type lva)
        {
            // we need to assign the address of the thread function to a
            // variable to  help the compiler to deduce the function type
            threads::thread_state_enum (*f)(threads::thread_state_ex_enum) =
                &Derived::template thread_function<threads::thread_state_ex_enum>;
            return f;
        }

        /// \brief This static \a construct_thread_function allows to construct
        /// a proper thread function for a \a thread without having to
        /// instantiate the base_action0 type. This is used by the \a applier in
        /// case a continuation has been supplied
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva)
        {
            return base_type::construct_continuation_thread_function_void(
                cont, F);
        }

        /// serialization support
        static void register_base()
        {
            util::void_cast_register_nonvirt<plain_base_action0, base_type>();
            base_type::register_base();
        }

    private:
        HPX_STD_FUNCTION<threads::thread_function_type>
        get_thread_function(naming::address::address_type lva)
        {
            return construct_thread_function(lva);
        }

        HPX_STD_FUNCTION<threads::thread_function_type>
        get_thread_function(continuation_type& cont,
            naming::address::address_type lva)
        {
            return construct_thread_function(cont, lva);
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
    template <void (*F)(),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    class plain_action0
      : public plain_base_action0<F,
            typename detail::action_type<
                plain_action0<F, Priority>, Derived
            >::type, Priority>
    {
    private:
        typedef typename detail::action_type<
            plain_action0<F, Priority>, Derived
        >::type derived_type;

        typedef plain_base_action0<F, derived_type, Priority> base_type;

    public:
        explicit plain_action0(threads::thread_priority priority = Priority)
          : base_type(priority)
        {}

        static util::unused_type
        execute_function(naming::address::address_type lva)
        {
            LTM_(debug)
                << "plain_action0::execute_function: name("
                << detail::get_action_name<derived_type>()
                << ")";
            F();
            return util::unused;
        }

        /// serialization support
        static void register_base()
        {
            util::void_cast_register_nonvirt<plain_action0, base_type>();
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
            data.func = this->construct_thread_function(lva);
            data.description = detail::get_action_name<derived_type>();
            data.parent_id =
                reinterpret_cast<threads::thread_id_type>(this->parent_id_);
            data.parent_phase = this->parent_phase_;
            data.parent_locality_id = this->parent_locality_;
            data.priority = this->priority_;
            return data;
        }

        threads::thread_init_data&
        get_thread_init_data(continuation_type& cont,
            naming::address::address_type lva,
            threads::thread_init_data& data)
        {
            data.lva = lva;
            data.func = this->construct_thread_function(cont, lva);
            data.description = detail::get_action_name<derived_type>();
            data.parent_id =
                reinterpret_cast<threads::thread_id_type>(this->parent_id_);
            data.parent_phase = this->parent_phase_;
            data.parent_locality_id = this->parent_locality_;
            data.priority = this->priority_;
            return data;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <void (*F)(), typename Derived = detail::this_type>
    class plain_direct_action0
      : public plain_base_action0<F,
            typename detail::action_type<
                plain_direct_action0<F>, Derived
            >::type>
    {
    private:
        typedef typename detail::action_type<
            plain_direct_action0<F>, Derived
        >::type derived_type;

        typedef plain_base_action0<F, derived_type> base_type;

    public:
        plain_direct_action0()
        {}

        explicit plain_direct_action0(threads::thread_priority)
        {}

    public:
        typedef boost::mpl::true_ direct_execution;

        static util::unused_type
        execute_function(naming::address::address_type lva)
        {
            LTM_(debug)
                << "plain_base_action0::execute_function: name("
                << detail::get_action_name<derived_type>()
                << ")";
            F();
            return util::unused;
        }

        /// serialization support
        static void register_base()
        {
            util::void_cast_register_nonvirt<plain_direct_action0, base_type>();
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
            data.func = this->construct_thread_function(lva);
            data.description = detail::get_action_name<derived_type>();
            data.parent_id = reinterpret_cast<threads::thread_id_type>(this->parent_id_);
            data.parent_phase = this->parent_phase_;
            data.parent_locality_id = this->parent_locality_;
            data.priority = this->priority_;
            return data;
        }

        threads::thread_init_data&
        get_thread_init_data(continuation_type& cont,
            naming::address::address_type lva,
            threads::thread_init_data& data)
        {
            data.lva = lva;
            data.func = this->construct_thread_function(cont, lva);
            data.description = detail::get_action_name<derived_type>();
            data.parent_id = reinterpret_cast<threads::thread_id_type>(this->parent_id_);
            data.parent_phase = this->parent_phase_;
            data.parent_locality_id = this->parent_locality_;
            data.priority = this->priority_;
            return data;
        }
    };

    template <void (*F)(),
        threads::thread_priority Priority,
        typename Derived>
    class plain_result_action0<void, F, Priority, Derived>
        : public plain_action0<F, Priority, Derived>
    {
        typedef plain_action0<F, Priority, Derived> base_type;

    public:
        explicit plain_result_action0(threads::thread_priority priority = Priority)
          : base_type(priority)
        {}

        /// serialization support
        static void register_base()
        {
            util::void_cast_register_nonvirt<plain_result_action0, base_type>();
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

// Disabling the guid initialization stuff for plain actions
namespace hpx { namespace actions { namespace detail {
        template <
            void (*F)()
          , hpx::threads::thread_priority Priority
          , typename Enable
        >
        struct needs_guid_initialization<
            hpx::actions::plain_action0<F, Priority>
          , Enable
        >
            : boost::mpl::false_
        {};

        template <
            void (*F)()
          , typename Derived
          , typename Enable
        >
        struct needs_guid_initialization<
            hpx::actions::plain_direct_action0<F, Derived>
          , Enable
        >
            : boost::mpl::false_
        {};

        template <
            typename R
          , R(*F)()
          , hpx::threads::thread_priority Priority
          , typename Enable
        >
        struct needs_guid_initialization<
            hpx::actions::plain_result_action0<R, F, Priority>
          , Enable
        >
            : boost::mpl::false_
        {};

        template <
            typename R
          , R(*F)()
          , typename Derived
          , typename Enable
        >
        struct needs_guid_initialization<
            hpx::actions::plain_direct_result_action0<R, F, Derived>
          , Enable
        >
            : boost::mpl::false_
        {};
}}}

///////////////////////////////////////////////////////////////////////////
// bring in the rest of the implementations
#include <hpx/runtime/actions/plain_action_implementations.hpp>

///////////////////////////////////////////////////////////////////////////////
/// The macro \a HPX_REGISTER_PLAIN_ACTION_DECLARATION is used create the
/// forward declarations for plain actions. This is only needed if the plain
/// action was declared in a header, and is defined in a source file. Use this
/// macro in the header, and \a HPX_REGISTER_PLAIN_ACTION in the source file
#define HPX_REGISTER_PLAIN_ACTION_DECLARATION(plain_action)                   \
    namespace hpx { namespace actions { namespace detail {                    \
        template <>                                                           \
        HPX_ALWAYS_EXPORT const char *                                        \
        get_action_name<plain_action>();                                      \
    }}}                                                                       \
/**/

#include <hpx/config/warnings_suffix.hpp>

#endif

