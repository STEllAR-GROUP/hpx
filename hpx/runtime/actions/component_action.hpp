//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_ACTIONS_COMPONENT_ACTION_MAR_26_2008_1054AM)
#define HPX_RUNTIME_ACTIONS_COMPONENT_ACTION_MAR_26_2008_1054AM

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
#include <hpx/util/unused.hpp>
#include <hpx/util/void_cast.hpp>

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repeat.hpp>
#if BOOST_WORKAROUND(BOOST_MSVC, == 1600)
#include <boost/mpl/identity.hpp>
#endif

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace actions
{
    ///////////////////////////////////////////////////////////////////////////
#define HPX_FUNCTION_ARG_ENUM(z, n, data)                                     \
        BOOST_PP_CAT(component_action_arg, BOOST_PP_INC(n)) =                 \
            component_action_base + BOOST_PP_INC(n),                          \
    /**/
#define HPX_FUNCTION_RETARG_ENUM(z, n, data)                                  \
        BOOST_PP_CAT(component_result_action_arg, BOOST_PP_INC(n)) =          \
            component_result_action_base + BOOST_PP_INC(n),                   \
    /**/

    enum component_action
    {
        /// remotely callable member function identifiers
        component_action_base = 1000,
        component_action_arg0 = component_action_base + 0,
        BOOST_PP_REPEAT(HPX_ACTION_ARGUMENT_LIMIT, HPX_FUNCTION_ARG_ENUM, _)

        /// remotely callable member function identifiers with result
        component_result_action_base = 2000,
        component_result_action_arg0 = component_result_action_base + 0,
        BOOST_PP_REPEAT(HPX_ACTION_ARGUMENT_LIMIT, HPX_FUNCTION_RETARG_ENUM, _)
    };

#undef HPX_FUNCTION_RETARG_ENUM
#undef HPX_FUNCTION_ARG_ENUM

    ///////////////////////////////////////////////////////////////////////////
    //  Specialized generic component action types allowing to hold a different
    //  number of arguments
    ///////////////////////////////////////////////////////////////////////////

    // zero argument version
    template <
        typename Component, typename Result, int Action,
        Result (Component::*F)(), typename Derived,
        threads::thread_priority Priority = threads::thread_priority_default>
    class base_result_action0
      : public action<Component, Action, Result, hpx::util::tuple0<>,
                      Derived, Priority>
    {
    public:
        typedef Result result_type;
        typedef hpx::util::tuple0<> arguments_type;
        typedef action<Component, Action, result_type, arguments_type,
                       Derived, Priority>
            base_type;

        explicit base_result_action0(threads::thread_priority priority = Priority)
          : base_type(priority)
        {}

    protected:
        /// The \a continuation_thread_function will be registered as the thread
        /// function of a thread. It encapsulates the execution of the
        /// original function (given by \a func), while ignoring the return
        /// value.
        template <typename Address>   // dummy template parameter
        static threads::thread_state_enum
        thread_function(Address lva)
        {
            try {
                LTM_(debug) << "Executing component action("
                            << detail::get_action_name<Derived>()
                            << ") lva(" << reinterpret_cast<void const*>
                                (get_lva<Component>::call(lva)) << ")";
                (get_lva<Component>::call(lva)->*F)();      // just call the function
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

        /// \brief This static \a construct_thread_function allows to construct
        /// a proper thread function for a \a thread without having to
        /// instantiate the \a base_result_action0 type. This is used by the \a
        /// applier in case no continuation has been supplied.
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(naming::address::address_type lva)
        {
            threads::thread_state_enum (*f)(naming::address::address_type) =
                &Derived::template thread_function<naming::address::address_type>;

            return HPX_STD_BIND(f, lva);
        }

        /// \brief This static \a construct_thread_function allows to construct
        /// a proper thread function for a \a thread without having to
        /// instantiate the \a base_result_action0 type. This is used by the \a
        /// applier in case a continuation has been supplied
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva)
        {
            return base_type::construct_continuation_thread_object_function(
                cont, F, get_lva<Component>::call(lva));
        }

        /// serialization support
        static void register_base()
        {
            util::void_cast_register_nonvirt<base_result_action0, base_type>();
            base_type::register_base();
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
        typename Component, typename Result, int Action,
        Result (Component::*F)(),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    class result_action0
      : public base_result_action0<Component, Result, Action, F,
            typename detail::action_type<
                result_action0<Component, Result, Action, F, Priority>,
                Derived
            >::type, Priority>
    {
    public:
        typedef typename detail::action_type<
            result_action0<Component, Result, Action, F, Priority>, Derived
        >::type derived_type;

    private:
        typedef base_result_action0<
            Component, Result, Action, F, derived_type, Priority>
        base_type;

    public:
        explicit result_action0(threads::thread_priority priority = Priority)
          : base_type(priority)
        {}

        static Result
        execute_function(naming::address::address_type lva)
        {
            LTM_(debug)
                << "result_action0::execute_function: name("
                << detail::get_action_name<derived_type>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";

            return (get_lva<Component>::call(lva)->*F)();
        }

        /// serialization support
        static void register_base()
        {
            util::void_cast_register_nonvirt<result_action0, base_type>();
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
            data.parent_locality_id = this->parent_locality_;
            data.priority = this->priority_;
            return data;
        }
    };

#if BOOST_WORKAROUND(BOOST_MSVC, <= 1700)
#if BOOST_WORKAROUND(BOOST_MSVC, == 1600)
    namespace detail
    {
        template <typename Obj, typename Result>
        struct synthesize_const_mf<Obj, Result (*)()>
        {
            typedef Result (Obj::*type)() const;
        };

        template <typename Obj, typename Result>
        struct synthesize_const_mf<Obj, Result (Obj::*)() const>
        {
            typedef Result (Obj::*type)() const;
        };

        template <typename Result>
        typename boost::mpl::identity<Result (*)()>::type
        replicate_type(Result (*p)());
    }
#endif

    template <typename Component, typename Result,
        Result (Component::*F)()>
    struct make_action<Result (Component::*)(), F, boost::mpl::false_>
      : result_action0<Component, Result, component_result_action_arg0, F>
    {};

    template <typename Component, typename Result,
        Result (Component::*F)() const>
    struct make_action<Result (Component::*)() const, F, boost::mpl::false_>
      : result_action0<Component const, Result, component_result_action_arg0, F>
    {};

#else

    template <typename Component, typename Result,
        Result (Component::*F)()>
    struct make_action<Result (Component::*)(), F, boost::mpl::false_>
      : boost::mpl::identity<result_action0<
            Component, Result, component_result_action_arg0, F> >
    {};

    template <typename Component, typename Result,
        Result (Component::*F)() const>
    struct make_action<Result (Component::*)() const, F, boost::mpl::false_>
      : boost::mpl::identity<result_action0<
            Component const, Result, component_result_action_arg0, F> >
    {};
#endif

    ///////////////////////////////////////////////////////////////////////////
    template <
        typename Component, typename Result, int Action,
        Result (Component::*F)(), typename Derived = detail::this_type>
    class direct_result_action0
      : public base_result_action0<Component, Result, Action, F,
            typename detail::action_type<
                direct_result_action0<Component, Result, Action, F>, Derived
            >::type>
    {
    public:
        typedef typename detail::action_type<
            direct_result_action0<Component, Result, Action, F>, Derived
        >::type derived_type;

    private:
        typedef base_result_action0<
            Component, Result, Action, F, derived_type>
        base_type;

    public:
        direct_result_action0()
        {}

        explicit direct_result_action0(threads::thread_priority)
        {}

    public:
        typedef boost::mpl::true_ direct_execution;

        static Result
        execute_function(naming::address::address_type lva)
        {
            LTM_(debug)
                << "direct_result_action0::execute_function: name("
                << detail::get_action_name<derived_type>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";

            return (get_lva<Component>::call(lva)->*F)();
        }

        /// serialization support
        static void register_base()
        {
            util::void_cast_register_nonvirt<direct_result_action0, base_type>();
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
            data.parent_locality_id = this->parent_locality_;
            data.priority = this->priority_;
            return data;
        }
    };

#if BOOST_WORKAROUND(BOOST_MSVC, <= 1700)
    template <typename Component, typename Result,
        Result (Component::*F)()>
    struct make_action<Result (Component::*)(), F, boost::mpl::true_>
      : direct_result_action0<Component, Result,
            component_result_action_arg0, F>
    {};

    template <typename Component, typename Result,
        Result (Component::*F)() const>
    struct make_action<Result (Component::*)() const, F, boost::mpl::true_>
      : direct_result_action0<Component const, Result,
            component_result_action_arg0, F>
    {};
#else
    template <typename Component, typename Result,
        Result (Component::*F)()>
    struct make_action<Result (Component::*)(), F, boost::mpl::true_>
      : boost::mpl::identity<direct_result_action0<Component, Result,
            component_result_action_arg0, F> >
    {};

    template <typename Component, typename Result,
        Result (Component::*F)() const>
    struct make_action<Result (Component::*)() const, F, boost::mpl::true_>
      : direct_result_action0<Component const, Result,
            component_result_action_arg0, F>
    {};
#endif

    ///////////////////////////////////////////////////////////////////////////
    //  zero parameter version, no result value
    template <typename Component, int Action, void (Component::*F)(), typename Derived,
      threads::thread_priority Priority = threads::thread_priority_default>
    class base_action0
      : public action<Component, Action, util::unused_type,
                      hpx::util::tuple0<>, Derived, Priority>
    {
    public:
        typedef util::unused_type result_type;
        typedef hpx::util::tuple0<> arguments_type;
        typedef action<Component, Action, result_type, arguments_type,
                       Derived, Priority>
            base_type;

        explicit base_action0(threads::thread_priority priority = Priority)
          : base_type(priority)
        {}

    protected:
        /// The \a continuation_thread_function will be registered as the thread
        /// function of a thread. It encapsulates the execution of the
        /// original function (given by \a func), while ignoring the return
        /// value.
        template <typename Address>   // dummy template parameter
        static threads::thread_state_enum
        thread_function(Address lva)
        {
            try {
                LTM_(debug) << "Executing component action("
                            << detail::get_action_name<Derived>()
                            << ") lva(" << reinterpret_cast<void const*>
                                (get_lva<Component>::call(lva)) << ")";
                (get_lva<Component>::call(lva)->*F)();      // just call the function
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

        /// \brief This static \a construct_thread_function allows to construct
        /// a proper thread function for a \a thread without having to
        /// instantiate the base_action0 type. This is used by the \a applier in
        /// case no continuation has been supplied.
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(naming::address::address_type lva)
        {
            threads::thread_state_enum (*f)(naming::address::address_type) =
                &Derived::template thread_function<naming::address::address_type>;

            return HPX_STD_BIND(f, lva);
        }

        /// \brief This static \a construct_thread_function allows to construct
        /// a proper thread function for a \a thread without having to
        /// instantiate the base_action0 type. This is used by the \a applier in
        /// case a continuation has been supplied
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva)
        {
            return base_type::construct_continuation_thread_object_function_void(
                cont, F, get_lva<Component>::call(lva));
        }

        /// serialization support
        static void register_base()
        {
            util::void_cast_register_nonvirt<base_action0, base_type>();
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
    template <typename Component, int Action, void (Component::*F)(),
        threads::thread_priority Priority = threads::thread_priority_default,
        typename Derived = detail::this_type>
    class action0
      : public base_action0<Component, Action, F,
            typename detail::action_type<
                action0<Component, Action, F, Priority>, Derived
            >::type, Priority>
    {
    public:
        typedef typename detail::action_type<
            action0<Component, Action, F, Priority>, Derived
        >::type derived_type;

    private:
        typedef base_action0<Component, Action, F, derived_type, Priority>
            base_type;

    public:
        explicit action0(threads::thread_priority priority = Priority)
          : base_type(priority)
        {}

        static util::unused_type
        execute_function(naming::address::address_type lva)
        {
            LTM_(debug)
                << "action0::execute_function: name("
                << detail::get_action_name<derived_type>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";

            (get_lva<Component>::call(lva)->*F)();
            return util::unused;
        }

        /// serialization support
        static void register_base()
        {
            util::void_cast_register_nonvirt<action0, base_type>();
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
            data.parent_locality_id = this->parent_locality_;
            data.priority = this->priority_;
            return data;
        }
    };

#if BOOST_WORKAROUND(BOOST_MSVC, <= 1700)
    template <typename Component, void (Component::*F)()>
    struct make_action<void (Component::*)(), F, boost::mpl::false_>
      : action0<Component, component_result_action_arg0, F>
    {};

    template <typename Component, void (Component::*F)() const>
    struct make_action<void (Component::*)() const, F, boost::mpl::false_>
      : action0<Component const, component_result_action_arg0, F>
    {};
#else
    template <typename Component, void (Component::*F)()>
    struct make_action<void (Component::*)(), F, boost::mpl::false_>
      : boost:mpl::identity<action0<
            Component, component_result_action_arg0, F> >
    {};

    template <typename Component, void (Component::*F)() const>
    struct make_action<void (Component::*)() const, F, boost::mpl::false_>
      : boost:mpl::identity<action0<
            Component const, component_result_action_arg0, F> >
    {};
#endif

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, int Action, void (Component::*F)(),
        typename Derived = detail::this_type>
    class direct_action0
      : public base_action0<Component, Action, F,
            typename detail::action_type<
                direct_action0<Component, Action, F>, Derived
            >::type>
    {
    public:
        typedef typename detail::action_type<
            direct_action0<Component, Action, F>, Derived
        >::type derived_type;

    private:
        typedef base_action0<Component, Action, F, derived_type> base_type;

    public:
        direct_action0()
        {}

        explicit direct_action0(threads::thread_priority)
        {}

    public:
        typedef boost::mpl::true_ direct_execution;

        static util::unused_type
        execute_function(naming::address::address_type lva)
        {
            LTM_(debug)
                << "direct_action0::execute_function: name("
                << detail::get_action_name<derived_type>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            (get_lva<Component>::call(lva)->*F)();
            return util::unused;
        }

        /// serialization support
        static void register_base()
        {
            util::void_cast_register_nonvirt<direct_action0, base_type>();
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
            data.parent_locality_id = this->parent_locality_;
            data.priority = this->priority_;
            return data;
        }
    };

#if BOOST_WORKAROUND(BOOST_MSVC, <= 1700)
    template <typename Component, void (Component::*F)()>
    struct make_action<void (Component::*)(), F, boost::mpl::true_>
      : direct_action0<Component, component_result_action_arg0, F>
    {};

    template <typename Component, void (Component::*F)() const>
    struct make_action<void (Component::*)() const, F, boost::mpl::true_>
      : direct_action0<Component const, component_result_action_arg0, F>
    {};
#else
    template <typename Component, void (Component::*F)()>
    struct make_action<void (Component::*)(), F, boost::mpl::true_>
      : boost::mpl::identity<direct_action0<
            Component, component_result_action_arg0, F> >
    {};

    template <typename Component, void (Component::*F)() const>
    struct make_action<void (Component::*)() const, F, boost::mpl::true_>
      : boost::mpl::identity<direct_action0<
            Component const, component_result_action_arg0, F> >
    {};
#endif

    template <
        typename Component, int Action,
        void (Component::*F)(),
        threads::thread_priority Priority,
        typename Derived>
    class result_action0<Component, void, Action, F, Priority, Derived>
        : public action0<Component, Action, F, Priority, Derived>
    {
        typedef action0<Component, Action, F, Priority, Derived> base_type;

    public:
        explicit result_action0(threads::thread_priority priority = Priority)
          : base_type(priority)
        {}

        /// serialization support
        static void register_base()
        {
            util::void_cast_register_nonvirt<result_action0, base_type>();
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
#define HPX_COMPONENT_ACTION(component, func, name)                           \
    typedef HPX_MAKE_COMPONENT_ACTION(component, func) name                   \
    /**/
#define HPX_COMPONENT_CONST_ACTION(component, func, name)                     \
    typedef HPX_MAKE_CONST_COMPONENT_ACTION(component, func) name             \
    /**/


///////////////////////////////////////////////////////////////////////////////
// bring in the rest of the implementations
#include <hpx/runtime/actions/component_action_implementations.hpp>
#include <hpx/runtime/actions/component_action_registration.hpp>

#include <hpx/config/warnings_suffix.hpp>

#endif

