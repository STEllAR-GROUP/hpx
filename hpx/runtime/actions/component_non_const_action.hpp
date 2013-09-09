//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file component_non_const_action.hpp

#if !defined(HPX_RUNTIME_ACTIONS_COMPONENT_NON_CONST_ACTION_MAR_26_2008_1054AM)
#define HPX_RUNTIME_ACTIONS_COMPONENT_NON_CONST_ACTION_MAR_26_2008_1054AM

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace actions
{
    /// \cond NOINTERNAL

    ///////////////////////////////////////////////////////////////////////////
    //  Specialized generic component action types allowing to hold a different
    //  number of arguments
    ///////////////////////////////////////////////////////////////////////////

    // zero argument version
    template <
        typename Component, typename Result,
        Result (Component::*F)(), typename Derived>
    class base_result_action0<Result (Component::*)(), F, Derived>
      : public action<Component, Result, hpx::util::tuple<>, Derived>
    {
    public:
        typedef Result result_type;
        typedef typename detail::remote_action_result<Result>::type
            remote_result_type;

        typedef hpx::util::tuple<> arguments_type;
        typedef action<Component, remote_result_type, arguments_type, Derived>
            base_type;

    protected:
        /// The \a continuation_thread_function will be registered as the thread
        /// function of a thread. It encapsulates the execution of the
        /// original function (given by \a func), while ignoring the return
        /// value.
        template <typename Address>   // dummy template parameter
        BOOST_FORCEINLINE static threads::thread_state_enum
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
                if (e.get_error() != hpx::thread_interrupted) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component>::call(lva)) << "): " << e.what();

                    // report this error to the console in any case
                    hpx::report_error(boost::current_exception());
                }
            }
            catch (...) {
                LTM_(error)
                    << "Unhandled exception while executing component action("
                    << detail::get_action_name<Derived>()
                    << ") lva(" << reinterpret_cast<void const*>
                        (get_lva<Component>::call(lva)) << ")";

                // report this error to the console in any case
                hpx::report_error(boost::current_exception());
            }

            // Verify that there are no more registered locks for this
            // OS-thread. This will throw if there are still any locks
            // held.
            util::force_error_on_lock();
            return threads::terminated;
        }

    public:
        /// \brief This static \a construct_thread_function allows to construct
        /// a proper thread function for a \a thread without having to
        /// instantiate the \a base_result_action0 type. This is used by the \a
        /// applier in case no continuation has been supplied.
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) /*args*/)
        {
            threads::thread_state_enum (*f)(naming::address::address_type) =
                &Derived::template thread_function<naming::address::address_type>;

            return boost::move(Derived::decorate_action(
                HPX_STD_BIND(f, lva), lva));
        }

        /// \brief This static \a construct_thread_function allows to construct
        /// a proper thread function for a \a thread without having to
        /// instantiate the \a base_result_action0 type. This is used by the \a
        /// applier in case a continuation has been supplied
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return boost::move(Derived::decorate_action(
                base_type::construct_continuation_thread_object_function(
                    cont, F, get_lva<Component>::call(lva),
                    boost::forward<Arguments>(args)), lva));
        }

        // direct execution
        template <typename Arguments>
        BOOST_FORCEINLINE static Result
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments))
        {
            LTM_(debug)
                << "base_result_action0::execute_function: name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";

            return (get_lva<Component>::call(lva)->*F)();
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <
        typename Component, typename Result,
        Result (Component::*F)(), typename Derived>
    struct result_action0<Result (Component::*)(), F, Derived>
      : base_result_action0<
            Result (Component::*)(), F,
            typename detail::action_type<
                result_action0<Result (Component::*)(), F, Derived>, Derived
            >::type>
    {
        typedef typename detail::action_type<
            result_action0, Derived
        >::type derived_type;

        typedef boost::mpl::false_ direct_execution;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <
        typename Component, typename Result,
        Result (Component::*F)(), typename Derived>
    struct make_action<Result (Component::*)(), F, Derived, boost::mpl::false_>
      : result_action0<Result (Component::*)(), F, Derived>
    {
        typedef result_action0<
            Result (Component::*)(), F, Derived
        > type;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <
        typename Component, typename Result,
        Result (Component::*F)(), typename Derived>
    struct direct_result_action0<Result (Component::*)(), F, Derived>
      : public base_result_action0<
            Result (Component::*)(), F,
            typename detail::action_type<
                direct_result_action0<Result (Component::*)(), F, Derived>, Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_result_action0, Derived
        >::type derived_type;

        typedef boost::mpl::true_ direct_execution;

        /// The function \a get_action_type returns whether this action needs
        /// to be executed in a new thread or directly.
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <
        typename Component, typename Result,
        Result (Component::*F)(), typename Derived>
    struct make_action<Result (Component::*)(), F, Derived, boost::mpl::true_>
      : direct_result_action0<Result (Component::*)(), F, Derived>
    {
        typedef direct_result_action0<
            Result (Component::*)(), F, Derived
        > type;
    };

    ///////////////////////////////////////////////////////////////////////////
    //  zero parameter version, no result value
    template <typename Component, void (Component::*F)(), typename Derived>
    class base_action0<void (Component::*)(), F, Derived>
      : public action<Component, util::unused_type,
            hpx::util::tuple<>, Derived>
    {
    public:
        typedef util::unused_type result_type;
        typedef util::unused_type remote_result_type;

        typedef hpx::util::tuple<> arguments_type;
        typedef action<Component, remote_result_type, arguments_type, Derived>
            base_type;

    protected:
        /// The \a continuation_thread_function will be registered as the thread
        /// function of a thread. It encapsulates the execution of the
        /// original function (given by \a func), while ignoring the return
        /// value.
        template <typename Address>   // dummy template parameter
        BOOST_FORCEINLINE static threads::thread_state_enum
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
                if (e.get_error() != hpx::thread_interrupted) {
                    LTM_(error)
                        << "Unhandled exception while executing component action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component>::call(lva)) << "): " << e.what();

                    // report this error to the console in any case
                    hpx::report_error(boost::current_exception());
                }
            }
            catch (...) {
                LTM_(error)
                    << "Unhandled exception while executing component action("
                    << detail::get_action_name<Derived>()
                    << ") lva(" << reinterpret_cast<void const*>
                        (get_lva<Component>::call(lva)) << ")";

                // report this error to the console in any case
                hpx::report_error(boost::current_exception());
            }

            // Verify that there are no more registered locks for this
            // OS-thread. This will throw if there are still any locks
            // held.
            util::force_error_on_lock();
            return threads::terminated;
        }

    public:
        /// \brief This static \a construct_thread_function allows to construct
        /// a proper thread function for a \a thread without having to
        /// instantiate the base_action0 type. This is used by the \a applier in
        /// case no continuation has been supplied.
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) /*args*/)
        {
            threads::thread_state_enum (*f)(naming::address::address_type) =
                &Derived::template thread_function<naming::address::address_type>;

            return boost::move(Derived::decorate_action(
                HPX_STD_BIND(f, lva), lva));
        }

        /// \brief This static \a construct_thread_function allows to construct
        /// a proper thread function for a \a thread without having to
        /// instantiate the base_action0 type. This is used by the \a applier in
        /// case a continuation has been supplied
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, BOOST_FWD_REF(Arguments) args)
        {
            return boost::move(Derived::decorate_action(
                base_type::construct_continuation_thread_object_function_void(
                    cont, F, get_lva<Component>::call(lva),
                    boost::forward<Arguments>(args)), lva));
        }

        // direct execution
        template <typename Arguments>
        BOOST_FORCEINLINE static util::unused_type
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments))
        {
            LTM_(debug)
                << "base_action0::execute_function: name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            (get_lva<Component>::call(lva)->*F)();
            return util::unused;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, void (Component::*F)(), typename Derived>
    struct action0<void (Component::*)(), F, Derived>
      : base_action0<
            void (Component::*)(), F,
            typename detail::action_type<
                action0<void (Component::*)(), F, Derived>, Derived
            >::type>
    {
        typedef typename detail::action_type<
            action0, Derived
        >::type derived_type;

        typedef boost::mpl::false_ direct_execution;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, void (Component::*F)(), typename Derived>
    struct make_action<void (Component::*)(), F, Derived, boost::mpl::false_>
      : action0<void (Component::*)(), F, Derived>
    {
        typedef action0<
            void (Component::*)(), F, Derived
        > type;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, void (Component::*F)(), typename Derived>
    struct direct_action0<void (Component::*)(), F, Derived>
      : base_action0<
            void (Component::*)(), F,
            typename detail::action_type<
                direct_action0<void (Component::*)(), F, Derived>, Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_action0, Derived
        >::type derived_type;

        typedef boost::mpl::true_ direct_execution;

        /// The function \a get_action_type returns whether this action needs
        /// to be executed in a new thread or directly.
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, void (Component::*F)(), typename Derived>
    struct make_action<void (Component::*)(), F, Derived, boost::mpl::true_>
      : direct_action0<void (Component::*)(), F, Derived>
    {
        typedef direct_action0<
            void (Component::*)(), F, Derived
        > type;
    };

    ///////////////////////////////////////////////////////////////////////////
    // the specialization for void return type is just a template alias
    template <typename Component, void (Component::*F)(), typename Derived>
    struct result_action0<void (Component::*)(), F, Derived>
      : action0<void (Component::*)(), F, Derived>
    {};

    /// \endcond
}}

/// \def HPX_DEFINE_COMPONENT_ACTION(component, func, action_type)
///
/// \brief Registers a non-const member function of a component as an action type
/// with HPX
///
/// The macro \a HPX_DEFINE_COMPONENT_CONST_ACTION can be used to register a
/// non-const member function of a component as an action type named \a action_type.
///
/// The parameter \a component is the type of the component exposing the
/// non-const member function \a func which should be associated with the newly
/// defined action type. The parameter \p action_type is the name of the action
/// type to register with HPX.
///
/// \par Example:
///
/// \code
///       namespace app
///       {
///           // Define a simple component exposing one action 'print_greating'
///           class HPX_COMPONENT_EXPORT server
///             : public hpx::components::simple_component_base<server>
///           {
///               void print_greating ()
///               {
///                   hpx::cout << "Hey, how are you?\n" << hpx::flush;
///               }
///
///               // Component actions need to be declared, this also defines the
///               // type 'print_greating_action' representing the action.
///               HPX_DEFINE_COMPONENT_ACTION(server, print_greating, print_greating_action);
///           };
///       }
/// \endcode
///
/// \note This macro should be used for non-const member functions only. Use
/// the macro \a HPX_DEFINE_COMPONENT_CONST_ACTION for const member functions.
///
/// The first argument must provide the type name of the component the
/// action is defined for.
///
/// The second argument must provide the member function name the action
/// should wrap.
///
/// \note The macro \a HPX_DEFINE_COMPONENT_ACTION can be used with 2 or
/// 3 arguments. The third argument is optional.
///
/// The default value for the third argument (the typename of the defined
/// action) is derived from the name of the function (as passed as the second
/// argument) by appending '_action'. The third argument can be omitted only
/// if the first argument with an appended suffix '_action' resolves to a valid,
/// unqualified C++ type name.
///
#define HPX_DEFINE_COMPONENT_ACTION(...)                                      \
    HPX_DEFINE_COMPONENT_ACTION_(__VA_ARGS__)                                 \
/**/

/// \cond NOINTERNAL
#define HPX_DEFINE_COMPONENT_ACTION_(...)                                     \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                            \
        HPX_DEFINE_COMPONENT_ACTION_, HPX_UTIL_PP_NARG(__VA_ARGS__)           \
    )(__VA_ARGS__))                                                           \

#define HPX_DEFINE_COMPONENT_ACTION_2(component, func)                        \
    typedef HPX_MAKE_COMPONENT_ACTION(component, func)::type                  \
        BOOST_PP_CAT(func, _action)                                           \
    /**/
#define HPX_DEFINE_COMPONENT_ACTION_3(component, func, action_type)           \
    typedef HPX_MAKE_COMPONENT_ACTION(component, func)::type action_type      \
    /**/
/// \endcond

/// \cond NOINTERNAL
#define HPX_DEFINE_COMPONENT_DIRECT_ACTION(...)                               \
    HPX_DEFINE_COMPONENT_DIRECT_ACTION_(__VA_ARGS__)                          \
/**/

#define HPX_DEFINE_COMPONENT_DIRECT_ACTION_(...)                              \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                            \
        HPX_DEFINE_COMPONENT_DIRECT_ACTION_, HPX_UTIL_PP_NARG(__VA_ARGS__)    \
    )(__VA_ARGS__))                                                           \

#define HPX_DEFINE_COMPONENT_DIRECT_ACTION_2(component, func)                 \
    typedef HPX_MAKE_DIRECT_COMPONENT_ACTION(component, func)::type           \
        BOOST_PP_CAT(func, _action)                                           \
    /**/
#define HPX_DEFINE_COMPONENT_DIRECT_ACTION_3(component, func, action_type)    \
    typedef HPX_MAKE_DIRECT_COMPONENT_ACTION(component, func)::type           \
        action_type                                                           \
    /**/

///////////////////////////////////////////////////////////////////////////////
// same as above, just for template functions
#define HPX_DEFINE_COMPONENT_ACTION_TPL(component, func, name)                \
    typedef typename HPX_MAKE_COMPONENT_ACTION_TPL(component, func)::type name\
    /**/
#define HPX_DEFINE_COMPONENT_DIRECT_ACTION_TPL(component, func, name)         \
    typedef typename                                                          \
        HPX_MAKE_DIRECT_COMPONENT_ACTION_TPL(component, func)::type name      \
    /**/
/// \endcond

/// \def HPX_DEFINE_COMPONENT_CONST_ACTION(component, func, action_type)
///
/// \brief Registers a const member function of a component as an action type
/// with HPX
///
/// The macro \a HPX_DEFINE_COMPONENT_CONST_ACTION can be used to register a
/// const member function of a component as an action type named \a action_type.
///
/// The parameter \a component is the type of the component exposing the const
/// member function \a func which should be associated with the newly defined
/// action type. The parameter \p action_type is the name of the action type to
/// register with HPX.
///
/// \par Example:
///
/// \code
///       namespace app
///       {
///           // Define a simple component exposing one action 'print_greating'
///           class HPX_COMPONENT_EXPORT server
///             : public hpx::components::simple_component_base<server>
///           {
///               void print_greating() const
///               {
///                   hpx::cout << "Hey, how are you?\n" << hpx::flush;
///               }
///
///               // Component actions need to be declared, this also defines the
///               // type 'print_greating_action' representing the action.
///               HPX_DEFINE_COMPONENT_CONST_ACTION(server, print_greating, print_greating_action);
///           };
///       }
/// \endcode
///
/// \note This macro should be used for const member functions only. Use
/// the macro \a HPX_DEFINE_COMPONENT_ACTION for non-const member functions.
///
/// The first argument must provide the type name of the component the
/// action is defined for.
///
/// The second argument must provide the member function name the action
/// should wrap.
///
/// \note The macro \a HPX_DEFINE_COMPONENT_CONST_ACTION can be used with 2 or
/// 3 arguments. The third argument is optional.
///
/// The default value for the third argument (the typename of the defined
/// action) is derived from the name of the function (as passed as the second
/// argument) by appending '_action'. The third argument can be omitted only
/// if the first argument with an appended suffix '_action' resolves to a valid,
/// unqualified C++ type name.
///
#define HPX_DEFINE_COMPONENT_CONST_ACTION(...)                                \
    HPX_DEFINE_COMPONENT_CONST_ACTION_(__VA_ARGS__)                           \
/**/

/// \cond NOINTERNAL
#define HPX_DEFINE_COMPONENT_CONST_ACTION_(...)                               \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                            \
        HPX_DEFINE_COMPONENT_CONST_ACTION_, HPX_UTIL_PP_NARG(__VA_ARGS__)     \
    )(__VA_ARGS__))                                                           \

#define HPX_DEFINE_COMPONENT_CONST_ACTION_2(component, func)                  \
    typedef HPX_MAKE_CONST_COMPONENT_ACTION(component, func)::type            \
        BOOST_PP_CAT(func, _action)                                           \
    /**/
#define HPX_DEFINE_COMPONENT_CONST_ACTION_3(component, func, action_type)     \
    typedef HPX_MAKE_CONST_COMPONENT_ACTION(component, func)::type action_type\
    /**/
/// \endcond

/// \cond NOINTERNAL
#define HPX_DEFINE_COMPONENT_CONST_DIRECT_ACTION(...)                         \
    HPX_DEFINE_COMPONENT_CONST_DIRECT_ACTION_(__VA_ARGS__)                    \
/**/

#define HPX_DEFINE_COMPONENT_CONST_DIRECT_ACTION_(...)                        \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                            \
        HPX_DEFINE_COMPONENT_CONST_DIRECT_ACTION_, HPX_UTIL_PP_NARG(__VA_ARGS__)\
    )(__VA_ARGS__))                                                           \

#define HPX_DEFINE_COMPONENT_CONST_DIRECT_ACTION_2(component, func)           \
    typedef HPX_MAKE_CONST_DIRECT_COMPONENT_ACTION(component, func)::type     \
        BOOST_PP_CAT(func, _action)                                           \
    /**/
#define HPX_DEFINE_COMPONENT_CONST_DIRECT_ACTION_3(component, func, action_type)\
    typedef HPX_MAKE_CONST_DIRECT_COMPONENT_ACTION(component, func)::type     \
        action_type                                                           \
    /**/

///////////////////////////////////////////////////////////////////////////////
// same as above, just for template functions
#define HPX_DEFINE_COMPONENT_CONST_ACTION_TPL(component, func, name)          \
    typedef typename                                                          \
        HPX_MAKE_CONST_COMPONENT_ACTION_TPL(component, func)::type name       \
    /**/
#define HPX_DEFINE_COMPONENT_CONST_DIRECT_ACTION_TPL(component, func, name)   \
    typedef typename                                                          \
        HPX_MAKE_CONST_DIRECT_COMPONENT_ACTION_TPL(component, func)::type name\
    /**/

/// \endcond

#include <hpx/config/warnings_suffix.hpp>

#endif

