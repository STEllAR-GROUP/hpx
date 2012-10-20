//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file component_action.hpp

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
#include <hpx/runtime/actions/component_action_enum.hpp>
#include <hpx/runtime/components/console_error_sink.hpp>
#include <hpx/util/unused.hpp>
#include <hpx/util/void_cast.hpp>
#include <hpx/util/detail/count_num_args.hpp>

#include <boost/preprocessor/cat.hpp>

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
        typename Component, typename Result, int Action,
        Result (Component::*F)(), typename Derived>
    class base_result_action0
      : public action<Component, Action, Result, hpx::util::tuple0<>, Derived>
    {
    public:
        typedef Result result_type;
        typedef hpx::util::tuple0<> arguments_type;
        typedef action<Component, Action, result_type, arguments_type, Derived>
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

            return boost::move(base_type::decorate_action(
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
            return boost::move(base_type::decorate_action(
                    base_type::construct_continuation_thread_object_function(
                        cont, F, get_lva<Component>::call(lva),
                        boost::forward<Arguments>(args)), lva));
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <
        typename Component, typename Result, int Action,
        Result (Component::*F)(), typename Derived = detail::this_type>
    struct result_action0
      : base_result_action0<Component, Result, Action, F,
            typename detail::action_type<
                result_action0<Component, Result, Action, F>, Derived
            >::type>
    {
        typedef typename detail::action_type<
            result_action0, Derived
        >::type derived_type;

        typedef boost::mpl::false_ direct_execution;
    };

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
        Result (Component::*F)(), typename Derived>
    struct make_action<Result (Component::*)(), F, Derived, boost::mpl::false_>
      : result_action0<Component, Result, component_result_action_arg0, F, Derived>
    {
        typedef result_action0<
            Component, Result, component_result_action_arg0, F, Derived
        > type;
    };

    template <typename Component, typename Result,
        Result (Component::*F)() const, typename Derived>
    struct make_action<Result (Component::*)() const, F, Derived, boost::mpl::false_>
      : result_action0<Component const, Result, component_result_action_arg0, 
            F, Derived> 
    {
        typedef result_action0<
            Component const, Result, component_result_action_arg0, F, Derived
        > type;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <
        typename Component, typename Result, int Action,
        Result (Component::*F)(), typename Derived = detail::this_type>
    struct direct_result_action0
      : public base_result_action0<Component, Result, Action, F,
            typename detail::action_type<
                direct_result_action0<Component, Result, Action, F>, Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_result_action0, Derived
        >::type derived_type;

        typedef boost::mpl::true_ direct_execution;

        template <typename Arguments>
        BOOST_FORCEINLINE static Result
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments))
        {
            LTM_(debug)
                << "direct_result_action0::execute_function: name("
                << detail::get_action_name<derived_type>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";

            return (get_lva<Component>::call(lva)->*F)();
        }

        /// The function \a get_action_type returns whether this action needs
        /// to be executed in a new thread or directly.
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };

    template <typename Component, typename Result,
        Result (Component::*F)(), typename Derived>
    struct make_action<Result (Component::*)(), F, Derived, boost::mpl::true_>
      : direct_result_action0<Component, Result, component_result_action_arg0, 
            F, Derived>
    {
        typedef direct_result_action0<
            Component, Result, component_result_action_arg0, F, Derived
        > type;
    };

    template <typename Component, typename Result,
        Result (Component::*F)() const, typename Derived>
    struct make_action<Result (Component::*)() const, F, Derived, boost::mpl::true_>
      : direct_result_action0<Component const, Result, 
            component_result_action_arg0, F, Derived>
    {
        typedef direct_result_action0<
            Component const, Result, component_result_action_arg0, F, Derived
        > type;
    };

    ///////////////////////////////////////////////////////////////////////////
    //  zero parameter version, no result value
    template <typename Component, int Action, void (Component::*F)(), typename Derived>
    class base_action0
      : public action<Component, Action, util::unused_type, 
            hpx::util::tuple0<>, Derived>
    {
    public:
        typedef util::unused_type result_type;
        typedef hpx::util::tuple0<> arguments_type;
        typedef action<Component, Action, result_type, arguments_type, Derived>
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

            return boost::move(base_type::decorate_action(
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
            return boost::move(base_type::decorate_action(
                    base_type::construct_continuation_thread_object_function_void(
                        cont, F, get_lva<Component>::call(lva),
                        boost::forward<Arguments>(args)), lva));
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, int Action, void (Component::*F)(),
        typename Derived = detail::this_type>
    struct action0
      : base_action0<Component, Action, F,
            typename detail::action_type<
                action0<Component, Action, F>, Derived
            >::type>
    {
        typedef typename detail::action_type<
            action0, Derived
        >::type derived_type;

        typedef boost::mpl::false_ direct_execution;
    };

    template <typename Component, void (Component::*F)(), typename Derived>
    struct make_action<void (Component::*)(), F, Derived, boost::mpl::false_>
      : action0<Component, component_result_action_arg0, F, Derived> 
    {
        typedef action0<
            Component, component_result_action_arg0, F, Derived
        > type;
    };

    template <typename Component, void (Component::*F)() const, typename Derived>
    struct make_action<void (Component::*)() const, F, Derived, boost::mpl::false_>
      : action0<Component const, component_result_action_arg0, F, Derived>
    {
        typedef action0<
            Component const, component_result_action_arg0, F, Derived
        > type;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, int Action, void (Component::*F)(),
        typename Derived = detail::this_type>
    struct direct_action0
      : base_action0<Component, Action, F,
            typename detail::action_type<
                direct_action0<Component, Action, F>, Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_action0, Derived
        >::type derived_type;

        typedef boost::mpl::true_ direct_execution;

        template <typename Arguments>
        BOOST_FORCEINLINE static util::unused_type
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments))
        {
            LTM_(debug)
                << "direct_action0::execute_function: name("
                << detail::get_action_name<derived_type>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            (get_lva<Component>::call(lva)->*F)();
            return util::unused;
        }

        /// The function \a get_action_type returns whether this action needs
        /// to be executed in a new thread or directly.
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };

    template <typename Component, void (Component::*F)(), typename Derived>
    struct make_action<void (Component::*)(), F, Derived, boost::mpl::true_>
      : direct_action0<Component, component_result_action_arg0, F, Derived>
    {
        typedef direct_action0<
            Component, component_result_action_arg0, F, Derived
        > type;
    };

    template <typename Component, void (Component::*F)() const, typename Derived>
    struct make_action<void (Component::*)() const, F, Derived, boost::mpl::true_>
      : direct_action0<Component const, component_result_action_arg0, F, Derived>
    {
        typedef direct_action0<
            Component const, component_result_action_arg0, F, Derived
        > type;
    };

    ///////////////////////////////////////////////////////////////////////////
    // the specialization for void return type is just a template alias
    template <typename Component, int Action, void (Component::*F)(), typename Derived>
    struct result_action0<Component, void, Action, F, Derived>
      : action0<Component, Action, F, Derived>
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
/// \note The macro \a HPX_DEFINE_COMPONENT_ACTION can be used with 1 or 
/// 2 arguments. The second argument is optional. The default value for the 
/// second argument (the typename of the defined action) is derived from the 
/// name of the function (as passed as the first argument) by appending '_action'.
/// The second argument can be omitted only if the first argument with an 
/// appended suffix '_action' resolves to a valid, unqualified C++ type name.
#define HPX_DEFINE_COMPONENT_ACTION(...)                                      \
    HPX_DEFINE_COMPONENT_ACTION_(__VA_ARGS__)                                 \
/**/

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
/// \note The macro \a HPX_DEFINE_COMPONENT_CONST_ACTION can be used with 1 or 
/// 2 arguments. The second argument is optional. The default value for the 
/// second argument (the typename of the defined action) is derived from the 
/// name of the function (as passed as the first argument) by appending '_action'.
/// The second argument can be omitted only if the first argument with an 
/// appended suffix '_action' resolves to a valid, unqualified C++ type name.
#define HPX_DEFINE_COMPONENT_CONST_ACTION(...)                                \
    HPX_DEFINE_COMPONENT_CONST_ACTION_(__VA_ARGS__)                           \
/**/

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
        BOOST_PP_CAT(func, _type)                                             \
    /**/
#define HPX_DEFINE_COMPONENT_DIRECT_ACTION_3(component, func, action_type)    \
    typedef HPX_MAKE_DIRECT_COMPONENT_ACTION(component, func)::type           \
        action_type                                                           \
    /**/


#define HPX_DEFINE_COMPONENT_CONST_DIRECT_ACTION(...)                         \
    HPX_DEFINE_COMPONENT_CONST_DIRECT_ACTION_(__VA_ARGS__)                    \
/**/

#define HPX_DEFINE_COMPONENT_CONST_DIRECT_ACTION_(...)                        \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                            \
        HPX_DEFINE_COMPONENT_CONST_DIRECT_ACTION_, HPX_UTIL_PP_NARG(__VA_ARGS__)\
    )(__VA_ARGS__))                                                           \

#define HPX_DEFINE_COMPONENT_CONST_DIRECT_ACTION_2(component, func)           \
    typedef HPX_MAKE_CONST_DIRECT_COMPONENT_ACTION(component, func)::type     \
        BOOST_PP_CAT(func, _type)                                             \
    /**/
#define HPX_DEFINE_COMPONENT_CONST_DIRECT_ACTION_3(component, func, action_type)\
    typedef HPX_MAKE_CONST_DIRECT_COMPONENT_ACTION(component, func)::type     \
        action_type                                                           \
    /**/

///////////////////////////////////////////////////////////////////////////////
// same as above, just for template functions
#define HPX_DEFINE_COMPONENT_ACTION_TPL(component, func, name)                \
    typedef typename HPX_MAKE_COMPONENT_ACTION_TPL(component, func)::type name\
    /**/
#define HPX_DEFINE_COMPONENT_CONST_ACTION_TPL(component, func, name)          \
    typedef HPX_MAKE_CONST_COMPONENT_ACTION_TPL(component, func)::type name   \
    /**/
#define HPX_DEFINE_COMPONENT_DIRECT_ACTION_TPL(component, func, name)         \
    typedef HPX_MAKE_DIRECT_COMPONENT_ACTION_TPL(component, func)::type name  \
    /**/
#define HPX_DEFINE_COMPONENT_CONST_DIRECT_ACTION_TPL(component, func, name)   \
    typedef HPX_MAKE_CONST_DIRECT_COMPONENT_ACTION_TPL(component, func)::type name \
    /**/

///////////////////////////////////////////////////////////////////////////////
// bring in the rest of the implementations
#include <hpx/runtime/actions/component_action_implementations.hpp>

///////////////////////////////////////////////////////////////////////////////
// Register the action templates with serialization.
HPX_SERIALIZATION_REGISTER_TEMPLATE(
    (template <typename Action>), (hpx::actions::transfer_action<Action>)
)

/// \endcond

#include <hpx/config/warnings_suffix.hpp>

#endif

