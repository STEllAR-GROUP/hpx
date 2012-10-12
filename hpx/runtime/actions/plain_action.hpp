//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file plain_action.hpp

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
#include <hpx/runtime/actions/plain_action_enum.hpp>
#include <hpx/runtime/components/console_error_sink.hpp>
#include <hpx/runtime/components/server/plain_function.hpp>
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
    //  Specialized generic plain (free) action types allowing to hold a
    //  different number of arguments
    ///////////////////////////////////////////////////////////////////////////

    // zero argument version
    template <typename Result, Result (*F)(), typename Derived>
    class plain_base_result_action0
      : public action<
            components::server::plain_function<Derived>,
            function_result_action_arg0, Result, hpx::util::tuple0<>, Derived>
    {
    public:
        typedef Result result_type;
        typedef hpx::util::tuple0<> arguments_type;
        typedef action<
            components::server::plain_function<Derived>,
            function_result_action_arg0, result_type, arguments_type, Derived
        > base_type;

    protected:
        /// The \a thread_function will be registered as the thread
        /// function of a thread. It encapsulates the execution of the
        /// original function (given by \a func), while ignoring the return
        /// value.
        template <typename State>   // dummy template parameter
        BOOST_FORCEINLINE static threads::thread_state_enum
        thread_function(State)
        {
            try {
                LTM_(debug) << "Executing plain action("
                            << detail::get_action_name<Derived>()
                            << ").";
                F();      // call the function, ignoring the return value
            }
            catch (hpx::exception const& e) {
                if (e.get_error() != hpx::thread_interrupted) {
                    LTM_(error)
                        << "Unhandled exception while executing plain action("
                        << detail::get_action_name<Derived>()
                        << "): " << e.what();

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
        /// instantiate the \a plain_base_result_action0 type. This is used by
        /// the \a applier in case no continuation has been supplied.
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(naming::address::address_type,
            BOOST_FWD_REF(Arguments) /*args*/)
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
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type, BOOST_FWD_REF(Arguments) args)
        {
            return boost::move(
                base_type::construct_continuation_thread_function(
                    cont, F, boost::forward<Arguments>(args)));
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Result, Result (*F)(),
        typename Derived = detail::this_type>
    struct plain_result_action0
      : plain_base_result_action0<Result, F,
            typename detail::action_type<
                plain_result_action0<Result, F>, Derived
            >::type>
    {
        typedef typename detail::action_type<
            plain_result_action0, Derived
        >::type derived_type;

        typedef boost::mpl::false_ direct_execution;
    };

    template <typename Result, Result (*F)(), typename Derived>
    struct make_action<Result (*)(), F, Derived, boost::mpl::false_>
      : plain_result_action0<Result, F, Derived> 
    {
        typedef plain_result_action0<Result, F, Derived> type;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Result, Result (*F)(),
        typename Derived = detail::this_type>
    struct plain_direct_result_action0
      : plain_base_result_action0<Result, F,
            typename detail::action_type<
                plain_direct_result_action0<Result, F>, Derived
            >::type>
    {
        typedef typename detail::action_type<
            plain_direct_result_action0, Derived
        >::type derived_type;

        typedef boost::mpl::true_ direct_execution;

        template <typename Arguments>
        BOOST_FORCEINLINE static Result
        execute_function(naming::address::address_type,
            BOOST_FWD_REF(Arguments) /*args*/)
        {
            LTM_(debug)
                << "plain_direct_result_action0::execute_function: name("
                << detail::get_action_name<derived_type>()
                << ")";
            return F();
        }
    };

    template <typename Result, Result (*F)(), typename Derived>
    struct make_action<Result (*)(), F, Derived, boost::mpl::true_>
      : plain_direct_result_action0<Result, F, Derived>
    {
        typedef plain_direct_result_action0<Result, F, Derived> type;
    };

    ///////////////////////////////////////////////////////////////////////////
    //  zero parameter version, no result value
    template <void (*F)(), typename Derived>
    class plain_base_action0
      : public action<
            components::server::plain_function<Derived>,
            function_action_arg0, util::unused_type, hpx::util::tuple0<>,
            Derived>
    {
    public:
        typedef util::unused_type result_type;
        typedef hpx::util::tuple0<> arguments_type;
        typedef action<
            components::server::plain_function<Derived>,
            function_action_arg0, result_type, arguments_type, Derived
        > base_type;

    protected:
        /// The \a continuation_thread_function will be registered as the thread
        /// function of a thread. It encapsulates the execution of the
        /// original function (given by \a func), while ignoring the return
        /// value.
        template <typename State>   // dummy template parameter
        BOOST_FORCEINLINE static threads::thread_state_enum
        thread_function(State)
        {
            try {
                LTM_(debug) << "Executing plain action("
                            << detail::get_action_name<Derived>()
                            << ").";
                F();      // call the function, ignoring the return value
            }
            catch (hpx::exception const& e) {
                if (e.get_error() != hpx::thread_interrupted) {
                    LTM_(error)
                        << "Unhandled exception while executing plain action("
                        << detail::get_action_name<Derived>()
                        << "): " << e.what();

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
        construct_thread_function(naming::address::address_type,
            BOOST_FWD_REF(Arguments) /*args*/)
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
        template <typename Arguments>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_thread_function(continuation_type& cont,
            naming::address::address_type, BOOST_FWD_REF(Arguments) args)
        {
            return boost::move(
                base_type::construct_continuation_thread_function_void(
                    cont, F, boost::forward<Arguments>(args)));
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <void (*F)(),
        typename Derived = detail::this_type>
    struct plain_action0
      : plain_base_action0<
            F, typename detail::action_type<plain_action0<F>, Derived>::type>
    {
        typedef typename detail::action_type<
            plain_action0, Derived
        >::type derived_type;

        typedef boost::mpl::false_ direct_execution;
    };

    template <void (*F)(), typename Derived>
    struct make_action<void (*)(), F, Derived, boost::mpl::false_>
      : plain_action0<F, Derived> 
    {
        typedef plain_action0<F, Derived> type;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <void (*F)(), typename Derived = detail::this_type>
    struct plain_direct_action0
      : plain_base_action0<F,
            typename detail::action_type<
                plain_direct_action0<F>, Derived
            >::type>
    {
        typedef typename detail::action_type<
            plain_direct_action0, Derived
        >::type derived_type;

        typedef boost::mpl::true_ direct_execution;

        template <typename Arguments>
        BOOST_FORCEINLINE static util::unused_type
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) /*args*/)
        {
            LTM_(debug)
                << "plain_direct_action0::execute_function: name("
                << detail::get_action_name<derived_type>()
                << ")";
            F();
            return util::unused;
        }

        /// The function \a get_action_type returns whether this action needs
        /// to be executed in a new thread or directly.
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };

    template <void (*F)(), typename Derived>
    struct make_action<void(*)(), F, Derived, boost::mpl::true_>
      : plain_direct_action0<F, Derived>
    {
        typedef plain_direct_action0<F, Derived> type;
    };

    ///////////////////////////////////////////////////////////////////////////
    // the specialization for void return type is just a template alias
    template <void (*F)(), typename Derived>
    struct plain_result_action0<void, F, Derived>
        : plain_action0<F, Derived>
    {};

    /// \endcond
}}


// Disabling the guid initialization stuff for plain actions
namespace hpx { namespace traits
{
    /// \cond NOINTERNAL
    template <void (*F)(), typename Derived, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_action0<F, Derived> > , Enable>
      : boost::mpl::false_
    {};

    template <void (*F)(), typename Derived, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_direct_action0<F, Derived> > , Enable>
      : boost::mpl::false_
    {};

    template <typename R, R(*F)(), typename Derived, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_result_action0<R, F, Derived> > , Enable>
      : boost::mpl::false_
    {};

    template <typename R, R(*F)(), typename Derived, typename Enable>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::actions::plain_direct_result_action0<R, F, Derived> >, Enable>
      : boost::mpl::false_
    {};
    /// \endcond
}}

/// \def HPX_REGISTER_PLAIN_ACTION(action_type)
/// \brief Registers an existing free function as a plain action with HPX
///
/// The macro \a HPX_REGISTER_PLAIN_ACTION can be used to register an existing
/// free function as a plain action. It additionally defines an action type
/// named \a action_type.
///
/// The parameter \p action_type is the name of the action type to register
/// with HPX.
///
/// \par Example:
///
/// \code
///       namespace app
///       {
///           void some_global_function(double d)
///           {
///               cout << d;
///           }
///
///           // This will define the action type 'app::some_global_action' which
///           // represents the function 'app::some_global_function'.
///           HPX_DEFINE_PLAIN_ACTION(some_global_function, some_global_action);
///       }
///
///       // The following macro expands to a series of definitions of global objects
///       // which are needed for proper serialization and initialization support
///       // enabling the remote invocation of the function `app::some_global_function`.
///       HPX_REGISTER_PLAIN_ACTION(app::some_global_action);
/// \endcode
///
/// \note Usually this macro will not be used in user code unless the intend is
/// to avoid defining the action_type in global namespace. Normally, the use of
/// the macro \a HPX_PLAIN_ACTION is recommend.
///
#define HPX_REGISTER_PLAIN_ACTION(...)                                        \
    HPX_REGISTER_PLAIN_ACTION_(__VA_ARGS__)                                   \
/**/

/// \def HPX_DEFINE_PLAIN_ACTION(func, name)
/// \brief Defines a plain action type
///
/// \par Example:
///
/// \code
///       namespace app
///       {
///           void some_global_function(double d)
///           {
///               cout << d;
///           }
///
///           // This will define the action type 'app::some_global_action' which
///           // represents the function 'app::some_global_function'.
///           HPX_DEFINE_PLAIN_ACTION(some_global_function, some_global_action);
///       }
///
///       // The following macro expands to a series of definitions of global objects
///       // which are needed for proper serialization and initialization support
///       // enabling the remote invocation of the function `app::some_global_function`.
///       HPX_REGISTER_PLAIN_ACTION(app::some_global_action);
/// \endcode
///
/// \note Usually this macro will not be used in user code unless the intend is
/// to avoid defining the action_type in global namespace. Normally, the use of
/// the macro \a HPX_PLAIN_ACTION is recommend.
///
#define HPX_DEFINE_PLAIN_ACTION(func, name)                                   \
    typedef HPX_MAKE_ACTION(func)::type name                                  \
    /**/

/// \def HPX_PLAIN_ACTION(func, name)
/// \brief Defines a plain action type based on the given function \a func and registers it with HPX.
///
/// The macro \a HPX_PLAIN_ACTION can be used to define a plain action (e.g. an
/// action encapsulating a global or free function) based on the given function
/// \a func. It defines the action type \a name representing the given function.
/// This macro additionally registers the newly define action type with HPX.
///
/// The parameter \p func is a global or free (non-member) function which
/// should be encapsulated into a plain action. The parameter \p name is the
/// name of the action type defined by this macro.
///
/// \par Example:
///
/// \code
///     namespace app
///     {
///         void some_global_function(double d)
///         {
///             cout << d;
///         }
///     }
///
///     // This will define the action type 'some_global_action' which represents
///     // the function 'app::some_global_function'.
///     HPX_PLAIN_ACTION(app::some_global_function, some_global_action);
/// \endcode
///
/// \note The macro \a HPX_PLAIN_ACTION has to be used at global namespace even
/// if the wrapped function is located in some other namespace. The newly
/// defined action type is placed into the global namespace as well.
/// 
/// \note The macro \a HPX_PLAIN_ACTION can be used with 1, 2, or 3 arguments.
/// The second and third arguments are optional. The default value for the 
/// second argument (the typename of the defined action) is derived from the 
/// name of the function (as passed as the first argument) by appending '_action'.
/// The second argument can be omitted only if the first argument with an 
/// appended suffix '_action' resolves to a valid, unqualified C++ type name.
/// The default value for the third argument is \a hpx::components::factory_check.
///
#define HPX_PLAIN_ACTION(...)                                                 \
    HPX_PLAIN_ACTION_(__VA_ARGS__)                                            \
/**/

// bring in the rest of the implementations
#include <hpx/runtime/actions/plain_action_implementations.hpp>

/// \cond NOINTERNAL

#define HPX_PLAIN_ACTION_(...)                                                \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                            \
        HPX_PLAIN_ACTION_, HPX_UTIL_PP_NARG(__VA_ARGS__)                      \
    )(__VA_ARGS__))                                                           \
/**/
#define HPX_PLAIN_ACTION_1(func)                                              \
    HPX_DEFINE_PLAIN_ACTION(func, BOOST_PP_CAT(func, _action));               \
    HPX_REGISTER_PLAIN_ACTION_1(BOOST_PP_CAT(func, _action))                  \
/**/
#define HPX_PLAIN_ACTION_2(func, name)                                        \
    HPX_DEFINE_PLAIN_ACTION(func, name);                                      \
    HPX_REGISTER_PLAIN_ACTION_1(name)                                         \
/**/
#define HPX_PLAIN_ACTION_3(func, name, state)                                 \
    HPX_DEFINE_PLAIN_ACTION(func, name);                                      \
    HPX_REGISTER_PLAIN_ACTION_3(name, name, state)                            \
/**/

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

/// \endcond

#include <hpx/config/warnings_suffix.hpp>

#endif

