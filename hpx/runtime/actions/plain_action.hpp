//  Copyright (c) 2007-2013 Hartmut Kaiser
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
#include <hpx/exception.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/actions/action_support.hpp>
#include <hpx/runtime/components/console_error_sink.hpp>
#include <hpx/runtime/components/plain_component_factory.hpp>
#include <hpx/runtime/components/server/plain_function.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/unused.hpp>
#include <hpx/util/void_cast.hpp>
#include <hpx/util/detail/count_num_args.hpp>
#include <hpx/util/detail/pack.hpp>

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

    ///////////////////////////////////////////////////////////////////////////
    //  N parameter version, with R
    template <
        typename R, typename ...Ps,
        typename TF, TF F, typename Derived>
    class basic_action_impl<R (*)(Ps...), TF, F, Derived>
      : public basic_action<
            components::server::plain_function<Derived>,
            R(Ps...), Derived>
    {
    public:
        typedef basic_action<
            components::server::plain_function<Derived>,
            R(Ps...), Derived>
            base_type;

        typedef boost::mpl::false_ needs_guid_initialization;

        // Only localities are valid targets for a plain action
        static bool is_target_valid(naming::id_type const& id)
        {
            return naming::is_locality(id);
        }

    protected:
        /// The \a thread_function will be registered as the thread
        /// function of a thread. It encapsulates the execution of the
        /// original function (given by \a func).
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;

            template <typename ...Ts>
            BOOST_FORCEINLINE result_type operator()(Ts&&... vs) const
            {
                try {
                    LTM_(debug) << "Executing plain action("
                                << detail::get_action_name<Derived>()
                                << ").";

                    // call the function, ignoring the return value
                    F(std::forward<Ts>(vs)...);
                }
                catch (hpx::thread_interrupted const&) { //-V565
                    /* swallow this exception */
                }
                catch (hpx::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing plain action("
                        << detail::get_action_name<Derived>()
                        << "): " << e.what();

                    // report this error to the console in any case
                    hpx::report_error(boost::current_exception());
                }
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing plain action("
                        << detail::get_action_name<Derived>() << ")";

                    // report this error to the console in any case
                    hpx::report_error(boost::current_exception());
                }

                // Verify that there are no more registered locks for this
                // OS-thread. This will throw if there are still any locks
                // held.
                util::force_error_on_lock();
                return threads::terminated;
            }
        };

    public:
        // This static construct_thread_function allows to construct
        // a proper thread function for a thread without having to
        // instantiate the base_action type. This is used by the applier in
        // case no continuation has been supplied.
        template <std::size_t ...Is, typename Args>
        static threads::thread_function_type
        construct_thread_function(naming::address::address_type lva,
            util::detail::pack_c<std::size_t, Is...>, Args&& args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                util::bind(util::one_shot(typename Derived::thread_function()),
                    util::get<Is>(std::forward<Args>(args))...));
        }

        template <typename Args>
        static threads::thread_function_type
        construct_thread_function(naming::address::address_type lva,
            Args&& args)
        {
            return construct_thread_function(lva,
                typename util::detail::make_index_pack<
                    util::tuple_size<typename util::decay<Args>::type>::value
                >::type(), std::forward<Args>(args));
        }

        // This static construct_thread_function allows to construct
        // a proper thread function for a thread without having to
        // instantiate the base_action type. This is used by the applier in
        // case a continuation has been supplied
        template <typename Args>
        static threads::thread_function_type
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, Args&& args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                base_type::construct_continuation_thread_function(
                    cont, F, std::forward<Args>(args)));
        }

        // direct execution
        template <std::size_t ...Is, typename Args>
        BOOST_FORCEINLINE static R
        execute_function(naming::address::address_type lva,
            util::detail::pack_c<std::size_t, Is...>, Args&& args)
        {
            LTM_(debug)
                << "basic_action_impl::execute_function name("
                << detail::get_action_name<Derived>() << ")";

            return F(util::get<Is>(std::forward<Args>(args))...);
        }

        template <typename Args>
        BOOST_FORCEINLINE static R
        execute_function(naming::address::address_type lva,
            Args&& args)
        {
            return execute_function(lva,
                typename util::detail::make_index_pack<
                    util::tuple_size<typename util::decay<Args>::type>::value
                >::type(), std::forward<Args>(args));
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    //  N parameter version, no result type
    template <typename ...Ps, typename TF, TF F, typename Derived>
    class basic_action_impl<void (*)(Ps...), TF, F, Derived>
      : public basic_action<
            components::server::plain_function<Derived>,
            util::unused_type(Ps...), Derived>
    {
    public:
        typedef basic_action<
            components::server::plain_function<Derived>,
            util::unused_type(Ps...), Derived>
            base_type;

        typedef boost::mpl::false_ needs_guid_initialization;

        // Only localities are valid targets for a plain action
        static bool is_target_valid(naming::id_type const& id)
        {
            return naming::is_locality(id);
        }

    protected:
        /// The \a thread_function will be registered as the thread
        /// function of a thread. It encapsulates the execution of the
        /// original function (given by \a func).
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;

            template <typename ...Ts>
            BOOST_FORCEINLINE result_type operator()(Ts&&... vs) const
            {
                try {
                    LTM_(debug) << "Executing plain action("
                                << detail::get_action_name<Derived>()
                                << ").";

                    // call the function, ignoring the return value
                    F(std::forward<Ts>(vs)...);
                }
                catch (hpx::thread_interrupted const&) { //-V565
                    /* swallow this exception */
                }
                catch (hpx::exception const& e) {
                    LTM_(error)
                        << "Unhandled exception while executing plain action("
                        << detail::get_action_name<Derived>()
                        << "): " << e.what();

                    // report this error to the console in any case
                    hpx::report_error(boost::current_exception());
                }
                catch (...) {
                    LTM_(error)
                        << "Unhandled exception while executing plain action("
                        << detail::get_action_name<Derived>() << ")";

                    // report this error to the console in any case
                    hpx::report_error(boost::current_exception());
                }

                // Verify that there are no more registered locks for this
                // OS-thread. This will throw if there are still any locks
                // held.
                util::force_error_on_lock();
                return threads::terminated;
            }
        };

    public:
        // This static construct_thread_function allows to construct
        // a proper thread function for a thread without having to
        // instantiate the base_action type. This is used by the applier in
        // case no continuation has been supplied.
        template <std::size_t ...Is, typename Args>
        static threads::thread_function_type
        construct_thread_function(naming::address::address_type lva,
            util::detail::pack_c<std::size_t, Is...>, Args&& args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                util::bind(util::one_shot(typename Derived::thread_function()),
                    util::get<Is>(std::forward<Args>(args))...));
        }

        template <typename Args>
        static threads::thread_function_type
        construct_thread_function(naming::address::address_type lva,
            Args&& args)
        {
            return construct_thread_function(lva,
                typename util::detail::make_index_pack<
                    util::tuple_size<typename util::decay<Args>::type>::value
                >::type(), std::forward<Args>(args));
        }

        // This static construct_thread_function allows to construct
        // a proper thread function for a thread without having to
        // instantiate the base_action type. This is used by the applier in
        // case a continuation has been supplied
        template <typename Args>
        static threads::thread_function_type
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, Args&& args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                base_type::construct_continuation_thread_function_void(
                    cont, F, std::forward<Args>(args)));
        }

        //  direct execution
        template <std::size_t ...Is, typename Args>
        BOOST_FORCEINLINE static util::unused_type
        execute_function(naming::address::address_type lva,
            util::detail::pack_c<std::size_t, Is...>, Args&& args)
        {
            LTM_(debug)
                << "basic_action_impl::execute_function name("
                << detail::get_action_name<Derived>() << ")";

            F(util::get<Is>(std::forward<Args>(args))...);
            return util::unused;
        }

        template <typename Args>
        BOOST_FORCEINLINE static util::unused_type
        execute_function(naming::address::address_type lva,
            Args&& args)
        {
            return execute_function(lva,
                typename util::detail::make_index_pack<
                    util::tuple_size<typename util::decay<Args>::type>::value
                >::type(), std::forward<Args>(args));
        }
    };

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
///       //
///       // The second argument used has to be the same as used for the
///       // HPX_DEFINE_PLAIN_ACTION above.
///       HPX_REGISTER_PLAIN_ACTION(app::some_global_action, some_global_action);
/// \endcode
///
/// \note Usually this macro will not be used in user code unless the intend is
/// to avoid defining the action_type in global namespace. Normally, the use of
/// the macro \a HPX_PLAIN_ACTION is recommend.
///
#define HPX_REGISTER_PLAIN_ACTION(...)                                        \
    HPX_REGISTER_PLAIN_ACTION_(__VA_ARGS__)                                   \
/**/

/// \def HPX_REGISTER_PLAIN_ACTION_TEMPLATE(template_, action_type)
/// \brief Registers an existing template-free function as a plain action with HPX
///
/// The macro \a HPX_REGISTER_PLAIN_ACTION can be used to register an existing
/// template-free function as a plain action. It relies on a separately defined
/// action type named \a action_type.
///
/// The parameter \p action_type is the name of the action type to register
/// with HPX.
///
/// \par Example:
///
/// \code
///       namespace app
///       {
///           template <typename T>
///           void some_global_function(T d)
///           {
///               cout << d;
///           }
///
///           // define a new template action type named some_global_action
///           template <typename T>
///           struct some_global_action
///             : hpx::actions::make_action<
///                     void (*)(T), &some_global_function<T>,
///                     some_global_action<T> >
///           {};
///       }
///
///       // The following macro expands to a series of definitions of global objects
///       // which are needed for proper serialization and initialization support
///       // enabling the remote invocation of the function `app::some_global_function`.
///       //
///       // Please note the extra parenthesis around both macro arguments.
///       HPX_REGISTER_PLAIN_ACTION_TEMPLATE((template <typename T>), (app::some_global_action<T>));
/// \endcode
///
#define HPX_REGISTER_PLAIN_ACTION_TEMPLATE(template_, action_type)            \
    HPX_REGISTER_ACTION_DECLARATION_TEMPLATE(template_, action_type);         \
    HPX_DEFINE_GET_COMPONENT_TYPE_TEMPLATE(template_,                         \
        (hpx::components::server::plain_function<HPX_UTIL_STRIP(action_type)>)) \
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
///       //
///       // The second argument used has to be the same as used for the
///       // HPX_DEFINE_PLAIN_ACTION above.
///       HPX_REGISTER_PLAIN_ACTION(app::some_global_action, some_global_action);
/// \endcode
///
/// \note Usually this macro will not be used in user code unless the intend is
/// to avoid defining the action_type in global namespace. Normally, the use of
/// the macro \a HPX_PLAIN_ACTION is recommend.
///
#define HPX_DEFINE_PLAIN_ACTION(func, name)                                   \
    typedef HPX_MAKE_ACTION(func)::type name                                  \
    /**/

/// \cond NOINTERNAL

#define HPX_DEFINE_PLAIN_DIRECT_ACTION(func, name)                            \
    typedef HPX_MAKE_DIRECT_ACTION(func)::type name                           \
    /**/

/// \endcond

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

/// \cond NOINTERNAL

#define HPX_PLAIN_DIRECT_ACTION(...)                                          \
    HPX_PLAIN_DIRECT_ACTION_(__VA_ARGS__)                                     \
/**/

#define HPX_REGISTER_PLAIN_ACTION_DYNAMIC(...)                                \
    HPX_REGISTER_PLAIN_ACTION_DYNAMIC_(__VA_ARGS__)                           \
/**/

/// \endcond

/// \cond NOINTERNAL

// macros for plain actions
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

// same for direct actions
#define HPX_PLAIN_DIRECT_ACTION_(...)                                         \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                            \
        HPX_PLAIN_DIRECT_ACTION_, HPX_UTIL_PP_NARG(__VA_ARGS__)               \
    )(__VA_ARGS__))                                                           \
/**/
#define HPX_PLAIN_DIRECT_ACTION_1(func)                                       \
    HPX_DEFINE_PLAIN_DIRECT_ACTION(func, BOOST_PP_CAT(func, _action));        \
    HPX_REGISTER_PLAIN_ACTION_1(BOOST_PP_CAT(func, _action))                  \
/**/
#define HPX_PLAIN_DIRECT_ACTION_2(func, name)                                 \
    HPX_DEFINE_PLAIN_DIRECT_ACTION(func, name);                               \
    HPX_REGISTER_PLAIN_ACTION_1(name)                                         \
/**/
#define HPX_PLAIN_DIRECT_ACTION_3(func, name, state)                          \
    HPX_DEFINE_PLAIN_DIRECT_ACTION(func, name);                               \
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

