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

    ///////////////////////////////////////////////////////////////////////////
    //  N parameter version, with result
    template <
        typename Component, typename R, typename ...Ps,
        typename TF, TF F, typename Derived>
    class basic_action_impl<
            R (Component::*)(Ps...), TF, F, Derived>
      : public basic_action<Component, R(Ps...), Derived>
    {
    public:
        typedef basic_action<Component, R(Ps...), Derived> base_type;

        // Let the component decide whether the id is valid
        static bool is_target_valid(naming::id_type const& id)
        {
            return Component::is_target_valid(id);
        }

    protected:
        /// The \a thread_function will be registered as the thread
        /// function of a thread. It encapsulates the execution of the
        /// original function (given by \a func).
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;

            template <typename ...Ts>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva, Ts&&... vs) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component>::call(lva)) << ")";

                    (get_lva<Component>::call(lva)->*F)(
                        std::forward<Ts>(vs)...);
                }
                catch (hpx::thread_interrupted const&) { //-V565
                    /* swallow this exception */
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
        };

    public:
        typedef boost::mpl::false_ direct_execution;

        // This static construct_thread_function allows to construct
        // a proper thread function for a thread without having to
        // instantiate the component_base_actionN type. This is used by the
        // applier in case no continuation has been supplied.
        template <std::size_t ...Is, typename Args>
        static threads::thread_function_type
        construct_thread_function(naming::address::address_type lva,
            util::detail::pack_c<std::size_t, Is...>, Args&& args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                util::bind(util::one_shot(typename Derived::thread_function()),
                    lva, util::get<Is>(std::forward<Args>(args))...));
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
        // instantiate the component_base_actionN type. This is used by the
        // applier in case a continuation has been supplied
        template <typename Args>
        static threads::thread_function_type
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, Args&& args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                base_type::construct_continuation_thread_object_function(
                    cont, F, get_lva<Component>::call(lva),
                    std::forward<Args>(args)));
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

            return (get_lva<Component>::call(lva)->*F)(
                util::get<Is>(std::forward<Args>(args))...);
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
    template <
        typename Component, typename ...Ps,
        typename TF, TF F, typename Derived>
    class basic_action_impl<
            void (Component::*)(Ps...), TF, F, Derived>
      : public basic_action<Component, util::unused_type(Ps...), Derived>
    {
    public:
        typedef basic_action<
            Component, util::unused_type(Ps...), Derived> base_type;

        // Let the component decide whether the id is valid
        static bool is_target_valid(naming::id_type const& id)
        {
            return Component::is_target_valid(id);
        }

    protected:
        /// The \a thread_function will be registered as the thread
        /// function of a thread. It encapsulates the execution of the
        /// original function (given by \a func).
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;

            template <typename ...Ts>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva, Ts&&... vs) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component>::call(lva)) << ")";

                    (get_lva<Component>::call(lva)->*F)(
                        std::forward<Ts>(vs)...);
                }
                catch (hpx::thread_interrupted const&) { //-V565
                    /* swallow this exception */
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
        };

    public:
        // This static construct_thread_function allows to construct
        // a proper thread function for a thread without having to
        // instantiate the component_base_actionN type. This is used by the applier in
        // case no continuation has been supplied.
        template <std::size_t ...Is, typename Args>
        static threads::thread_function_type
        construct_thread_function(naming::address::address_type lva,
            util::detail::pack_c<std::size_t, Is...>, Args&& args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                util::bind(util::one_shot(typename Derived::thread_function()),
                    lva, util::get<Is>(std::forward<Args>(args))...));
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
        // instantiate the component_base_actionN type. This is used by the applier in
        // case a continuation has been supplied
        template <typename Args>
        static threads::thread_function_type
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, Args&& args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                base_type::construct_continuation_thread_object_function_void(
                    cont, F, get_lva<Component>::call(lva),
                    std::forward<Args>(args)));
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

            (get_lva<Component>::call(lva)->*F)(
                util::get<Is>(std::forward<Args>(args))...);
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
    /**/

#define HPX_DEFINE_COMPONENT_ACTION_2(component, func)                        \
    typedef HPX_MAKE_ACTION(component::func)::type                            \
        BOOST_PP_CAT(func, _action)                                           \
    /**/
#define HPX_DEFINE_COMPONENT_ACTION_3(component, func, action_type)           \
    typedef HPX_MAKE_ACTION(component::func)::type action_type                \
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
    /**/

#define HPX_DEFINE_COMPONENT_DIRECT_ACTION_2(component, func)                 \
    typedef HPX_MAKE_DIRECT_ACTION(component::func)::type                     \
        BOOST_PP_CAT(func, _action)                                           \
    /**/
#define HPX_DEFINE_COMPONENT_DIRECT_ACTION_3(component, func, action_type)    \
    typedef HPX_MAKE_DIRECT_ACTION(component::func)::type                     \
        action_type                                                           \
    /**/

///////////////////////////////////////////////////////////////////////////////
// same as above, just for template functions
#define HPX_DEFINE_COMPONENT_ACTION_TPL(...)                                  \
    HPX_DEFINE_COMPONENT_ACTION_TPL_(__VA_ARGS__)                             \
    /**/

#define HPX_DEFINE_COMPONENT_ACTION_TPL_(...)                                 \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                            \
        HPX_DEFINE_COMPONENT_ACTION_TPL_, HPX_UTIL_PP_NARG(__VA_ARGS__)       \
    )(__VA_ARGS__))                                                           \
    /**/

#define HPX_DEFINE_COMPONENT_ACTION_TPL_2(component, func)                    \
    typedef typename HPX_MAKE_ACTION(component::func)::type                   \
        BOOST_PP_CAT(func, _action)                                           \
    /**/
#define HPX_DEFINE_COMPONENT_ACTION_TPL_3(component, func, action_type)       \
    typedef typename HPX_MAKE_ACTION(component::func)::type                   \
        action_type                                                           \
    /**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_DEFINE_COMPONENT_DIRECT_ACTION_TPL(...)                           \
    HPX_DEFINE_COMPONENT_DIRECT_ACTION_TPL_(__VA_ARGS__)                      \
    /**/

#define HPX_DEFINE_COMPONENT_DIRECT_ACTION_TPL_(...)                          \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                            \
        HPX_DEFINE_COMPONENT_DIRECT_ACTION_TPL_, HPX_UTIL_PP_NARG(__VA_ARGS__)\
    )(__VA_ARGS__))                                                           \
    /**/

#define HPX_DEFINE_COMPONENT_DIRECT_ACTION_TPL_2(component, func)             \
    typedef typename HPX_MAKE_DIRECT_ACTION(component::func)::type            \
        BOOST_PP_CAT(func, _action)                                           \
    /**/
#define HPX_DEFINE_COMPONENT_DIRECT_ACTION_TPL_3(component, func, action_type)\
    typedef typename HPX_MAKE_DIRECT_ACTION(component::func)::type action_type\
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
    /**/

#define HPX_DEFINE_COMPONENT_CONST_ACTION_2(component, func)                  \
    typedef HPX_MAKE_ACTION(component::func)::type                            \
        BOOST_PP_CAT(func, _action)                                           \
    /**/
#define HPX_DEFINE_COMPONENT_CONST_ACTION_3(component, func, action_type)     \
    typedef HPX_MAKE_ACTION(component::func)::type action_type                \
    /**/
/// \endcond

/// \cond NOINTERNAL
#define HPX_DEFINE_COMPONENT_CONST_DIRECT_ACTION(...)                         \
    HPX_DEFINE_COMPONENT_CONST_DIRECT_ACTION_(__VA_ARGS__)                    \
    /**/

#define HPX_DEFINE_COMPONENT_CONST_DIRECT_ACTION_(...)                        \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                            \
        HPX_DEFINE_COMPONENT_CONST_DIRECT_ACTION_,                            \
            HPX_UTIL_PP_NARG(__VA_ARGS__)                                     \
    )(__VA_ARGS__))                                                           \
    /**/

#define HPX_DEFINE_COMPONENT_CONST_DIRECT_ACTION_2(component, func)           \
    typedef HPX_MAKE_DIRECT_ACTION(component::func)::type                     \
        BOOST_PP_CAT(func, _action)                                           \
    /**/
#define HPX_DEFINE_COMPONENT_CONST_DIRECT_ACTION_3(component, func,           \
        action_type)                                                          \
    typedef HPX_MAKE_DIRECT_ACTION(component::func)::type                     \
        action_type                                                           \
    /**/

///////////////////////////////////////////////////////////////////////////////
// same as above, just for template functions
#define HPX_DEFINE_COMPONENT_CONST_ACTION_TPL(...)                            \
    HPX_DEFINE_COMPONENT_CONST_ACTION_TPL_(__VA_ARGS__)                       \
    /**/

#define HPX_DEFINE_COMPONENT_CONST_ACTION_TPL_(...)                           \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                            \
        HPX_DEFINE_COMPONENT_CONST_ACTION_TPL_, HPX_UTIL_PP_NARG(__VA_ARGS__) \
    )(__VA_ARGS__))                                                           \
    /**/

#define HPX_DEFINE_COMPONENT_CONST_ACTION_TPL_2(component, func)              \
    typedef typename HPX_MAKE_ACTION(component::func)::type                   \
        BOOST_PP_CAT(func, _action)                                           \
    /**/
#define HPX_DEFINE_COMPONENT_CONST_ACTION_TPL_3(component, func, action_type) \
    typedef typename HPX_MAKE_ACTION(component::func)::type action_type       \
    /**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_DEFINE_COMPONENT_CONST_DIRECT_ACTION_TPL(...)                     \
    HPX_DEFINE_COMPONENT_CONST_DIRECT_ACTION_TPL_(__VA_ARGS__)                \
    /**/

#define HPX_DEFINE_COMPONENT_CONST_DIRECT_ACTION_TPL_(...)                    \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                            \
        HPX_DEFINE_COMPONENT_CONST_DIRECT_ACTION_TPL_,                        \
            HPX_UTIL_PP_NARG(__VA_ARGS__)                                     \
    )(__VA_ARGS__))                                                           \
    /**/

#define HPX_DEFINE_COMPONENT_CONST_DIRECT_ACTION_TPL_2(component, func)       \
    typedef typename HPX_MAKE_DIRECT_ACTION(component::func)::type            \
        BOOST_PP_CAT(func, _action)                                           \
    /**/
#define HPX_DEFINE_COMPONENT_CONST_DIRECT_ACTION_TPL_3(component, func,       \
        action_type)                                                          \
    typedef typename HPX_MAKE_DIRECT_ACTION(component::func)::type action_type\
    /**/
/// \endcond

#include <hpx/config/warnings_suffix.hpp>

#endif

