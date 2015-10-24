//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file plain_action.hpp

#if !defined(HPX_RUNTIME_ACTIONS_PLAIN_ACTION_NOV_14_2008_0706PM)
#define HPX_RUNTIME_ACTIONS_PLAIN_ACTION_NOV_14_2008_0706PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/config.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/actions/basic_action.hpp>
#include <hpx/runtime/components/console_error_sink.hpp>
#include <hpx/util/unused.hpp>
#include <hpx/util/detail/count_num_args.hpp>
#include <hpx/util/detail/pp_strip_parens.hpp>
#include <hpx/util/detail/pack.hpp>

#include <boost/preprocessor/cat.hpp>

#include <cstdlib>
#include <stdexcept>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace actions
{
    /// \cond NOINTERNAL

    namespace detail
    {
        struct plain_function
        {
            template <typename F>
            static threads::thread_function_type
            decorate_action(naming::address_type, F && f)
            {
                return std::forward<F>(f);
            }

            static void schedule_thread(naming::address_type,
                threads::thread_init_data& data,
                threads::thread_state_enum initial_state)
            {
                hpx::threads::register_work_plain(data, initial_state); //-V106
            }
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    //  Specialized generic plain (free) action types allowing to hold a
    //  different number of arguments
    ///////////////////////////////////////////////////////////////////////////
    template <
        typename R, typename ...Ps,
        typename TF, TF F, typename Derived>
    class basic_action_impl<R (*)(Ps...), TF, F, Derived>
      : public basic_action<detail::plain_function, R(Ps...), Derived>
    {
    public:

        typedef void is_plain_action;

        static std::string get_action_name(naming::address::address_type /*lva*/)
        {
            std::stringstream name;
            name << "plain action(" << detail::get_action_name<Derived>() << ")";
            return name.str();
        }

        // Only localities are valid targets for a plain action
        static bool is_target_valid(naming::id_type const& id)
        {
            return naming::is_locality(id);
        }

        template <typename ...Ts>
        static R invoke(naming::address::address_type /*lva*/, Ts&&... vs)
        {
            basic_action<detail::plain_function, R(Ps...), Derived>::
                increment_invocation_count();
            return F(std::forward<Ts>(vs)...);
        }
    };

    /// \endcond
}}

namespace hpx { namespace traits
{
    template <> HPX_ALWAYS_EXPORT
    inline components::component_type
    component_type_database<hpx::actions::detail::plain_function>::get()
    {
        return hpx::components::component_plain_function;
    }

    template <> HPX_ALWAYS_EXPORT
    inline void
    component_type_database<hpx::actions::detail::plain_function>::set(
        components::component_type)
    {
        HPX_ASSERT(false);      // shouldn't be ever called
    }
}}

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
/// \endcode
///
/// \note Usually this macro will not be used in user code unless the intend is
/// to avoid defining the action_type in global namespace. Normally, the use of
/// the macro \a HPX_PLAIN_ACTION is recommend.
///
/// \note The macro \a HPX_DEFINE_PLAIN_ACTION can be used with 1 or 2
/// arguments. The second argument is optional. The default value for the
/// second argument (the typename of the defined action) is derived from the
/// name of the function (as passed as the first argument) by appending '_action'.
/// The second argument can be omitted only if the first argument with an
/// appended suffix '_action' resolves to a valid, unqualified C++ type name.
///
#define HPX_DEFINE_PLAIN_ACTION(...)                                          \
    HPX_DEFINE_PLAIN_ACTION_(__VA_ARGS__)                                     \
    /**/

/// \cond NOINTERNAL

#define HPX_DEFINE_PLAIN_DIRECT_ACTION(...)                                   \
    HPX_DEFINE_PLAIN_DIRECT_ACTION_(__VA_ARGS__)                              \
    /**/

#define HPX_DEFINE_PLAIN_ACTION_(...)                                         \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                            \
        HPX_DEFINE_PLAIN_ACTION_, HPX_UTIL_PP_NARG(__VA_ARGS__)               \
    )(__VA_ARGS__))                                                           \
    /**/

#define HPX_DEFINE_PLAIN_DIRECT_ACTION_(...)                                  \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                            \
        HPX_DEFINE_PLAIN_DIRECT_ACTION_, HPX_UTIL_PP_NARG(__VA_ARGS__)        \
    )(__VA_ARGS__))                                                           \
    /**/

#define HPX_DEFINE_PLAIN_ACTION_1(func)                                       \
    HPX_DEFINE_PLAIN_ACTION_2(func, BOOST_PP_CAT(func, _action))              \
    /**/

#define HPX_DEFINE_PLAIN_ACTION_2(func, name)                                 \
    struct name : hpx::actions::make_action<                                  \
        decltype(&func), &func, name>::type {}                                \
    /**/

#define HPX_DEFINE_PLAIN_DIRECT_ACTION_1(func)                                \
    HPX_DEFINE_PLAIN_DIRECT_ACTION_2(func, BOOST_PP_CAT(func, _action))       \
    /**/

#define HPX_DEFINE_PLAIN_DIRECT_ACTION_2(func, name)                          \
    struct name : hpx::actions::make_direct_action<                           \
        decltype(&func), &func, name>::type {}                                \
    /**/

/// \endcond

/// \def HPX_PLAIN_ACTION(func, name)
///
/// \brief Defines a plain action type based on the given function
/// \a func and registers it with HPX.
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
/// \note The macro \a HPX_PLAIN_ACTION_ID can be used with 1, 2, or 3 arguments.
/// The second and third arguments are optional. The default value for the
/// second argument (the typename of the defined action) is derived from the
/// name of the function (as passed as the first argument) by appending '_action'.
/// The second argument can be omitted only if the first argument with an
/// appended suffix '_action' resolves to a valid, unqualified C++ type name.
/// The default value for the third argument is \a hpx::components::factory_check.
///
/// \note Only one of the forms of this macro \a HPX_PLAIN_ACTION or
///       \a HPX_PLAIN_ACTION_ID should be used for a particular action,
///       never both.
///
#define HPX_PLAIN_ACTION(...)                                                 \
    HPX_PLAIN_ACTION_(__VA_ARGS__)                                            \
/**/

/// \def HPX_PLAIN_ACTION_ID(func, actionname, actionid)
///
/// \brief Defines a plain action type based on the given function \a func and
///   registers it with HPX.
///
/// The macro \a HPX_PLAIN_ACTION_ID can be used to define a plain action (e.g. an
/// action encapsulating a global or free function) based on the given function
/// \a func. It defines the action type \a actionname representing the given function.
/// The parameter \a actionid
///
/// The parameter \a actionid specifies an unique integer value which will be
/// used to represent the action during serialization.
///
/// The parameter \p func is a global or free (non-member) function which
/// should be encapsulated into a plain action. The parameter \p name is the
/// name of the action type defined by this macro.
///
/// The second parameter has to be usable as a plain (non-qualified) C++
/// identifier, it should not contain special characters which cannot be part
/// of a C++ identifier, such as '<', '>', or ':'.
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
///     HPX_PLAIN_ACTION_ID(app::some_global_function, some_global_action,
///       some_unique_id);
/// \endcode
///
/// \note The macro \a HPX_PLAIN_ACTION_ID has to be used at global namespace even
/// if the wrapped function is located in some other namespace. The newly
/// defined action type is placed into the global namespace as well.
///
/// \note Only one of the forms of this macro \a HPX_PLAIN_ACTION or
///       \a HPX_PLAIN_ACTION_ID should be used for a particular action,
///       never both.
///
#define HPX_PLAIN_ACTION_ID(func, name, id)                                   \
    HPX_DEFINE_PLAIN_ACTION(func, name);                                      \
    HPX_REGISTER_ACTION_DECLARATION(name, name);                              \
    HPX_REGISTER_ACTION_ID(name, name, id);                                   \
/**/

/// \cond NOINTERNAL

#define HPX_PLAIN_DIRECT_ACTION(...)                                          \
    HPX_PLAIN_DIRECT_ACTION_(__VA_ARGS__)                                     \
/**/

/// \endcond

/// \cond NOINTERNAL

// macros for plain actions
#define HPX_PLAIN_ACTION_(...)                                                \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                            \
        HPX_PLAIN_ACTION_, HPX_UTIL_PP_NARG(__VA_ARGS__)                      \
    )(__VA_ARGS__))                                                           \
/**/
#define HPX_PLAIN_ACTION_2(func, name)                                        \
    HPX_DEFINE_PLAIN_ACTION(func, name);                                      \
    HPX_REGISTER_ACTION_DECLARATION(name, name);                              \
    HPX_REGISTER_ACTION(name, name);                                          \
/**/
#define HPX_PLAIN_ACTION_1(func)                                              \
    HPX_PLAIN_ACTION_2(func, BOOST_PP_CAT(func, _action));                    \
/**/

// same for direct actions
#define HPX_PLAIN_DIRECT_ACTION_(...)                                         \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                            \
        HPX_PLAIN_DIRECT_ACTION_, HPX_UTIL_PP_NARG(__VA_ARGS__)               \
    )(__VA_ARGS__))                                                           \
/**/
#define HPX_PLAIN_DIRECT_ACTION_2(func, name)                                 \
    HPX_DEFINE_PLAIN_DIRECT_ACTION(func, name);                               \
    HPX_REGISTER_ACTION_DECLARATION(name, name);                              \
    HPX_REGISTER_ACTION(name, name);                                          \
/**/
#define HPX_PLAIN_DIRECT_ACTION_1(func)                                       \
    HPX_PLAIN_DIRECT_ACTION_2(func, BOOST_PP_CAT(func, _action));             \
/**/
#define HPX_PLAIN_DIRECT_ACTION_ID(func, name, id)                            \
    HPX_DEFINE_PLAIN_DIRECT_ACTION(func, name);                               \
    HPX_REGISTER_ACTION_DECLARATION(name, name);                              \
    HPX_REGISTER_ACTION_ID(name, name, id);                                   \
/**/

/// \endcond

#include <hpx/config/warnings_suffix.hpp>

#endif

