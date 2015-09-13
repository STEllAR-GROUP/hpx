//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file component_action.hpp

#if !defined(HPX_RUNTIME_ACTIONS_COMPONENT_ACTION_MAR_26_2008_1054AM)
#define HPX_RUNTIME_ACTIONS_COMPONENT_ACTION_MAR_26_2008_1054AM

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

#include <boost/preprocessor/cat.hpp>

#include <cstdlib>
#include <stdexcept>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace actions
{
    /// \cond NOINTERNAL

    ///////////////////////////////////////////////////////////////////////////
    //  Specialized generic non-conts component action types allowing to hold
    //  a different number of arguments
    ///////////////////////////////////////////////////////////////////////////
    template <
        typename Component, typename R, typename ...Ps,
        typename TF, TF F, typename Derived>
    class basic_action_impl<
            R (Component::*)(Ps...), TF, F, Derived>
      : public basic_action<Component, R(Ps...), Derived>
    {
    public:
        static std::string get_action_name(naming::address::address_type lva)
        {
            std::stringstream name;
            name << "component action(" << detail::get_action_name<Derived>() <<
                ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            return name.str();
        }

        // Let the component decide whether the id is valid
        static bool is_target_valid(naming::id_type const& id)
        {
            return Component::is_target_valid(id);
        }

        template <typename ...Ts>
        static R invoke(naming::address::address_type lva, Ts&&... vs)
        {
            return (get_lva<Component>::call(lva)->*F)
                (std::forward<Ts>(vs)...);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    //  Specialized generic const component action types allowing to hold a
    //  different number of arguments
    ///////////////////////////////////////////////////////////////////////////
    template <
        typename Component, typename R, typename ...Ps,
        typename TF, TF F, typename Derived>
    class basic_action_impl<
            R (Component::*)(Ps...) const, TF, F, Derived>
      : public basic_action<Component const, R(Ps...), Derived>
    {
    public:
        static std::string get_action_name(naming::address::address_type lva)
        {
            std::stringstream name;
            name << "component action(" << detail::get_action_name<Derived>() <<
                ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component>::call(lva)) << ")";
            return name.str();
        }

        // Let the component decide whether the id is valid
        static bool is_target_valid(naming::id_type const& id)
        {
            return Component::is_target_valid(id);
        }

        template <typename ...Ts>
        static R invoke(naming::address::address_type lva, Ts&&... vs)
        {
            return (get_lva<Component const>::call(lva)->*F)
                (std::forward<Ts>(vs)...);
        }
    };

    /// \endcond
}}

/// \def HPX_DEFINE_COMPONENT_ACTION(component, func, action_type)
///
/// \brief Registers a  member function of a component as an action type
/// with HPX
///
/// The macro \a HPX_DEFINE_COMPONENT_ACTION can be used to register a
/// member function of a component as an action type named \a action_type.
///
/// The parameter \a component is the type of the component exposing the
/// member function \a func which should be associated with the newly defined
/// action type. The parameter \p action_type is the name of the action type to
/// register with HPX.
///
/// \par Example:
///
/// \code
///       namespace app
///       {
///           // Define a simple component exposing one action 'print_greeting'
///           class HPX_COMPONENT_EXPORT server
///             : public hpx::components::simple_component_base<server>
///           {
///               void print_greeting() const
///               {
///                   hpx::cout << "Hey, how are you?\n" << hpx::flush;
///               }
///
///               // Component actions need to be declared, this also defines the
///               // type 'print_greeting_action' representing the action.
///               HPX_DEFINE_COMPONENT_ACTION(server, print_greeting,
///                   print_greeting_action);
///           };
///       }
/// \endcode
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

#define HPX_DEFINE_COMPONENT_ACTION_3(component, func, name)                  \
    struct name : hpx::actions::make_action<                                  \
        decltype(&component::func), &component::func, name>::type {}          \
    /**/
#define HPX_DEFINE_COMPONENT_ACTION_2(component, func)                        \
    HPX_DEFINE_COMPONENT_ACTION_3(component, func,                            \
        BOOST_PP_CAT(func, _action))                                          \
    /**/
/// \endcond

/// \cond NOINTERNAL
#define HPX_DEFINE_COMPONENT_DIRECT_ACTION(...)                               \
    HPX_DEFINE_COMPONENT_DIRECT_ACTION_(__VA_ARGS__)                          \
    /**/

#define HPX_DEFINE_COMPONENT_DIRECT_ACTION_(...)                              \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                            \
        HPX_DEFINE_COMPONENT_DIRECT_ACTION_,                                  \
            HPX_UTIL_PP_NARG(__VA_ARGS__)                                     \
    )(__VA_ARGS__))                                                           \
    /**/

#define HPX_DEFINE_COMPONENT_DIRECT_ACTION_3(component, func, name)           \
    struct name : hpx::actions::make_direct_action<                           \
        decltype(&component::func), &component::func, name>::type {}          \
    /**/

#define HPX_DEFINE_COMPONENT_DIRECT_ACTION_2(component, func)                 \
    HPX_DEFINE_COMPONENT_DIRECT_ACTION_3(component, func,                     \
        BOOST_PP_CAT(func, _action))                                          \
    /**/
/// \endcond

#include <hpx/config/warnings_suffix.hpp>

#endif
