//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file component_action.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/actions_base/basic_action.hpp>
#include <hpx/actions_base/traits/is_client.hpp>
#include <hpx/futures/traits/future_access.hpp>
#include <hpx/futures/traits/is_future.hpp>
#include <hpx/modules/components_base.hpp>
#include <hpx/modules/format.hpp>
#include <hpx/naming_base/address.hpp>
#include <hpx/preprocessor/cat.hpp>
#include <hpx/preprocessor/expand.hpp>
#include <hpx/preprocessor/nargs.hpp>

#include <cstdlib>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace actions {

    /// \cond NOINTERNAL
    namespace detail {

        ///////////////////////////////////////////////////////////////////////
        inline std::string make_component_action_name(
            std::string_view action_name, void const* lva)
        {
            return hpx::util::format(
                "component action({}) lva({})", action_name, lva);
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename Component, typename R, typename F, typename... Ts>
        R component_invoke(naming::address_type lva,
            naming::component_type /*comptype*/, F Component::*f, Ts&&... vs)
        {
            Component* component = get_lva<Component>::call(lva);
            if constexpr (traits::is_future_v<R> || traits::is_client_v<R>)
            {
                // additional pinning is required such that the object becomes
                // unpinned only after the returned future has become ready
                components::pinned_ptr p =
                    components::pinned_ptr::create<Component>(lva);

                R result = (component->*f)(HPX_FORWARD(Ts, vs)...);

                if (!result.is_ready())
                {
                    traits::detail::get_shared_state(result)->set_on_completed(
                        [p = HPX_MOVE(p)]() {});
                }
                return result;
            }
            else
            {
                return (component->*f)(HPX_FORWARD(Ts, vs)...);
            }
        }
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    //  Specialized generic non-const component action types allowing to hold
    //  a different number of arguments
    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, typename R, typename... Ps,
        R (Component::*F)(Ps...), typename Derived>
    struct action<R (Component::*)(Ps...), F, Derived>
      : public basic_action<Component, R(Ps...),
            detail::action_type_t<action<R (Component::*)(Ps...), F, Derived>,
                Derived>>
    {
    public:
        using derived_type = detail::action_type_t<action, Derived>;

        static std::string get_action_name(naming::address_type lva)
        {
            return detail::make_component_action_name(
                detail::get_action_name<derived_type>(),
                get_lva<Component>::call(lva));
        }

        template <typename... Ts>
        static R invoke(naming::address_type lva,
            naming::component_type comptype, Ts&&... vs)
        {
            basic_action<Component, R(Ps...),
                derived_type>::increment_invocation_count();

            return detail::component_invoke<Component, R>(
                lva, comptype, F, HPX_FORWARD(Ts, vs)...);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    //  Specialized generic const component action types allowing to hold a
    //  different number of arguments
    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, typename R, typename... Ps,
        R (Component::*F)(Ps...) const, typename Derived>
    struct action<R (Component::*)(Ps...) const, F, Derived>
      : public basic_action<Component const, R(Ps...),
            detail::action_type_t<
                action<R (Component::*)(Ps...) const, F, Derived>, Derived>>
    {
    public:
        using derived_type = detail::action_type_t<action, Derived>;

        static std::string get_action_name(naming::address_type lva)
        {
            return detail::make_component_action_name(
                detail::get_action_name<derived_type>(),
                get_lva<Component>::call(lva));
        }

        template <typename... Ts>
        static R invoke(naming::address_type lva,
            naming::component_type comptype, Ts&&... vs)
        {
            basic_action<Component const, R(Ps...),
                derived_type>::increment_invocation_count();

            return detail::component_invoke<Component const, R>(
                lva, comptype, F, HPX_FORWARD(Ts, vs)...);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    //  Specialized generic non-const noexcept component action types allowing
    //  to hold a different number of arguments
    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, typename R, typename... Ps,
        R (Component::*F)(Ps...) noexcept, typename Derived>
    struct action<R (Component::*)(Ps...) noexcept, F, Derived>
      : public basic_action<Component, R(Ps...),
            detail::action_type_t<
                action<R (Component::*)(Ps...) noexcept, F, Derived>, Derived>>
    {
    public:
        using derived_type = detail::action_type_t<action, Derived>;

        static std::string get_action_name(naming::address_type lva)
        {
            return detail::make_component_action_name(
                detail::get_action_name<derived_type>(),
                get_lva<Component>::call(lva));
        }

        template <typename... Ts>
        static R invoke(naming::address_type lva,
            naming::component_type comptype, Ts&&... vs)
        {
            basic_action<Component, R(Ps...),
                derived_type>::increment_invocation_count();

            return detail::component_invoke<Component, R>(
                lva, comptype, F, HPX_FORWARD(Ts, vs)...);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    //  Specialized generic const noexcept component action types allowing to
    //  hold a different number of arguments
    template <typename Component, typename R, typename... Ps,
        R (Component::*F)(Ps...) const noexcept, typename Derived>
    struct action<R (Component::*)(Ps...) const noexcept, F, Derived>
      : public basic_action<Component const, R(Ps...),
            detail::action_type_t<
                action<R (Component::*)(Ps...) const noexcept, F, Derived>,
                Derived>>
    {
    public:
        using derived_type = detail::action_type_t<action, Derived>;

        static std::string get_action_name(naming::address_type lva)
        {
            return detail::make_component_action_name(
                detail::get_action_name<derived_type>(),
                get_lva<Component>::call(lva));
        }

        template <typename... Ts>
        static R invoke(naming::address_type lva,
            naming::component_type comptype, Ts&&... vs)
        {
            basic_action<Component const, R(Ps...),
                derived_type>::increment_invocation_count();

            return detail::component_invoke<Component const, R>(
                lva, comptype, F, HPX_FORWARD(Ts, vs)...);
        }
    };

    /// \endcond
}}    // namespace hpx::actions

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
///             : public hpx::components::component_base<server>
///           {
///               void print_greeting() const
///               {
///                   hpx::cout << "Hey, how are you?\n" << std::flush;
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
/// if the second argument with an appended suffix '_action' resolves to a valid,
/// unqualified C++ type name.
///
#if defined(HPX_COMPUTE_DEVICE_CODE)
#define HPX_DEFINE_COMPONENT_ACTION(...)                            /**/
#define HPX_DEFINE_COMPONENT_ACTION_3(component, func, name)        /**/
#define HPX_DEFINE_COMPONENT_ACTION_2(component, func)              /**/
#define HPX_DEFINE_COMPONENT_DIRECT_ACTION(...)                     /**/
#define HPX_DEFINE_COMPONENT_DIRECT_ACTION_3(component, func, name) /**/
#define HPX_DEFINE_COMPONENT_DIRECT_ACTION_2(component, func)       /**/
#else
#define HPX_DEFINE_COMPONENT_ACTION(...)                                       \
    HPX_DEFINE_COMPONENT_ACTION_(__VA_ARGS__)                                  \
    /**/

/// \cond NOINTERNAL
#define HPX_DEFINE_COMPONENT_ACTION_(...)                                      \
    HPX_PP_EXPAND(HPX_PP_CAT(                                                  \
        HPX_DEFINE_COMPONENT_ACTION_, HPX_PP_NARGS(__VA_ARGS__))(__VA_ARGS__)) \
    /**/

#define HPX_DEFINE_COMPONENT_ACTION_3(component, func, name)                   \
    struct name                                                                \
      : hpx::actions::make_action_t<decltype(&component::func),                \
            &component::func, name>                                            \
    {                                                                          \
    }; /**/
#define HPX_DEFINE_COMPONENT_ACTION_2(component, func)                         \
    HPX_DEFINE_COMPONENT_ACTION_3(component, func, HPX_PP_CAT(func, _action))  \
    /**/
/// \endcond

/// \cond NOINTERNAL
#define HPX_DEFINE_COMPONENT_DIRECT_ACTION(...)                                \
    HPX_DEFINE_COMPONENT_DIRECT_ACTION_(__VA_ARGS__)                           \
    /**/

#define HPX_DEFINE_COMPONENT_DIRECT_ACTION_(...)                               \
    HPX_PP_EXPAND(HPX_PP_CAT(HPX_DEFINE_COMPONENT_DIRECT_ACTION_,              \
        HPX_PP_NARGS(__VA_ARGS__))(__VA_ARGS__))                               \
    /**/

#define HPX_DEFINE_COMPONENT_DIRECT_ACTION_3(component, func, name)            \
    struct name                                                                \
      : hpx::actions::make_direct_action_t<decltype(&component::func),         \
            &component::func, name>                                            \
    {                                                                          \
    }; /**/

#define HPX_DEFINE_COMPONENT_DIRECT_ACTION_2(component, func)                  \
    HPX_DEFINE_COMPONENT_DIRECT_ACTION_3(                                      \
        component, func, HPX_PP_CAT(func, _action))                            \
    /**/
/// \endcond
#endif

#include <hpx/config/warnings_suffix.hpp>
