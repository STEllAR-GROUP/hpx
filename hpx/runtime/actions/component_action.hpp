//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file component_action.hpp

#if !defined(HPX_RUNTIME_ACTIONS_COMPONENT_ACTION_MAR_26_2008_1054AM)
#define HPX_RUNTIME_ACTIONS_COMPONENT_ACTION_MAR_26_2008_1054AM

#include <hpx/config.hpp>
#include <hpx/runtime/actions/basic_action.hpp>
#include <hpx/runtime/components/pinned_ptr.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/traits/is_future.hpp>
#include <hpx/util/detail/pp/cat.hpp>
#include <hpx/util/detail/pp/expand.hpp>
#include <hpx/util/detail/pp/nargs.hpp>

#include <boost/utility/string_ref.hpp>

#include <cstdlib>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace actions
{
    /// \cond NOINTERNAL
    namespace detail
    {
        struct keep_object_pinned
        {
            void operator()() const {}

            components::pinned_ptr p_;
        };

        ///////////////////////////////////////////////////////////////////////
        inline std::string make_component_action_name(
            boost::string_ref action_name, void const* lva)
        {
            std::stringstream name;
            name << "component action(" << action_name << ") lva(" << lva << ")";
            return name.str();
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename Component, typename R, typename F, typename ...Ts>
        R component_invoke(std::false_type,
            naming::address::address_type lva,
            naming::address::component_type /*comptype*/,
            F Component::*f, Ts&&... vs)
        {
            Component* component = get_lva<Component>::call(lva);
            return (component->*f)(std::forward<Ts>(vs)...);
        }

        template <typename Component, typename R, typename F, typename ...Ts>
        R component_invoke(std::true_type,
            naming::address::address_type lva,
            naming::address::component_type comptype,
            F Component::*f, Ts&&... vs)
        {
            // additional pinning is required such that the object becomes
            // unpinned only after the returned future has become ready
            components::pinned_ptr p =
                components::pinned_ptr::create<Component>(lva);

            Component* component = get_lva<Component>::call(lva);
            R result = (component->*f)(std::forward<Ts>(vs)...);

            traits::detail::get_shared_state(result)->set_on_completed(
                detail::keep_object_pinned{std::move(p)});

            return result;
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    //  Specialized generic non-const component action types allowing to hold
    //  a different number of arguments
    ///////////////////////////////////////////////////////////////////////////
    template <
        typename Component, typename R, typename ...Ps,
        typename TF, TF F, typename Derived>
    class basic_action_impl<R (Component::*)(Ps...), TF, F, Derived>
      : public basic_action<Component, R(Ps...), Derived>
    {
    public:
        static std::string get_action_name(naming::address::address_type lva)
        {
            return detail::make_component_action_name(
                detail::get_action_name<Derived>(),
                get_lva<Component>::call(lva));
        }

        template <typename ...Ts>
        static R invoke(
            naming::address::address_type lva,
            naming::address::component_type comptype, Ts&&... vs)
        {
            basic_action<Component, R(Ps...), Derived>::
                increment_invocation_count();

            using is_future = typename traits::is_future<R>::type;
            return detail::component_invoke<Component, R>(is_future{},
                lva, comptype, F, std::forward<Ts>(vs)...);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    //  Specialized generic const component action types allowing to hold a
    //  different number of arguments
    ///////////////////////////////////////////////////////////////////////////
    template <
        typename Component, typename R, typename ...Ps,
        typename TF, TF F, typename Derived>
    class basic_action_impl<R (Component::*)(Ps...) const, TF, F, Derived>
      : public basic_action<Component const, R(Ps...), Derived>
    {
    public:
        static std::string get_action_name(naming::address::address_type lva)
        {
            return detail::make_component_action_name(
                detail::get_action_name<Derived>(),
                get_lva<Component>::call(lva));
        }

        template <typename ...Ts>
        static R invoke(
            naming::address::address_type lva,
            naming::address::component_type comptype, Ts&&... vs)
        {
            basic_action<Component const, R(Ps...), Derived>::
                increment_invocation_count();

            using is_future = typename traits::is_future<R>::type;
            return detail::component_invoke<Component const, R>(is_future{},
                lva, comptype, F, std::forward<Ts>(vs)...);
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
/// if the second argument with an appended suffix '_action' resolves to a valid,
/// unqualified C++ type name.
///
#define HPX_DEFINE_COMPONENT_ACTION(...)                                      \
    HPX_DEFINE_COMPONENT_ACTION_(__VA_ARGS__)                                 \
    /**/

/// \cond NOINTERNAL
#define HPX_DEFINE_COMPONENT_ACTION_(...)                                     \
    HPX_PP_EXPAND(HPX_PP_CAT(                                                 \
        HPX_DEFINE_COMPONENT_ACTION_, HPX_PP_NARGS(__VA_ARGS__)               \
    )(__VA_ARGS__))                                                           \
    /**/

#define HPX_DEFINE_COMPONENT_ACTION_3(component, func, name)                  \
    struct name : hpx::actions::make_action<                                  \
        decltype(&component::func), &component::func, name>::type {}          \
    /**/
#define HPX_DEFINE_COMPONENT_ACTION_2(component, func)                        \
    HPX_DEFINE_COMPONENT_ACTION_3(component, func,                            \
        HPX_PP_CAT(func, _action))                                            \
    /**/
/// \endcond

/// \cond NOINTERNAL
#define HPX_DEFINE_COMPONENT_DIRECT_ACTION(...)                               \
    HPX_DEFINE_COMPONENT_DIRECT_ACTION_(__VA_ARGS__)                          \
    /**/

#define HPX_DEFINE_COMPONENT_DIRECT_ACTION_(...)                              \
    HPX_PP_EXPAND(HPX_PP_CAT(                                                 \
        HPX_DEFINE_COMPONENT_DIRECT_ACTION_,                                  \
            HPX_PP_NARGS(__VA_ARGS__)                                         \
    )(__VA_ARGS__))                                                           \
    /**/

#define HPX_DEFINE_COMPONENT_DIRECT_ACTION_3(component, func, name)           \
    struct name : hpx::actions::make_direct_action<                           \
        decltype(&component::func), &component::func, name>::type {}          \
    /**/

#define HPX_DEFINE_COMPONENT_DIRECT_ACTION_2(component, func)                 \
    HPX_DEFINE_COMPONENT_DIRECT_ACTION_3(component, func,                     \
        HPX_PP_CAT(func, _action))                                            \
    /**/
/// \endcond

#include <hpx/config/warnings_suffix.hpp>

#endif
