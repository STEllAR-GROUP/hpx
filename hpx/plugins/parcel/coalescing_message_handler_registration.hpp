//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PLUGINS_PARCEL_COALESCING_MESSAGE_HANDLER_REGISTRATION_HPP
#define HPX_PLUGINS_PARCEL_COALESCING_MESSAGE_HANDLER_REGISTRATION_HPP

#include <hpx/config.hpp>

// the module itself should not register any actions which coalesce parcels
#if defined(HPX_HAVE_PARCEL_COALESCING) && !defined(HPX_PARCEL_COALESCING_MODULE_EXPORTS)

#include <hpx/error_code.hpp>
#include <hpx/runtime/message_handler_fwd.hpp>
#include <hpx/runtime/parcelset/policies/message_handler.hpp>
#include <hpx/runtime/parcelset_fwd.hpp>
#include <hpx/traits/action_message_handler.hpp>
#include <hpx/util/detail/pp/cat.hpp>
#include <hpx/util/detail/pp/stringize.hpp>

#include <cstddef>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Action> char const* get_action_coalescing_name();

    template <typename Action>
    struct register_coalescing_for_action
    {
        register_coalescing_for_action()
        {
            // ignore all errors as the module might not be available
            hpx::error_code ec (hpx::lightweight);
            hpx::register_message_handler(
                "coalescing_message_handler",
                get_action_coalescing_name<Action>(),
                ec
            );
        }
        static register_coalescing_for_action instance_;
    };

    template <typename Action>
    register_coalescing_for_action<Action>
        register_coalescing_for_action<Action>::instance_;
}}

///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_COALESCING_COUNTERS(Action, coalescing_name)             \
    namespace hpx { namespace parcelset                                       \
    {                                                                         \
        template<> HPX_ALWAYS_EXPORT                                          \
        char const* get_action_coalescing_name< Action>()                     \
        {                                                                     \
            return coalescing_name;                                           \
        }                                                                     \
        template register_coalescing_for_action< Action>                      \
            register_coalescing_for_action< Action>::instance_;               \
    }}                                                                        \
/**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_ACTION_USES_MESSAGE_COALESCING_DECLARATION(...)                   \
    HPX_ACTION_USES_MESSAGE_COALESCING_DECLARATION_(__VA_ARGS__)              \
/**/

#define HPX_ACTION_USES_MESSAGE_COALESCING_DECLARATION_(...)                  \
    HPX_PP_EXPAND(HPX_PP_CAT(                                                 \
        HPX_ACTION_USES_MESSAGE_COALESCING_DECLARATION_,                      \
            HPX_PP_NARGS(__VA_ARGS__)                                         \
    )(__VA_ARGS__))                                                           \
/**/

#define HPX_ACTION_USES_MESSAGE_COALESCING_DECLARATION_1(action_type)         \
    HPX_ACTION_USES_MESSAGE_COALESCING_DECLARATION_4(action_type,             \
        HPX_PP_STRINGIZE(action_type), std::size_t(-1), std::size_t(-1))      \
/**/

#define HPX_ACTION_USES_MESSAGE_COALESCING_DECLARATION_2(action_type, num)    \
    HPX_ACTION_USES_MESSAGE_COALESCING_DECLARATION_3(action_type,             \
        HPX_PP_STRINGIZE(action_type), num, std::size_t(-1))                  \
/**/

#define HPX_ACTION_USES_MESSAGE_COALESCING_DECLARATION_3(                     \
        action_type, num, interval)                                           \
    HPX_ACTION_USES_MESSAGE_COALESCING_DECLARATION_3(action_type,             \
        HPX_PP_STRINGIZE(action_type), num, interval)                         \
/**/

#define HPX_ACTION_USES_MESSAGE_COALESCING_DECLARATION_4(                     \
        action_type, action_name, num, interval)                              \
    namespace hpx { namespace traits                                          \
    {                                                                         \
        template <>                                                           \
        struct action_message_handler< action_type>                           \
        {                                                                     \
            static parcelset::policies::message_handler* call(                \
                parcelset::parcelhandler* ph, parcelset::locality const& loc, \
                parcelset::parcel const& /*p*/)                               \
            {                                                                 \
                return parcelset::get_message_handler(ph, action_name,        \
                    "coalescing_message_handler", num, interval, loc);        \
            }                                                                 \
        };                                                                    \
    }}                                                                        \
/**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_ACTION_USES_MESSAGE_COALESCING_DEFINITION(...)                    \
    HPX_ACTION_USES_MESSAGE_COALESCING_DEFINITION_(__VA_ARGS__)               \
/**/

#define HPX_ACTION_USES_MESSAGE_COALESCING_DEFINITION_(...)                   \
    HPX_PP_EXPAND(HPX_PP_CAT(                                                 \
        HPX_ACTION_USES_MESSAGE_COALESCING_DEFINITION_,                       \
            HPX_PP_NARGS(__VA_ARGS__)                                         \
    )(__VA_ARGS__))                                                           \
/**/

#define HPX_ACTION_USES_MESSAGE_COALESCING_DEFINITION_1(action_type)          \
    HPX_ACTION_USES_MESSAGE_COALESCING_DEFINITION_4(action_type,              \
        HPX_PP_STRINGIZE(action_type), std::size_t(-1), std::size_t(-1))      \
/**/

#define HPX_ACTION_USES_MESSAGE_COALESCING_DEFINITION_2(action_type, num)     \
    HPX_ACTION_USES_MESSAGE_COALESCING_DEFINITION_3(action_type,              \
        HPX_PP_STRINGIZE(action_type), num, std::size_t(-1))                  \
/**/

#define HPX_ACTION_USES_MESSAGE_COALESCING_DEFINITION_3(                      \
        action_type, num, interval)                                           \
    HPX_ACTION_USES_MESSAGE_COALESCING_DEFINITION_3(action_type,              \
        HPX_PP_STRINGIZE(action_type), num, interval)                         \
/**/

#define HPX_ACTION_USES_MESSAGE_COALESCING_DEFINITION_4(                      \
        action_type, action_name, num, interval)                              \
    HPX_REGISTER_COALESCING_COUNTERS(action_type, action_name);               \
/**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_ACTION_USES_MESSAGE_COALESCING(...)                               \
    HPX_ACTION_USES_MESSAGE_COALESCING_DECLARATION(__VA_ARGS__)               \
    HPX_ACTION_USES_MESSAGE_COALESCING_DEFINITION(__VA_ARGS__)                \
/**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_ACTION_USES_MESSAGE_COALESCING_NOTHROW_DECLARATION(               \
        action_type, action_name, num, interval)                              \
    namespace hpx { namespace traits                                          \
    {                                                                         \
        template <>                                                           \
        struct action_message_handler< action_type>                           \
        {                                                                     \
            static parcelset::policies::message_handler* call(                \
                parcelset::parcelhandler* ph, parcelset::locality const& loc, \
                parcelset::parcel const& /*p*/)                               \
            {                                                                 \
                error_code ec(lightweight);                                   \
                return parcelset::get_message_handler(ph, action_name,        \
                    "coalescing_message_handler", num, interval, loc, ec);    \
            }                                                                 \
        };                                                                    \
    }}                                                                        \
/**/

#define HPX_ACTION_USES_MESSAGE_COALESCING_NOTHROW_DEFINITION(                \
        action_type, action_name, num, interval)                              \
    HPX_REGISTER_COALESCING_COUNTERS(action_type, action_name);               \

#define HPX_ACTION_USES_MESSAGE_COALESCING_NOTHROW(...)                       \
    HPX_ACTION_USES_MESSAGE_COALESCING_NOTHROW_DECLARATION(__VA_ARGS__)       \
    HPX_ACTION_USES_MESSAGE_COALESCING_NOTHROW_DEFINITION(__VA_ARGS__)        \
/**/

#else

#define HPX_ACTION_USES_MESSAGE_COALESCING_DECLARATION(...)
#define HPX_ACTION_USES_MESSAGE_COALESCING_DEFINITION(...)
#define HPX_ACTION_USES_MESSAGE_COALESCING(...)

#define HPX_ACTION_USES_MESSAGE_COALESCING_NOTHROW_DECLARATION(...)
#define HPX_ACTION_USES_MESSAGE_COALESCING_NOTHROW_DEFINITION(...)
#define HPX_ACTION_USES_MESSAGE_COALESCING_NOTHROW(...)

#endif

#endif /*HPX_PLUGINS_PARCEL_COALESCING_MESSAGE_HANDLER_REGISTRATION_HPP*/
