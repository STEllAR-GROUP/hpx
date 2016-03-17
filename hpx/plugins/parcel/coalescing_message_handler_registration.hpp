//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_COALESCING_MESSAGE_HANDLER_REGISTRATION_MAR_17_2016_0150PM)
#define HPX_RUNTIME_COALESCING_MESSAGE_HANDLER_REGISTRATION_MAR_17_2016_0150PM

#include <hpx/config.hpp>

#if defined(HPX_HAVE_PARCEL_COALESCING)

#include <hpx/traits/action_message_handler.hpp>
#include <hpx/runtime/message_handler_fwd.hpp>
#include <hpx/runtime/parcelset/policies/message_handler.hpp>

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
            hpx::register_message_handler(
                "coalescing_message_handler",
                get_action_coalescing_name<Action>()
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
        template<>                                                            \
        inline char const* get_action_coalescing_name<Action>()               \
        {                                                                     \
            return coalescing_name;                                           \
        }                                                                     \
        template register_coalescing_for_action<Action>                       \
            register_coalescing_for_action<Action>::instance_;                \
    }}                                                                        \
/**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_ACTION_USES_MESSAGE_COALESCING(...)                               \
    HPX_ACTION_USES_MESSAGE_COALESCING_(__VA_ARGS__)                          \
/**/

#define HPX_ACTION_USES_MESSAGE_COALESCING_(...)                              \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                            \
        HPX_ACTION_USES_MESSAGE_COALESCING_, HPX_UTIL_PP_NARG(__VA_ARGS__)    \
    )(__VA_ARGS__))                                                           \
/**/

#define HPX_ACTION_USES_MESSAGE_COALESCING_1(action_type)                     \
    HPX_ACTION_USES_MESSAGE_COALESCING_4(action_type,                         \
        BOOST_PP_STRINGIZE(action_type), std::size_t(-1), std::size_t(-1))    \
/**/

#define HPX_ACTION_USES_MESSAGE_COALESCING_2(action_type, num)                \
    HPX_ACTION_USES_MESSAGE_COALESCING_3(action_type,                         \
        BOOST_PP_STRINGIZE(action_type), num, std::size_t(-1))                \
/**/

#define HPX_ACTION_USES_MESSAGE_COALESCING_3(action_type, num, interval)      \
    HPX_ACTION_USES_MESSAGE_COALESCING_3(action_type,                         \
        BOOST_PP_STRINGIZE(action_type), num, interval)                       \
/**/

#define HPX_ACTION_USES_MESSAGE_COALESCING_4(                                 \
        action_type, action_name, num, interval)                              \
    namespace hpx { namespace traits                                          \
    {                                                                         \
        template <>                                                           \
        struct action_message_handler<action_type>                            \
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
    HPX_REGISTER_COALESCING_COUNTERS(action_type, action_name);               \
/**/

#define HPX_ACTION_USES_MESSAGE_COALESCING_NOTHROW(                           \
        action_type, action_name, num, interval)                              \
    namespace hpx { namespace traits                                          \
    {                                                                         \
        template <>                                                           \
        struct action_message_handler<action_type>                            \
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
    HPX_REGISTER_COALESCING_COUNTERS(action_type, action_name);               \
/**/

#else

#define HPX_ACTION_USES_MESSAGE_COALESCING(...)
#define HPX_ACTION_USES_MESSAGE_COALESCING_NOTHROW(...)

#endif

#endif
