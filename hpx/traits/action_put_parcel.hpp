//  Copyright (c) 2016 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_ACTION_PUT_PARCEL_AUG_2016)
#define HPX_TRAITS_ACTION_PUT_PARCEL_AUG_2016

#include <hpx/config.hpp>
#include <hpx/runtime/threads/thread_enums.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/parcelset/put_parcel.hpp>
#include <hpx/runtime/parcelset/policies/message_handler.hpp>
#include <hpx/traits/extract_action.hpp>
#include <utility>

namespace hpx { namespace traits
{
    namespace detail {

        ///////////////////////////////////////////////////////////////////////
        template <typename Action>
        inline naming::address&& complement_addr(naming::address& addr)
        {
            if (components::component_invalid == addr.type_)
            {
                addr.type_ = components::get_component_type<
                    typename Action::component_type>();
            }
            return std::move(addr);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // Customization point for action put parcel
    template <typename Action, typename Enable = void>
    struct action_put_parcel
    {
        template <typename ...Ts>
        static inline bool
        call(naming::id_type const& id, naming::address&& addr,
            threads::thread_priority priority, Ts&&... vs)
        {
            typedef
                typename hpx::traits::extract_action<Action>::type
                action_type;
            action_type act;

            parcelset::put_parcel(id, detail::complement_addr<action_type>(addr),
                act, priority, std::forward<Ts>(vs)...);

            return false;     // destinations are remote
        }

        template <typename Continuation, typename ...Ts>
        static inline bool
        call_cont(naming::id_type const& id, naming::address&& addr,
            threads::thread_priority priority,
            Continuation && cont, Ts&&... vs)
        {
            typedef
                typename hpx::traits::extract_action<Action>::type
                action_type;
            action_type act;

            parcelset::put_parcel(id, detail::complement_addr<action_type>(addr),
                std::forward<Continuation>(cont),
                act, priority, std::forward<Ts>(vs)...);

            return false;     // destinations are remote
        }

        template <typename ...Ts>
        static inline bool
        call_cb(naming::id_type const& id, naming::address&& addr,
            threads::thread_priority priority,
            hpx::parcelset::write_handler_type const& cb, Ts&&... vs)
        {
            typedef
                typename hpx::traits::extract_action<Action>::type
                action_type;
            action_type act;

            parcelset::put_parcel_cb(cb, id,
                detail::complement_addr<action_type>(addr),
                act, priority, std::forward<Ts>(vs)...);

            return false;     // destinations are remote
        }

        template <typename ...Ts>
        static inline bool
        call_cb(naming::id_type const& id, naming::address&& addr,
            threads::thread_priority priority,
            parcelset::policies::message_handler::write_handler_type && cb, Ts&&... vs)
        {
            typedef
                typename hpx::traits::extract_action<Action>::type
                action_type;
            action_type act;

            parcelset::put_parcel_cb(std::move(cb), id,
                detail::complement_addr<action_type>(addr),
                act, priority, std::forward<Ts>(vs)...);

            return false;     // destinations are remote
        }

        template <typename Continuation, typename Handler, typename ...Ts>
        static inline bool
        call_cont_cb(naming::id_type const& id,
            naming::address&& addr, threads::thread_priority priority,
            Continuation && cont,
            Handler const & cb, Ts&&... vs)
        {
            typedef
                typename hpx::traits::extract_action<Action>::type
                action_type;
            action_type act;

            parcelset::put_parcel_cb(cb, id,
                detail::complement_addr<action_type>(addr),
                std::forward<Continuation>(cont),
                act, priority, std::forward<Ts>(vs)...);

            return false;     // destinations are remote
        }

        template <typename Continuation, typename Handler, typename ...Ts>
        static inline bool
        call_cont_cb(naming::id_type const& id,
            naming::address&& addr, threads::thread_priority priority,
            Continuation && cont,
            Handler &&cb, Ts&&... vs)
        {
            typedef
                typename hpx::traits::extract_action<Action>::type
                action_type;
            action_type act;

            parcelset::put_parcel_cb(std::move(cb), id,
                detail::complement_addr<action_type>(addr),
                std::forward<Continuation>(cont),
                act, priority, std::forward<Ts>(vs)...);

            return false;     // destinations are remote
        }
    };
}}

#endif

