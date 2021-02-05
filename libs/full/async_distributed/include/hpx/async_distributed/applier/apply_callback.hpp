//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/actions_base/traits/action_is_target_valid.hpp>
#include <hpx/actions_base/traits/action_priority.hpp>
#include <hpx/actions_base/traits/extract_action.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/format.hpp>
#include <hpx/traits/is_continuation.hpp>
#include <hpx/traits/is_distribution_policy.hpp>
#include <hpx/type_support/pack.hpp>

#include <hpx/async_distributed/applier/apply.hpp>

#include <cstddef>
#include <system_error>
#include <type_traits>
#include <utility>

namespace hpx {
#if defined(HPX_HAVE_NETWORKING)
    ///////////////////////////////////////////////////////////////////////////
    namespace applier { namespace detail {
        template <typename Action, typename Callback, typename... Ts>
        inline bool apply_r_p_cb(naming::address&& addr,
            naming::id_type const& id, threads::thread_priority priority,
            Callback&& cb, Ts&&... vs)
        {
            // If remote, create a new parcel to be sent to the destination
            // Create a new parcel with the gid, action, and arguments
            return detail::put_parcel_cb<Action>(id, std::move(addr), priority,
                std::forward<Callback>(cb), std::forward<Ts>(vs)...);
        }

        template <typename Action, typename Callback, typename... Ts>
        inline bool apply_r_cb(naming::address&& addr,
            naming::id_type const& gid, Callback&& cb, Ts&&... vs)
        {
            return apply_r_p_cb<Action>(std::move(addr), gid,
                actions::action_priority<Action>(), std::forward<Callback>(cb),
                std::forward<Ts>(vs)...);
        }
    }}    // namespace applier::detail
#endif

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Callback, typename... Ts>
    inline bool apply_p_cb(naming::id_type const& gid,
        threads::thread_priority priority, Callback&& cb, Ts&&... vs)
    {
        return hpx::detail::apply_cb_impl<Action>(
            gid, priority, std::forward<Callback>(cb), std::forward<Ts>(vs)...);
    }

    template <typename Action, typename Callback, typename... Ts>
    inline bool apply_cb(naming::id_type const& gid, Callback&& cb, Ts&&... vs)
    {
        return apply_p_cb<Action>(gid, actions::action_priority<Action>(),
            std::forward<Callback>(cb), std::forward<Ts>(vs)...);
    }

    template <typename Component, typename Signature, typename Derived,
        typename Callback, typename... Ts>
    inline bool apply_cb(
        hpx::actions::basic_action<Component, Signature, Derived> /*act*/,
        naming::id_type const& gid, Callback&& cb, Ts&&... vs)
    {
        return apply_p_cb<Derived>(gid, actions::action_priority<Derived>(),
            std::forward<Callback>(cb), std::forward<Ts>(vs)...);
    }

    template <typename Action, typename DistPolicy, typename Callback,
        typename... Ts>
    inline typename std::enable_if<
        traits::is_distribution_policy<DistPolicy>::value, bool>::type
    apply_p_cb(DistPolicy const& policy, threads::thread_priority priority,
        Callback&& cb, Ts&&... vs)
    {
        return policy.template apply_cb<Action>(
            priority, std::forward<Callback>(cb), std::forward<Ts>(vs)...);
    }

    template <typename Action, typename DistPolicy, typename Callback,
        typename... Ts>
    inline typename std::enable_if<
        traits::is_distribution_policy<DistPolicy>::value, bool>::type
    apply_cb(DistPolicy const& policy, Callback&& cb, Ts&&... vs)
    {
        return apply_p_cb<Action>(policy, actions::action_priority<Action>(),
            std::forward<Callback>(cb), std::forward<Ts>(vs)...);
    }

    template <typename Component, typename Signature, typename Derived,
        typename DistPolicy, typename Callback, typename... Ts>
    inline typename std::enable_if<
        traits::is_distribution_policy<DistPolicy>::value, bool>::type
    apply_cb(hpx::actions::basic_action<Component, Signature, Derived> /*act*/,
        DistPolicy const& policy, Callback&& cb, Ts&&... vs)
    {
        return apply_p_cb<Derived>(policy, actions::action_priority<Derived>(),
            std::forward<Callback>(cb), std::forward<Ts>(vs)...);
    }

#if defined(HPX_HAVE_NETWORKING)
    ///////////////////////////////////////////////////////////////////////////
    namespace applier { namespace detail {
        template <typename Action, typename Continuation, typename Callback,
            typename... Ts>
        inline bool apply_r_p_cb(naming::address&& addr, Continuation&& c,
            naming::id_type const& id, threads::thread_priority priority,
            Callback&& cb, Ts&&... vs)
        {
            // If remote, create a new parcel to be sent to the destination
            // Create a new parcel with the gid, action, and arguments
            return detail::put_parcel_cont_cb<Action>(id, std::move(addr),
                priority, std::forward<Continuation>(c),
                std::forward<Callback>(cb), std::forward<Ts>(vs)...);
        }

        template <typename Action, typename Continuation, typename Callback,
            typename... Ts>
        inline bool apply_r_cb(naming::address&& addr, Continuation&& c,
            naming::id_type const& gid, Callback&& cb, Ts&&... vs)
        {
            return apply_r_p_cb<Action>(std::move(addr),
                std::forward<Continuation>(c), gid,
                actions::action_priority<Action>(), std::forward<Callback>(cb),
                std::forward<Ts>(vs)...);
        }
    }}    // namespace applier::detail
#endif

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Continuation, typename Callback,
        typename... Ts>
    inline bool apply_p_cb(Continuation&& c, naming::address&& addr,
        naming::id_type const& gid, threads::thread_priority priority,
        Callback&& cb, Ts&&... vs)
    {
        if (!traits::action_is_target_valid<Action>::call(gid))
        {
            HPX_THROW_EXCEPTION(bad_parameter, "apply_p_cb",
                hpx::util::format("the target (destination) does not match the "
                                  "action type ({})",
                    hpx::actions::detail::get_action_name<Action>()));
            return false;
        }

        // Determine whether the gid is local or remote
        if (addr.locality_ == hpx::get_locality())
        {
            // apply locally
            bool result = applier::detail::apply_l_p<Action>(
                std::forward<Continuation>(c), gid, std::move(addr), priority,
                std::forward<Ts>(vs)...);

            // invoke callback
#if defined(HPX_HAVE_NETWORKING)
            cb(std::error_code(), parcelset::parcel());
#else
            cb();
#endif
            return result;
        }

#if defined(HPX_HAVE_NETWORKING)
        // apply remotely
        return applier::detail::apply_r_p_cb<Action>(std::move(addr),
            std::forward<Continuation>(c), gid, priority,
            std::forward<Callback>(cb), std::forward<Ts>(vs)...);
#else
        HPX_THROW_EXCEPTION(invalid_status, "hpx::apply_cb",
            "unexpected attempt to send a parcel with networking disabled");
#endif
    }

    template <typename Action, typename Continuation, typename Callback,
        typename... Ts>
    inline bool apply_p_cb(Continuation&& c, naming::id_type const& gid,
        threads::thread_priority priority, Callback&& cb, Ts&&... vs)
    {
        return hpx::detail::apply_cb_impl<Action>(std::forward<Continuation>(c),
            gid, priority, std::forward<Callback>(cb), std::forward<Ts>(vs)...);
    }

    template <typename Action, typename Continuation, typename Callback,
        typename... Ts>
    inline bool apply_cb(
        Continuation&& c, naming::id_type const& gid, Callback&& cb, Ts&&... vs)
    {
        return apply_p_cb<Action>(std::forward<Continuation>(c), gid,
            actions::action_priority<Action>(), std::forward<Callback>(cb),
            std::forward<Ts>(vs)...);
    }

    template <typename Component, typename Continuation, typename Signature,
        typename Derived, typename Callback, typename... Ts>
    inline bool apply_cb(Continuation&& c,
        hpx::actions::basic_action<Component, Signature, Derived> /*act*/,
        naming::id_type const& gid, Callback&& cb, Ts&&... vs)
    {
        return apply_p<Derived>(std::forward<Continuation>(c), gid,
            actions::action_priority<Derived>(), std::forward<Callback>(cb),
            std::forward<Ts>(vs)...);
    }

    template <typename Action, typename Continuation, typename DistPolicy,
        typename Callback, typename... Ts>
    inline
        typename std::enable_if<traits::is_continuation<Continuation>::value &&
                traits::is_distribution_policy<DistPolicy>::value,
            bool>::type
        apply_p_cb(Continuation&& c, DistPolicy const& policy,
            threads::thread_priority priority, Callback&& cb, Ts&&... vs)
    {
        return policy.template apply_cb<Action>(std::forward<Continuation>(c),
            priority, std::forward<Callback>(cb), std::forward<Ts>(vs)...);
    }

    template <typename Action, typename Continuation, typename DistPolicy,
        typename Callback, typename... Ts>
    inline
        typename std::enable_if<traits::is_continuation<Continuation>::value &&
                traits::is_distribution_policy<DistPolicy>::value,
            bool>::type
        apply_cb(Continuation&& c, DistPolicy const& policy, Callback&& cb,
            Ts&&... vs)
    {
        return apply_p_cb<Action>(std::forward<Continuation>(c), policy,
            actions::action_priority<Action>(), std::forward<Callback>(cb),
            std::forward<Ts>(vs)...);
    }

    template <typename Component, typename Continuation, typename Signature,
        typename Derived, typename DistPolicy, typename Callback,
        typename... Ts>
    inline typename std::enable_if<
        traits::is_distribution_policy<DistPolicy>::value, bool>::type
    apply_cb(Continuation&& c,
        hpx::actions::basic_action<Component, Signature, Derived> /*act*/,
        DistPolicy const& policy, Callback&& cb, Ts&&... vs)
    {
        return apply_p<Derived>(std::forward<Continuation>(c), policy,
            actions::action_priority<Derived>(), std::forward<Callback>(cb),
            std::forward<Ts>(vs)...);
    }

#if defined(HPX_HAVE_NETWORKING)
    ///////////////////////////////////////////////////////////////////////////
    namespace applier { namespace detail {
        template <typename Action, typename Callback, typename... Ts>
        inline bool apply_c_p_cb(naming::address&& addr,
            naming::id_type const& contgid, naming::id_type const& gid,
            threads::thread_priority priority, Callback&& cb, Ts&&... vs)
        {
            typedef
                typename hpx::traits::extract_action<Action>::remote_result_type
                    remote_result_type;
            typedef
                typename hpx::traits::extract_action<Action>::local_result_type
                    local_result_type;

            return apply_r_p_cb<Action>(std::move(addr),
                actions::typed_continuation<local_result_type,
                    remote_result_type>(contgid),
                gid, priority, std::forward<Callback>(cb),
                std::forward<Ts>(vs)...);
        }

        template <typename Action, typename Callback, typename... Ts>
        inline bool apply_c_cb(naming::address&& addr,
            naming::id_type const& contgid, naming::id_type const& gid,
            Callback&& cb, Ts&&... vs)
        {
            typedef
                typename hpx::traits::extract_action<Action>::remote_result_type
                    remote_result_type;
            typedef
                typename hpx::traits::extract_action<Action>::local_result_type
                    local_result_type;

            return apply_r_p_cb<Action>(std::move(addr),
                actions::typed_continuation<local_result_type,
                    remote_result_type>(contgid),
                gid, actions::action_priority<Action>(),
                std::forward<Callback>(cb), std::forward<Ts>(vs)...);
        }
    }}    // namespace applier::detail
#endif

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Callback, typename... Ts>
    inline bool apply_c_p_cb(naming::id_type const& contgid,
        naming::id_type const& gid, threads::thread_priority priority,
        Callback&& cb, Ts&&... vs)
    {
        typedef typename hpx::traits::extract_action<Action>::remote_result_type
            remote_result_type;
        typedef typename hpx::traits::extract_action<Action>::local_result_type
            local_result_type;

        return apply_p_cb<Action>(
            actions::typed_continuation<local_result_type, remote_result_type>(
                contgid),
            gid, priority, std::forward<Callback>(cb), std::forward<Ts>(vs)...);
    }

    template <typename Action, typename Callback, typename... Ts>
    inline bool apply_c_cb(naming::id_type const& contgid,
        naming::id_type const& gid, Callback&& cb, Ts&&... vs)
    {
        typedef typename hpx::traits::extract_action<Action>::remote_result_type
            remote_result_type;
        typedef typename hpx::traits::extract_action<Action>::local_result_type
            local_result_type;

        return apply_p_cb<Action>(
            actions::typed_continuation<local_result_type, remote_result_type>(
                contgid),
            gid, actions::action_priority<Action>(), std::forward<Callback>(cb),
            std::forward<Ts>(vs)...);
    }

    template <typename Action, typename Callback, typename... Ts>
    inline bool apply_c_p_cb(naming::id_type const& contgid,
        naming::address&& addr, naming::id_type const& gid,
        threads::thread_priority priority, Callback&& cb, Ts&&... vs)
    {
        typedef typename hpx::traits::extract_action<Action>::remote_result_type
            remote_result_type;
        typedef typename hpx::traits::extract_action<Action>::local_result_type
            local_result_type;

        return apply_p_cb<Action>(
            actions::typed_continuation<local_result_type, remote_result_type>(
                contgid),
            std::move(addr), gid, priority, std::forward<Callback>(cb),
            std::forward<Ts>(vs)...);
    }

    template <typename Action, typename Callback, typename... Ts>
    inline bool apply_c_cb(naming::id_type const& contgid,
        naming::address&& addr, naming::id_type const& gid, Callback&& cb,
        Ts&&... vs)
    {
        typedef typename hpx::traits::extract_action<Action>::remote_result_type
            remote_result_type;
        typedef typename hpx::traits::extract_action<Action>::local_result_type
            local_result_type;

        return apply_p_cb<Action>(
            actions::typed_continuation<local_result_type, remote_result_type>(
                contgid),
            std::move(addr), gid, actions::action_priority<Action>(),
            std::forward<Callback>(cb), std::forward<Ts>(vs)...);
    }

    namespace functional {
        template <typename Action, typename Callback, typename... Ts>
        struct apply_c_p_cb_impl
        {
        public:
            typedef hpx::tuple<Ts...> tuple_type;

            template <typename... Ts_>
            apply_c_p_cb_impl(naming::id_type const& contid,
                naming::address&& addr, naming::id_type const& id,
                threads::thread_priority p, Callback&& cb, Ts_&&... vs)
              : contid_(contid)
              , addr_(std::move(addr))
              , id_(id)
              , p_(p)
              , cb_(std::move(cb))
              , args_(std::forward<Ts_>(vs)...)
            {
            }

            apply_c_p_cb_impl(apply_c_p_cb_impl&& rhs)
              : contid_(std::move(rhs.contid_))
              , addr_(std::move(rhs.addr_))
              , id_(std::move(rhs.id_))
              , p_(std::move(rhs.p_))
              , cb_(std::move(rhs.cb_))
              , args_(std::move(rhs.args_))
            {
            }

            apply_c_p_cb_impl& operator=(apply_c_p_cb_impl&& rhs)
            {
                contid_ = std::move(rhs.contid_);
                addr_ = std::move(rhs.addr_);
                id_ = std::move(rhs.id_);
                p_ = std::move(rhs.p_);
                cb_ = std::move(rhs.cb_);
                args_ = std::move(rhs.args_);
                return *this;
            }

            void operator()()
            {
                apply_action(
                    typename util::make_index_pack<sizeof...(Ts)>::type());
            }

        protected:
            template <std::size_t... Is>
            void apply_action(util::index_pack<Is...>)
            {
                if (addr_)
                {
                    hpx::apply_c_p_cb<Action>(contid_, std::move(addr_), id_,
                        p_, std::move(cb_),
                        hpx::get<Is>(std::forward<tuple_type>(args_))...);
                }
                else
                {
                    hpx::apply_c_p_cb<Action>(contid_, id_, p_, std::move(cb_),
                        hpx::get<Is>(std::forward<tuple_type>(args_))...);
                }
            }

        private:
            naming::id_type contid_;
            naming::address addr_;
            naming::id_type id_;
            threads::thread_priority p_;
            Callback cb_;
            tuple_type args_;
        };

        template <typename Action, typename Callback, typename... Ts>
        apply_c_p_cb_impl<Action, typename std::decay<Callback>::type,
            typename std::decay<Ts>::type...>
        apply_c_p_cb(naming::id_type const& contid, naming::address&& addr,
            naming::id_type const& id, threads::thread_priority p,
            Callback&& cb, Ts&&... vs)
        {
            typedef apply_c_p_cb_impl<Action,
                typename std::decay<Callback>::type,
                typename std::decay<Ts>::type...>
                result_type;

            return result_type(contid, std::move(addr), id, p,
                std::forward<Callback>(cb), std::forward<Ts>(vs)...);
        }
    }    // namespace functional
}    // namespace hpx
