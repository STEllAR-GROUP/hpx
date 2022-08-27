//  Copyright (c) 2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING)
#include <hpx/assert.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/execution.hpp>
#include <hpx/modules/type_support.hpp>

#include <hpx/actions/actions_fwd.hpp>
#include <hpx/actions/transfer_action.hpp>
#include <hpx/actions_base/traits/is_continuation.hpp>
#include <hpx/async_distributed/put_parcel_fwd.hpp>
#include <hpx/async_distributed/transfer_continuation_action.hpp>
#include <hpx/functional/traits/is_action.hpp>
#include <hpx/naming/credit_handling.hpp>
#include <hpx/naming/split_gid.hpp>
#include <hpx/naming_base/address.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/parcelset/parcel.hpp>
#include <hpx/parcelset/parcelhandler.hpp>
#include <hpx/parcelset/parcelset_fwd.hpp>
#include <hpx/runtime_local/runtime_local.hpp>

#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

namespace hpx::parcelset {

    namespace detail {

        template <typename Action, typename Continuation, typename... Args>
        std::unique_ptr<actions::base_action> make_parcel_action_impl(
            std::true_type, Continuation&& cont, Action, Args&&... args)
        {
            static_assert(traits::is_action_v<Action>,
                "We need an action to construct a parcel");

            return std::unique_ptr<actions::base_action>(
                new actions::transfer_continuation_action<Action>(
                    std::forward<Continuation>(cont),
                    std::forward<Args>(args)...));
        }

        template <typename Action, typename... Args>
        std::unique_ptr<actions::base_action> make_parcel_action_impl(
            std::false_type, Action, Args&&... args)
        {
            static_assert(traits::is_action_v<Action>,
                "We need an action to construct a parcel");

            return std::unique_ptr<actions::base_action>(
                new actions::transfer_action<Action>(
                    std::forward<Args>(args)...));
        }

        template <typename Arg0, typename... Args>
        std::unique_ptr<actions::base_action> make_parcel_action(
            Arg0&& arg0, Args&&... args)
        {
            // Is the first argument a continuation?
            using is_continuation = traits::is_continuation<Arg0>;
            return make_parcel_action_impl(is_continuation{},
                std::forward<Arg0>(arg0), std::forward<Args>(args)...);
        }

        template <typename... Args>
        parcelset::parcel create_parcel::call(
            naming::gid_type&& dest, naming::address&& addr, Args&&... args)
        {
            return parcelset::parcel(
                new detail::parcel(HPX_MOVE(dest), HPX_MOVE(addr),
                    make_parcel_action(HPX_FORWARD(Args, args)...)));
        }

        parcelset::parcel create_parcel::call_with_action(
            naming::gid_type&& dest, naming::address&& addr,
            std::unique_ptr<actions::base_action>&& action)
        {
            return parcelset::parcel(new detail::parcel(
                HPX_MOVE(dest), HPX_MOVE(addr), HPX_MOVE(action)));
        }

        template <typename PutParcel>
        void put_parcel_cont<PutParcel>::operator()(
            hpx::future<naming::gid_type> f)
        {
            pp(detail::create_parcel::call_with_action(
                f.get(), HPX_MOVE(addr), HPX_MOVE(action)));
        }

        template <typename PutParcel>
        void put_parcel_impl(PutParcel&& pp, hpx::id_type dest,
            naming::address&& addr,
            std::unique_ptr<actions::base_action>&& action)
        {
            if (dest.get_management_type() ==
                hpx::id_type::management_type::unmanaged)
            {
                naming::gid_type gid = dest.get_gid();
                naming::detail::strip_credits_from_gid(gid);

                // NOLINTNEXTLINE(bugprone-use-after-move)
                HPX_ASSERT(gid);

                pp(detail::create_parcel::call_with_action(
                    HPX_MOVE(gid), HPX_MOVE(addr), HPX_MOVE(action)));
            }
            else if (dest.get_management_type() ==
                hpx::id_type::management_type::managed_move_credit)
            {
                naming::gid_type gid = naming::detail::move_gid(dest.get_gid());

                // NOLINTNEXTLINE(bugprone-use-after-move)
                HPX_ASSERT(gid);

                pp(detail::create_parcel::call_with_action(
                    HPX_MOVE(gid), HPX_MOVE(addr), HPX_MOVE(action)));
            }
            else
            {
                future<naming::gid_type> split_gid =
                    naming::detail::split_gid_if_needed(dest.get_gid());

                if (split_gid.is_ready())
                {
                    pp(detail::create_parcel::call_with_action(
                        split_gid.get(), HPX_MOVE(addr), HPX_MOVE(action)));
                }
                else
                {
                    split_gid.then(hpx::launch::sync,
                        put_parcel_cont<PutParcel>{HPX_FORWARD(PutParcel, pp),
                            HPX_MOVE(dest), HPX_MOVE(addr), HPX_MOVE(action)});
                }
            }
        }

        struct put_parcel_handler
        {
            void operator()(parcelset::parcel&& p) const
            {
                hpx::parcelset::put_parcel(HPX_MOVE(p));
            }
        };

        template <typename Callback>
        struct put_parcel_handler_cb
        {
            std::decay_t<Callback> cb_;

            void operator()(parcelset::parcel&& p)
            {
                hpx::parcelset::put_parcel(HPX_MOVE(p), HPX_MOVE(cb_));
            }
        };
    }    // namespace detail

    template <typename... Args>
    void put_parcel(
        hpx::id_type const& dest, naming::address&& addr, Args&&... args)
    {
        detail::put_parcel_impl(detail::put_parcel_handler(), dest,
            HPX_MOVE(addr),
            detail::make_parcel_action(HPX_FORWARD(Args, args)...));
    }

    template <typename Callback, typename... Args>
    void put_parcel_cb(Callback&& cb, hpx::id_type const& dest,
        naming::address&& addr, Args&&... args)
    {
        detail::put_parcel_impl(
            detail::put_parcel_handler_cb<Callback>{HPX_FORWARD(Callback, cb)},
            dest, HPX_MOVE(addr),
            detail::make_parcel_action(HPX_FORWARD(Args, args)...));
    }
}    // namespace hpx::parcelset

#endif
