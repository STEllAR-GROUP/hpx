//  Copyright (c) 2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING)

#include <hpx/actions/actions_fwd.hpp>
#include <hpx/actions/transfer_action.hpp>
#include <hpx/actions/transfer_continuation_action.hpp>
#include <hpx/assert.hpp>
#include <hpx/functional/traits/is_action.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/execution.hpp>
#include <hpx/naming/credit_handling.hpp>
#include <hpx/naming/split_gid.hpp>
#include <hpx/naming_base/address.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/runtime/parcelset/parcel.hpp>
#include <hpx/runtime/parcelset/parcelhandler.hpp>
#include <hpx/runtime/parcelset/put_parcel_fwd.hpp>
#include <hpx/runtime/parcelset_fwd.hpp>
#include <hpx/runtime/runtime_fwd.hpp>
#include <hpx/runtime_local/runtime_local.hpp>
#include <hpx/traits/is_continuation.hpp>
#include <hpx/type_support/unused.hpp>

#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

namespace hpx { namespace parcelset
{
    namespace detail
    {
        template <typename Action, typename Continuation, typename... Args>
        std::unique_ptr<actions::base_action> make_parcel_action_impl(
            std::true_type /* Continuation */,
            Continuation&& cont,
            Action, Args&&... args)
        {
            static_assert(traits::is_action<Action>::value,
                "We need an action to construct a parcel");
            return std::unique_ptr<actions::base_action>(
                    new actions::transfer_continuation_action<Action>(
                        std::forward<Continuation>(cont),
                        std::forward<Args>(args)...
                    )
                );
        }

        template <typename Action, typename... Args>
        std::unique_ptr<actions::base_action> make_parcel_action_impl(
            std::false_type /* Continuation */,
            Action, Args&&... args)
        {
            static_assert(traits::is_action<Action>::value,
                "We need an action to construct a parcel");
            return std::unique_ptr<actions::base_action>(
                    new actions::transfer_action<Action>(
                        std::forward<Args>(args)...
                    )
                );
        }

        template <typename Arg0, typename... Args>
        std::unique_ptr<actions::base_action> make_parcel_action(
            Arg0&& arg0, Args&&... args)
        {
            // Is the first argument a continuation?
            using is_continuation = traits::is_continuation<Arg0>;
            return make_parcel_action_impl(
                is_continuation{},
                std::forward<Arg0>(arg0), std::forward<Args>(args)...);
        }

        template <typename... Args>
        parcel create_parcel::call(
            naming::gid_type&& dest, naming::address&& addr,
            Args&&... args)
        {
            return parcel(
                std::move(dest), std::move(addr),
                detail::make_parcel_action(std::forward<Args>(args)...));
        }

        parcel create_parcel::call_with_action(
            naming::gid_type&& dest, naming::address&& addr,
            std::unique_ptr<actions::base_action>&& action)
        {
            return parcel(
                std::move(dest), std::move(addr),
                std::move(action));
        }

        template <typename PutParcel>
        void put_parcel_cont<PutParcel>::operator()(
            hpx::future<naming::gid_type> f)
        {
            pp(detail::create_parcel::call_with_action(
                f.get(), std::move(addr),
                std::move(action)
            ));
        }

        template <typename PutParcel>
        void put_parcel_impl(PutParcel&& pp,
            naming::id_type dest, naming::address&& addr,
            std::unique_ptr<actions::base_action>&& action)
        {
            if (dest.get_management_type() == naming::id_type::unmanaged)
            {
                naming::gid_type gid = dest.get_gid();
                naming::detail::strip_credits_from_gid(gid);
                // NOLINTNEXTLINE(bugprone-use-after-move)
                HPX_ASSERT(gid);

                pp(detail::create_parcel::call_with_action(
                    std::move(gid), std::move(addr),
                    std::move(action)
                ));
            }
            else if (dest.get_management_type() == naming::id_type::managed_move_credit)
            {
                naming::gid_type gid = naming::detail::move_gid(dest.get_gid());
                // NOLINTNEXTLINE(bugprone-use-after-move)
                HPX_ASSERT(gid);

                pp(detail::create_parcel::call_with_action(
                    std::move(gid), std::move(addr),
                    std::move(action)
                ));
            }
            else
            {
                future<naming::gid_type> split_gid =
                    naming::detail::split_gid_if_needed(dest.get_gid());
                if (split_gid.is_ready())
                {
                    pp(detail::create_parcel::call_with_action(
                        split_gid.get(), std::move(addr),
                        std::move(action)
                    ));
                }
                else
                {
                    split_gid.then(
                        hpx::launch::sync,
                        put_parcel_cont<PutParcel>{
                            std::forward<PutParcel>(pp),
                            std::move(dest), std::move(addr),
                            std::move(action)
                        });
                }
            }
        }

        struct HPX_EXPORT put_parcel_handler
        {
            void operator()(parcel&& p) const;
        };

        template <typename Callback>
        struct put_parcel_handler_cb
        {
            typename std::decay<Callback>::type cb_;

            void operator()(parcel&& p)
            {
                hpx::parcelset::put_parcel(std::move(p), std::move(cb_));
            }
        };
    }

    template <typename... Args>
    void put_parcel(
        naming::id_type const& dest, naming::address&& addr, Args&&... args)
    {
        detail::put_parcel_impl(
            detail::put_parcel_handler(),
            dest, std::move(addr),
            detail::make_parcel_action(std::forward<Args>(args)...));
    }

    template <typename Callback, typename... Args>
    void put_parcel_cb(Callback&& cb,
        naming::id_type const& dest, naming::address&& addr, Args&&... args)
    {
        detail::put_parcel_impl(
            detail::put_parcel_handler_cb<Callback>{std::forward<Callback>(cb)},
            dest, std::move(addr),
            detail::make_parcel_action(std::forward<Args>(args)...));
    }
}}

#endif
