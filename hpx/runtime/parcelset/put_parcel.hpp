//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_PUT_PARCEL_HPP
#define HPX_PARCELSET_PUT_PARCEL_HPP

#include <hpx/runtime.hpp>
#include <hpx/runtime/actions_fwd.hpp>
#include <hpx/runtime/parcelset_fwd.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/parcelset/parcel.hpp>
#include <hpx/runtime/parcelset/parcelhandler.hpp>
#include <hpx/traits/is_action.hpp>
#include <hpx/traits/is_continuation.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/detail/pack.hpp>

#include <memory>
#include <utility>

namespace hpx { namespace parcelset {
    namespace detail {
        struct create_parcel
        {
            template <typename Action, typename Continuation, typename... Args>
            static parcel call(
                std::true_type /* Continuation */,
                std::true_type /* Action */,
                naming::id_type const& dest,
                naming::address&& addr,
                Continuation&& cont,
                Action,
                Args&&... args)
            {
                return parcel(
                    dest,
                    std::move(addr),
                    std::unique_ptr<actions::continuation>(
                        new typename util::decay<Continuation>::type(
                            std::forward<Continuation>(cont)
                        )
                    ),
                    std::unique_ptr<actions::base_action>(
                        new actions::transfer_action<Action>(
                            std::forward<Args>(args)...
                        )
                    )
                );
            }
            template <typename Action, typename... Args>
            static parcel call(
                std::false_type /* Continuation */,
                std::true_type /* Action */,
                naming::id_type const& dest,
                naming::address&& addr,
                std::unique_ptr<actions::continuation> cont,
                Action,
                Args&&... args)
            {
                static_assert(traits::is_action<Action>::value,
                    "We need an action to construct a parcel");
                return parcel(
                    dest,
                    std::move(addr),
                    std::move(cont),
                    std::unique_ptr<actions::base_action>(
                        new actions::transfer_action<Action>(
                            std::forward<Args>(args)...
                        )
                    )
                );
            }

            template <typename Action, typename... Args>
            static parcel call(
                std::false_type /* Continuation */,
                std::false_type /* Action */,
                naming::id_type const& dest,
                naming::address&& addr,
                Action,
                Args&&... args)
            {
                static_assert(traits::is_action<Action>::value,
                    "We need an action to construct a parcel");
                return parcel(
                    dest,
                    std::move(addr),
                    std::unique_ptr<actions::continuation>(),
                    std::unique_ptr<actions::base_action>(
                        new actions::transfer_action<Action>(
                            std::forward<Args>(args)...
                        )
                    )
                );
            }
        };
    }

    template <typename... Args>
    void put_parcel(
        naming::id_type const& dest, naming::address&& addr, Args&&... args)
    {
        typedef
            typename util::detail::at_index<0, Args...>::type
            arg0_type;
        std::integral_constant<bool,
            traits::is_continuation<
                arg0_type
            >::value &&
            !std::is_same<
                std::unique_ptr<actions::continuation>,
                arg0_type
            >::value
        >
        is_continuation;

        traits::is_action<
            typename util::detail::at_index<1, Args...>::type
        >
        is_action;

        parcelset::parcelhandler& ph =
            hpx::get_runtime().get_parcel_handler();
        ph.put_parcel(
            detail::create_parcel::call(
                is_continuation, is_action,
                dest, std::move(addr), std::forward<Args>(args)...
            )
        );
    }

    template <typename Callback, typename... Args>
    void put_parcel_cb(Callback&& cb,
        naming::id_type const& dest, naming::address&& addr, Args&&... args)
    {
        typedef
            typename util::detail::at_index<0, Args...>::type
            arg0_type;
        std::integral_constant<bool,
            traits::is_continuation<
                arg0_type
            >::value &&
            !std::is_same<
                std::unique_ptr<actions::continuation>,
                arg0_type
            >::value
        >
        is_continuation;

        traits::is_action<
            typename util::detail::at_index<1, Args...>::type
        >
        is_action;

        parcelset::parcelhandler& ph =
            hpx::get_runtime().get_parcel_handler();
        ph.put_parcel(
            detail::create_parcel::call(
                is_continuation, is_action,
                dest, std::move(addr), std::forward<Args>(args)...
            ),
            std::forward<Callback>(cb)
        );
    }
}}

#endif
