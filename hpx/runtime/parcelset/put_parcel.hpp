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
#include <hpx/runtime/naming/split_gid.hpp>
#include <hpx/runtime/parcelset/parcel.hpp>
#include <hpx/runtime/parcelset/parcelhandler.hpp>
#include <hpx/traits/is_action.hpp>
#include <hpx/traits/is_continuation.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/detail/pack.hpp>

#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

namespace hpx { namespace parcelset {
    namespace detail {
        struct create_parcel
        {
            template <typename Action, typename Continuation, typename... Args>
            static parcel call(
                std::true_type /* Continuation */,
                std::true_type /* Action */,
                naming::gid_type&& dest,
                naming::address&& addr,
                Continuation&& cont,
                Action,
                Args&&... args)
            {
                return parcel(
                    std::move(dest),
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
                    ),
                    util::detail::all_of<
                        traits::is_bitwise_serializable<Args>...>::value
                );
            }
            template <typename Action, typename... Args>
            static parcel call(
                std::false_type /* Continuation */,
                std::true_type /* Action */,
                naming::gid_type&& dest,
                naming::address&& addr,
                std::unique_ptr<actions::continuation> cont,
                Action,
                Args&&... args)
            {
                static_assert(traits::is_action<Action>::value,
                    "We need an action to construct a parcel");
                return parcel(
                    std::move(dest),
                    std::move(addr),
                    std::move(cont),
                    std::unique_ptr<actions::base_action>(
                        new actions::transfer_action<Action>(
                            std::forward<Args>(args)...
                        )
                    ),
                    util::detail::all_of<
                        traits::is_bitwise_serializable<Args>...>::value
                );
            }

            template <typename Action, typename... Args>
            static parcel call(
                std::false_type /* Continuation */,
                std::false_type /* Action */,
                naming::gid_type&& dest,
                naming::address&& addr,
                Action,
                Args&&... args)
            {
                static_assert(traits::is_action<Action>::value,
                    "We need an action to construct a parcel");
                return parcel(
                    std::move(dest),
                    std::move(addr),
                    std::unique_ptr<actions::continuation>(),
                    std::unique_ptr<actions::base_action>(
                        new actions::transfer_action<Action>(
                            std::forward<Args>(args)...
                        )
                    ),
                    util::detail::all_of<
                        traits::is_bitwise_serializable<Args>...>::value
                );
            }
        };


        template <typename PutParcel, typename... Args>
        void put_parcel_impl(PutParcel&& pp,
            naming::id_type dest, naming::address&& addr, Args&&... args)
        {
            typedef
                typename util::detail::at_index<0, Args...>::type
                arg0_type;
            typedef
                typename util::detail::at_index<1, Args...>::type
                arg1_type;

            // Is the first argument a continuation?
            std::integral_constant<bool,
                traits::is_continuation<arg0_type>::value &&
                // We need to tread unique pointers to continuations
                // differently
                !std::is_same<
                    std::unique_ptr<actions::continuation>, arg0_type
                >::value
            > is_continuation;

            // Is the second paramter a action?
            traits::is_action<arg1_type> is_action;

            if (dest.get_management_type() == naming::id_type::unmanaged)
            {
                naming::gid_type gid = dest.get_gid();
                naming::detail::strip_credits_from_gid(gid);
                HPX_ASSERT(gid);

                pp(detail::create_parcel::call(
                    is_continuation, is_action,
                    std::move(gid), std::move(addr),
                    std::forward<Args>(args)...
                ));
            }
            else if (dest.get_management_type() == naming::id_type::managed_move_credit)
            {
                naming::gid_type gid = naming::detail::move_gid(dest.get_gid());
                HPX_ASSERT(gid);

                pp(detail::create_parcel::call(
                    is_continuation, is_action,
                    std::move(gid), std::move(addr),
                    std::forward<Args>(args)...
                ));
            }
            else
            {
                future<naming::gid_type> splitted_gid =
                    naming::detail::split_gid_if_needed(dest.get_gid());
                if (splitted_gid.is_ready())
                {
                    pp(detail::create_parcel::call(
                        is_continuation, is_action,
                        splitted_gid.get(), std::move(addr),
                        std::forward<Args>(args)...
                    ));
                }
                else
                {
                    splitted_gid.then(
                        hpx::util::bind(
                            hpx::util::one_shot(
                                [is_continuation, is_action, dest]
                                (hpx::future<naming::gid_type> f,
                                 typename util::decay<PutParcel>::type&& pp_,
                                 naming::address&& addr_,
                                 typename util::decay<Args>::type&&... args_)
                                {
                                    pp_(detail::create_parcel::call(
                                        is_continuation, is_action,
                                        f.get(), std::move(addr_),
                                        std::move(args_)...
                                    ));
                                }
                            ),
                            hpx::util::placeholders::_1,
                            std::forward<PutParcel>(pp), std::move(addr),
                            std::forward<Args>(args)...
                        )
                    );
                }
            }
        }

        struct put_parcel_handler
        {
            void operator()(parcel&& p)
            {
                parcelset::parcelhandler& ph =
                    hpx::get_runtime().get_parcel_handler();
                ph.put_parcel(std::move(p));
            }
        };

        template <typename Callback>
        struct put_parcel_handler_cb
        {
            template <typename Callback_>
            put_parcel_handler_cb(Callback_ cb)
              : cb_(std::forward<Callback_>(cb))
            {
            }

            void operator()(parcel&& p)
            {
                parcelset::parcelhandler& ph =
                    hpx::get_runtime().get_parcel_handler();
                ph.put_parcel(std::move(p), std::move(cb_));
            }

            typename hpx::util::decay<Callback>::type cb_;
        };
    }

    template <typename... Args>
    void put_parcel(
        naming::id_type const& dest, naming::address&& addr, Args&&... args)
    {
        detail::put_parcel_impl(detail::put_parcel_handler(),
            dest, std::move(addr), std::forward<Args>(args)...);
    }

    template <typename Callback, typename... Args>
    void put_parcel_cb(Callback&& cb,
        naming::id_type const& dest, naming::address&& addr, Args&&... args)
    {
        detail::put_parcel_impl(
            detail::put_parcel_handler_cb<Callback>(std::forward<Callback>(cb)),
            dest, std::move(addr), std::forward<Args>(args)...);
    }
}}

#endif
