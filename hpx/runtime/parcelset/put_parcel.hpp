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
#include <hpx/runtime/serialization/detail/preprocess.hpp>
#include <hpx/runtime/serialization/output_archive.hpp>
#include <hpx/traits/is_action.hpp>
#include <hpx/traits/is_continuation.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/protect.hpp>
#include <hpx/util/detail/pack.hpp>

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
                    )
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
                    )
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
                    )
                );
            }
        };

        template <typename PutParcel>
        struct parcel_await
          : std::enable_shared_from_this<parcel_await<PutParcel>>
        {
            template <typename PutParcel_, typename... Args>
            parcel_await(PutParcel_&& pp,
                naming::address&& addr, Args&&... args)
              : put_parcel_(std::forward<PutParcel_>(pp)),
                p_(
                    create_parcel::call(
                        // is the first parameter of args a continuation?
                        std::integral_constant<bool,
                            traits::is_continuation<
                                typename util::detail::at_index<0, Args...>::type
                            >::value &&
                            // we need to treat unique pointers to continuations
                            // differently
                            !std::is_same<
                                std::unique_ptr<actions::continuation>,
                                typename util::detail::at_index<0, Args...>::type
                            >::value
                        >(),
                        // is the second parameter of args a action?
                        traits::is_action<
                            typename util::detail::at_index<1, Args...>::type
                        >(),
                        naming::gid_type(), std::move(addr),
                        std::forward<Args>(args)...
                    )
                ),
                size_(0)
            {
            }

            void apply(naming::gid_type&& gid)
            {
                p_.set_destination_id(std::move(gid));
                (*this)();
            }

            void operator()()
            {
                preprocess_.reset();
                typedef hpx::serialization::output_archive archive_type;
                std::shared_ptr<archive_type> archive(new archive_type(preprocess_));
                (*archive) << p_;

                // We are doing a fixed point iteration until we are sure that the
                // serialization process requires nothing more to wait on ...
                // Things where we need waiting:
                //  - (shared_)future<id_type>: when the future wasn't ready yet, we
                //      need to do another await round for the id splitting
                //  - id_type: we need to await, if and only if, the credit of the
                //      needs to split.
                if(preprocess_.has_futures())
                {
                    auto this_ = this->shared_from_this();
                    preprocess_([this_, archive](){ (*this_)(); });
                    return;
                }
                HPX_ASSERT(preprocess_.size() == archive->bytes_written());
                p_.size() = preprocess_.size();
                p_.set_splitted_gids(std::move(preprocess_.splitted_gids_));
                put_parcel_(std::move(p_));
            }

            typename hpx::util::decay<PutParcel>::type put_parcel_;
            parcel p_;
            hpx::serialization::detail::preprocess preprocess_;
            std::size_t size_;
        };

        template <typename PutParcel, typename... Args>
        void put_parcel_impl(PutParcel&& pp,
            naming::id_type dest, naming::address&& addr, Args&&... args)
        {
            typedef parcel_await<PutParcel> parcel_awaiter_type;
            std::shared_ptr<parcel_awaiter_type> parcel_awaiter(
                new parcel_awaiter_type(
                    std::forward<PutParcel>(pp), std::move(addr),
                    std::forward<Args>(args)...));

            if (dest.get_management_type() == naming::id_type::unmanaged)
            {
                naming::gid_type gid = dest.get_gid();
                naming::detail::strip_credits_from_gid(gid);
                HPX_ASSERT(gid);

                parcel_awaiter->apply(std::move(gid));
            }
            else if (dest.get_management_type() == naming::id_type::managed_move_credit)
            {
                naming::gid_type gid = naming::detail::move_gid(dest.get_gid());
                HPX_ASSERT(gid);
                parcel_awaiter->apply(std::move(gid));
            }
            else
            {
                future<naming::gid_type> splitted_gid =
                    naming::detail::split_gid_if_needed(dest.get_gid());
                if (splitted_gid.is_ready())
                {
                    parcel_awaiter->apply(splitted_gid.get());
                }
                else
                {
                    splitted_gid.then(
                        [dest, parcel_awaiter]
                        (hpx::future<naming::gid_type> f)
                        {
                            parcel_awaiter->apply(f.get());
                        }
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
