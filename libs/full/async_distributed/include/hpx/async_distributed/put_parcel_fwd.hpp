//  Copyright (c) 2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING)
#include <hpx/actions_base/actions_base_fwd.hpp>
#include <hpx/futures/future_fwd.hpp>
#include <hpx/naming_base/address.hpp>
#include <hpx/naming_base/gid_type.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/parcelset/parcel.hpp>
#include <hpx/parcelset/parcelset_fwd.hpp>

#include <memory>
#include <type_traits>

namespace hpx::parcelset {

    namespace detail {

        struct create_parcel
        {
            template <typename... Args>
            static inline parcelset::parcel call(naming::gid_type&& dest,
                naming::address&& addr, Args&&... args);

            static inline parcelset::parcel call_with_action(
                naming::gid_type&& dest, naming::address&& addr,
                std::unique_ptr<actions::base_action>&& action);
        };

        template <typename PutParcel>
        struct put_parcel_cont
        {
            std::decay_t<PutParcel> pp;
            hpx::id_type dest;
            naming::address addr;
            std::unique_ptr<actions::base_action> action;

            void operator()(hpx::future<naming::gid_type> f);
        };

        template <typename PutParcel>
        void put_parcel_impl(PutParcel&& pp, hpx::id_type dest,
            naming::address&& addr,
            std::unique_ptr<actions::base_action>&& action);
    }    // namespace detail

    template <typename... Args>
    void put_parcel(
        hpx::id_type const& dest, naming::address&& addr, Args&&... args);

    template <typename Callback, typename... Args>
    void put_parcel_cb(Callback&& cb, hpx::id_type const& dest,
        naming::address&& addr, Args&&... args);
}    // namespace hpx::parcelset

#endif
