//  Copyright (c) 2015-2021 Hartmut Kaiser
//  Copyright (c) 2015-2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/assert.hpp>
#include <hpx/modules/datastructures.hpp>
#include <hpx/modules/naming_base.hpp>
#include <hpx/modules/thread_support.hpp>
#include <hpx/naming/credit_handling.hpp>
#include <hpx/serialization/detail/extra_archive_data.hpp>
#include <hpx/synchronization/spinlock.hpp>

#include <cstddef>
#include <map>
#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace serialization { namespace detail {

    ///////////////////////////////////////////////////////////////////////////
    // This class allows to handle credit splitting for gid_types during
    // serialization.
    class preprocess_gid_types
    {
        using mutex_type = hpx::lcos::local::spinlock;

    public:
        using split_gids_map =
            std::map<naming::gid_type const*, naming::gid_type>;

        preprocess_gid_types() = default;

        ~preprocess_gid_types()
        {
            std::unique_lock<mutex_type> l(mtx_);
            reset_locked(l);
        }

        preprocess_gid_types(preprocess_gid_types&& rhs) noexcept = delete;
        preprocess_gid_types& operator=(
            preprocess_gid_types&& rhs) noexcept = delete;

        template <typename Lock>
        void reset_locked(Lock& l)
        {
            HPX_ASSERT_OWNS_LOCK(l);

            // If there are gids left in the map then several parcels got
            // preprocessed separately, but serialized into the same archive.
            // In this case we must explicitly return the credits for those
            // gids before clearing the map.
            if (!split_gids_.empty())
            {
                std::vector<naming::gid_type> gids;
                gids.reserve(split_gids_.size());

                for (auto&& e : split_gids_)
                {
                    gids.push_back(e.second);
                }

                split_gids_.clear();

                // now return credits to AGAS
                util::unlock_guard<Lock> ul(l);

                for (auto const& gid : gids)
                {
                    naming::decrement_refcnt(gid);
                }
            }
        }

        void add_gid(
            naming::gid_type const& gid, naming::gid_type const& split_gid)
        {
            std::lock_guard<mutex_type> l(mtx_);

            HPX_ASSERT(split_gids_.find(&gid) == split_gids_.end());
            split_gids_.insert(split_gids_map::value_type(&gid, split_gid));
        }

        bool has_gid(naming::gid_type const& gid) const noexcept
        {
            std::lock_guard<mutex_type> l(mtx_);
            return split_gids_.find(&gid) != split_gids_.end();
        }

        naming::gid_type get_new_gid(naming::gid_type const& gid)
        {
            std::lock_guard<mutex_type> l(mtx_);
            auto it = split_gids_.find(&gid);

            HPX_ASSERT(it != split_gids_.end());
            HPX_ASSERT(it->second != naming::invalid_gid);

            naming::gid_type new_gid = it->second;
            split_gids_.erase(it);
            return new_gid;
        }

        split_gids_map move_split_gids() noexcept
        {
            std::lock_guard<mutex_type> l(mtx_);
            return std::move(split_gids_);
        }
        void set_split_gids(split_gids_map&& gids)
        {
            std::unique_lock<mutex_type> l(mtx_);

            reset_locked(l);
            split_gids_ = std::move(gids);
        }

    private:
        mutable mutex_type mtx_;
        split_gids_map split_gids_;
    };

    // This is explicitly instantiated to ensure that the id is stable across
    // shared libraries.
    template <>
    struct extra_archive_data_helper<preprocess_gid_types>
    {
        HPX_EXPORT static extra_archive_data_id_type id() noexcept;
        static constexpr void reset(preprocess_gid_types*) noexcept {}
    };
}}}    // namespace hpx::serialization::detail
