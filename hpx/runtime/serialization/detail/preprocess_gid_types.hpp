//  Copyright (c) 2015-2019 Hartmut Kaiser
//  Copyright (c) 2015-2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_SERIALIZATION_DETAIL_PREPROCESS_GID_TYPES_HPP)
#define HPX_SERIALIZATION_DETAIL_PREPROCESS_GID_TYPES_HPP

#include <hpx/assertion.hpp>
#include <hpx/datastructures.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/naming_fwd.hpp>

#include <cstddef>
#include <map>
#include <mutex>
#include <type_traits>
#include <utility>

////////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace serialization {

    namespace detail {

        // This class allows to handle credit splitting fr gid_types during
        // serialization.
        class preprocess_gid_types
        {
            using mutex_type = hpx::lcos::local::spinlock;

        public:
            using split_gids_map =
                std::map<naming::gid_type const*, naming::gid_type>;

            preprocess_gid_types() = default;

            preprocess_gid_types(preprocess_gid_types&& rhs) noexcept
              : mtx_()
              , split_gids_(std::move(rhs.split_gids_))
            {}

            ~preprocess_gid_types()
            {
                HPX_ASSERT(split_gids_.empty());
            }

            preprocess_gid_types& operator=(preprocess_gid_types&& rhs) noexcept
            {
                split_gids_ = std::move(rhs.split_gids_);
                return *this;
            }

            void reset()
            {
                split_gids_.clear();
            }

            void add_gid(
                naming::gid_type const& gid, naming::gid_type const& split_gid)
            {
                std::lock_guard<mutex_type> l(mtx_);
                HPX_ASSERT(split_gids_[&gid] == naming::invalid_gid);
                split_gids_[&gid] = split_gid;
            }

            bool has_gid(naming::gid_type const& gid)
            {
                std::lock_guard<mutex_type> l(mtx_);
                return split_gids_.find(&gid) != split_gids_.end();
            }

            naming::gid_type get_new_gid(naming::gid_type const& gid)
            {
                auto it = split_gids_.find(&gid);
                HPX_ASSERT(it != split_gids_.end());
                HPX_ASSERT(it->second != naming::invalid_gid);

                naming::gid_type new_gid = it->second;
#if defined(HPX_DEBUG)
                split_gids_.erase(it);
#endif
                return new_gid;
            }

            split_gids_map move_split_gids()
            {
                return std::move(split_gids_);
            }
            void set_split_gids(split_gids_map&& gids)
            {
                split_gids_ = std::move(gids);
            }

        private:
            mutex_type mtx_;
            split_gids_map split_gids_;
        };
    }    // namespace detail
}}    // namespace hpx::serialization

#endif
