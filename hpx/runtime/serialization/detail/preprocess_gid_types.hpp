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

            preprocess_gid_types& operator=(preprocess_gid_types&& rhs) noexcept
            {
                split_gids_ = std::move(rhs.split_gids_);
                return *this;
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
                std::lock_guard<mutex_type> l(mtx_);
                if (split_gids_.empty())
                    return naming::gid_type();

                auto it = split_gids_.find(&gid);
                HPX_ASSERT(it != split_gids_.end());
                HPX_ASSERT(it->second != naming::invalid_gid);

                naming::gid_type new_gid = it->second;
#if defined(HPX_DEBUG)
                it->second = naming::invalid_gid;
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

            // We add this solely for the purpose of making moveonly_any compile.
            // Comparing instances of this type does not make any sense,
            // conceptually.
            friend bool operator==(
                preprocess_gid_types const&, preprocess_gid_types const&)
            {
                HPX_ASSERT(false);    // shouldn't ever be called
                return false;
            }

        private:
            mutex_type mtx_;
            split_gids_map split_gids_;
        };
    }    // namespace detail

    ////////////////////////////////////////////////////////////////////////////
    // serialization support for gid_type (handles credit-splitting)
    constexpr std::size_t extra_output_split_credits = 1;
    constexpr std::size_t encode_parcel_extra_output_data_size = 2;

    template <>
    inline util::moveonly_any_nonser
    init_extra_output_data_item<extra_output_split_credits>()
    {
        return util::moveonly_any_nonser{detail::preprocess_gid_types{}};
    }

    template <>
    inline void reset_extra_output_data_item<extra_output_split_credits>(
        extra_archive_data_type& data)
    {
        // nothing to do
    }
}}    // namespace hpx::serialization

#endif
