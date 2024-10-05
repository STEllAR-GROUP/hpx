//  Copyright (c) 2014 Bibek Ghimire
//  Copyright (c) 2014-2024 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/actions_base/traits/is_distribution_policy.hpp>
#include <hpx/assert.hpp>
#include <hpx/components_base/agas_interface.hpp>
#include <hpx/distribution_policies/default_distribution_policy.hpp>
#include <hpx/serialization/serialize.hpp>
#include <hpx/serialization/shared_ptr.hpp>
#include <hpx/serialization/vector.hpp>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx {

    ///////////////////////////////////////////////////////////////////////////
    // This class specifies the block chunking policy parameters to use for the
    // partitioning of the data in a hpx::partitioned_vector
    struct container_distribution_policy
      : components::default_distribution_policy
    {
        container_distribution_policy() = default;

        container_distribution_policy operator()(std::size_t num_partitions,
            traits::create_mode mode = traits::create_mode::resize) const
        {
            return {num_partitions, get_localities(), mode};
        }

        container_distribution_policy operator()(traits::create_mode mode) const
        {
            return container_distribution_policy(mode);
        }

        container_distribution_policy operator()(hpx::id_type const& locality,
            traits::create_mode mode = traits::create_mode::resize) const
        {
            return {locality, mode};
        }

        container_distribution_policy operator()(
            std::vector<id_type> const& localities,
            traits::create_mode mode = traits::create_mode::resize) const
        {
            if (num_partitions_ != static_cast<std::size_t>(-1))
            {
                return {num_partitions_, localities, mode};
            }
            return {localities.size(), localities, mode};
        }

        container_distribution_policy operator()(
            std::vector<id_type>&& localities,
            traits::create_mode mode = traits::create_mode::resize) const
        {
            if (num_partitions_ != static_cast<std::size_t>(-1))
            {
                return {num_partitions_, HPX_MOVE(localities), mode};
            }
            return {localities.size(), HPX_MOVE(localities), mode};
        }

        container_distribution_policy operator()(std::size_t num_partitions,
            std::vector<id_type> const& localities,
            traits::create_mode mode = traits::create_mode::resize) const
        {
            return {num_partitions, localities, mode};
        }

        container_distribution_policy operator()(std::size_t num_partitions,
            std::vector<id_type>&& localities,
            traits::create_mode mode = traits::create_mode::resize) const
        {
            return {num_partitions, HPX_MOVE(localities), mode};
        }

        ///////////////////////////////////////////////////////////////////////
        [[nodiscard]] std::size_t get_num_partitions() const
        {
            if (localities_)
            {
                std::size_t const num_parts =
                    (num_partitions_ == static_cast<std::size_t>(-1)) ?
                    localities_->size() :
                    num_partitions_;
                return (std::max)(num_parts, static_cast<std::size_t>(1));
            }
            return static_cast<std::size_t>(1);
        }

        [[nodiscard]] std::vector<hpx::id_type> get_localities() const
        {
            if (!localities_)
            {
                // use this locality, if this object was default constructed
                return std::vector<id_type>(1,
                    naming::get_id_from_locality_id(agas::get_locality_id()));
            }

            HPX_ASSERT(!localities_->empty());
            return *localities_;
        }

        [[nodiscard]] constexpr traits::create_mode get_create_mode() const
        {
            return mode_;
        }

    private:
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive& ar, const unsigned int /* version */)
        {
            // clang-format off
            ar & localities_ & num_partitions_ & mode_;
            // clang-format on
        }

        container_distribution_policy(std::size_t num_partitions,
            std::vector<id_type> const& localities, traits::create_mode mode)
          : components::default_distribution_policy(localities)
          , num_partitions_(num_partitions)
          , mode_(mode)
        {
        }

        container_distribution_policy(std::size_t num_partitions,
            std::vector<id_type>&& localities, traits::create_mode mode)
          : components::default_distribution_policy(HPX_MOVE(localities))
          , num_partitions_(num_partitions)
          , mode_(mode)
        {
        }

        container_distribution_policy(
            hpx::id_type const& locality, traits::create_mode mode)
          : components::default_distribution_policy(locality)
          , mode_(mode)
        {
        }

        explicit container_distribution_policy(traits::create_mode mode)
          : mode_(mode)
        {
        }

        // number of chunks to create
        std::size_t num_partitions_ = static_cast<std::size_t>(-1);
        traits::create_mode mode_ = traits::create_mode::resize;
    };

    static container_distribution_policy const container_layout{};

    ///////////////////////////////////////////////////////////////////////////
    namespace traits {

        template <>
        struct is_distribution_policy<container_distribution_policy>
          : std::true_type
        {
        };

        template <>
        struct num_container_partitions<container_distribution_policy>
        {
            static std::size_t call(container_distribution_policy const& policy)
            {
                return policy.get_num_partitions();
            }
        };

        template <>
        struct allocation_mode<container_distribution_policy>
        {
            static create_mode call(container_distribution_policy const& policy)
            {
                return policy.get_create_mode();
            }
        };
    }    // namespace traits
}    // namespace hpx
