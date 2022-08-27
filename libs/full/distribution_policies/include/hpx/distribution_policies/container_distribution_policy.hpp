//  Copyright (c) 2014 Bibek Ghimire
//  Copyright (c) 2014-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/actions_base/traits/is_distribution_policy.hpp>
#include <hpx/components_base/agas_interface.hpp>
#include <hpx/runtime_components/default_distribution_policy.hpp>

#include <hpx/assert.hpp>
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
    public:
        container_distribution_policy()
          : num_partitions_(std::size_t(-1))
        {
        }

        container_distribution_policy operator()(
            std::size_t num_partitions) const
        {
            return container_distribution_policy(
                num_partitions, get_localities());
        }

        container_distribution_policy operator()(
            hpx::id_type const& locality) const
        {
            return container_distribution_policy(locality);
        }

        container_distribution_policy operator()(
            std::vector<id_type> const& localities) const
        {
            if (num_partitions_ != std::size_t(-1))
            {
                return container_distribution_policy(
                    num_partitions_, localities);
            }
            return container_distribution_policy(localities.size(), localities);
        }

        container_distribution_policy operator()(
            std::vector<id_type>&& localities) const
        {
            if (num_partitions_ != std::size_t(-1))
            {
                return container_distribution_policy(
                    num_partitions_, HPX_MOVE(localities));
            }
            return container_distribution_policy(
                localities.size(), HPX_MOVE(localities));
        }

        container_distribution_policy operator()(std::size_t num_partitions,
            std::vector<id_type> const& localities) const
        {
            return container_distribution_policy(num_partitions, localities);
        }

        container_distribution_policy operator()(
            std::size_t num_partitions, std::vector<id_type>&& localities) const
        {
            return container_distribution_policy(
                num_partitions, HPX_MOVE(localities));
        }

        ///////////////////////////////////////////////////////////////////////
        std::size_t get_num_partitions() const
        {
            if (localities_)
            {
                std::size_t num_parts = (num_partitions_ == std::size_t(-1)) ?
                    localities_->size() :
                    num_partitions_;
                return (std::max)(num_parts, std::size_t(1));
            }
            return std::size_t(1);
        }

        std::vector<hpx::id_type> get_localities() const
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

    private:
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive& ar, const unsigned int /* version */)
        {
            // clang-format off
            ar & localities_ & num_partitions_;
            // clang-format on
        }

        container_distribution_policy(
            std::size_t num_partitions, std::vector<id_type> const& localities)
          : components::default_distribution_policy(localities)
          , num_partitions_(num_partitions)
        {
        }

        container_distribution_policy(
            std::size_t num_partitions, std::vector<id_type>&& localities)
          : components::default_distribution_policy(HPX_MOVE(localities))
          , num_partitions_(num_partitions)
        {
        }

        explicit container_distribution_policy(hpx::id_type const& locality)
          : components::default_distribution_policy(locality)
          , num_partitions_(1)
        {
        }

    private:
        std::size_t num_partitions_;    // number of chunks to create
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
    }    // namespace traits
}    // namespace hpx
