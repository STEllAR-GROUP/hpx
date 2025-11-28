//  Copyright (c) 2017 Christopher Taylor
//  Copyright (c) 2024 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/components/containers/partitioned_vector/partitioned_vector_decl.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/serialization.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace hpx::serialization {

    template <typename T, typename Data>
    void serialize(
        input_archive& ar, hpx::partitioned_vector<T, Data>& v, unsigned)
    {
        using partitioned_vector = hpx::partitioned_vector<T, Data>;
        bool has_id = false;
        ar >> has_id;
        if (has_id)
        {
            hpx::id_type id;
            std::size_t size;
            std::vector<typename partitioned_vector::partition_data_type>
                partitions;

            ar >> id >> size >> partitions;
            v = partitioned_vector::create_from(
                HPX_MOVE(id), size, HPX_MOVE(partitions));
        }
        else
        {
            std::string registered_name;
            ar >> registered_name;
            v.connect_to(registered_name).get();
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    template <typename T, typename Data>
    void serialize(
        output_archive& ar, hpx::partitioned_vector<T, Data> const& v, unsigned)
    {
        bool has_id = v.valid();
        ar << has_id;
        if (has_id)
        {
            ar << v.get_id() << v.size() << v.partitions();
        }
        else
        {
            std::string const registered_name = v.registered_name();
            if (registered_name.empty())
            {
                HPX_THROW_EXCEPTION(hpx::error::invalid_status,
                    "hpx::serialization::serialize",
                    "partitioned_vector is not registered");
            }
            ar << registered_name;
        }
    }
}    // namespace hpx::serialization
