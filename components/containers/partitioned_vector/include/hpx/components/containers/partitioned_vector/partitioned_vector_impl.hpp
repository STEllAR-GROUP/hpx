//  Copyright (c) 2014 Anuj R. Sharma
//  Copyright (c) 2014-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/async_combinators/wait_all.hpp>
#include <hpx/async_combinators/when_all.hpp>
#include <hpx/functional/bind_back.hpp>
#include <hpx/assert.hpp>
#include <hpx/modules/async_distributed.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/execution.hpp>
#include <hpx/runtime/components/client_base.hpp>
#include <hpx/runtime/components/copy_component.hpp>
#include <hpx/runtime/components/new.hpp>
#include <hpx/runtime/components/server/distributed_metadata_base.hpp>
#include <hpx/runtime/get_ptr.hpp>
#include <hpx/traits/is_distribution_policy.hpp>
#include <hpx/type_support/unused.hpp>

#include <hpx/components/containers/container_distribution_policy.hpp>
#include <hpx/components/containers/partitioned_vector/partitioned_vector_decl.hpp>
#include <hpx/components/containers/partitioned_vector/partitioned_vector_component_impl.hpp>
#include <hpx/components/containers/partitioned_vector/partitioned_vector_segmented_iterator.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{

    template <typename T, typename Data /*= std::vector<T> */>
    HPX_PARTITIONED_VECTOR_SPECIALIZATION_EXPORT std::size_t
    partitioned_vector<T, Data>::get_partition_size() const
    {
        std::size_t num_parts = partitions_.size();
        return num_parts ? ((size_ + num_parts - 1) / num_parts) : 0;
    }

    template <typename T, typename Data /*= std::vector<T> */>
    HPX_PARTITIONED_VECTOR_SPECIALIZATION_EXPORT std::size_t
    partitioned_vector<T, Data>::get_global_index(
        std::size_t segment, std::size_t part_size, size_type local_index) const
    {
        return segment * part_size + local_index;
    }

    template <typename T, typename Data /*= std::vector<T> */>
    HPX_PARTITIONED_VECTOR_SPECIALIZATION_EXPORT void
    partitioned_vector<T, Data>::get_data_helper(
        id_type id, server::partitioned_vector_config_data data)
    {
        partitions_.clear();
        partitions_.reserve(data.partitions_.size());

        size_ = data.size_;
        std::move(data.partitions_.begin(), data.partitions_.end(),
            std::back_inserter(partitions_));

        std::uint32_t this_locality = get_locality_id();
        std::vector<future<void>> ptrs;

        typedef typename partitions_vector_type::const_iterator const_iterator;

        std::size_t l = 0;
        const_iterator end = partitions_.cend();
        for (const_iterator it = partitions_.cbegin(); it != end; ++it, ++l)
        {
            if (it->locality_id_ == this_locality)
            {
                ptrs.push_back(
                    get_ptr<partitioned_vector_partition_server>(it->partition_)
                        .then(get_ptr_helper{l, partitions_}));
            }
        }
        wait_all(ptrs);

        partition_size_ = get_partition_size();
        this->base_type::reset(std::move(id));
    }

    template <typename T, typename Data /*= std::vector<T> */>
    HPX_PARTITIONED_VECTOR_SPECIALIZATION_EXPORT hpx::future<void>
    partitioned_vector<T, Data>::connect_to_helper(id_type id)
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        typedef typename components::server::distributed_metadata_base<
            server::partitioned_vector_config_data>::get_action act;

        return async(act(), id).then(
            [=](future<server::partitioned_vector_config_data>&& f) -> void {
                return get_data_helper(id, f.get());
            });
#else
        HPX_ASSERT(false);
        HPX_UNUSED(id);
        return hpx::make_ready_future();
#endif
    }

    template <typename T, typename Data /*= std::vector<T> */>
    HPX_PARTITIONED_VECTOR_SPECIALIZATION_EXPORT hpx::future<void>
    partitioned_vector<T, Data>::connect_to(std::string const& symbolic_name)
    {
        this->base_type::connect_to(symbolic_name);
        return this->base_type::share().then(
            [=](shared_future<id_type>&& f) -> hpx::future<void> {
                return connect_to_helper(f.get());
            });
    }

    template <typename T, typename Data /*= std::vector<T> */>
    HPX_PARTITIONED_VECTOR_SPECIALIZATION_EXPORT void
    partitioned_vector<T, Data>::connect_to(
        launch::sync_policy, std::string const& symbolic_name)
    {
        connect_to(symbolic_name).get();
    }

    template <typename T, typename Data /*= std::vector<T> */>
    HPX_PARTITIONED_VECTOR_SPECIALIZATION_EXPORT hpx::future<void>
    partitioned_vector<T, Data>::register_as(std::string const& symbolic_name)
    {
        std::vector<server::partitioned_vector_config_data::partition_data>
            partitions;
        partitions.reserve(partitions_.size());

        std::copy(
            partitions_.begin(), partitions_.end(), std::back_inserter(partitions));

        server::partitioned_vector_config_data data(size_, std::move(partitions));
        this->base_type::reset(
            hpx::new_<components::server::distributed_metadata_base<
                server::partitioned_vector_config_data>>(
                hpx::find_here(), std::move(data)));

        return this->base_type::register_as(symbolic_name);
    }

    template <typename T, typename Data /*= std::vector<T> */>
    HPX_PARTITIONED_VECTOR_SPECIALIZATION_EXPORT void
    partitioned_vector<T, Data>::register_as(
        launch::sync_policy, std::string const& symbolic_name)
    {
        register_as(symbolic_name).get();
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Data /*= std::vector<T> */>
    HPX_PARTITIONED_VECTOR_SPECIALIZATION_EXPORT
    partitioned_vector<T, Data>::partitioned_vector(future<id_type>&& f)
    {
        f.share().then(
            [=](shared_future<id_type>&& f) -> hpx::future<void> {
                return connect_to_helper(f.get());
            });
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Data /*= std::vector<T> */>
    HPX_PARTITIONED_VECTOR_SPECIALIZATION_EXPORT std::size_t
    partitioned_vector<T, Data>::get_partition(size_type global_index) const
    {
        if (global_index == size_)
            return partitions_.size();

        std::size_t part_size = partition_size_;
        if (part_size != 0)
            return (part_size != size_) ? (global_index / part_size) : 0;

        return partitions_.size();
    }

    template <typename T, typename Data /*= std::vector<T> */>
    HPX_PARTITIONED_VECTOR_SPECIALIZATION_EXPORT std::size_t
    partitioned_vector<T, Data>::get_local_index(size_type global_index) const
    {
        if (global_index == size_ || partition_size_ == std::size_t(-1) ||
            partition_size_ == 0)
        {
            return std::size_t(-1);
        }

        return (partition_size_ != size_) ? (global_index % partition_size_) :
                                            global_index;
    }

    template <typename T, typename Data /*= std::vector<T> */>
    HPX_PARTITIONED_VECTOR_SPECIALIZATION_EXPORT
        std::vector<typename partitioned_vector<T, Data>::size_type>
        partitioned_vector<T, Data>::get_local_indices(
            std::vector<size_type> indices) const
    {
        for (size_type& index : indices)
            index = get_local_index(index);
        return indices;
    }

    template <typename T, typename Data /*= std::vector<T> */>
    HPX_PARTITIONED_VECTOR_SPECIALIZATION_EXPORT
        typename partitioned_vector<T, Data>::local_iterator
        partitioned_vector<T, Data>::get_local_iterator(
            size_type global_index) const
    {
        HPX_ASSERT(global_index != std::size_t(-1));

        std::size_t part = get_partition(global_index);
        if (part == partitions_.size())
        {
            // return an iterator to the end of the last partition
            auto const& back = partitions_.back();
            return local_iterator(back.partition_, back.size_, back.local_data_);
        }

        std::size_t local_index = get_local_index(global_index);
        HPX_ASSERT(local_index != std::size_t(-1));

        return local_iterator(partitions_[part].partition_, local_index,
            partitions_[part].local_data_);
    }

    template <typename T, typename Data /*= std::vector<T> */>
    HPX_PARTITIONED_VECTOR_SPECIALIZATION_EXPORT
        typename partitioned_vector<T, Data>::const_local_iterator
        partitioned_vector<T, Data>::get_const_local_iterator(
            size_type global_index) const
    {
        HPX_ASSERT(global_index != std::size_t(-1));

        std::size_t part = get_partition(global_index);
        if (part == partitions_.size())
        {
            // return an iterator to the end of the last partition
            auto const& back = partitions_.back();
            return local_iterator(back.partition_, back.size_, back.local_data_);
        }

        std::size_t local_index = get_local_index(global_index);
        HPX_ASSERT(local_index != std::size_t(-1));

        return const_local_iterator(partitions_[part].partition_, local_index,
            partitions_[part].local_data_);
    }

    template <typename T, typename Data /*= std::vector<T> */>
    HPX_PARTITIONED_VECTOR_SPECIALIZATION_EXPORT
        typename partitioned_vector<T, Data>::segment_iterator
        partitioned_vector<T, Data>::get_segment_iterator(size_type global_index)
    {
        std::size_t part = get_partition(global_index);
        if (part == partitions_.size())
            return segment_iterator(partitions_.end(), this);

        return segment_iterator(partitions_.begin() + part, this);
    }

    template <typename T, typename Data /*= std::vector<T> */>
    HPX_PARTITIONED_VECTOR_SPECIALIZATION_EXPORT
        typename partitioned_vector<T, Data>::const_segment_iterator
        partitioned_vector<T, Data>::get_const_segment_iterator(
            size_type global_index) const
    {
        std::size_t part = get_partition(global_index);
        if (part == partitions_.size())
            return const_segment_iterator(partitions_.cend(), this);

        return const_segment_iterator(partitions_.cbegin() + part, this);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Data /*= std::vector<T> */>
    template <typename DistPolicy>
    HPX_PARTITIONED_VECTOR_SPECIALIZATION_EXPORT hpx::future<
        std::vector<typename partitioned_vector<T, Data>::bulk_locality_result>>
    partitioned_vector<T, Data>::create_helper1(
        DistPolicy const& policy, std::size_t count, std::size_t size)
    {
        typedef typename partitioned_vector_partition_client::server_component_type
            component_type;

        return policy.template bulk_create<component_type>(count, size);
    }

    template <typename T, typename Data /*= std::vector<T> */>
    template <typename DistPolicy>
    HPX_PARTITIONED_VECTOR_SPECIALIZATION_EXPORT hpx::future<
        std::vector<typename partitioned_vector<T, Data>::bulk_locality_result>>
    partitioned_vector<T, Data>::create_helper2(
        DistPolicy const& policy, std::size_t count, std::size_t size, T const& val)
    {
        typedef typename partitioned_vector_partition_client::server_component_type
            component_type;

        return policy.template bulk_create<component_type>(count, size, val);
    }

    template <typename T, typename Data /*= std::vector<T> */>
    struct partitioned_vector<T, Data>::get_ptr_helper
    {
        std::size_t loc;
        partitions_vector_type& partitions;

        void operator()(
            future<std::shared_ptr<partitioned_vector_partition_server>>&& f) const
        {
            partitions[loc].local_data_ = f.get();
        }
    };

    template <typename T, typename Data /*= std::vector<T> */>
    template <typename DistPolicy, typename Create>
    void partitioned_vector<T, Data>::create(
        DistPolicy const& policy, Create&& creator)
    {
        std::size_t num_parts =
            traits::num_container_partitions<DistPolicy>::call(policy);
        std::size_t part_size = (size_ + num_parts - 1) / num_parts;

        // create as many partitions as required
        hpx::future<std::vector<bulk_locality_result>> f =
            creator(policy, num_parts, part_size);

        // now initialize our data structures
        std::uint32_t this_locality = get_locality_id();
        std::vector<future<void>> ptrs;

        std::size_t num_part = 0;
        std::size_t allocated_size = 0;

        std::size_t l = 0;

        // Fixing the size of partitions to avoid race conditions between
        // possible reallocations during push back and the continuation
        // to set the local partition data
        partitions_.resize(num_parts);
        for (bulk_locality_result const& r : f.get())
        {
            using naming::get_locality_id_from_id;
            std::uint32_t locality = get_locality_id_from_id(r.first);
            for (hpx::id_type const& id : r.second)
            {
                std::size_t size = (std::min)(part_size, size_ - allocated_size);
                partitions_[l] = partition_data(id, size, locality);

                if (locality == this_locality)
                {
                    ptrs.push_back(
                        get_ptr<partitioned_vector_partition_server>(id).then(
                            get_ptr_helper{l, partitions_}));
                }
                ++l;

                allocated_size += size;
                if (++num_part == num_parts)
                {
                    HPX_ASSERT(allocated_size == size_);

                    // shrink last partition, if appropriate
                    if (size != part_size)
                    {
                        partitioned_vector_partition_client(
                            partitions_[l - 1].partition_)
                            .resize(size);
                    }
                    break;
                }
                else
                {
                    HPX_ASSERT(size == part_size);
                }
                HPX_ASSERT(l < num_parts);
            }
        }
        HPX_ASSERT(l == num_parts);

        when_all(ptrs).get();

        // cache our partition size
        partition_size_ = get_partition_size();
    }

    template <typename T, typename Data /*= std::vector<T> */>
    template <typename DistPolicy>
    HPX_PARTITIONED_VECTOR_SPECIALIZATION_EXPORT void
    partitioned_vector<T, Data>::create(DistPolicy const& policy)
    {
        create(policy, &partitioned_vector::create_helper1<DistPolicy>);
    }

    template <typename T, typename Data /*= std::vector<T> */>
    template <typename DistPolicy>
    HPX_PARTITIONED_VECTOR_SPECIALIZATION_EXPORT void
    partitioned_vector<T, Data>::create(T const& val, DistPolicy const& policy)
    {
        create(policy, util::bind_back(
            &partitioned_vector::create_helper2<DistPolicy>, std::ref(val)));
    }

    template <typename T, typename Data /*= std::vector<T> */>
    HPX_PARTITIONED_VECTOR_SPECIALIZATION_EXPORT void
    partitioned_vector<T, Data>::copy_from(partitioned_vector const& rhs)
    {
        typedef typename partitions_vector_type::const_iterator const_iterator;

        std::vector<future<id_type>> objs;
        const_iterator end = rhs.partitions_.end();
        for (const_iterator it = rhs.partitions_.begin(); it != end; ++it)
        {
            typedef
                typename partitioned_vector_partition_client::server_component_type
                    component_type;
            objs.push_back(hpx::components::copy<component_type>(it->partition_));
        }
        wait_all(objs);

        std::uint32_t this_locality = get_locality_id();
        std::vector<future<void>> ptrs;

        partitions_vector_type partitions;
        // Fixing the size of partitions to avoid race conditions between
        // possible reallocations during push back and the continuation
        // to set the local partition data
        partitions.resize(rhs.partitions_.size());
        for (std::size_t i = 0; i != rhs.partitions_.size(); ++i)
        {
            std::uint32_t locality = rhs.partitions_[i].locality_id_;

            partitions[i] =
                partition_data(objs[i].get(), rhs.partitions_[i].size_, locality);

            if (locality == this_locality)
            {
                ptrs.push_back(
                    get_ptr<partitioned_vector_partition_server>(
                        partitions[i].partition_)
                        .then(get_ptr_helper{i, partitions}));
            }
        }

        when_all(ptrs).get();

        size_ = rhs.size_;
        partition_size_ = rhs.partition_size_;
        std::swap(partitions_, partitions);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Data /*= std::vector<T> */>
    HPX_PARTITIONED_VECTOR_SPECIALIZATION_EXPORT
    partitioned_vector<T, Data>::partitioned_vector()
      : size_(0)
      , partition_size_(std::size_t(-1))
    {
    }

    template <typename T, typename Data /*= std::vector<T> */>
    HPX_PARTITIONED_VECTOR_SPECIALIZATION_EXPORT
    partitioned_vector<T, Data>::partitioned_vector(size_type size)
      : size_(size)
      , partition_size_(std::size_t(-1))
    {
        if (size != 0)
            create(hpx::container_layout);
    }

    template <typename T, typename Data /*= std::vector<T> */>
    HPX_PARTITIONED_VECTOR_SPECIALIZATION_EXPORT
    partitioned_vector<T, Data>::partitioned_vector(size_type size, T const& val)
      : size_(size)
      , partition_size_(std::size_t(-1))
    {
        if (size != 0)
            create(val, hpx::container_layout);
    }

    template <typename T, typename Data /*= std::vector<T> */>
    template <typename DistPolicy>
    HPX_PARTITIONED_VECTOR_SPECIALIZATION_EXPORT
    partitioned_vector<T, Data>::partitioned_vector(size_type size,
        DistPolicy const& policy,
        typename std::enable_if<
            traits::is_distribution_policy<DistPolicy>::value>::type*)
      : size_(size)
      , partition_size_(std::size_t(-1))
    {
        if (size != 0)
            create(policy);
    }

    template <typename T, typename Data /*= std::vector<T> */>
    template <typename DistPolicy>
    HPX_PARTITIONED_VECTOR_SPECIALIZATION_EXPORT
    partitioned_vector<T, Data>::partitioned_vector(size_type size, T const& val,
        DistPolicy const& policy,
        typename std::enable_if<
            traits::is_distribution_policy<DistPolicy>::value>::type* /*= nullptr*/)
      : size_(size)
      , partition_size_(std::size_t(-1))
    {
        if (size != 0)
            create(val, policy);
    }
}

