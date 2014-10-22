//  Copyright (c) 2014 Anuj R. Sharma
//  Copyright (c) 2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/components/vector/vector.hpp

#ifndef HPX_VECTOR_HPP
#define HPX_VECTOR_HPP

/// \brief The hpx::vector and its API's are defined here.
///
/// The hpx::vector is a segmented data structure which is a collection of one
/// or more hpx::partition_vectors. The hpx::vector stores the global IDs of each
/// hpx::partition_vector and the index (with respect to whole vector) of the first
/// element in that hpx::partition_vector. These two are stored in std::pair.

#include <hpx/include/lcos.hpp>
#include <hpx/include/util.hpp>
#include <hpx/include/components.hpp>

#include <hpx/components/vector/segmented_iterator.hpp>
#include <hpx/components/vector/partition_vector_component.hpp>
#include <hpx/components/vector/vector_configuration.hpp>
#include <hpx/components/vector/distribution_policy.hpp>

#include <cstdint>
#include <iostream>
#include <memory>

#include <boost/format.hpp>
#include <boost/cstdint.hpp>

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    /// \brief This is the vector class which define hpx::vector functionality.
    ///
    ///  This contains the client side implementation of the hpx::vector. This
    ///  class defines the synchronous and asynchronous API's for each of the
    ///  exposed functionalities.
    ///
    template <typename T>
    class vector
      : hpx::components::client_base<vector<T>, server::vector_configuration>
    {
    public:
        typedef std::allocator<T> allocator_type;

        typedef std::size_t size_type;
        typedef std::ptrdiff_t difference_type;

        typedef T value_type;
        typedef T reference;
        typedef T const const_reference;
        typedef typename std::allocator_traits<allocator_type>::pointer pointer;
        typedef typename std::allocator_traits<allocator_type>::const_pointer
            const_pointer;

    private:
        typedef hpx::components::client_base<
                vector, server::vector_configuration
            > base_type;

        typedef hpx::server::partition_vector partition_vector_server;
        typedef hpx::stubs::partition_vector<T> partition_vector_stub;
        typedef hpx::partition_vector<T> partition_vector_client;

        // The list of partitions belonging to this vector.
        // Each partition is described by it's corresponding client object, its
        // size, and locality id.
        typedef server::vector_configuration::partition_data partition_data;
        typedef std::vector<partition_data> partitions_vector_type;

        size_type size_;                // overall size of the vector
        size_type block_size_;          // cycle stride

        // This is the vector representing the base_index and corresponding
        // global ID's of the underlying partition_vectors.
        partitions_vector_type partitions_;

        // parameters taken from distribution policy
        BOOST_SCOPED_ENUM(distribution_policy) policy_;     // policy to use

    public:
        typedef vector_iterator<T> iterator;
        typedef const_vector_iterator<T> const_iterator;
        typedef std::reverse_iterator<iterator> reverse_iterator;
        typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

        typedef local_vector_iterator<T> local_iterator;
        typedef const_local_vector_iterator<T> const_local_iterator;

        typedef segment_vector_iterator<
                T, typename partitions_vector_type::iterator
            > segment_iterator;
        typedef const_segment_vector_iterator<
                T, typename partitions_vector_type::const_iterator
            > const_segment_iterator;

    private:
        friend class vector_iterator<T>;
        friend class const_vector_iterator<T>;

        std::size_t get_partition_size() const
        {
            std::size_t num_parts = partitions_.size();
            return num_parts ? ((size_ + num_parts - 1) / num_parts) : 0;
        }

        std::size_t get_global_index(std::size_t segment, std::size_t part_size,
            size_type local_index) const
        {
            switch (policy_)
            {
            case distribution_policy::block:
                return segment * part_size + local_index;

            case distribution_policy::cyclic:
                return segment + local_index * (part_size - 1);

            case distribution_policy::block_cyclic:
                return (segment * part_size/block_size_) + local_index * (part_size - 1);

            default:
                break;
            }
            return std::size_t(-1);
        }

        void verify_consistency()
        {
            // verify consistency of parameters
            switch (policy_)
            {
            case distribution_policy::block:
                break;      // no limitations apply

            case distribution_policy::cyclic:
                // overall size must be multiple of partition size
                {
                    std::size_t part_size = get_partition_size();
                    if (part_size != std::size_t(-1) &&
                        (size_ % part_size) != 0)
                    {
                        HPX_THROW_EXCEPTION(bad_parameter,
                            "hpx::vector::create",
                            boost::str(boost::format(
                                "cyclic distribution policy requires that the "
                                "overall size(%1%) of the vector must be a "
                                "multiple of the partition size(%2%)"
                            ) % size_ % part_size));
                    }
                }
                break;

            case distribution_policy::block_cyclic:
                {
                    if (block_size_ == std::size_t(-1))
                        block_size_ = get_partition_size();

                    std::size_t part_size = get_partition_size();
                    if (part_size != std::size_t(-1) &&
                        (size_ % part_size) != 0)
                    {
                        HPX_THROW_EXCEPTION(bad_parameter,
                            "hpx::vector::create",
                            boost::str(boost::format(
                                "block_cyclic distribution policy requires "
                                "that the overall size(%1%) of the vector must "
                                "be a multiple of the partition size(%2%)"
                            ) % size_ % part_size));
                        break;
                    }

                    HPX_ASSERT(block_size_ != 0);
                    if ((part_size % block_size_) != 0)
                    {
                        HPX_THROW_EXCEPTION(bad_parameter,
                            "hpx::vector::create",
                            boost::str(boost::format(
                                "block_cyclic distribution policy requires "
                                "that the overall partition size(%1%) of the "
                                "vector must be a multiple of the block "
                                "size(%2%)"
                            ) % part_size % block_size_));
                        break;
                    }
                }
                break;

            default:
                break;
            }
        }

        ///////////////////////////////////////////////////////////////////////
        // Connect this vector to the existing vector using the given symbolic
        // name.
        void get_data_helper(id_type id,
            future<server::vector_configuration::config_data> f)
        {
            server::vector_configuration::config_data data = f.get();
            size_ = data.size_;
            block_size_ = data.block_size_;
            partitions_ = std::move(data.partitions_);
            policy_ = static_cast<BOOST_SCOPED_ENUM(distribution_policy)>(
                data.policy_);

            base_type::reset(std::move(id));
        }

        // this will be called by the base class once the registered id becomes
        // available
        future<void> connect_to_helper(future<id_type> f)
        {
            using util::placeholders::_1;
            id_type id = f.get();
            return async(server::vector_configuration::get_action(), id)
                .then(util::bind(&vector::get_data_helper, this, id, _1));
        }

    public:
        future<void> connect_to(char const* symbolic_name)
        {
            using util::placeholders::_1;
            return base_type::connect_to(symbolic_name,
                util::bind(&vector::connect_to_helper, this, _1));
        }

        future<void> connect_to(std::string const& symbolic_name)
        {
            return connect_to(symbolic_name.c_str());
        }

        // Register this vector with AGAS using the given symbolic name
        future<void> register_as(char const* symbolic_name)
        {
            server::vector_configuration::config_data data(
                size_, block_size_, partitions_, int(policy_));
            base_type::reset(base_type::create_async(hpx::find_here(), data));
            return base_type::register_as(symbolic_name);
        }

        future<void> register_as(std::string const& symbolic_name)
        {
            return register_as(symbolic_name.c_str());
        }

    public:
        // Return the sequence number of the segment corresponding to the
        // given global index
        std::size_t get_partition(size_type global_index) const
        {
            if (global_index == size_)
                return std::size_t(-1);

            switch (policy_)
            {
            case distribution_policy::block:
                {
                    std::size_t part_size = get_partition_size();
                    if (part_size != 0)
                        return global_index / part_size;
                }
                break;

            case distribution_policy::cyclic:
                {
                    std::size_t num_parts = partitions_.size();
                    if (num_parts != 0)
                        return global_index % num_parts;
                }
                break;

            case distribution_policy::block_cyclic:
                {
                    HPX_ASSERT(block_size_ != 0);
                    std::size_t num_blocks = size_ / block_size_;
                    if (num_blocks != 0)
                    {
                        std::size_t num_parts = partitions_.size();
                        if (num_parts != 0)
                        {
                            std::size_t block_num = global_index % num_blocks;
                            return block_num / (num_blocks / num_parts);
                        }
                    }
                }
                break;

            default:
                break;
            }
            return std::size_t(-1);
        }

        // Return the local index inside the segment corresponding to the
        // given global index
        std::size_t get_local_index(size_type global_index) const
        {
            if (global_index == size_)
                return std::size_t(-1);

            switch (policy_)
            {
            case distribution_policy::block:
                {
                    std::size_t part_size = get_partition_size();
                    if (part_size != 0)
                        return global_index % part_size;
                }
                break;

            case distribution_policy::cyclic:
                {
                    std::size_t num_parts = partitions_.size();
                    if (num_parts != 0)
                        return global_index / num_parts;
                }
                break;

            case distribution_policy::block_cyclic:
                {
                    HPX_ASSERT(block_size_ != 0);
                    std::size_t num_blocks = size_ / block_size_;
                    if (num_blocks != 0)
                    {
                        std::size_t num_parts = partitions_.size();
                        if (num_parts != 0)
                        {
                            // block number inside its partitions
                            std::size_t block_num = global_index % num_blocks;
                            block_num %= (num_blocks / num_parts);

                            // blocks below current index + index inside block
                            std::size_t block_idx = global_index / num_blocks;
                            return block_size_ * block_num + block_idx;
                        }
                    }
                }
                break;

            default:
                break;
            }
            return std::size_t(-1);
        }

        // Return the global index corresponding to the local index inside the
        // given segment.
        std::size_t get_global_index(segment_iterator const& it,
            size_type local_index)
        {
            std::size_t part_size = get_partition_size();
            if (part_size == 0)
                return std::size_t(-1);

            std::size_t segment = std::distance(partitions_.begin(), it.base());
            return get_global_index(segment, part_size, local_index);
        }

        std::size_t get_global_index(const_segment_iterator const& it,
            size_type local_index) const
        {
            std::size_t part_size = get_partition_size();
            if (part_size == 0)
                return std::size_t(-1);

            std::size_t segment = std::distance(partitions_.cbegin(), it.base());
            return get_global_index(segment, part_size, local_index);
        }

        // Return the local iterator referencing an element inside a segment
        // based on the given global index.
        local_iterator get_local_iterator(size_type global_index) const
        {
            std::size_t part = get_partition(global_index);
            if (part == std::size_t(-1))
                return local_iterator();

            std::size_t local_index = get_local_index(global_index);
            HPX_ASSERT(local_index != std::size_t(-1));

            return local_iterator(partitions_[part].partition_, local_index);
        }

        const_local_iterator get_const_local_iterator(size_type global_index) const
        {
            std::size_t part = get_partition(global_index);
            if (part == std::size_t(-1))
                return const_local_iterator();

            std::size_t local_index = get_local_index(global_index);
            HPX_ASSERT(local_index != std::size_t(-1));

            return const_local_iterator(partitions_[part].partition_, local_index);
        }

        // Return the segment iterator referencing a segment based on the
        // given global index.
        segment_iterator get_segment_iterator(size_type global_index)
        {
            std::size_t part = get_partition(global_index);
            if (part == std::size_t(-1))
                return segment_iterator(this, partitions_.end());

            return segment_iterator(this, partitions_.begin() + part,
                partitions_.end());
        }

        const_segment_iterator get_const_segment_iterator(
            size_type global_index) const
        {
            std::size_t part = get_partition(global_index);
            if (part == std::size_t(-1))
                return const_segment_iterator(this, partitions_.cend());

            return const_segment_iterator(this, partitions_.cbegin() + part,
                partitions_.cend());
        }

    protected:
        template <typename DistPolicy>
        void create(std::vector<id_type> const& localities,
            DistPolicy const& policy)
        {
            std::size_t num_parts = policy.get_num_partitions();
            std::size_t part_size = (size_ + num_parts - 1) / num_parts;
            std::size_t num_localities = localities.size();
            std::size_t num_parts_per_loc =
                (num_parts + num_localities - 1) / num_localities;

            // create as many partitions as required
            std::size_t num_part = 0;
            std::size_t allocated_size = 0;
            for (std::size_t loc = 0; loc != num_localities; ++loc)
            {
                // create as many partitions on a given locality as required
                for (std::size_t l = 0; l != num_parts_per_loc; ++l)
                {
                    id_type const& locality = localities[loc];
                    std::size_t size = (std::min)(part_size, size_-allocated_size);
                    partitions_.push_back(partition_data(
                        partition_vector_client::create_async(locality, size),
                        size, hpx::naming::get_locality_id_from_id(locality)
                    ));

                    allocated_size += size;
                    if (++num_part == num_parts)
                        return;
                }
            }
        }

        template <typename DistPolicy>
        void create(T const& val, std::vector<id_type> const& localities,
            DistPolicy const& policy)
        {
            std::size_t num_parts = policy.get_num_partitions();
            std::size_t part_size = (size_ + num_parts - 1) / num_parts;
            std::size_t num_localities = localities.size();
            std::size_t num_parts_per_loc =
                (num_parts + num_localities - 1) / num_localities;

            // create as many partitions as required
            std::size_t num_part = 0;
            std::size_t allocated_size = 0;
            for (std::size_t loc = 0; loc != num_localities; ++loc)
            {
                // create as many partitions on a given locality as required
                for (std::size_t l = 0; l != num_parts_per_loc; ++l)
                {
                    id_type const& locality = localities[loc];
                    std::size_t size = (std::min)(part_size, size_-allocated_size);
                    partitions_.push_back(partition_data(
                        partition_vector_client::create_async(locality, size, val),
                        size, hpx::naming::get_locality_id_from_id(locality)
                    ));

                    allocated_size += size;
                    if (++num_part == num_parts)
                        return;
                }
            }
        }

        // This function is called when we are creating the vector. It
        // initializes the partitions based on the give parameters.
        template <typename DistPolicy>
        void create(DistPolicy const& policy)
        {
            std::vector<id_type> const& localities = policy.get_localities();
            if (localities.empty())
                create(std::vector<id_type>(1, find_here()), policy);
            else
                create(localities, policy);

            verify_consistency();
        }

        template <typename DistPolicy>
        void create(T const& val, DistPolicy const& policy)
        {
            std::vector<id_type> const& localities = policy.get_localities();
            if (localities.empty())
                create(val, std::vector<id_type>(1, find_here()), policy);
            else
                create(val, localities, policy);

            verify_consistency();
        }

// //        future<size_type>
// //            max_size_helper(size_type num_partitions) const
// //        {
// //            if(num_partitions < 1)
// //            {
// //                HPX_ASSERT(num_partitions >= 0);
// //                return partition_vector_stub::max_size_async(
// //                        ((partitions_.at(num_partitions)).second).get()
// //                                                                );
// //            }
// //            else
// //                return hpx::lcos::local::dataflow(
// //                    [](future<size_type> s1,
// //                       future<size_type> s2) -> size_type
// //                    {
// //                        return s1.get() + s2.get();
// //                    },
// //                    partition_vector_stub::max_size_async(
// //                        ((partitions_.at(num_partitions)).second).get()
// //                                                             ),
// //                    max_size_helper(num_partitions - 1)
// //                                                );
// //            }//end of max_size_helper
//
//
//         //FASTER VERSION OF MAX_SIZE_HELPER
//
//         // PROGRAMMER DOCUMENTATION:
//         //  This helper function return the number of element in the hpx::vector.
//         //  Here we are dividing the sequence of partition_description_types into half and
//         //  computing the max_size of the individual partition_vector and then adding
//         //  them. Note this create the binary tree of height. Equal to log
//         //  (num_partition_description_types in partitions_). Hence it might be efficient
//         //  than previous implementation
//         //
//         // NOTE: This implementation does not need all the partition_vector of same
//         //       size.
//         //
//         future<size_type> max_size_helper(partition_vector_type::const_iterator it_begin,
//                                     partition_vector_type::const_iterator it_end) const
//         {
//             if((it_end - it_begin) == 1 )
//                 return partition_vector_stub::max_size_async(
//                                                     (it_begin->first).get()
//                                                                 );
//             else
//             {
//                 int mid = (it_end - it_begin)/2;
//                 future<size_type> left_tree_size = max_size_helper(it_begin,
//                                                              it_begin + mid);
//                 future<size_type> right_tree_size = hpx::async(
//                                                 launch::async,
//                                                 hpx::util::bind(
//                                                     &vector::max_size_helper,
//                                                     this,
//                                                     (it_begin + mid),
//                                                     it_end
//                                                                 )
//                                                         );
//
//                 return hpx::lcos::local::dataflow(
//                             [](future<size_type> s1, future<size_type> s2) -> size_type
//                             {
//                                 return s1.get() + s2.get();
//                             },
//                             std::move(left_tree_size),
//                             std::move(right_tree_size)
//                                                  );
//             }
//         }//end of max_size_helper
//
//
// //        future<size_type>
// //            capacity_helper(size_type num_partitions) const
// //        {
// //            if(num_partitions < 1)
// //            {
// //                HPX_ASSERT(num_partitions >= 0);
// //                return partition_vector_stub::capacity_async(
// //                          ((partitions_.at(num_partitions)).second).get()
// //                                                         );
// //            }
// //            else
// //                return hpx::lcos::local::dataflow(
// //                    [](future<size_type> s1,
// //                       future<size_type> s2) -> size_type
// //                    {
// //                        return s1.get() + s2.get();
// //                    },
// //                    partition_vector_stub::capacity_async(
// //                        ((partitions_.at(num_partitions)).second).get()
// //                                                       ),
// //                    capacity_helper(num_partitions - 1)
// //                                                );
// //            }//end of capacity_helper
//
//         //FASTER VERSION OF CAPACITY_HELPER
//
//         // PROGRAMMER DOCUMENTATION:
//         //  This helper function return the number of element in the hpx::vector.
//         //  Here we are dividing the sequence of partition_description_types into half and
//         //  computing the capacity of the individual partition_vector and then adding
//         //  them. Note this create the binary tree of height Equal to log
//         //  (num_partition_description_types in partitions_). Hence it might be efficient
//         //  than previous implementation.
//         //
//         // NOTE: This implementation does not need all the partition_vector of same
//         //       size.
//         //
//         future<size_type> capacity_helper(partition_vector_type::const_iterator it_begin,
//                                     partition_vector_type::const_iterator it_end) const
//         {
//             if((it_end - it_begin) == 1 )
//                 return partition_vector_stub::capacity_async(
//                                                     (it_begin->first).get()
//                                                           );
//             else
//             {
//                 int mid = (it_end - it_begin)/2;
//                 future<size_type> left_tree_size = capacity_helper(it_begin,
//                                                              it_begin + mid);
//                 future<size_type> right_tree_size = hpx::async(
//                                                 launch::async,
//                                                 hpx::util::bind(
//                                                     &vector::capacity_helper,
//                                                     this,
//                                                     (it_begin + mid),
//                                                     it_end
//                                                                 )
//                                                         );
//
//                 return hpx::lcos::local::dataflow(
//                             [](future<size_type> s1, future<size_type> s2) -> size_type
//                             {
//                                 return s1.get() + s2.get();
//                             },
//                             std::move(left_tree_size),
//                             std::move(right_tree_size)
//                                                  );
//             }
//         }//end of capacity_helper

    public:
        /// Default Constructor which create hpx::vector with
        /// \a num_partitions = 1 and \a partition_size = 0. Hence overall size
        /// of the vector is 0.
        ///
        vector(char const* symbolic_name = 0)
          : size_(0),
            block_size_(std::size_t(-1)),
            policy_(distribution_policy::block)
        {
            if (symbolic_name)
                connect_to(symbolic_name).get();
        }

        vector(std::string const& symbolic_name)
          : size_(0),
            block_size_(std::size_t(-1)),
            policy_(distribution_policy::block)
        {
            connect_to(symbolic_name).get();
        }

        /// Constructor which create hpx::vector with the given overall \a size
        ///
        /// \param size   The overall size of the vector
        ///
        explicit vector(size_type size, char const* symbolic_name = 0)
          : size_(size),
            block_size_(std::size_t(-1)),
            policy_(distribution_policy::block)
        {
            if (size != 0)
                create(hpx::block);

            if (symbolic_name)
                register_as(symbolic_name).get();
        }

        vector(size_type size, std::string const& symbolic_name)
          : size_(size),
            block_size_(std::size_t(-1)),
            policy_(distribution_policy::block)
        {
            if (size != 0)
                create(hpx::block);

            register_as(symbolic_name).get();
        }

        /// Constructor which create and initialize vector with the
        /// given \a where all elements are initialized with \a val.
        ///
        /// \param size   The overall size of the vector
        /// \param val    Default value for the elements in vector
        ///
        vector(size_type size, T const& val, char const* symbolic_name = 0)
          : size_(size),
            block_size_(std::size_t(-1)),
            policy_(distribution_policy::block)
        {
            if (size != 0)
                create(val, hpx::block);

            if (symbolic_name)
                register_as(symbolic_name).get();
        }

        vector(size_type size, T const& val, std::string const& symbolic_name)
          : size_(size),
            block_size_(std::size_t(-1)),
            policy_(distribution_policy::block)
        {
            if (size != 0)
                create(val, hpx::block);

            register_as(symbolic_name).get();
        }

        /// Constructor which create and initialize vector of size
        /// \a size using the given distribution policy.
        ///
        /// \param size   The overall size of the vector
        /// \param policy The distribution policy to use (default: block)
        ///
        template <typename DistPolicy>
        vector(size_type size, DistPolicy const& policy,
                char const* symbolic_name = 0)
          : size_(size),
            block_size_(policy.get_block_size()),
            policy_(policy.get_policy_type())
        {
            if (size != 0)
                create(policy);

            if (symbolic_name)
                register_as(symbolic_name).get();
        }

        template <typename DistPolicy>
        vector(size_type size, DistPolicy const& policy,
                std::string const& symbolic_name)
          : size_(size),
            block_size_(policy.get_block_size()),
            policy_(policy.get_policy_type())
        {
            if (size != 0)
                create(policy);

            register_as(symbolic_name).get();
        }

        /// Constructor which create and initialize vector with the
        /// given \a where all elements are initialized with \a val and
        /// using the given distribution policy.
        ///
        /// \param size   The overall size of the vector
        /// \param val    Default value for the elements in vector
        /// \param policy The distribution policy to use (default: block)
        ///
        template <typename DistPolicy>
        vector(size_type size, T const& val, DistPolicy const& policy,
                char const* symbolic_name = 0)
          : size_(size),
            block_size_(policy.get_block_size()),
            policy_(policy.get_policy_type())
        {
            if (size != 0)
                create(val, policy);

            if (symbolic_name)
                register_as(symbolic_name).get();
        }

        template <typename DistPolicy>
        vector(size_type size, T const& val, DistPolicy const& policy,
                std::string const& symbolic_name)
          : size_(size),
            block_size_(policy.get_block_size()),
            policy_(policy.get_policy_type())
        {
            if (size != 0)
                create(val, policy);

            register_as(symbolic_name).get();
        }

    public:
        /// \brief Array subscript operator. This does not throw any exception.
        ///
        /// \param pos Position of the element in the vector [Note the first
        ///             position in the partition is 0]
        ///
        /// \return Returns the value of the element at position represented by
        ///         \a pos.
        ///
        /// \note The non-const version of is operator returns a proxy object
        ///       instead of a real reference to the element.
        ///
        detail::vector_value_proxy<T> operator[](size_type pos)
        {
            return detail::vector_value_proxy<T>(*this, pos);
        }
        T operator[](size_type pos) const
        {
            return get_value(pos);
        }

//         /** @brief Copy assignment operator.
//          *
//          *  @param other    This the hpx::vector object which is to be copied
//          *
//          *  @return This return the reference to the newly created vector
//          */
//         vector& operator=(vector const& other)
//         {
//             this->partitions_ = other.partitions_;
//             return *this;
//         }

        ///////////////////////////////////////////////////////////////////////
        // Capacity related API's in vector class
        ///////////////////////////////////////////////////////////////////////

        /// \brief Compute the size as the number of elements it contains.
        ///
        /// \return Return the number of elements in the vector
        ///
        size_type size() const
        {
            return size_;
        }

//         /** @brief Asynchronous API for size().
//          *
//          * @return This return the hpx::future of return value of size()
//          */
//         future<size_type> size_async() const
//         {
//             HPX_ASSERT(partitions_.size() > 1);
//             //Here end -1 is because we have the LAST in the vector
//             return size_helper(partitions_.begin(),
//                                partitions_.end() - 1);
//         }
//
//         //MAX_SIZE
//         /**  @brief Compute the maximum size of hpx::vector in terms of
//          *           number of elements.
//          *  @return Return maximum number of elements the vector can hold
//          */
//         size_type max_size() const
//         {
//             HPX_ASSERT(partitions_.size() > 1);
//             //Here end -1 is because we have the LAST in the vector
//             return max_size_helper(partitions_.begin(),
//                                    partitions_.end() - 1
//                                    ).get();
//         }
//
//         /**  @brief Asynchronous API for max_size().
//          *
//          *  @return Return the hpx::future of return value of max_size()
//          */
//         future<size_type> max_size_async() const
//         {
//             HPX_ASSERT(partitions_.size() > 1);
//             //Here end -1 is because we have the LAST in the vector
//             return max_size_helper(partitions_.begin(),
//                                    partitions_.end() - 1);
//         }
//
// //            //RESIZE (without value)
// //
// //            void resize(size_type n)
// //            {
// //                if(n == 0)
// //                    HPX_THROW_EXCEPTION(hpx::invalid_vector_error,
// //                                        "resize",
// //                                        "Invalid Vector: new_partition_size should be greater than zero");
// //
// //                std::vector<future<void>> resize_lazy_sync;
// //                //Resizing the vector partitions
// //                //AS we have to iterate until we hit LAST
// //                BOOST_FOREACH(partition_vector_type const& p, std::make_pair(partitions_.begin(), partitions_.end() - 1) )
// //                {
// //                    resize_lazy_sync.push_back(partition_vector_stub::resize_async((p.second).get(), n));
// //                }
// //                HPX_ASSERT(partitions_.size() > 1); //As this function changes the size we should have LAST always.
// //                //waiting for the resizing
// //                hpx::wait_all(resize_lazy_sync);
// //                adjust_base_index(partitions_.begin(), partitions_.end() - 1, n);
// //            }
// //            future<void> resize_async(size_type n)
// //            {
// //                //static_cast to resolve ambiguity of the overloaded function
// //                return hpx::async(launch::async, hpx::util::bind(static_cast<void(vector::*)(std::size_t)>(&vector::resize), this, n));
// //            }
//
//         // RESIZE (with value)
//         // SEMANTIC DIFFERENCE:
//         //    It is resize with respective partition not whole vector
//         /** @brief Resize each partition so that it contain n elements. If
//          *          the \a val is not it use default constructor instead.
//          *
//          *  This function resize the each partition so that it contains \a n
//          *   elements. [Note that the \a n does not represent the total size of
//          *   vector it is the size of each partition. This mean if \a n is 10 and
//          *   num_partitions is 5 then total size of vector after resize is 10*5 = 50]
//          *
//          *  @param n    New size of the each partition
//          *  @param val  Value to be copied if \a n is greater than the current
//          *               size [Default is 0]
//          *
//          *  @exception hpx::invalid_vector_error If the \a n is equal to zero
//          *              then it throw \a hpx::invalid_vector_error exception.
//          */
//         void resize(size_type n, VALUE_TYPE const& val = VALUE_TYPE())
//         {
//             if(n == 0)
//                 HPX_THROW_EXCEPTION(
//                     hpx::invalid_vector_error,
//                     "resize",
//                     "Invalid Vector: new_partition_size should be greater than zero"
//                                     );
//
//             std::vector<future<void>> resize_lazy_sync;
//             BOOST_FOREACH(partition_description_type const& p,
//                           std::make_pair(partitions_.begin(),
//                                          partitions_.end() - 1)
//                          )
//             {
//                 resize_lazy_sync.push_back(
//                                 partition_vector_stub::resize_async(
//                                                         (p.first).get(),
//                                                          n,
//                                                          val)
//                                            );
//             }
//             hpx::wait_all(resize_lazy_sync);
//
//             //To maintain the consistency in the base_index of each partition_description_type.
//             adjust_base_index(partitions_.begin(),
//                               partitions_.end() - 1,
//                               n);
//         }
//
//         /** @brief Asynchronous API for resize().
//          *
//          *  @param n    New size of the each partition
//          *  @param val  Value to be copied if \a n is greater than the current size
//          *               [Default is 0]
//          *
//          *  @return This return the hpx::future of type void [The void return
//          *           type can help to check whether the action is completed or
//          *           not]
//          *
//          *  @exception hpx::invalid_vector_error If the \a n is equal to zero
//          *              then it throw \a hpx::invalid_vector_error exception.
//          */
//         future<void> resize_async(size_type n,
//                                  VALUE_TYPE const& val = VALUE_TYPE())
//         {
//             //static_cast to resolve ambiguity of the overloaded function
//             return hpx::async(launch::async,
//                               hpx::util::bind(
//                                 static_cast<
//                                 void(vector::*)(size_type,
//                                                 VALUE_TYPE const&)
//                                             >
//                                             (&vector::resize),
//                                               this,
//                                               n,
//                                               val)
//                               );
//         }
//
//         //CAPACITY
//
//         /** @brief Compute the size of currently allocated storage capacity for
//          *          vector.
//          *
//          *  @return Returns capacity of vector, expressed in terms of elements
//          */
//         size_type capacity() const
//         {
//             HPX_ASSERT(partitions_.size() > 1);
//             //Here end -1 is because we have the LAST in the vector
//             return capacity_helper(partitions_.begin(),
//                                    partitions_.end() - 1
//                                    ).get();
//         }
//
//         /** @brief Asynchronous API for capacity().
//          *
//          *  @return Returns the hpx::future of return value of capacity()
//          */
//         future<size_type> capacity_async() const
//         {
//             HPX_ASSERT(partitions_.size() > 1);
//             //Here end -1 is because we have the LAST in the vector
//             return capacity_helper(partitions_.begin(),
//                                    partitions_.end() - 1);
//         }
//
//         //EMPTY
//         /** @brief Return whether the vector is empty.
//          *
//          *  @return Return true if vector size is 0, false otherwise
//          */
//         bool empty() const
//         {
//             return !(this->size());
//         }
//
//         /** @brief Asynchronous API for empty().
//          *
//          *  @return The hpx::future of return value empty()
//          */
//         future<bool> empty_async() const
//         {
//             return hpx::async(launch::async,
//                               hpx::util::bind(&vector::empty, this));
//         }
//
//         //RESERVE
//         /** @brief Request the change in each partition capacity so that it
//          *          can hold \a n elements. Throws the
//          *          \a hpx::partition_error exception.
//          *
//          *  This function request for each partition capacity should be at
//          *   least enough to contain \a n elements. For all partition in vector
//          *   if its capacity is less than \a n then their reallocation happens
//          *   to increase their capacity to \a n (or greater). In other cases
//          *   the partition capacity does not got affected. It does not change the
//          *   partition size. Hence the size of the vector does not affected.
//          *
//          * @param n Minimum capacity of partition
//          *
//          * @exception hpx::partition_error If \a n is greater than maximum size for
//          *             at least one partition then function throw
//          *             \a hpx::partition_error exception.
//          */
//         void reserve(size_type n)
//         {
//             std::vector<future<void>> reserve_lazy_sync;
//             BOOST_FOREACH(partition_description_type const& p,
//                           std::make_pair(partitions_.begin(),
//                                          partitions_.end() - 1)
//                           )
//             {
//                 reserve_lazy_sync.push_back(
//                         partition_vector_stub::reserve_async((p.first).get(), n)
//                                             );
//             }
//             hpx::wait_all(reserve_lazy_sync);
//         }
//
//         /** @brief Asynchronous API for reserve().
//          *
//          *  @param n Minimum capacity of partition
//          *
//          *  @return This return the hpx::future of type void [The void return
//          *           type can help to check whether the action is completed or
//          *           not]
//          *
//          *  @exception hpx::partition_error If \a n is greater than maximum size
//          *              for at least one partition then function throw
//          *              \a hpx::partition_error exception.
//          */
//         future<void> reserve_async(size_type n)
//         {
//             return hpx::async(launch::async,
//                               hpx::util::bind(&vector::reserve, this, n));
//         }

        //
        //  Element access API's in vector class
        //

        /// Returns the element at position \a pos in the vector container.
        ///
        /// \param pos Position of the element in the vector [Note the first
        ///         position in the partition is 0]
        ///
        /// \return Returns the value of the element at position represented by
        ///         \a pos.
        ///
        T get_value(size_type pos) const
        {
            return get_value_async(pos).get();
        }

        /// Asynchronous API for get_value().
        ///
        /// \param pos Position of the element in the vector
        ///
        /// \return Returns the hpx::future to value of the element at position
        ///         represented by \a pos.
        ///
        future<T> get_value_async(size_type pos) const
        {
            std::size_t part = get_partition(pos);
            std::size_t index = get_local_index(pos);
            return partition_vector_client(partitions_[part].partition_)
                .get_value_async(index);
        }

//         //FRONT (never throws exception)
//         /** @brief Access the value of first element in the vector.
//          *
//          *  Calling the function on empty container cause undefined behavior.
//          *
//          * @return Return the value of the first element in the vector
//          */
//         VALUE_TYPE front() const
//         {
//             return partition_vector_stub::front_async(
//                                     (partitions_.front().first).get()
//                                                   ).get();
//         }//end of front_value
//
//         /** @brief Asynchronous API for front().
//          *
//          *  Calling the function on empty container cause undefined behavior.
//          *
//          * @return Return the hpx::future to return value of front()
//          */
//         hpx::future< VALUE_TYPE > front_async() const
//         {
//             return partition_vector_stub::front_async(
//                                     (partitions_.front().first).get()
//                                                   );
//         }//end of front_async
//
//         //BACK (never throws exception)
//         /** @brief Access the value of last element in the vector.
//          *
//          *  Calling the function on empty container cause undefined behavior.
//          *
//          * @return Return the value of the last element in the vector
//          */
//         VALUE_TYPE back() const
//         {
//             // As the LAST pair is there and then decrement operator to that
//             // LAST is undefined hence used the end() function rather than back()
//             return partition_vector_stub::back_async(
//                             ((partitions_.end() - 2)->first).get()
//                                                  ).get();
//         }//end of back_value
//
//         /** @brief Asynchronous API for back().
//          *
//          *  Calling the function on empty container cause undefined behavior.
//          *
//          * @return Return hpx::future to the return value of back()
//          */
//         hpx::future< VALUE_TYPE > back_async() const
//         {
//             //As the LAST pair is there
//             return partition_vector_stub::back_async(
//                             ((partitions_.end() - 2)->first).get()
//                                                  );
//         }//end of back_async
//
//         //
//         // Modifier component action
//         //
//
//         //ASSIGN
//         /** @brief Assigns new contents to each partition, replacing its
//          *          current contents and modifying each partition size
//          *          accordingly.
//          *
//          *  @param n     New size of each partition
//          *  @param val   Value to fill the partition with
//          *
//          *  @exception hpx::invalid_vector_error If the \a n is equal to zero
//          *              then it throw \a hpx::invalid_vector_error exception.
//          */
//         void assign(size_type n, VALUE_TYPE const& val)
//         {
//             if(n == 0)
//                 HPX_THROW_EXCEPTION(
//                     hpx::invalid_vector_error,
//                     "assign",
//                     "Invalid Vector: new_partition_size should be greater than zero"
//                                     );
//
//             std::vector<future<void>> assign_lazy_sync;
//             BOOST_FOREACH(partition_description_type const& p,
//                           std::make_pair(partitions_.begin(),
//                                          partitions_.end() - 1)
//                           )
//             {
//                 assign_lazy_sync.push_back(
//                     partition_vector_stub::assign_async((p.first).get(), n, val)
//                                           );
//             }
//             hpx::wait_all(assign_lazy_sync);
//             adjust_base_index(partitions_.begin(),
//                               partitions_.end() - 1,
//                               n);
//         }//End of assign
//
//         /** @brief Asynchronous API for assign().
//          *
//          *  @param n     New size of each partition
//          *  @param val   Value to fill the partition with
//          *
//          *  @exception hpx::invalid_vector_error If the \a n is equal to zero
//          *              then it throw \a hpx::invalid_vector_error exception.
//          *
//          *  @return This return the hpx::future of type void [The void return
//          *           type can help to check whether the action is completed or
//          *           not]
//          */
//         future<void> assign_async(size_type n, VALUE_TYPE const& val)
//         {
//             return hpx::async(launch::async,
//                               hpx::util::bind(&vector::assign,
//                                               this,
//                                               n,
//                                               val)
//                               );
//         }
//
//         //PUSH_BACK
//         /** @brief Add new element at the end of vector. The added element
//          *          contain the \a val as value.
//          *
//          *  The value is added to the back to the last partition.
//          *
//          *  @param val Value to be copied to new element
//          */
//         void push_back(VALUE_TYPE const& val)
//         {
//             partition_vector_stub::push_back_async(
//                             ((partitions_.end() - 2 )->first).get(),
//                                                 val
//                                                 ).get();
//         }
//
//         /** @brief Asynchronous API for push_back().
//          *
//          *  @param val Value to be copied to new element
//          *
//          *  @return This return the hpx::future of type void [The void return
//          *           type can help to check whether the action is completed or
//          *           not]
//          */
//         future<void> push_back_async(VALUE_TYPE const& val)
//         {
//             return partition_vector_stub::push_back_async(
//                             ((partitions_.end() - 2)->first).get(),
//                                                         val
//                                                         );
//         }
//
//         //PUSH_BACK (With rval)
//         /** @brief Add new element at the end of vector. The added element
//          *          contain the \a val as value.
//          *
//          *  The value is added to the back to the last partition.
//          *
//          *  @param val Value to be moved to new element
//          */
//         void push_back(VALUE_TYPE const&& val)
//         {
//             partition_vector_stub::push_back_rval_async(
//                             ((partitions_.end() - 2)->first).get(),
//                                                     std::move(val)
//                                                      ).get();
//         }
//
//         /** @brief Asynchronous API for push_back(VALUE_TYPE const&& val).
//          *
//          *  @param val Value to be moved to new element
//          */
//         future<void> push_back_async(VALUE_TYPE const&& val)
//         {
//             return partition_vector_stub::push_back_rval_async(
//                             ((partitions_.end() - 2)->first).get(),
//                                                     std::move(val)
//                                                             );
//         }
//
//         //POP_BACK (Never throw exception)
// //            void pop_back()
// //            {
// //                partition_vector_stub::pop_back_async(( (partitions_.end() - 2)->second).get()).get();
// //                //TODO if following change the affect back() and further pop_back function
// //                //checking if last element from the particular gid is popped up then delete that..
// //                // (-2)I am retaining one gid in vector as otherwise it goes to invalid state and it makes a compulsion that we need to keep at least one element that is not good
// //                if(partition_vector_stub::empty_async(( (partitions_.end() - 2)->second).get()).get() && partitions_.size() > 2)
// //                    partitions_.pop_back();
// //                HPX_ASSERT(partitions_.size() > 1); //As this function changes the size we should have LAST always.
// //            }
//

        /// Copy the value of \a val in the element at position \a pos in
        /// the vector container.
        ///
        /// @param pos   Position of the element in the vector
        /// @param val   The value to be copied
        ///
        template <typename T_>
        void set_value(size_type pos, T_ && val)
        {
            set_value_async(pos, std::forward<T_>(val)).get();
        }

        /// Asynchronous API for set_value().
        ///
        /// \param pos   Position of the element in the vector
        /// \param val   The value to be copied
        ///
        template <typename T_>
        future<void> set_value_async(size_type pos, T_ && val)
        {
            std::size_t part = get_partition(pos);
            std::size_t index = get_local_index(pos);
            return partition_vector_client(partitions_[part].partition_)
                .set_value_async(index, std::forward<T_>(val));
        }

//             //CLEAR
//             //TODO if number of partitions is kept constant every time then clear should modified (clear each partition_vector one by one).
// //            void clear()
// //            {
// //                //It is keeping one gid hence iterator does not go in an invalid state
// //                partitions_.erase(partitions_.begin() + 1,
// //                                           partitions_.end()-1);
// //                partition_vector_stub::clear_async((partitions_[0].second).get()).get();
// //                HPX_ASSERT(partitions_.size() > 1); //As this function changes the size we should have LAST always.
// //            }
//
//             //
//             // HPX CUSTOM API's
//             //
//
// //            //CREATE_partition
// //            //TODO This statement can create Data Inconsistency :
// //             //If size of partitions_ calculated and added to the base_index but not whole creation is completed and in betwen this som push_back on hpx::vector is done then that operation is losted
// //            void create_partition(hpx::naming::id locality, std::size_t partition_size = 0, VALUE_TYPE val = 0.0)
// //            {
// //                partitions_.push_back(
// //                        std::make_pair(
// //                            partitions_.size(),
// //                             hpx::components::new_<partition_partition_vector_type>(locality, partition_size, val)
// //                                      )
// //                                                    );
// //            }//end of create partition

        ///////////////////////////////////////////////////////////////////////
        /// \brief Return the iterator at the beginning of the vector.
        iterator begin()
        {
            return iterator(this, 0);
        }

        /// \brief Return the const_iterator at the beginning of the vector.
        const_iterator begin() const
        {
            return const_iterator(this, 0);
        }

        /// \brief Return the iterator at the end of the vector.
        iterator end()
        {
            return iterator(this, size_);
        }

        /// \brief Return the const_iterator at the end of the vector.
        const_iterator end() const
        {
            return const_iterator(this, size_);
        }

        /// \brief Return the const_iterator at the beginning of the vector.
        const_iterator cbegin() const
        {
            return const_iterator(this, 0);
        }

        /// \brief Return the const_iterator at the end of the vector.
        const_iterator cend() const
        {
            return const_iterator(this, size_);
        }

        ///////////////////////////////////////////////////////////////////////
        segment_iterator
        segment_begin(boost::uint32_t id = invalid_locality_id)
        {
            return segment_iterator(this, partitions_.begin(),
                partitions_.end(), id);
        }

        const_segment_iterator
        segment_begin(boost::uint32_t id = invalid_locality_id) const
        {
            return const_segment_iterator(this, partitions_.cbegin(),
                partitions_.cend(), id);
        }

        const_segment_iterator
        segment_cbegin(boost::uint32_t id = invalid_locality_id) const
        {
            return const_segment_iterator(this, partitions_.cbegin(),
                partitions_.cend(), id);
        }

        segment_iterator
        segment_end(boost::uint32_t id = invalid_locality_id)
        {
            return segment_iterator(this, partitions_.end());
        }

        const_segment_iterator
        segment_end(boost::uint32_t id = invalid_locality_id) const
        {
            return const_segment_iterator(this, partitions_.cend());
        }

        const_segment_iterator
        segment_cend(boost::uint32_t id = invalid_locality_id) const
        {
            return const_segment_iterator(this, partitions_.cend());
        }
    };
}

#endif // VECTOR_HPP
