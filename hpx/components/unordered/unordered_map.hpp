//  Copyright (c) 2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/components/vector/vector.hpp

#if !defined(HPX_UNORDERED_MAP_NOV_11_2014_0852PM)
#define HPX_UNORDERED_MAP_NOV_11_2014_0852PM

#include <hpx/include/lcos.hpp>
#include <hpx/include/util.hpp>
#include <hpx/include/components.hpp>

#include <hpx/components/unordered/distribution_policy.hpp>
#include <hpx/components/unordered/unordered_map_segmented_iterator.hpp>
#include <hpx/components/unordered/partition_unordered_map_component.hpp>

#include <cstdint>
#include <memory>

#include <boost/cstdint.hpp>

/// The hpx::unordered_map and its API's are defined here.
///
/// The hpx::unordered_map is a segmented data structure which is a collection
/// of one or more hpx::partition_unordered_maps. The hpx::unordered_map stores
/// the global IDs of each hpx::partition_unordered_maps.

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    struct unordered_map_config_data
    {
        // Each partition is described by it's corresponding client object, its
        // size, and locality id.
        struct partition_data
        {
            partition_data()
              : locality_id_(naming::invalid_locality_id)
            {}

            partition_data(future<id_type> && part, boost::uint32_t locality_id)
              : partition_(part.share()),
                locality_id_(locality_id)
            {}

            partition_data(id_type const& part, boost::uint32_t locality_id)
              : partition_(make_ready_future(part).share()),
                locality_id_(locality_id)
            {}

            id_type get_id() const
            {
                return partition_.get();
            }

            hpx::shared_future<id_type> partition_;
            boost::uint32_t locality_id_;

        private:
            friend class boost::serialization::access;

            template <typename Archive>
            void serialize(Archive& ar, unsigned)
            {
                ar & partition_ & locality_id_;
            }
        };

        unordered_map_config_data()
          : policy_(0)
        {}

        unordered_map_config_data(std::vector<partition_data> const& partitions,
                int policy)
          : partitions_(std::move(partitions)),
            policy_(policy)
        {}

        std::vector<partition_data> partitions_;
        int policy_;

    private:
        friend class boost::serialization::access;

        template <typename Archive>
        void serialize(Archive& ar, unsigned)
        {
            ar & partitions_ & policy_;
        }
    };
}}

HPX_DISTRIBUTED_METADATA_DECLARATION(hpx::server::unordered_map_config_data,
    hpx_server_unordered_map_config_data);

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    /// This is the unordered_map class which defines hpx::unordered_map
    /// functionality.
    ///
    ///  This contains the client side implementation of the hpx::unordered_map.
    ///  This class defines the synchronous and asynchronous API's for each of
    ///  the exposed functionalities.
    ///
    template <typename Key, typename T, typename Hash, typename KeyEqual>
    class unordered_map
      : hpx::components::client_base<
            unordered_map<Key, T, Hash, KeyEqual>,
            server::unordered_map_config_data>
    {
    public:
        typedef std::allocator<T> allocator_type;

        typedef std::size_t size_type;
        typedef std::ptrdiff_t difference_type;

        typedef T value_type;
        typedef T reference;
        typedef T const const_reference;

#if !defined(HPX_GCC_VERSION) || HPX_GCC_VERSION >= 40700
        typedef typename std::allocator_traits<allocator_type>::pointer pointer;
        typedef typename std::allocator_traits<allocator_type>::const_pointer
            const_pointer;
#else
        typedef T* pointer;
        typedef T const* const_pointer;
#endif

    private:
        typedef hpx::components::client_base<
                unordered_map, server::unordered_map_config_data
            > base_type;

        // The list of partitions belonging to this vector.
        //
        // Each partition is described by it's corresponding client object, its
        // size, and locality id.
        typedef std::vector<server::unordered_map_config_data::partition_data>
            partitions_vector_type;

        // This is the vector representing the base_index and corresponding
        // global ID's of the underlying partition_vectors.
        partitions_vector_type partitions_;

        // parameters taken from distribution policy
        BOOST_SCOPED_ENUM(unordered_distribution_policy) policy_; // policy to use

        // will be set for created (non-attached) objects
        std::string registered_name_;

        ///////////////////////////////////////////////////////////////////////
        // Connect this vector to the existing vector using the given symbolic
        // name.
        void get_data_helper(id_type id,
            future<server::unordered_map_config_data> && f)
        {
            server::unordered_map_config_data data = f.get();

            std::swap(partitions_, data.partitions_);
            policy_ = static_cast<BOOST_SCOPED_ENUM(unordered_distribution_policy)>(
                data.policy_);
            base_type::reset(std::move(id));
        }

        // this will be called by the base class once the registered id becomes
        // available
        future<void> connect_to_helper(future<id_type> && f)
        {
            using util::placeholders::_1;
            typedef typename base_type::server_component_type::get_action act;

            id_type id = f.get();
            return async(act(), id).then(
                util::bind(&unordered_map::get_data_helper, this, id, _1));
        }

    public:
        future<void> connect_to(std::string const& symbolic_name)
        {
            using util::placeholders::_1;
            return base_type::connect_to(symbolic_name,
                util::bind(&unordered_map::connect_to_helper, this, _1));
        }

        // Register this vector with AGAS using the given symbolic name
        future<void> register_as(std::string const& symbolic_name)
        {
            server::unordered_map_config_data data(partitions_, int(policy_));

            base_type::reset(hpx::new_<
                    typename base_type::server_component_type> >(
                hpx::find_here(), std::move(data)));

            registered_name_ = symbolic_name;
            return base_type::register_as(symbolic_name);
        }

    public:
        /// Default Constructor which create hpx::unordered_map with
        /// \a num_partitions = 1 and \a partition_size = 0. Hence overall size
        /// of the unordered_map is 0.
        unordered_map()
          : policy_(unordered_distribution_policy::hash)
        {
        }

        explicit unordered_map(std::size_t bucket_count,
                Hash const& hash = Hash(), KeyEqual const& equal = KeyEqual())
          : policy_(unordered_distribution_policy::hash)
        {
            create(bucket_count, hpx::hash);
        }

        template <typename DistPolicy>
        unordered_map(DistPolicy const& policy,
                typename boost::enable_if<
                        is_unordered_distribution_policy<DistPolicy>
                    >::type* = 0)
          : policy_(policy_.get_policy_type())
        {
            create(bucket_count, policy);
        }

        template <typename DistPolicy>
        unordered_map(std::size_t bucket_count,
                DistPolicy const& policy, Hash const& hash = Hash(),
                KeyEqual const& equal = KeyEqual(),
                typename boost::enable_if<
                        is_unordered_distribution_policy<DistPolicy>
                    >::type* = 0)
          : policy_(policy_.get_policy_type())
        {
            create(bucket_count, policy);
        }

        /// Construct a new unordered_map representation from the data
        /// associated with the given symbolic name.
        unordered_map(std::string const& symbolic_name)
        {
            connect_to(symbolic_name).get();
        }

        /// Construct a new unordered_map
        unordered_map(std::size_t bucket_count)
          : policy_(unordered_distribution_policy::hash)
        {
            create(bucket_count, hpx::hash);
        }

        unordered_map(unordered_map const& rhs)
          : policy_(rhs.policy_)
        {}

        unordered_map(unordered_map && rhs)
          : policy_(rhs.policy_)
        {}

        ~unordered_map()
        {}

        unordered_map& operator=(unordered_map const& rhs)
        {
            return *this;
        }
        unordered_map& operator=(unordered_map && rhs)
        {
            return *this;
        }

        /// \brief Array subscript operator. This does not throw any exception.
        ///
        /// \param pos Position of the element in the vector [Note the first
        ///            position in the partition is 0]
        ///
        /// \return Returns the value of the element at position represented by
        ///         \a pos.
        ///
        /// \note The non-const version of is operator returns a proxy object
        ///       instead of a real reference to the element.
        ///
        detail::unordered_map_value_proxy<Key, T, Hash, KeyEqual>
        operator[](Key const& pos)
        {
            return detail::unordered_map_value_proxy<
                    Key, T, Hash, KeyEqual
                >(*this, pos);
        }
        T operator[](Key const& pos) const
        {
            return get_value_sync(pos);
        }

        /// Returns the element at position \a pos in the vector container.
        ///
        /// \param pos Position of the element in the vector
        ///
        /// \return Returns the value of the element at position represented by
        ///         \a pos.
        ///
        T get_value_sync(Key const& pos) const
        {
            return get_value_sync(get_partition(pos), pos);
        }

        /// Returns the element at position \a pos in the vector container.
        ///
        /// \param part  Sequence number of the partition
        /// \param pos   Position of the element in the partition
        ///
        /// \return Returns the value of the element at position represented by
        ///         \a pos.
        ///
        T get_value_sync(size_type part, Key const& pos) const
        {
            partition_data const& part_data = partitions_[part];
            if (part_data.local_data_)
                return part_data.local_data_->get_value(pos);

            return partition_unordered_map_client(part_data.partition_)
                .get_value_sync(pos);
        }

        /// Returns the element at position \a pos in the vector container
        /// asynchronously.
        ///
        /// \param pos Position of the element in the vector
        ///
        /// \return Returns the hpx::future to value of the element at position
        ///         represented by \a pos.
        ///
        future<T> get_value(Key const& pos) const
        {
            return get_value(get_partition(pos), pos);
        }

        /// Returns the element at position \a pos in the given partition in
        /// the vector container asynchronously.
        ///
        /// \param part  Sequence number of the partition
        /// \param pos   Position of the element in the partition
        ///
        /// \return Returns the hpx::future to value of the element at position
        ///         represented by \a pos.
        ///
        future<T> get_value(size_type part, Key const& pos) const
        {
            if (partitions_[part].local_data_)
            {
                return make_ready_future(
                    partitions_[part].local_data_->get_value(pos));
            }

            return partition_unordered_map_client(partitions_[part].partition_)
                .get_value(pos);
        }

        /// Copy the value of \a val in the element at position \a pos in
        /// the vector container.
        ///
        /// \param pos   Position of the element in the vector
        /// \param val   The value to be copied
        ///
        template <typename T_>
        void set_value_sync(Key const& pos, T_ && val)
        {
            return set_value_sync(get_partition(pos), pos,
                std::forward<T_>(val));
        }

        /// Copy the value of \a val in the element at position \a pos in
        /// the vector container.
        ///
        /// \param part  Sequence number of the partition
        /// \param pos   Position of the element in the partition
        /// \param val   The value to be copied
        ///
        template <typename T_>
        void set_value_sync(size_type part, Key const& pos, T_ && val)
        {
            partition_data const& part_data = partitions_[part];
            if (part_data.local_data_)
            {
                part_data.local_data_->set_value(pos, std::forward<T_>(val));
            }
            else
            {
                partition_unordered_map_client(part_data.partition_)
                    .set_value_sync(pos, std::forward<T_>(val));
            }
        }

        /// Asynchronous set the element at position \a pos of the partition
        /// \a part to the given value \a val.
        ///
        /// \param pos   Position of the element in the vector
        /// \param val   The value to be copied
        ///
        /// \return This returns the hpx::future of type void which gets ready
        ///         once the operation is finished.
        ///
        template <typename T_>
        future<void> set_value(Key const& pos, T_ && val)
        {
            return set_value(get_partition(pos), pos, std::forward<T_>(val));
        }

        /// Asynchronously set the element at position \a pos in
        /// the partition \part to the given value \a val.
        ///
        /// \param part  Sequence number of the partition
        /// \param pos   Position of the element in the partition
        /// \param val   The value to be copied
        ///
        /// \return This returns the hpx::future of type void which gets ready
        ///         once the operation is finished.
        ///
        template <typename T_>
        future<void> set_value(size_type part, Key const& pos, T_ && val)
        {
            partition_data const& part_data = partitions_[part];
            if (part_data.local_data_)
            {
                part_data.local_data_->set_value(pos, std::forward<T_>(val));
                return make_ready_future();
            }

            return partition_unordered_map_client(part_data.partition_)
                .set_value(pos, std::forward<T_>(val));
        }

        /// \brief Compute the size as the number of elements it contains.
        ///
        /// \return Return the number of elements in the vector
        ///
        std::size_t size() const
        {
            return 0;
        }
    };
}

#endif
