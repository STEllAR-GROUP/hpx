//  Copyright (c) 2014 Anuj R. Sharma
//  Copyright (c) 2014-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/components/partitioned_vector/partitioned_vector.hpp

#ifndef HPX_PARTITIONED_VECTOR_HPP
#define HPX_PARTITIONED_VECTOR_HPP

#include <hpx/include/lcos.hpp>
#include <hpx/include/util.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/serialization.hpp>

#include <hpx/components/containers/container_distribution_policy.hpp>
#include <hpx/components/containers/partitioned_vector/partitioned_vector_segmented_iterator.hpp>
#include <hpx/components/containers/partitioned_vector/partitioned_vector_component.hpp>

#include <cstdint>
#include <memory>
#include <iterator>
#include <algorithm>
#include <type_traits>

#include <boost/cstdint.hpp>

///////////////////////////////////////////////////////////////////////////////
/// \cond NOINTERNAL
namespace hpx { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    struct partitioned_vector_config_data
    {
        // Each partition is described by it's corresponding client object, its
        // size, and locality id.
        struct partition_data
        {
            partition_data()
              : size_(0), locality_id_(naming::invalid_locality_id)
            {}

            partition_data(id_type const& part, std::size_t size,
                    boost::uint32_t locality_id)
              : partition_(part),
                size_(size), locality_id_(locality_id)
            {}

            id_type const& get_id() const
            {
                return partition_;
            }

            hpx::id_type partition_;
            std::size_t size_;
            boost::uint32_t locality_id_;

        private:
            friend class hpx::serialization::access;

            template <typename Archive>
            void serialize(Archive& ar, unsigned)
            {
                ar & partition_ & size_ & locality_id_;
            }
        };

        partitioned_vector_config_data()
          : size_(0)
        {}

        partitioned_vector_config_data(std::size_t size,
                std::vector<partition_data> && partitions)
          : size_(size),
            partitions_(std::move(partitions))
        {}

        std::size_t size_;
        std::vector<partition_data> partitions_;

    private:
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive& ar, unsigned)
        {
            ar & size_ & partitions_;
        }
    };
}}

HPX_DISTRIBUTED_METADATA_DECLARATION(
    hpx::server::partitioned_vector_config_data,
    hpx_server_partitioned_vector_config_data);

/// \endcond

namespace hpx
{
    /// hpx::partitioned_vector is a sequence container that encapsulates
    /// dynamic size arrays.
    ///
    /// \note A hpx::partitioned_vector does not stores all elements in a
    ///       contiguous block of memory. Memory is contiguous inside each of
    ///       the segmented partitions only.
    ///
    /// The hpx::partitioned_vector is a segmented data structure which is a
    /// collection of one
    /// or more hpx::server::partition_vectors. The hpx::partitioned_vector
    /// stores the global
    /// ids of each hpx::server::partition_vector and the size of each
    /// hpx::server::partition_vector.
    ///
    /// The storage of the vector is handled automatically, being expanded and
    /// contracted as needed. Vectors usually occupy more space than static arrays,
    /// because more memory is allocated to handle future growth. This way a vector
    /// does not need to reallocate each time an element is inserted, but only when
    /// the additional memory is exhausted.
    ///
    ///  This contains the client side implementation of the
    ///  hpx::partitioned_vector. This
    ///  class defines the synchronous and asynchronous API's for each of the
    ///  exposed functionalities.
    ///
    /// \tparam T   The type of the elements. The requirements that are imposed
    ///             on the elements depend on the actual operations performed
    ///             on the container. Generally, it is required that element type
    ///             is a complete type and meets the requirements of Erasable,
    ///             but many member functions impose stricter requirements.
    ///
    template <typename T>
    class partitioned_vector
      : hpx::components::client_base<partitioned_vector<T>,
            hpx::components::server::distributed_metadata_base<
                server::partitioned_vector_config_data> >
    {
    public:
        typedef std::allocator<T> allocator_type;

        typedef std::size_t size_type;
        typedef std::ptrdiff_t difference_type;

        typedef T value_type;
        typedef T reference;
        typedef T const const_reference;

#if (defined(HPX_GCC_VERSION) && HPX_GCC_VERSION < 40700) || defined(HPX_NATIVE_MIC)
        typedef T* pointer;
        typedef T const* const_pointer;
#else
        typedef typename std::allocator_traits<allocator_type>::pointer pointer;
        typedef typename std::allocator_traits<allocator_type>::const_pointer
            const_pointer;
#endif

    private:
        typedef hpx::components::client_base<
                partitioned_vector,
                hpx::components::server::distributed_metadata_base<
                    server::partitioned_vector_config_data>
            > base_type;

        typedef hpx::server::partitioned_vector<T> partition_vector_server;
        typedef hpx::partition_vector<T> partition_vector_client;

        struct partition_data
          : server::partitioned_vector_config_data::partition_data
        {
            typedef server::partitioned_vector_config_data::partition_data base_type;

            partition_data(id_type const& part, std::size_t size,
                    boost::uint32_t locality_id)
              : base_type(part, size, locality_id)
            {}

            partition_data(base_type && base)
              : base_type(std::move(base))
            {}

            boost::shared_ptr<partition_vector_server> local_data_;
        };

        // The list of partitions belonging to this vector.
        // Each partition is described by it's corresponding client object, its
        // size, and locality id.
        typedef std::vector<partition_data> partitions_vector_type;

        size_type size_;                // overall size of the vector
        size_type partition_size_;      // cached partition size

        // This is the vector representing the base_index and corresponding
        // global ID's of the underlying partition_vectors.
        partitions_vector_type partitions_;

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

        typedef local_segment_vector_iterator<
                T, typename partitions_vector_type::iterator
            > local_segment_iterator;
        typedef local_segment_vector_iterator<
                T, typename partitions_vector_type::const_iterator
            > const_local_segment_iterator;

    private:
        friend class vector_iterator<T>;
        friend class const_vector_iterator<T>;

        friend class segment_vector_iterator<
            T, typename partitions_vector_type::iterator>;
        friend class const_segment_vector_iterator<
            T, typename partitions_vector_type::const_iterator>;

        std::size_t get_partition_size() const
        {
            std::size_t num_parts = partitions_.size();
            return num_parts ? ((size_ + num_parts - 1) / num_parts) : 0;
        }

        std::size_t get_global_index(std::size_t segment,
            std::size_t part_size, size_type local_index) const
        {
            return segment * part_size + local_index;
        }

        ///////////////////////////////////////////////////////////////////////
        // Connect this vector to the existing vector using the given symbolic
        // name.
        void get_data_helper(id_type id,
            future<server::partitioned_vector_config_data> && f)
        {
            server::partitioned_vector_config_data data = f.get();

            partitions_.clear();
            partitions_.reserve(data.partitions_.size());

            size_ = data.size_;
            std::move(data.partitions_.begin(), data.partitions_.end(),
                std::back_inserter(partitions_));

            boost::uint32_t this_locality = get_locality_id();
            std::vector<future<void> > ptrs;

            typedef typename partitions_vector_type::const_iterator const_iterator;

            std::size_t l = 0;
            const_iterator end = partitions_.cend();
            for (const_iterator it = partitions_.cbegin(); it != end; ++it, ++l)
            {
                if (it->locality_id_ == this_locality)
                {
                    using util::placeholders::_1;
                    ptrs.push_back(
                        get_ptr<partition_vector_server>(it->partition_)
                        .then(
                            util::bind(&partitioned_vector::get_ptr_helper,
                                l, std::ref(partitions_), _1
                            )
                        )
                    );
                }
            }
            wait_all(ptrs);

            partition_size_ = get_partition_size();
            this->base_type::reset(std::move(id));
        }

        // this will be called by the base class once the registered id becomes
        // available
        future<void> connect_to_helper(shared_future<id_type> && f)
        {
            using util::placeholders::_1;
            typedef typename components::server::distributed_metadata_base<
                    server::partitioned_vector_config_data
                >::get_action act;

            id_type id = f.get();
            return async(act(), id).then(
                util::bind(&partitioned_vector::get_data_helper, this, id, _1));
        }

    public:
        future<void> connect_to(std::string const& symbolic_name)
        {
            using util::placeholders::_1;
            this->base_type::connect_to(symbolic_name);
            return this->base_type::share().then(
                util::bind(&partitioned_vector::connect_to_helper, this, _1));
        }

        void connect_to_sync(std::string const& symbolic_name)
        {
            connect_to(symbolic_name).get();
        }

        // Register this vector with AGAS using the given symbolic name
        future<void> register_as(std::string const& symbolic_name)
        {
            std::vector<
                server::partitioned_vector_config_data::partition_data
            > partitions;
            partitions.reserve(partitions_.size());

            std::copy(partitions_.begin(), partitions_.end(),
                std::back_inserter(partitions));

            server::partitioned_vector_config_data data(
                size_, std::move(partitions));
            this->base_type::reset(hpx::new_<
                    components::server::distributed_metadata_base<
                        server::partitioned_vector_config_data> >(
                    hpx::find_here(), std::move(data)));

            return this->base_type::register_as(symbolic_name);
        }
        void register_as_sync(std::string const& symbolic_name)
        {
            register_as(symbolic_name).get();
        }

    public:
        // Return the sequence number of the segment corresponding to the
        // given global index
        std::size_t get_partition(size_type global_index) const
        {
            if (global_index == size_)
                return partitions_.size();

            std::size_t part_size = partition_size_;
            if (part_size != 0)
                return (part_size != size_) ? (global_index / part_size) : 0;

            return partitions_.size();
        }

        // Return the local index inside the segment corresponding to the
        // given global index
        std::size_t get_local_index(size_type global_index) const
        {
            if (global_index == size_ || partition_size_ == std::size_t(-1) ||
                partition_size_ == 0)
            {
                return std::size_t(-1);
            }

            return (partition_size_ != size_) ?
                (global_index % partition_size_) : global_index;
        }

        // Return the local indices inside the segment corresponding to the
        // given global indices
        std::vector<size_type>
        get_local_indices(std::vector<size_type> indices) const
        {
            for (size_type& index: indices)
                index = get_local_index(index);
            return indices;
        }

        // Return the global index corresponding to the local index inside the
        // given segment.
        template <typename SegmentIter>
        std::size_t get_global_index(SegmentIter const& it,
            size_type local_index) const
        {
            std::size_t part_size = partition_size_;
            if (part_size == std::size_t(-1) || part_size == 0)
                return size_;

            std::size_t segment = it.base() - partitions_.cbegin();
            if (segment == partitions_.size())
                return size_;

            return get_global_index(segment, part_size, local_index);
        }

        template <typename SegmentIter>
        std::size_t get_partition(SegmentIter const& it) const
        {
            return std::distance(partitions_.begin(), it.base());
        }

        // Return the local iterator referencing an element inside a segment
        // based on the given global index.
        local_iterator get_local_iterator(size_type global_index) const
        {
            HPX_ASSERT(global_index != std::size_t(-1));

            std::size_t part = get_partition(global_index);
            if (part == partitions_.size())
            {
                // return an iterator to the end of the last partition
                return local_iterator(partitions_.back().partition_,
                    partitions_.back().size_, partitions_.back().local_data_);
            }

            std::size_t local_index = get_local_index(global_index);
            HPX_ASSERT(local_index != std::size_t(-1));

            return local_iterator(partitions_[part].partition_, local_index,
                partitions_[part].local_data_);
        }

        const_local_iterator get_const_local_iterator(size_type global_index) const
        {
            HPX_ASSERT(global_index != std::size_t(-1));

            std::size_t part = get_partition(global_index);
            if (part == partitions_.size())
            {
                // return an iterator to the end of the last partition
                return const_local_iterator(partitions_.back().partition_,
                    partitions_.back().size_, partitions_.back().local_data_);
            }

            std::size_t local_index = get_local_index(global_index);
            HPX_ASSERT(local_index != std::size_t(-1));

            return const_local_iterator(partitions_[part].partition_,
                local_index, partitions_[part].local_data_);
        }

        // Return the segment iterator referencing a segment based on the
        // given global index.
        segment_iterator get_segment_iterator(size_type global_index)
        {
            std::size_t part = get_partition(global_index);
            if (part == partitions_.size())
                return segment_iterator(partitions_.end(), this);

            return segment_iterator(partitions_.begin() + part, this);
        }

        const_segment_iterator get_const_segment_iterator(
            size_type global_index) const
        {
            std::size_t part = get_partition(global_index);
            if (part == partitions_.size())
                return const_segment_iterator(partitions_.cend(), this);

            return const_segment_iterator(partitions_.cbegin() + part, this);
        }

    protected:
        /// \cond NOINTERNAL
        typedef std::pair<hpx::id_type, std::vector<hpx::id_type> >
            bulk_locality_result;
        /// \endcond

        template <typename DistPolicy>
        static hpx::future<std::vector<bulk_locality_result> >
        create_helper1(DistPolicy const& policy, std::size_t count,
            std::size_t size)
        {
            typedef typename partition_vector_client::server_component_type
                component_type;

            return policy.template bulk_create<component_type>(
                count, size);
        }

        template <typename DistPolicy>
        static hpx::future<std::vector<bulk_locality_result> >
        create_helper2(DistPolicy const& policy, std::size_t count,
            std::size_t size, T const& val)
        {
            typedef typename partition_vector_client::server_component_type
                component_type;

            return policy.template bulk_create<component_type>(
                count, size, val);
        }

        static void get_ptr_helper(std::size_t loc,
            partitions_vector_type& partitions,
            future<boost::shared_ptr<partition_vector_server> > && f)
        {
            partitions[loc].local_data_ = f.get();
        }

        // This function is called when we are creating the vector. It
        // initializes the partitions based on the give parameters.
        template <typename DistPolicy, typename Create>
        void create(DistPolicy const& policy, Create && creator)
        {
            std::size_t num_parts =
                traits::num_container_partitions<DistPolicy>::call(policy);
            std::size_t part_size = (size_ + num_parts - 1) / num_parts;

            // create as many partitions as required
            hpx::future<std::vector<bulk_locality_result> > f =
                creator(policy, num_parts, part_size);

            // now initialize our data structures
            boost::uint32_t this_locality = get_locality_id();
            std::vector<future<void> > ptrs;

            std::size_t num_part = 0;
            std::size_t allocated_size = 0;

            std::size_t l = 0;
            for (bulk_locality_result const& r: f.get())
            {
                using naming::get_locality_id_from_id;
                boost::uint32_t locality = get_locality_id_from_id(r.first);
                for (hpx::id_type const& id: r.second)
                {
                    std::size_t size = (std::min)(part_size, size_-allocated_size);
                    partitions_.push_back(partition_data(id, size, locality));

                    if (locality == this_locality)
                    {
                        using util::placeholders::_1;
                        ptrs.push_back(
                            get_ptr<partition_vector_server>(id).then(
                                util::bind(&partitioned_vector::get_ptr_helper,
                                    l, std::ref(partitions_), _1
                                )
                            )
                        );
                    }
                    ++l;

                    allocated_size += size;
                    if (++num_part == num_parts)
                    {
                        HPX_ASSERT(allocated_size == size_);

                        // shrink last partition, if appropriate
                        if (size != part_size)
                        {
                            partition_vector_client(
                                    partitions_.back().partition_
                                ).resize(size);
                        }
                        break;
                    }
                    else
                    {
                        HPX_ASSERT(size == part_size);
                    }
                }
            }

            wait_all(ptrs);

            // cache our partition size
            partition_size_ = get_partition_size();
        }

        template <typename DistPolicy>
        void create(DistPolicy const& policy)
        {
            using util::placeholders::_1;
            using util::placeholders::_2;
            using util::placeholders::_3;

            create(policy, util::bind(
                &partitioned_vector::create_helper1<DistPolicy>, _1, _2, _3));
        }

        template <typename DistPolicy>
        void create(T const& val, DistPolicy const& policy)
        {
            using util::placeholders::_1;
            using util::placeholders::_2;
            using util::placeholders::_3;

            create(policy, util::bind(&partitioned_vector::create_helper2<DistPolicy>,
                _1, _2, _3, std::ref(val)));
        }

        // Perform a deep copy from the given vector
        void copy_from(partitioned_vector const& rhs)
        {
            typedef typename partitions_vector_type::const_iterator const_iterator;

            std::vector<future<id_type> > objs;
            const_iterator end = rhs.partitions_.end();
            for (const_iterator it = rhs.partitions_.begin(); it != end; ++it)
            {
                typedef typename partition_vector_client::server_component_type
                    component_type;
                objs.push_back(hpx::components::copy<component_type>(
                    it->partition_));
            }
            wait_all(objs);

            boost::uint32_t this_locality = get_locality_id();
            std::vector<future<void> > ptrs;

            partitions_vector_type partitions;
            partitions.reserve(rhs.partitions_.size());
            for (std::size_t i = 0; i != rhs.partitions_.size(); ++i)
            {
                boost::uint32_t locality = rhs.partitions_[i].locality_id_;

                partitions.push_back(partition_data(objs[i].get(),
                    rhs.partitions_[i].size_, locality));

                if (locality == this_locality)
                {
                    using util::placeholders::_1;
                    ptrs.push_back(get_ptr<partition_vector_server>(
                        partitions[i].partition_).then(
                            util::bind(&partitioned_vector::get_ptr_helper,
                                i, std::ref(partitions), _1)));
                }
            }

            wait_all(ptrs);

            size_ = rhs.size_;
            partition_size_ = rhs.partition_size_;
            std::swap(partitions_, partitions);
        }

    public:
        /// Default Constructor which create hpx::partitioned_vector with
        /// \a num_partitions = 0 and \a partition_size = 0. Hence overall size
        /// of the vector is 0.
        ///
        partitioned_vector()
          : size_(0),
            partition_size_(std::size_t(-1))
        {}

        /// Constructor which create hpx::partitioned_vector with the given
        /// overall \a size
        ///
        /// \param size             The overall size of the vector
        ///
        partitioned_vector(size_type size)
          : size_(size),
            partition_size_(std::size_t(-1))
        {
            if (size != 0)
                create(hpx::container_layout);
        }

        /// Constructor which create and initialize vector with the
        /// given \a where all elements are initialized with \a val.
        ///
        /// \param size             The overall size of the vector
        /// \param val              Default value for the elements in vector
        /// \param symbolic_name    The (optional) name to register the newly
        ///                         created vector
        ///
        partitioned_vector(size_type size, T const& val)
          : size_(size),
            partition_size_(std::size_t(-1))
        {
            if (size != 0)
                create(val, hpx::container_layout);
        }

        /// Constructor which create and initialize vector of size
        /// \a size using the given distribution policy.
        ///
        /// \param size             The overall size of the vector
        /// \param policy           The distribution policy to use
        /// \param symbolic_name    The (optional) name to register the newly
        ///                         created vector
        ///
        template <typename DistPolicy>
        partitioned_vector(size_type size, DistPolicy const& policy,
                typename std::enable_if<
                    traits::is_distribution_policy<DistPolicy>::value
                >::type* = 0)
          : size_(size),
            partition_size_(std::size_t(-1))
        {
            if (size != 0)
                create(policy);
        }

        /// Constructor which create and initialize vector with the
        /// given \a where all elements are initialized with \a val and
        /// using the given distribution policy.
        ///
        /// \param size             The overall size of the vector
        /// \param val              Default value for the elements in vector
        /// \param policy           The distribution policy to use
        /// \param symbolic_name    The (optional) name to register the newly
        ///                         created vector
        ///
        template <typename DistPolicy>
        partitioned_vector(size_type size, T const& val, DistPolicy const& policy,
                typename std::enable_if<
                    traits::is_distribution_policy<DistPolicy>::value
                >::type* = 0)
          : size_(size),
            partition_size_(std::size_t(-1))
        {
            if (size != 0)
                create(val, policy);
        }

        /// Copy construction performs a deep copy of the right hand side
        /// vector.
        partitioned_vector(partitioned_vector const& rhs)
          : base_type(),
            size_(0)
        {
            if (rhs.size_ != 0)
                copy_from(rhs);
        }

        partitioned_vector(partitioned_vector && rhs)
          : base_type(std::move(rhs)),
            size_(rhs.size_),
            partition_size_(rhs.partition_size_),
            partitions_(std::move(rhs.partitions_))
        {
            rhs.size_ = 0;
            rhs.partition_size_ = std::size_t(-1);
        }

    public:

        std::vector< hpx::id_type > get_partitions_ids()
        {
            std::vector< hpx::id_type > ids;

            for(auto const part_data : partitions_)
            {
                ids.push_back( part_data.partition_);
            }
            return ids;
        }

        /// \brief Array subscript operator. This does not throw any exception.
        ///
        /// \param pos Position of the element in the vector [Note the first
        ///            position in the partition is 0]
        ///
        /// \return Returns a proxy object which represents the indexed element.
        ///         A (possibly remote) access operation is performed only once
        ///         this proxy instance is used.
        ///
        detail::vector_value_proxy<T> operator[](size_type pos)
        {
            return detail::vector_value_proxy<T>(*this, pos);
        }

        /// \brief Array subscript operator. This does not throw any exception.
        ///
        /// \param pos Position of the element in the vector [Note the first
        ///            position in the partition is 0]
        ///
        /// \return Returns the value of the element at position represented by
        ///         \a pos.
        ///
        /// \note This function does not return a reference to the actual
        ///       element but a copy of its value.
        ///
        T operator[](size_type pos) const
        {
            return get_value_sync(pos);
        }

        /// Copy assignment operator, performs deep copy of the right hand side
        /// vector.
        ///
        /// \param rhs    This the hpx::partitioned_vector object which is to
        ///               be copied
        ///
        partitioned_vector& operator=(partitioned_vector const& rhs)
        {
            if (this != &rhs && rhs.size_ != 0)
                copy_from(rhs);
            return *this;
        }

        partitioned_vector& operator=(partitioned_vector && rhs)
        {
            if (this != &rhs)
            {
                this->base_type::operator=(std::move(rhs));

                size_ = rhs.size_;
                partition_size_ = rhs.partition_size_;
                partitions_ = std::move(rhs.partitions_);

                rhs.size_ = 0;
                rhs.partition_size_ = std::size_t(-1);
            }
            return *this;
        }

        ///////////////////////////////////////////////////////////////////////
        // Capacity related API's in vector class

        /// \brief Compute the size as the number of elements it contains.
        ///
        /// \return Return the number of elements in the vector
        ///
        size_type size() const
        {
            return size_;
        }

        //
        //  Element access API's in vector class
        //

        /// Returns the element at position \a pos in the vector container.
        ///
        /// \param pos Position of the element in the vector
        ///
        /// \return Returns the value of the element at position represented by
        ///         \a pos.
        ///
        T get_value_sync(size_type pos) const
        {
            return get_value_sync(get_partition(pos), get_local_index(pos));
        }

        /// Returns the element at position \a pos in the vector container.
        ///
        /// \param part  Sequence number of the partition
        /// \param pos   Position of the element in the partition
        ///
        /// \return Returns the value of the element at position represented by
        ///         \a pos.
        ///
        T get_value_sync(size_type part, size_type pos) const
        {
            partition_data const& part_data = partitions_[part];
            if (part_data.local_data_)
                return part_data.local_data_->get_value(pos);

            return partition_vector_client(part_data.partition_)
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
        future<T> get_value(size_type pos) const
        {
            return get_value(get_partition(pos), get_local_index(pos));
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
        future<T> get_value(size_type part, size_type pos) const
        {
            if (partitions_[part].local_data_)
            {
                return make_ready_future(
                    partitions_[part].local_data_->get_value(pos));
            }

            return partition_vector_client(partitions_[part].partition_)
                .get_value(pos);
        }

        /// Returns the elements at the positions \a pos from the given
        /// partition in the vector container.
        ///
        /// \param part  Sequence number of the partition
        /// \param pos   Position of the element in the partition
        ///
        /// \return Returns the value of the element at position represented by
        ///         \a pos.
        ///
        std::vector<T>
        get_values_sync(size_type part, std::vector<size_type> const& pos) const
        {
            partition_data const& part_data = partitions_[part];
            if (part_data.local_data_)
                return part_data.local_data_->get_values(pos);

            return partition_vector_client(part_data.partition_)
                .get_values_sync(pos);
        }

        /// Asynchronously returns the elements at the positions \a pos from
        /// the given partition in the vector container.
        ///
        /// \param part  Sequence number of the partition
        /// \param pos   Positions of the elements in the vector
        ///
        /// \return Returns the hpx::future to values of the elements at the
        ///         given positions represented by \a pos.
        ///
        future<std::vector<T> >
        get_values(size_type part, std::vector<size_type> const& pos) const
        {
            partition_data const& part_data = partitions_[part];
            if (part_data.local_data_)
                return make_ready_future(part_data.local_data_->get_values(pos));

            return partition_vector_client(part_data.partition_)
                .get_values(pos);
        }

        /// Returns the elements at the positions \a pos
        /// in the vector container.
        ///
        /// \param pos   Global position of the element in the vector
        ///
        /// \return Returns the value of the element at position represented by
        ///         \a pos.
        ///
        future<std::vector<T> >
        get_values(std::vector<size_type> const & pos_vec) const
        {
            // check if position vector is empty
            // the following code needs at least one element.
            if (pos_vec.empty())
                return make_ready_future(std::vector<T>());

            // current partition index of the block
            size_type part_cur = get_partition(pos_vec[0]);

            // iterator to the begin of current block
            std::vector<size_type>::const_iterator part_begin = pos_vec.begin();

            // vector holding futures of the values for all blocks
            std::vector<future<std::vector<T> > > part_values_future;
            for (std::vector<size_type>::const_iterator it = pos_vec.begin();
                 it != pos_vec.end(); ++it)
            {
                // get the partition of the current position
                size_type part = get_partition(*it);

                // if the partition of the current position is the same
                // as the rest of the current block go to next position
                if (part == part_cur)
                    continue;

                // if the partition of the current position is NOT the same
                // as the positions before the block ends here
                else
                {
                    // this is the end of a block containing indexes ('pos')
                    // of the same partition ('part').
                    // get async values for this block
                    part_values_future.push_back(get_values(part_cur,
                        get_local_indices(std::vector<size_type>(part_begin, it))));

                    // reset block variables to start a new one from here
                    part_cur = part;
                    part_begin = it;
                }
            }

            // the end of the vector is also an end of a block
            // get async values for this block
            part_values_future.push_back(get_values(part_cur,
                get_local_indices(std::vector<size_type>(
                    part_begin, pos_vec.end()))));

            // This helper function unwraps the vectors from each partition
            // and merge them to one vector
            auto merge_func =
                [&pos_vec](std::vector<future<std::vector<T> > > && part_values_f)
                    -> std::vector<T>
                {
                    std::vector<T> values;
                    values.reserve(pos_vec.size());

                    for (future<std::vector<T> >& part_f: part_values_f)
                    {
                        std::vector<T> part_values = part_f.get();
                        std::move(part_values.begin(), part_values.end(),
                            std::back_inserter(values));
                    }
                    return values;
                };

            // when all values are here merge them to one vector
            // and return a future to this vector
            return lcos::local::dataflow(launch::async, merge_func,
                std::move(part_values_future));
        }

        /// Returns the elements at the positions \a pos
        /// in the vector container.
        ///
        /// \param pos   Global position of the element in the vector
        ///
        /// \return Returns the value of the element at position represented by
        ///         \a pos.
        ///
        std::vector<T>
        get_values_sync(std::vector<size_type> const & pos_vec) const
        {
            return get_values(pos_vec).get();
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
//             for (partition_description_type const& p,
//                 boost::make_iterator_range(partitions_.begin(),
//                                            partitions_.end() - 1)
//                 )
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

        /// Copy the value of \a val in the element at position \a pos in
        /// the vector container.
        ///
        /// \param pos   Position of the element in the vector
        /// \param val   The value to be copied
        ///
        template <typename T_>
        void set_value_sync(size_type pos, T_ && val)
        {
            return set_value_sync(get_partition(pos), get_local_index(pos),
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
        void set_value_sync(size_type part, size_type pos, T_ && val)
        {
            partition_data const& part_data = partitions_[part];
            if (part_data.local_data_)
            {
                part_data.local_data_->set_value(pos, std::forward<T_>(val));
            }
            else
            {
                partition_vector_client(part_data.partition_)
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
        future<void> set_value(size_type pos, T_ && val)
        {
            return set_value(get_partition(pos), get_local_index(pos),
                std::forward<T_>(val));
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
        future<void> set_value(size_type part, size_type pos, T_ && val)
        {
            partition_data const& part_data = partitions_[part];
            if (part_data.local_data_)
            {
                part_data.local_data_->set_value(pos, std::forward<T_>(val));
                return make_ready_future();
            }

            return partition_vector_client(part_data.partition_)
                .set_value(pos, std::forward<T_>(val));
        }

        /// Copy the values of \a val to the elements at positions \a pos in
        /// the partition \part of the vector container.
        ///
        /// \param part  Sequence number of the partition
        /// \param pos   Position of the element in the vector
        /// \param val   The value to be copied
        ///
        void set_values_sync(size_type part, std::vector<size_type> const& pos,
            std::vector<T> const& val)
        {
            set_value(pos, val).get();
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
        future<void>
        set_values(size_type part, std::vector<size_type> const& pos,
            std::vector<T> const& val)
        {
            HPX_ASSERT(pos.size() == val.size());

            if (partitions_[part].local_data_)
            {
                partitions_[part].local_data_->set_values(pos, val);
                return make_ready_future();
            }

            return partition_vector_client(partitions_[part].partition_)
                .set_values(pos, val);
        }

        /// Asynchronously set the element at position \a pos
        /// to the given value \a val.
        ///
        /// \param pos   Global position of the element in the vector
        /// \param val   The value to be copied
        ///
        /// \return This returns the hpx::future of type void which gets ready
        ///         once the operation is finished.
        ///
        future<void>
        set_values(std::vector<size_type> const& pos, std::vector<T> const& val)
        {
            HPX_ASSERT(pos.size() == val.size());

            // check if position vector is empty
            // the following code needs at least one element.
            if (pos.empty())
                return make_ready_future();

            // partition index of the current block
            size_type part_cur = get_partition(pos[0]);

            // iterator to the begin of current block
            std::vector<size_type>::const_iterator pos_block_begin = pos.begin();
            typename std::vector<T>::const_iterator val_block_begin = val.begin();

            // vector holding futures of the state for all blocks
            std::vector<future<void> > part_futures;

            // going through the position vector
            std::vector<size_type>::const_iterator pos_it = pos.begin();
            typename std::vector<T>::const_iterator val_it = val.begin();
            for (/**/; pos_it != pos.end(); ++pos_it, ++val_it)
            {
                // get the partition of the current position
                size_type part = get_partition(*pos_it);

                // if the partition of the current position is the same
                // as the rest of the current block go to next position
                if (part == part_cur)
                    continue;

                // if the partition of the current position is NOT the same
                // as the positions before the block ends here
                else
                {
                    // this is the end of a block containing indexes ('pos')
                    // of the same partition ('part').
                    // set asynchronous values for this block
                    part_futures.push_back(set_values(part_cur,
                        get_local_indices(std::vector<size_type>(
                            pos_block_begin, pos_it)),
                        std::vector<T>(val_block_begin, val_it)));

                    // reset block variables to start a new one from here
                    part_cur = part;
                    pos_block_begin = pos_it;
                    val_block_begin = val_it;
                }
            }

            // the end of the vector is also an end of a block
            // get asynchronous values for this block
            part_futures.push_back(set_values(part_cur,
                get_local_indices(std::vector<size_type>(
                    pos_block_begin, pos.end())),
                std::vector<T>(val_block_begin, val.end())));

            return when_all(part_futures);
        }

        void set_values_sync(std::vector<size_type> const& pos,
            std::vector<T> const& val)
        {
            return set_value(pos, val).get();
        }

//             //CLEAR
//             //TODO if number of partitions is kept constant every time then
//             // clear should modified (clear each partition_vector one by one).
// //            void clear()
// //            {
// //                //It is keeping one gid hence iterator does not go
// //                //in an invalid state
// //                partitions_.erase(partitions_.begin() + 1,
// //                                           partitions_.end()-1);
// //                partition_vector_stub::clear_async((partitions_[0].second).get())
// //                        .get();
// //                HPX_ASSERT(partitions_.size() > 1);
// //                //As this function changes the size we should have LAST always.
// //            }

        ///////////////////////////////////////////////////////////////////////
        /// Return the iterator at the beginning of the first segment located
        /// on the given locality.
        iterator begin()
        {
            return iterator(this, get_global_index(segment_cbegin(), 0));
        }

        /// \brief Return the const_iterator at the beginning of the vector.
        const_iterator begin() const
        {
            return const_iterator(this, get_global_index(segment_cbegin(), 0));
        }

        /// \brief Return the const_iterator at the beginning of the vector.
        const_iterator cbegin() const
        {
            return const_iterator(this, get_global_index(segment_cbegin(), 0));
        }

        /// \brief Return the iterator at the end of the vector.
        iterator end()
        {
            return iterator(this, get_global_index(segment_cend(), 0));
        }

        /// \brief Return the const_iterator at the end of the vector.
        const_iterator end() const
        {
            return const_iterator(this, get_global_index(segment_cend(), 0));
        }

        /// \brief Return the const_iterator at the end of the vector.
        const_iterator cend() const
        {
            return const_iterator(this, get_global_index(segment_cend(), 0));
        }

        ///////////////////////////////////////////////////////////////////////
        /// Return the iterator at the beginning of the first partition of the
        /// vector on the given locality.
        iterator begin(boost::uint32_t id)
        {
            return iterator(this, get_global_index(segment_begin(id), 0));
        }

        /// Return the iterator at the beginning of the first partition of the
        /// vector on the given locality.
        const_iterator begin(boost::uint32_t id) const
        {
            return const_iterator(this, get_global_index(segment_cbegin(id), 0));
        }

        /// Return the iterator at the beginning of the first partition of the
        /// vector on the given locality.
        const_iterator cbegin(boost::uint32_t id) const
        {
            return const_iterator(this, get_global_index(segment_cbegin(id), 0));
        }

        /// Return the iterator at the end of the last partition of the
        /// vector on the given locality.
        iterator end(boost::uint32_t id)
        {
            return iterator(this, get_global_index(segment_end(id), 0));
        }

        /// Return the iterator at the end of the last partition of the
        /// vector on the given locality.
        const_iterator end(boost::uint32_t id) const
        {
            return const_iterator(this, get_global_index(segment_cend(id), 0));
        }

        /// Return the iterator at the end of the last partition of the
        /// vector on the given locality.
        const_iterator cend(boost::uint32_t id) const
        {
            return const_iterator(this, get_global_index(segment_cend(id), 0));
        }

        ///////////////////////////////////////////////////////////////////////
        /// Return the iterator at the beginning of the first segment located
        /// on the given locality.
        iterator begin(id_type const& id)
        {
            HPX_ASSERT(naming::is_locality(id));
            return begin(naming::get_locality_from_id(id));
        }

        /// Return the iterator at the beginning of the first segment located
        /// on the given locality.
        const_iterator begin(id_type const& id) const
        {
            HPX_ASSERT(naming::is_locality(id));
            return begin(naming::get_locality_from_id(id));
        }

        /// Return the iterator at the beginning of the first segment located
        /// on the given locality.
        const_iterator cbegin(id_type const& id) const
        {
            HPX_ASSERT(naming::is_locality(id));
            return cbegin(naming::get_locality_from_id(id));
        }

        /// Return the iterator at the end of the last segment located
        /// on the given locality.
        iterator end(id_type const& id)
        {
            HPX_ASSERT(naming::is_locality(id));
            return end(naming::get_locality_from_id(id));
        }

        /// Return the iterator at the end of the last segment located
        /// on the given locality.
        const_iterator end(id_type const& id) const
        {
            HPX_ASSERT(naming::is_locality(id));
            return end(naming::get_locality_from_id(id));
        }

        /// Return the iterator at the end of the last segment located
        /// on the given locality.
        const_iterator cend(id_type const& id) const
        {
            HPX_ASSERT(naming::is_locality(id));
            return cend(naming::get_locality_from_id(id));
        }

        ///////////////////////////////////////////////////////////////////////
        // Return global segment iterator
        segment_iterator segment_begin()
        {
            return segment_iterator(partitions_.begin(), this);
        }

        const_segment_iterator segment_begin() const
        {
            return const_segment_iterator(partitions_.cbegin(), this);
        }

        const_segment_iterator segment_cbegin() const //-V524
        {
            return const_segment_iterator(partitions_.cbegin(), this);
        }

        segment_iterator segment_end()
        {
            return segment_iterator(partitions_.end(), this);
        }

        const_segment_iterator segment_end() const
        {
            return const_segment_iterator(partitions_.cend(), this);
        }

        const_segment_iterator segment_cend() const //-V524
        {
            return const_segment_iterator(partitions_.cend(), this);
        }

        ///////////////////////////////////////////////////////////////////////
        // Return local segment iterator
        local_segment_iterator segment_begin(boost::uint32_t id)
        {
            // local_segement_iterators are only valid on the locality where
            // the data lives
            HPX_ASSERT(id == hpx::get_locality_id());
            return local_segment_iterator(partitions_.begin(),
                partitions_.end(), id);
        }

        const_local_segment_iterator segment_begin(boost::uint32_t id) const
        {
            // local_segement_iterators are only valid on the locality where
            // the data lives
            HPX_ASSERT(id == hpx::get_locality_id());
            return const_local_segment_iterator(partitions_.cbegin(),
                partitions_.cend(), id);
        }

        const_local_segment_iterator segment_cbegin(boost::uint32_t id) const
        {
            // local_segement_iterators are only valid on the locality where
            // the data lives
            HPX_ASSERT(id == hpx::get_locality_id());
            return const_local_segment_iterator(partitions_.cbegin(),
                partitions_.cend(), id);
        }

        local_segment_iterator segment_end(boost::uint32_t id)
        {
            // local_segement_iterators are only valid on the locality where
            // the data lives
            HPX_ASSERT(id == hpx::get_locality_id());
            return local_segment_iterator(partitions_.end());
        }

        const_local_segment_iterator segment_end(boost::uint32_t id) const
        {
            // local_segement_iterators are only valid on the locality where
            // the data lives
            HPX_ASSERT(id == hpx::get_locality_id());
            return const_local_segment_iterator(partitions_.cend());
        }

        const_local_segment_iterator segment_cend(boost::uint32_t id) const
        {
            // local_segement_iterators are only valid on the locality where
            // the data lives
            HPX_ASSERT(id == hpx::get_locality_id());
            return const_local_segment_iterator(partitions_.cend());
        }

        ///////////////////////////////////////////////////////////////////////
        local_segment_iterator segment_begin(id_type const& id)
        {
            HPX_ASSERT(naming::is_locality(id));
            return segment_begin(naming::get_locality_from_id(id));
        }

        const_local_segment_iterator segment_begin(id_type const& id) const
        {
            HPX_ASSERT(naming::is_locality(id));
            return segment_begin(naming::get_locality_from_id(id));
        }

        const_local_segment_iterator segment_cbegin(id_type const& id) const
        {
            HPX_ASSERT(naming::is_locality(id));
            return segment_cbegin(naming::get_locality_from_id(id));
        }

        local_segment_iterator segment_end(id_type const& id)
        {
            HPX_ASSERT(naming::is_locality(id));
            return segment_end(naming::get_locality_from_id(id));
        }

        const_local_segment_iterator segment_end(id_type const& id) const
        {
            HPX_ASSERT(naming::is_locality(id));
            return segment_end(naming::get_locality_from_id(id));
        }

        const_local_segment_iterator segment_cend(id_type const& id) const
        {
            HPX_ASSERT(naming::is_locality(id));
            return segment_cend(naming::get_locality_from_id(id));
        }
    };
}

#endif // VECTOR_HPP
