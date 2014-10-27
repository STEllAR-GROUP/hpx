//  Copyright (c) 2014 Bibek Ghimire
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_BLOCK_DISTRIBUTION_POLICY_HPP
#define HPX_BLOCK_DISTRIBUTION_POLICY_HPP

#include <hpx/hpx_fwd.hpp>
#include <boost/detail/scoped_enum_emulation.hpp>

#include <vector>
#include <iostream>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    BOOST_SCOPED_ENUM_START(distribution_policy)
    {
        block         = 0,      ///< block distribution policy
        cyclic        = 1,      ///< cyclic distribution policy
        block_cyclic  = 2       ///< block-cyclic distribution policy
    };
    BOOST_SCOPED_ENUM_END

    ///////////////////////////////////////////////////////////////////////////
    // This class specifies the block chunking policy parameters to use for the
    // partitioning of the data in a hpx::vector
    struct block_distribution_policy
    {
    public:
        block_distribution_policy()
          : num_partitions_(1)
        {}

        block_distribution_policy operator()(std::size_t num_partitions) const
        {
            return block_distribution_policy(num_partitions, localities_);
        }

        block_distribution_policy operator()(
            std::vector<id_type> const& localities) const
        {
            return block_distribution_policy(num_partitions_, localities);
        }

        block_distribution_policy operator()(std::size_t num_partitions,
            std::vector<id_type> const& localities) const
        {
            return block_distribution_policy(num_partitions, localities);
        }

        ///////////////////////////////////////////////////////////////////////
        std::vector<id_type> const& get_localities() const
        {
            return localities_;
        }

        std::size_t get_block_size() const
        {
            return std::size_t(-1);
        }

        std::size_t get_num_partitions() const
        {
            return num_partitions_;
        }

        BOOST_SCOPED_ENUM(distribution_policy) get_policy_type() const
        {
            return distribution_policy::block;
        }

    private:
        friend class boost::serialization::access;

        template <typename Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            ar & localities_ & num_partitions_;
        }

        block_distribution_policy(std::size_t num_partitions,
             std::vector<id_type> const& localities)
          : localities_(localities),
            num_partitions_(num_partitions)
        {}

    private:
        std::vector<id_type> localities_;   // localities to create chunks on
        std::size_t num_partitions_;            // number of chunks to create
    };

    static block_distribution_policy const block;

    ///////////////////////////////////////////////////////////////////////////
    // This class specifies the cyclic chunking policy parameters to use for
    // the partitioning of the data in a hpx::vector
    struct cyclic_distribution_policy
    {
    public:
        cyclic_distribution_policy()
          : num_partitions_(1)
        {}

        cyclic_distribution_policy operator()(std::size_t num_partitions) const
        {
            return cyclic_distribution_policy(num_partitions, localities_);
        }

        cyclic_distribution_policy operator()(
            std::vector<id_type> const& localities) const
        {
            return cyclic_distribution_policy(num_partitions_, localities);
        }

        cyclic_distribution_policy operator()(std::size_t num_partitions,
            std::vector<id_type> const& localities) const
        {
            return cyclic_distribution_policy(num_partitions, localities);
        }

        std::vector<id_type> const& get_localities() const
        {
            return localities_;
        }

        std::size_t get_block_size() const
        {
            return std::size_t(-1);
        }

        std::size_t get_num_partitions() const
        {
            return num_partitions_;
        }

        BOOST_SCOPED_ENUM(distribution_policy) get_policy_type() const
        {
            return distribution_policy::cyclic;
        }

    private:
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            ar & localities_ & num_partitions_;
        }

        cyclic_distribution_policy(std::size_t num_partitions,
                std::vector<id_type> const& localities)
          : localities_(localities),
            num_partitions_(num_partitions)
        {}

    private:
        std::vector<id_type> localities_;
        std::size_t num_partitions_;
    };

    static cyclic_distribution_policy const cyclic;

    ///////////////////////////////////////////////////////////////////////////
    // This class specifies the block-cyclic chunking policy parameters to
    // use for the partitioning of the data in a hpx::vector
    struct block_cyclic_distribution_policy
    {
    public:
        block_cyclic_distribution_policy()
          : num_partitions_(1),
            block_size_(std::size_t(-1))
        {}

        block_cyclic_distribution_policy operator()(std::size_t num_partitions) const
        {
            return block_cyclic_distribution_policy(
                num_partitions, localities_, block_size_);
        }

        block_cyclic_distribution_policy operator()(std::size_t num_partitions,
            std::vector<id_type> const& localities) const
        {
            return block_cyclic_distribution_policy(
                num_partitions, localities, block_size_);
        }

        block_cyclic_distribution_policy operator()(std::size_t num_partitions,
            std::vector<id_type> const& localities, std::size_t block_size) const
        {
            return block_cyclic_distribution_policy(
                num_partitions, localities, block_size);
        }

        block_cyclic_distribution_policy operator()(
            std::vector<id_type> const& localities, std::size_t block_size) const
        {
            return block_cyclic_distribution_policy(
                num_partitions_, localities, block_size);
        }

        block_cyclic_distribution_policy operator()(
            std::vector<id_type> const& localities) const
        {
            return block_cyclic_distribution_policy(
                num_partitions_, localities, block_size_);
        }

        block_cyclic_distribution_policy operator()(std::size_t num_partitions,
            std::size_t block_size) const
        {
            return block_cyclic_distribution_policy(
                num_partitions, localities_, block_size);
        }

        std::vector<id_type> const& get_localities() const
        {
            return localities_;
        }

        std::size_t get_num_partitions() const
        {
            return num_partitions_;
        }

        BOOST_SCOPED_ENUM(distribution_policy) get_policy_type() const
        {
            return distribution_policy::block_cyclic;
        }

        std::size_t get_block_size() const
        {
            return block_size_;
        }

    private:
        friend class boost::serialization::access;

        template <typename Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            ar & localities_ & num_partitions_ & block_size_;
        }

        block_cyclic_distribution_policy(std::size_t num_partitions,
                std::vector<id_type> const& localities, std::size_t block_size)
          : localities_(localities),
            num_partitions_(num_partitions),
            block_size_(block_size)
        {}

    private:
        std::vector<id_type> localities_;   // localities to create chunks on
        std::size_t num_partitions_;            // number of chunks to create
        std::size_t block_size_;            // size of a cyclic block
    };

    static block_cyclic_distribution_policy const block_cyclic;
}

#endif
