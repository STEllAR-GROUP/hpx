//  Copyright (c) 2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UNORDERED_DISTRIBUTION_POLICY_HPP
#define HPX_UNORDERED_DISTRIBUTION_POLICY_HPP

#include <hpx/hpx_fwd.hpp>
#include <boost/detail/scoped_enum_emulation.hpp>

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    BOOST_SCOPED_ENUM_START(unordered_distribution_policy)
    {
        hash = 0,       ///< find locality based on a given hash
    };
    BOOST_SCOPED_ENUM_END

    ///////////////////////////////////////////////////////////////////////////
    // This class specifies the block chunking policy parameters to use for the
    // partitioning of the data in a hpx::vector
    struct hash_distribution_policy
    {
    public:
        hash_distribution_policy()
          : num_partitions_(1)
        {}

        hash_distribution_policy operator()(std::size_t num_partitions) const
        {
            return hash_distribution_policy(num_partitions, localities_);
        }

        hash_distribution_policy operator()(
            std::vector<id_type> const& localities) const
        {
            return hash_distribution_policy(num_partitions_, localities);
        }

        hash_distribution_policy operator()(std::size_t num_partitions,
            std::vector<id_type> const& localities) const
        {
            return hash_distribution_policy(num_partitions, localities);
        }

        ///////////////////////////////////////////////////////////////////////
        std::vector<id_type> const& get_localities() const
        {
            return localities_;
        }

        std::size_t get_num_partitions() const
        {
            return num_partitions_;
        }

        BOOST_SCOPED_ENUM(unordered_distribution_policy) get_policy_type() const
        {
            return unordered_distribution_policy::hash;
        }

    private:
        friend class boost::serialization::access;

        template <typename Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            ar & localities_ & num_partitions_;
        }

        hash_distribution_policy(std::size_t num_partitions,
                std::vector<id_type> const& localities)
          : localities_(localities),
            num_partitions_(num_partitions)
        {}

    private:
        std::vector<id_type> localities_;   // localities to create chunks on
        std::size_t num_partitions_;            // number of chunks to create
    };

    static hash_distribution_policy const hash;

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename T>
        struct is_unordered_distribution_policy
          : boost::mpl::false_
        {};

        template <>
        struct is_unordered_distribution_policy<hash_distribution_policy>
          : boost::mpl::true_
        {};
        // \endcond
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    struct is_unordered_distribution_policy
      : detail::is_unordered_distribution_policy<typename hpx::util::decay<T>::type>
    {};
}

#endif
