//  Copyright (c) 2014 Bibek Ghimire
//  Copyright (c) 2014-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_VECTOR_DISTRIBUTION_POLICY_HPP
#define HPX_VECTOR_DISTRIBUTION_POLICY_HPP

#include <hpx/hpx_fwd.hpp>
#include <boost/detail/scoped_enum_emulation.hpp>

#include <vector>
#include <iostream>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    // This class specifies the block chunking policy parameters to use for the
    // partitioning of the data in a hpx::vector
    struct distribution_policy
    {
    public:
        distribution_policy()
          : num_partitions_(1)
        {}

        distribution_policy operator()(std::size_t num_partitions) const
        {
            return distribution_policy(num_partitions, localities_);
        }

        distribution_policy operator()(
            std::vector<id_type> const& localities) const
        {
            return distribution_policy(num_partitions_, localities);
        }

        distribution_policy operator()(std::size_t num_partitions,
            std::vector<id_type> const& localities) const
        {
            return distribution_policy(num_partitions, localities);
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

    private:
        friend class boost::serialization::access;

        template <typename Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            ar & localities_ & num_partitions_;
        }

        distribution_policy(std::size_t num_partitions,
             std::vector<id_type> const& localities)
          : localities_(localities),
            num_partitions_(num_partitions)
        {}

    private:
        std::vector<id_type> localities_;   // localities to create chunks on
        std::size_t num_partitions_;        // number of chunks to create
    };

    static distribution_policy const layout;

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename T>
        struct is_vector_distribution_policy
          : boost::mpl::false_
        {};

        template <>
        struct is_vector_distribution_policy<distribution_policy>
          : boost::mpl::true_
        {};
        // \endcond
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    struct is_vector_distribution_policy
      : detail::is_vector_distribution_policy<typename hpx::util::decay<T>::type>
    {};
}

#endif
