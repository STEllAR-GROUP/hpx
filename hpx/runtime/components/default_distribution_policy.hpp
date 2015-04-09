//  Copyright (c) 2014-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file distribution_policy.hpp

#if !defined(HPX_COMPONENTS_DISTRIBUTION_POLICY_APR_07_2015_1246PM)
#define HPX_COMPONENTS_DISTRIBUTION_POLICY_APR_07_2015_1246PM

#include <hpx/config.hpp>

#include <algorithm>

namespace hpx { namespace components
{
    ///////////////////////////////////////////////////////////////////////////
    /// \cond NOINTERNAL
    namespace detail
    {
        BOOST_FORCEINLINE std::size_t
        round_to_multiple(std::size_t n1, std::size_t n2, std::size_t n3)
        {
            return (n1 / n2) * n3;
        }
    }
    /// \endcond

    /// This class specifies the parameters for a simple distribution policy
    /// to use for creating (and evenly distributing) a given number of items
    /// on a given set of localities.
    struct default_distribution_policy
    {
    public:
        /// Default-construct a new instance of a \a default_distribution_policy.
        /// This policy will represent one locality (the local locality).
        default_distribution_policy()
        {}

        /// Create a new \a default_distribution policy representing the given
        /// set of localities.
        ///
        /// \param locs     [in] The list of localities the new instance should
        ///                 represent
        default_distribution_policy operator()(
            std::vector<id_type> const& locs) const
        {
#if defined(HPX_DEBUG)
            for (id_type const& loc: locs)
            {
                HPX_ASSERT(naming::is_locality(loc));
            }
#endif
            return default_distribution_policy(locs);
        }

        /// Create a new \a default_distribution policy representing the given
        /// locality
        ///
        /// \param loc     [in] The locality the new instance should
        ///                 represent
        default_distribution_policy operator()(id_type const& loc) const
        {
            HPX_ASSERT(naming::is_locality(loc));
            return default_distribution_policy(loc);
        }

        /// Return the list of localities represented by this
        /// \a distribution_policy
        std::vector<id_type> const& get_localities() const
        {
            return localities_;
        }

        ///////////////////////////////////////////////////////////////////////
        /// Return the number of items to place on the given locality.
        ///
        /// \param items    [in] The overall number of items to create based on
        ///                 the given \a distribution_policy.
        /// \param loc      [in] The locality for which the function will
        ///                 return how many items to create.
        ///
        /// \note This function will calculate the number of items to be
        ///       created on the given locality. A \a distribution_policy will
        ///       evenly distribute the overall number of items over the given
        ///       localities it represents.
        ///
        /// \returns The number of items to be created on the given locality
        ///          \a loc.
        ///
        std::size_t
        get_num_items(std::size_t items, hpx::id_type const& loc) const
        {
            // make sure the given id is known to this distribution policy
            HPX_ASSERT(
                std::find(localities_.begin(), localities_.end(), loc) !=
                    localities_.end() ||
                (localities_.empty() && loc == hpx::find_here())
            );

            // this distribution policy places an equal number of items onto
            // each locality
            std::size_t locs = (std::max)(std::size_t(1), localities_.size());

            // the overall number of items to create is smaller than the number
            // of localities
            if (items < locs)
            {
                auto it = std::find(localities_.begin(), localities_.end(), loc);
                std::size_t num_loc = std::distance(localities_.begin(), it);
                return (items < num_loc) ? 1 : 0;
            }

            // the last locality might get less items
            if (localities_.size() > 1 && loc == localities_.back())
                return items - detail::round_to_multiple(items, locs, locs-1);

            // otherwise just distribute evenly
            return (items + locs - 1) / locs;
        }

    protected:
        /// \cond NOINTERNAL
        default_distribution_policy(std::vector<id_type> const& localities)
          : localities_(localities)
        {}

        default_distribution_policy(id_type const& locality)
        {
            localities_.push_back(locality);
        }

        std::vector<id_type> localities_;   // localities to create things on
        /// \endcond
    };

    /// A predefined instance of the default \a distribution_policy. It will
    /// represent the local locality and will place all items to create here.
    static default_distribution_policy const default_layout;
}}

/// \cond NOINTERNAL
namespace hpx
{
    using hpx::components::default_distribution_policy;
    using hpx::components::default_layout;
}
/// \endcond

#endif
