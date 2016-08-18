//  Copyright (c) 2014-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file target_distribution_policy.hpp

#if !defined(HPX_COMPUTE_TARGET_DISTRIBUTION_POLICY)
#define HPX_COMPUTE_TARGET_DISTRIBUTION_POLICY

#include <hpx/config.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/runtime/serialization/serialization_fwd.hpp>

#include <mutex>
#include <utility>
#include <vector>

#include <boost/atomic.hpp>

namespace hpx { namespace compute { namespace detail
{
    /// This class specifies the parameters for a simple distribution policy
    /// to use for creating (and evenly distributing) a given number of items
    /// on a given set of localities.
    template <typename Target>
    struct target_distribution_policy
    {
    private:
        typedef hpx::lcos::local::spinlock mutex_type;

    public:
        typedef Target target_type;

        /// Default-construct a new instance of a \a target_distribution_policy.
        target_distribution_policy()
          : targets_(),
            num_partitions_(1),
            next_target_(0)
        {}

        target_distribution_policy(target_distribution_policy const& rhs)
          : targets_(rhs.targets_),
            num_partitions_(rhs.num_partitions_),
            next_target_(0)
        {}

        target_distribution_policy(target_distribution_policy && rhs)
          : targets_(std::move(rhs.targets_)),
            num_partitions_(rhs.num_partitions_),
            next_target_(0)
        {}

        target_distribution_policy& operator=(
            target_distribution_policy const& rhs)
        {
            if (this != &rhs)
            {
                targets_ = rhs.targets_;
                num_partitions_ = rhs.num_partitions_;
                next_target_ = 0;
            }
            return *this;
        }

        target_distribution_policy& operator=(
            target_distribution_policy && rhs)
        {
            if (this != &rhs)
            {
                targets_ = std::move(rhs.targets_);
                num_partitions_ = rhs.num_partitions_;
                next_target_ = 0;
            }
            return *this;
        }

        /// Returns the locality which is anticipated to be used for the next
        /// async operation
        target_type get_next_target() const
        {
            std::lock_guard<mutex_type> l(mtx_);
            if (targets_.empty())
                targets_ = Target::get_local_targets();
            return targets_[next_target_++ % targets_.size()];
        }

        std::size_t get_num_partitions() const
        {
            std::lock_guard<mutex_type> l(mtx_);
            if (targets_.empty())
                targets_ = Target::get_local_targets();

            std::size_t num_parts = (num_partitions_ == std::size_t(-1)) ?
                targets_.size() : num_partitions_;
            return (std::max)(num_parts, std::size_t(1));
        }

    protected:
        /// \cond NOINTERNAL
        HPX_FORCEINLINE static std::size_t
        round_to_multiple(std::size_t n1, std::size_t n2, std::size_t n3)
        {
            return (n1 / n2) * n3;
        }

        std::size_t get_num_items(std::size_t items, target_type const& t) const
        {
            std::lock_guard<mutex_type> l(mtx_);
            if (targets_.empty())
                targets_ = Target::get_local_targets();

            // this distribution policy places an equal number of items onto
            // each target
            std::size_t sites = (std::max)(std::size_t(1), targets_.size());

            // the overall number of items to create is smaller than the number
            // of sites
            if (items < sites)
            {
                auto it = std::find(targets_.begin(), targets_.end(), t);
                std::size_t num_loc = std::distance(targets_.begin(), it);
                return (items < num_loc) ? 1 : 0;
            }

            // the last locality might get less items
            if (!targets_.empty() && t == targets_.back())
            {
                return items - round_to_multiple(items, sites, sites-1);
            }

            // otherwise just distribute evenly
            return (items + sites - 1) / sites;
        }
        /// \endcond

    protected:
        /// \cond NOINTERNAL
        target_distribution_policy(std::vector<target_type> const& targets,
                std::size_t num_partitions)
          : targets_(targets),
            num_partitions_(num_partitions),
            next_target_(0)
        {}

        target_distribution_policy(std::vector<target_type> && targets,
                std::size_t num_partitions)
          : targets_(std::move(targets)),
            num_partitions_(num_partitions),
            next_target_(0)
        {}

        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive& ar, unsigned int const)
        {
            ar & targets_ & num_partitions_;
        }

        mutable mutex_type mtx_;
        mutable std::vector<target_type> targets_;   // targets
        std::size_t num_partitions_;
        mutable std::size_t next_target_;
        /// \endcond
    };
}}}

#endif
