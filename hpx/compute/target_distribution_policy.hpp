//  Copyright (c) 2014-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file target_distribution_policy.hpp

#if !defined(HPX_COMPUTE_TARGET_DISTRIBUTION_POLICY)
#define HPX_COMPUTE_TARGET_DISTRIBUTION_POLICY

#include <hpx/config.hpp>

#include <hpx/dataflow.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/components/containers/container_distribution_policy.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/runtime/serialization/serialization_fwd.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/traits/is_distribution_policy.hpp>

#include <algorithm>
#include <map>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/atomic.hpp>

namespace hpx { namespace compute
{
    /// This class specifies the parameters for a simple distribution policy
    /// to use for creating (and evenly distributing) a given number of items
    /// on a given set of localities.
    template <typename Target>
    struct target_distribution_policy
    {
    public:
        typedef Target target_type;

        /// Default-construct a new instance of a \a target_distribution_policy.
        /// This policy will represent all devices on the current locality.
        ///
        target_distribution_policy()
          : targets_(Target::get_local_targets()),
            num_partitions_(1), next_target_(0)
        {}

        /// Create a new \a target_distribution_policy representing the given
        /// set of targets
        ///
        /// \param targets [in] The targets the new instances should represent
        ///
        target_distribution_policy operator()(
            std::vector<target_type> const& targets,
            std::size_t num_partitions = std::size_t(-1)) const
        {
            if (num_partitions == std::size_t(-1))
                num_partitions = targets.size();
            return target_distribution_policy(targets, num_partitions);
        }

        /// Create a new \a target_distribution_policy representing the given
        /// set of targets
        ///
        /// \param targets [in] The targets the new instances should represent
        ///
        target_distribution_policy operator()(
            std::vector<target_type> && targets,
            std::size_t num_partitions = std::size_t(-1)) const
        {
            if (num_partitions == std::size_t(-1))
                num_partitions = targets.size();
            return target_distribution_policy(std::move(targets), num_partitions);
        }

        /// Create a new \a target_distribution_policy representing the given
        /// target
        ///
        /// \param target [in] The target the new instances should represent
        ///
        target_distribution_policy operator()(target_type const& target,
            std::size_t num_partitions = 1) const
        {
            return target_distribution_policy(target, num_partitions);
        }

        /// Create one object on one of the localities associated by
        /// this policy instance
        ///
        /// \param ts  [in] The arguments which will be forwarded to the
        ///            constructor of the new object.
        ///
        /// \note This function is part of the placement policy implemented by
        ///       this class
        ///
        /// \returns A future holding the global address which represents
        ///          the newly created object
        ///
        template <typename Component, typename ... Ts>
        hpx::future<hpx::id_type> create(Ts &&... ts) const
        {
            target_type t = get_next_target();
            hpx::id_type target_locality = t.get_locality();
            return components::stub_base<Component>::create_async(
                target_locality, std::forward<Ts>(ts)..., std::move(t));
        }

        /// \cond NOINTERNAL
        typedef std::pair<hpx::id_type, std::vector<hpx::id_type> >
            bulk_locality_result;
        /// \endcond

        /// Create multiple objects on the localities associated by
        /// this policy instance
        ///
        /// \param count [in] The number of objects to create
        /// \param vs   [in] The arguments which will be forwarded to the
        ///             constructors of the new objects.
        ///
        /// \note This function is part of the placement policy implemented by
        ///       this class
        ///
        /// \returns A future holding the list of global addresses which
        ///          represent the newly created objects
        ///
        template <typename Component, typename ...Ts>
        hpx::future<std::vector<bulk_locality_result> >
        bulk_create(std::size_t count, Ts &&... ts) const
        {
            // collect all targets per locality
            std::map<hpx::id_type, std::vector<target_type> > targets;
            for(target_type const& t : targets_)
            {
                hpx::id_type target_locality = t.get_locality();
                targets[target_locality].push_back(t);
            }

            std::vector<hpx::id_type> localities;

            std::vector<hpx::future<std::vector<hpx::id_type> > > objs;
            objs.reserve(targets_.size());

            auto end = targets.end();
            for (auto it = targets.begin(); it != end; ++it)
            {
                hpx::id_type target_locality = it->first;
                localities.push_back(target_locality);

                std::size_t num_partitions = 0;
                for (target_type const& t : it->second)
                {
                    num_partitions += get_num_items(count, t);
                }

                objs.push_back(
                    components::stub_base<Component>::bulk_create_async(
                        target_locality, num_partitions, ts..., it->second));
            }

            return hpx::dataflow(
                [=](std::vector<hpx::future<std::vector<hpx::id_type> > > && v)
                    mutable -> std::vector<bulk_locality_result>
                {
                    HPX_ASSERT(localities.size() == v.size());

                    std::vector<bulk_locality_result> result;
                    result.reserve(v.size());

                    for (std::size_t i = 0; i != v.size(); ++i)
                    {
#if !defined(HPX_GCC_VERSION) || HPX_GCC_VERSION >= 408000
                        result.emplace_back(
                                std::move(localities[i]), v[i].get()
                            );
#else
                        result.push_back(std::make_pair(
                                std::move(localities[i]), v[i].get()
                            ));
#endif
                    }
                    return result;
                },
                std::move(objs));
        }

        /// Returns the locality which is anticipated to be used for the next
        /// async operation
        target_type get_next_target() const
        {
            return targets_[next_target_++ % targets_.size()];
        }

        std::size_t get_num_partitions() const
        {
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
        target_distribution_policy(std::vector<target_type> const& targets)
          : targets_(targets), next_target_(0)
        {}

        target_distribution_policy(std::vector<target_type> && targets)
          : targets_(std::move(targets)), next_target_(0)
        {}

        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive& ar, unsigned int const)
        {
            ar & targets_;
        }

        std::vector<target_type> targets_;   // targets
        std::size_t num_partitions_;
        boost::atomic<std::size_t> next_target_;
        /// \endcond
    };
}}

/// \cond NOINTERNAL
namespace hpx { namespace traits
{
    template <typename Target>
    struct is_distribution_policy<compute::target_distribution_policy<Target> >
      : std::true_type
    {};

    template <typename Target>
    struct num_container_partitions<compute::target_distribution_policy<Target> >
    {
        static std::size_t
        call(compute::target_distribution_policy<Target> const& policy)
        {
            return policy.get_num_partitions();
        }
    };
}}
/// \endcond

#endif
