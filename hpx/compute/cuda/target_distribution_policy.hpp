//  Copyright (c) 2014-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file target_distribution_policy.hpp

#if !defined(HPX_COMPUTE_CUDA_TARGET_DISTRIBUTION_POLICY)
#define HPX_COMPUTE_CUDA_TARGET_DISTRIBUTION_POLICY

#include <hpx/config.hpp>

#if defined(HPX_HAVE_CUDA) && defined(__CUDACC__)
#include <hpx/traits/is_distribution_policy.hpp>
#include <hpx/runtime/serialization/serialization_fwd.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/dataflow.hpp>

#include <hpx/compute/cuda/get_targets.hpp>

#include <algorithm>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/atomic.hpp>

namespace hpx { namespace components
{
    /// This class specifies the parameters for a simple distribution policy
    /// to use for creating (and evenly distributing) a given number of items
    /// on a given set of localities.
    struct target_distribution_policy
    {
    public:
        /// Default-construct a new instance of a \a target_distribution_policy.
        /// This policy will represent all devices on the current locality.
        ///
        target_distribution_policy()
          : targets_(get_targets()), num_partitions_(1), next_target_(0)
        {}

        /// Create a new \a target_distribution_policy representing the given
        /// set of targets
        ///
        /// \param targets [in] The targets the new instances should represent
        ///
        target_distribution_policy operator()(
            std::vector<cuda::target> const& targets,
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
            std::vector<cuda::target> && targets,
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
        target_distribution_policy operator()(cuda::target const& target,
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
            cuda::target t = get_next_target();
            hpx::id_type target_locality = t.native_handle().get_locality();
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
            std::vector<hpx::id_type> localities;
            std::vector<hpx::future<hpx::id_type> > partitions;
            for(cuda::target const& t : targets_)
            {
                hpx::id_type target_locality = t.native_handle().get_locality();
                localities.push_back(target_locality);
                partitions.push_back(
                    components::stub_base<Component>::create_async(
                        target_locality, ts..., t));
            }

            return hpx::dataflow(
                [=](std::vector<hpx::future<hpx::id_type> > && partitions)
                {
                    HPX_ASSERT(localities.size() == partitions.size());

                    std::vector<bulk_locality_result> result;
                    result.reserve(partitions.size());

                    for (hpx::id_type const& locality : localities)
                    {
                        hpx::id_type locality = t.native_handle().get_locality();
#if !defined(HPX_GCC_VERSION) || HPX_GCC_VERSION >= 408000
                        result.emplace_back(std::move(locality), v[i].get());
#else
                        result.push_back(std::make_pair(
                            std::move(locality), v[i].get()));
#endif
                    }
                    return result;
                });
        }

        /// Returns the locality which is anticipated to be used for the next
        /// async operation
        cuda::target get_next_target() const
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
        target_distribution_policy(std::vector<cuda::target> const& targets)
          : targets_(targets), next_target_(0)
        {}

        target_distribution_policy(std::vector<cuda::target> && targets)
          : targets_(std::move(targets)), next_target_(0)
        {}

        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive& ar, unsigned int const)
        {
            ar & targets_;
        }

        std::vector<cuda::target> targets_;   // target devices
        std::size_t num_partitions_;
        boost::atomic<std::size_t> next_target_;
        /// \endcond
    };

    /// A predefined instance of the \a target_distribution_policy. It will
    /// represent the local locality and will place all items to create here.
    static target_distribution_policy const target;
}}

/// \cond NOINTERNAL
namespace hpx { namespace traits
{
    template <>
    struct is_distribution_policy<
            compute::cuda::target_distribution_policy>
        : std::true_type
    {};
}}
/// \endcond

#endif
