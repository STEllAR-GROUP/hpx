//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file host/target_distribution_policy.hpp

#if !defined(HPX_COMPUTE_HOST_TARGET_DISTRIBUTION_POLICY)
#define HPX_COMPUTE_HOST_TARGET_DISTRIBUTION_POLICY

#include <hpx/config.hpp>

#include <hpx/dataflow.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/runtime/serialization/base_object.hpp>
#include <hpx/traits/is_distribution_policy.hpp>
#include <hpx/util/assert.hpp>

#include <hpx/compute/detail/target_distribution_policy.hpp>
#include <hpx/compute/host/target.hpp>

#include <algorithm>
#include <map>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace compute { namespace host
{
    /// A target_distribution_policy used for CPU bound localities.
    struct target_distribution_policy
      : compute::detail::target_distribution_policy<host::target>
    {
        typedef compute::detail::target_distribution_policy<host::target>
            base_type;

        /// Default-construct a new instance of a \a target_distribution_policy.
        /// This policy will represent all devices on the current locality.
        ///
        target_distribution_policy() {}

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
        target_distribution_policy operator()(
            target_type const& target, std::size_t num_partitions = 1) const
        {
            std::vector<target_type> targets;
            targets.push_back(target);
            return target_distribution_policy(std::move(targets), num_partitions);
        }

        /// Create a new \a target_distribution_policy representing the given
        /// target
        ///
        /// \param target [in] The target the new instances should represent
        ///
        target_distribution_policy operator()(
            target_type && target, std::size_t num_partitions = 1) const
        {
            std::vector<target_type> targets;
            targets.push_back(std::move(target));
            return target_distribution_policy(std::move(targets), num_partitions);
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
            target_type t = this->get_next_target();
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
            std::map<hpx::id_type, std::vector<target_type> > m;
            for(target_type const& t : this->targets_)
            {
                m[t.get_locality()].push_back(t);
            }

            std::vector<hpx::id_type> localities;
            localities.reserve(m.size());

            std::vector<hpx::future<std::vector<hpx::id_type> > > objs;
            objs.reserve(m.size());

            auto end = m.end();
            for (auto it = m.begin(); it != end; ++it)
            {
                localities.push_back(std::move(it->first));

                std::size_t num_partitions = 0;
                for (target_type const& t : it->second)
                {
                    num_partitions += this->get_num_items(count, t);
                }

                objs.push_back(
                    components::stub_base<Component>::bulk_create_async(
                        localities.back(), num_partitions, ts...,
                        std::move(it->second)));
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

    protected:
        /// \cond NOINTERNAL
        target_distribution_policy(std::vector<target_type> const& targets,
                std::size_t num_partitions)
          : base_type(targets, num_partitions)
        {}

        target_distribution_policy(std::vector<target_type> && targets,
                std::size_t num_partitions)
          : base_type(std::move(targets), num_partitions)
        {}

        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive& ar, unsigned int const)
        {
            ar & serialization::base_object<base_type>(*this);
        }
        /// \endcond
    };

    /// A predefined instance of the \a target_distribution_policy for
    /// localities. It will represent all NUMA domains of the given locality
    /// and will place all items to create here.
    static target_distribution_policy const target_layout;
}}}

/// \cond NOINTERNAL
namespace hpx { namespace traits
{
    template <>
    struct is_distribution_policy<compute::host::target_distribution_policy>
      : std::true_type
    {};

    template <>
    struct num_container_partitions<compute::host::target_distribution_policy>
    {
        static std::size_t
        call(compute::host::target_distribution_policy const& policy)
        {
            return policy.get_num_partitions();
        }
    };
}}
/// \endcond

#endif
