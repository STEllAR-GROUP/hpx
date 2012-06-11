//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_DISTRIBUTING_FACTORY_JUN_20_2008_0948PM)
#define HPX_COMPONENTS_DISTRIBUTING_FACTORY_JUN_20_2008_0948PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/util/locality_result.hpp>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/foreach.hpp>

#include <vector>
#include <algorithm>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    /// \brief The partition_info data structure describes the dimensionality
    ///        and the size of the partitions for which components need to be
    ///        created.
    struct HPX_COMPONENT_EXPORT partition_info
    {
        partition_info()
          : dims_(0)
        {}

        partition_info(std::size_t dims, std::vector<std::size_t> const& sizes)
          : dims_(dims), dim_sizes_(sizes)
        {}

        // return the overall size of a partition described by this info
        std::size_t size() const
        {
            return std::accumulate(dim_sizes_.begin(), dim_sizes_.end(),
                std::size_t(1), std::multiplies<std::size_t>());
        }

        std::size_t dims_;                    ///< dimensionality of the partitions
        std::vector<std::size_t> dim_sizes_;  ///< size of dimension of the partitions
    };

    ///////////////////////////////////////////////////////////////////////////
    class HPX_COMPONENT_EXPORT distributing_factory
      : public simple_component_base<distributing_factory>
    {
    public:
        // parcel action code: the action to be performed on the destination
        // object
        enum actions
        {
            factory_create_components = 0,  // create new components
            factory_create_partitioned = 1  // create components in partitions
        };

        // constructor
        distributing_factory()
        {}

        typedef std::vector<util::remote_locality_result> remote_result_type;
        typedef std::vector<util::locality_result> result_type;

        typedef util::locality_result_iterator iterator_type;
        typedef
            std::pair<util::locality_result_iterator, util::locality_result_iterator>
        iterator_range_type;

        /// \brief Action to create new components
        remote_result_type create_components(components::component_type type,
            std::size_t count) const;

        /// \brief Action to create new components based on the given
        ///        partitioning information
        ///
        /// This function will create new components, assuming that \a count
        /// newly created components are to be placed into each partition,
        /// where all the partitions are equal in size as described by the
        /// passed \a partition_info. The number of partitions to create is
        /// specified by \a part_count.
        ///
        /// For example:
        /// <code>
        ///       partition_info pi(3, {2, 2, 2});
        ///       create_partitioned(t, 2, 16, pi);
        /// </code>
        /// will create 32 new components in 16 partitions (2 in each
        /// partition), where each partition is assumed to span 8 localities.
        remote_result_type create_partitioned(components::component_type type,
            std::size_t count, std::size_t part_count,
            partition_info const& info) const;

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into a action
        // type, allowing to generate all require boilerplate code for threads,
        // serialization, etc.
        typedef hpx::actions::result_action2<
            distributing_factory const, remote_result_type,
            factory_create_components, components::component_type, std::size_t,
            &distributing_factory::create_components
        > create_components_action;

        typedef hpx::actions::result_action4<
            distributing_factory const, remote_result_type,
            factory_create_partitioned, components::component_type, std::size_t,
            std::size_t, partition_info const&,
            &distributing_factory::create_partitioned
        > create_partitioned_action;
    };
}}}

///////////////////////////////////////////////////////////////////////////////
// Serialization of partition_info
namespace boost { namespace serialization
{
    template <typename Archive>
    void serialize(Archive&, hpx::components::server::partition_info&,
        unsigned int const);
}}


///////////////////////////////////////////////////////////////////////////////
// Declaration of serialization support for the distributing_factory actions
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::components::server::distributing_factory::create_components_action
  , distributing_factory_create_components_action
)

HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::components::server::distributing_factory::create_partitioned_action
  , distributing_factory_create_partitioned_action
)

#endif
