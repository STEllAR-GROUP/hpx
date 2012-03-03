//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_DISTRIBUTING_FACTORY_JUN_20_2008_0948PM)
#define HPX_COMPONENTS_DISTRIBUTING_FACTORY_JUN_20_2008_0948PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/traits/get_remote_result.hpp>

#include <boost/make_shared.hpp>
#include <boost/iterator_adaptors.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/foreach.hpp>

#include <vector>
#include <algorithm>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    class HPX_COMPONENT_EXPORT locality_result_iterator;

    ///////////////////////////////////////////////////////////////////////////
    // exposed functionality of this component
    struct remote_locality_result
    {
        typedef std::vector<naming::gid_type>::iterator iterator;
        typedef std::vector<naming::gid_type>::const_iterator const_iterator;
        typedef std::vector<naming::gid_type>::value_type value_type;

        remote_locality_result()
        {}

        remote_locality_result(naming::gid_type const& prefix,
                components::component_type type)
          : prefix_(prefix), type_(type)
        {}

        naming::gid_type prefix_;             ///< prefix of the locality
        std::vector<naming::gid_type> gids_;  ///< gids of the created components
        components::component_type type_;     ///< type of created components

        iterator begin() { return gids_.begin(); }
        const_iterator begin() const { return gids_.begin(); }
        iterator end() { return gids_.end(); }
        const_iterator end() const { return gids_.end(); }

    private:
        // serialization support
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int)
        {
            ar & prefix_ & gids_ & type_;
        }
    };

    // same as remote_locality_result, except it stores id_type's
    struct locality_result
    {
        typedef std::vector<naming::id_type>::iterator iterator;
        typedef std::vector<naming::id_type>::const_iterator const_iterator;
        typedef std::vector<naming::id_type>::value_type value_type;

        locality_result()
        {}

        locality_result(remote_locality_result const& results)
          : prefix_(results.prefix_), type_(results.type_)
        {
            BOOST_FOREACH(naming::gid_type const& gid, results.gids_)
            {
                gids_.push_back(naming::id_type(gid, naming::id_type::managed));
            }
        }

        iterator begin() { return gids_.begin(); }
        const_iterator begin() const { return gids_.begin(); }
        iterator end() { return gids_.end(); }
        const_iterator end() const { return gids_.end(); }

        naming::gid_type prefix_;             ///< prefix of the locality
        std::vector<naming::id_type> gids_;   ///< gids of the created components
        components::component_type type_;     ///< type of created components
    };

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
            factory_create_partitioned = 1, // create components in partitions
        };

        // constructor
        distributing_factory()
        {}

        typedef std::vector<remote_locality_result> remote_result_type;
        typedef std::vector<locality_result> result_type;

        typedef locality_result_iterator iterator_type;
        typedef
            std::pair<locality_result_iterator, locality_result_iterator>
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

    ///////////////////////////////////////////////////////////////////////////
    /// Special segmented iterator allowing to iterate over all gids referenced
    /// by an instance of a \a distributing_factory#result_type
    class HPX_COMPONENT_EXPORT locality_result_iterator
      : public boost::iterator_facade<
            locality_result_iterator, naming::id_type,
            boost::forward_traversal_tag, naming::id_type const&>
    {
    private:
        typedef distributing_factory::result_type result_type;
        typedef result_type::value_type locality_result_type;

        struct HPX_COMPONENT_EXPORT data
        {
            data();
            data(result_type::const_iterator begin, result_type::const_iterator end);

            void increment();
            bool equal(data const& rhs) const;
            naming::id_type const& dereference() const;

            result_type::const_iterator current_;
            result_type::const_iterator end_;
            locality_result_type::const_iterator current_gid_;

            bool is_at_end_;
        };

    public:
        /// construct begin iterator
        locality_result_iterator(result_type const& results)
          : data_(new data(results.begin(), results.end()))
        {}

        /// construct end iterator
        locality_result_iterator()
          : data_(boost::make_shared<data>())
        {}

    private:
        boost::shared_ptr<data> data_;

        /// support functions needed for a forward iterator
        friend class boost::iterator_core_access;

        void increment()
        {
            data_->increment();
        }

        void decrement() {}
        void advance(difference_type) {}

        bool equal(locality_result_iterator const& rhs) const
        {
            return data_->equal(*rhs.data_);
        }

        naming::id_type const& dereference() const
        {
            return data_->dereference();
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    HPX_COMPONENT_EXPORT
    std::pair<locality_result_iterator, locality_result_iterator>
        locality_results(distributing_factory::result_type const& v);
}}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace traits
{
    // we need to specialize this template to allow for automatic conversion of
    // the vector<remote_locality_result> to a vector<locality_result>
    template <>
    struct get_remote_result<
        std::vector<components::server::locality_result>,
        std::vector<components::server::remote_locality_result> >
    {
        typedef std::vector<components::server::locality_result> result_type;
        typedef std::vector<components::server::remote_locality_result>
            remote_result_type;

        static result_type call(remote_result_type const& rhs)
        {
            result_type result;
            BOOST_FOREACH(remote_result_type::value_type const& r, rhs)
            {
                result.push_back(result_type::value_type(r));
            }
            return result;
        }
    };
}}

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

HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::lcos::base_lco_with_value<
        hpx::components::server::distributing_factory::remote_result_type
    >::set_result_action
  , set_result_action_distributing_factory_result);

#endif
