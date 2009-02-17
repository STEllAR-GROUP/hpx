//  Copyright (c) 2007-2009 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_DISTRIBUTING_FACTORY_JUN_20_2008_0948PM)
#define HPX_COMPONENTS_DISTRIBUTING_FACTORY_JUN_20_2008_0948PM

#include <vector>

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/iterator_adaptors.hpp>
#include <boost/serialization/serialization.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server 
{
    ///////////////////////////////////////////////////////////////////////////
    class HPX_COMPONENT_EXPORT locality_result_iterator;

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
            factory_free_components = 1,    // free existing components
        };

        // constructor
        distributing_factory()
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component
        struct locality_result
        {
            locality_result()
            {}

            locality_result(naming::id_type const& prefix, 
                    naming::id_type const& first_gid, std::size_t count,
                    components::component_type type)
              : prefix_(prefix), first_gid_(first_gid), count_(count), 
                type_(type)
            {}

            naming::id_type prefix_;    ///< prefix of the locality 
            naming::id_type first_gid_; ///< gid of the first created component
            std::size_t count_;         ///< number of created components
            components::component_type type_; ///< type of created components

        private:
            // serialization support
            friend class boost::serialization::access;

            template<class Archive>
            void serialize(Archive& ar, const unsigned int)
            {
                ar & prefix_ & first_gid_ & count_ & type_;
            }
        };

        typedef std::vector<locality_result> result_type;
        typedef locality_result_iterator iterator_type;
        typedef 
            std::pair<locality_result_iterator, locality_result_iterator>
        iterator_range_type;

        /// \brief Action to create new components
        result_type create_components(components::component_type type, 
            std::size_t count); 

        /// \brief Action to delete existing components
        void free_components(result_type const& gids, bool sync); 

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into a action
        // type, allowing to generate all require boilerplate code for threads,
        // serialization, etc.
        typedef hpx::actions::result_action2<
            distributing_factory, result_type, factory_create_components, 
            components::component_type, std::size_t, 
            &distributing_factory::create_components
        > create_components_action;

        typedef hpx::actions::action2<
            distributing_factory, factory_free_components, 
            result_type const&, bool, &distributing_factory::free_components
        > free_components_action;
    };

    ///////////////////////////////////////////////////////////////////////////
    /// Special segmented iterator allowing to iterate over all gids referenced
    /// by an instance of a \a distributing_factory#result_type 
    class HPX_COMPONENT_EXPORT locality_result_iterator
      : public boost::iterator_facade<
            locality_result_iterator, naming::id_type, 
            boost::forward_traversal_tag, naming::id_type const&
        >
    {
    private:
        typedef distributing_factory::result_type result_type;

        struct HPX_COMPONENT_EXPORT data
        {
            data();
            data(result_type::const_iterator begin, result_type::const_iterator end);

            void increment(); 
            bool equal(data const& rhs) const;
            naming::id_type const& dereference() const;

            result_type::const_iterator current_;
            result_type::const_iterator end_;
            std::size_t count_;
            naming::id_type value_;
            bool is_at_end_;
        };

    public:
        /// construct begin iterator
        locality_result_iterator(result_type const& results)
          : data_(new data(results.begin(), results.end()))
        {}

        /// construct end iterator
        locality_result_iterator()
          : data_(new data())
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

#endif
