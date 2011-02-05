//  Copyright (c) 2007-2011 Hartmut Kaiser
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
#include <boost/serialization/vector.hpp>
#include <boost/foreach.hpp>

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

        typedef std::vector<remote_locality_result> remote_result_type;
        typedef std::vector<locality_result> result_type;

        typedef locality_result_iterator iterator_type;
        typedef 
            std::pair<locality_result_iterator, locality_result_iterator>
        iterator_range_type;

        /// \brief Action to create new components
        remote_result_type create_components(components::component_type type, 
            std::size_t count); 

        /// \brief Action to delete existing components
//         void free_components(result_type const& gids, bool sync); 

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into a action
        // type, allowing to generate all require boilerplate code for threads,
        // serialization, etc.
        typedef hpx::actions::result_action2<
            distributing_factory, remote_result_type, factory_create_components, 
            components::component_type, std::size_t, 
            &distributing_factory::create_components
        > create_components_action;

//         typedef hpx::actions::action2<
//             distributing_factory, factory_free_components, 
//             result_type const&, bool, &distributing_factory::free_components
//         > free_components_action;
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

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace detail 
{
    // we need to specialize this template to allow for automatic conversion of
    // the vector<remote_locality_result> to a vector<locality_result>
    template <typename Result, typename RemoteResult>
    struct get_result;

    template <>
    struct get_result<
        std::vector<components::server::locality_result>, 
        std::vector<components::server::remote_locality_result> >
    {
        typedef std::vector<components::server::locality_result> result_type;
        typedef std::vector<components::server::remote_locality_result> remote_result_type;

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
}}}

#endif
