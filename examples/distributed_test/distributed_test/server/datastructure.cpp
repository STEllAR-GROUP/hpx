//  Copyright (c) 2012 Vinay C Amatya
//
//  Distributed under the Boost Software License, Version 1.0. (Seec accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/component_factory.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/assert.hpp>
#include "datastructure.hpp"

#include <iostream>

namespace distributed { 
    config_comp::config_comp() {}
    config_comp::~config_comp(){}
    config_comp::config_comp(std::string const& symbolic_name
        , std::size_t num_instances, std::size_t my_cardinality)
        : symbolic_name_(symbolic_name), num_instances_(num_instances)
        , my_cardinality_(my_cardinality)
    {
        std::cout << "Component Configured" << std::endl;
    }
}

namespace distributed { namespace server
{
//  Initialize data for this component
    datastructure::datastructure()
    {
        config_data_.my_cardinality_ = 0;
        config_data_.num_instances_ = 0;
    }
    datastructure::~datastructure(){}

    typedef std::vector<std::size_t> data_type;

    void datastructure::data_init(std::string const& symbolic_name
        , std::size_t num_instances, std::size_t my_cardinality
        , std::size_t init_length, std::size_t init_value)
    {
        BOOST_ASSERT(data_.size() == 0);
        while(data_.size() != init_length)
        {
            data_.push_back(init_value+my_cardinality);
        }
        config_data_.symbolic_name_ = symbolic_name;
        config_data_.num_instances_ = num_instances;
        config_data_.my_cardinality_ = my_cardinality;
        
        std::cout << "My Cardinality:" << my_cardinality << std::endl;
        std::cout << "Num Instances:" << num_instances << std::endl;
        //BOOST_FOREACH(std::size_t temp, data_)
        //{
        //    std::cout << temp << std::endl;
        //} 
        std::vector<size_t>::iterator itr;
        itr = data_.begin();
        while(itr != data_.end())
        {
            std::cout << *itr << std::endl;
            ++itr;
        }
    }

    void datastructure::data_write(std::string const& symbolic_name
        , std::size_t num_instances, std::size_t my_cardinality
        , data_type client_data)
    {
        // get_config(gid_component);  implement?
        //distributed::config_comp config_data = get_config();
        if(config_data_.my_cardinality_ != my_cardinality)
            config_data_.my_cardinality_ = my_cardinality;
        if(data_.size() != client_data.size())
            data_.resize(client_data.size());
        data_ = client_data;
        for (data_type::iterator itr = data_.begin(); itr != data_.end(); ++itr)
            std::cout << "Data:" << *itr << std::endl;
        std::cout<< "Write Data Part for component:" << my_cardinality << std::endl;
    }

    distributed::config_comp datastructure::get_config_info() const
    {
        return config_data_;
    }

    data_type datastructure::get_data()
    {
        return data_;
    }

    std::size_t datastructure::get_data_at(std::size_t pos)
    {
        return data_.at(pos);
    }
}}


///////////////////////////////////////////////////////////////////////////////
namespace boost { namespace serialization
{
    //Implement seriialization functions.
    template <typename Archive>
    void serialize(Archive& ar, distributed::config_comp& cfg, 
        unsigned int const)
    {
        ar & cfg.symbolic_name_;
        ar & cfg.num_instances_;
        ar & cfg.my_cardinality_;                         
    }
    //////////////////////////////////////////////////////////////
    // Explicit instantiation for the correct archive types.
#if HPX_USE_PORTABLE_ARCHIVES != 0
    template HPX_COMPONENT_EXPORT void
    serialize(hpx::util::portable_binary_iarchive&, distributed::config_comp&,
        unsigned int const);
   template HPX_COMPONENT_EXPORT void                                           
    serialize(hpx::util::portable_binary_oarchive&, distributed::config_comp&,       
        unsigned int const);                                                     
#else                                                                            
    template HPX_COMPONENT_EXPORT void                                           
    serialize(boost::archive::binary_iarchive&, distributed::config_comp&,           
        unsigned int const);                                                     
    template HPX_COMPONENT_EXPORT void                                           
    serialize(boost::archive::binary_oarchive&, distributed::config_comp&,           
        unsigned int const);                                                     
#endif                       
}}

//////////////////////////////////////////////////////////////////////// 
typedef distributed::server::datastructure datastructure_type;

//////////////////////////////////////////////////////////////////////
// Serialization support for the actions
HPX_REGISTER_ACTION_EX(datastructure_type::init_action,
    distributed_datastructure_init_action);
HPX_REGISTER_ACTION_EX(datastructure_type::write_action,
    distributed_datastructure_write_action);
HPX_REGISTER_ACTION_EX(distributed::server::datastructure::get_config_action,
    distributed_datastructure_get_config_action);
HPX_REGISTER_ACTION_EX(distributed::server::datastructure::get_data_action,
    distributed_datastructure_get_data_action);
HPX_REGISTER_ACTION_EX(distributed::server::datastructure::get_data_at_action,
    distributed_datastructure_get_data_at_action);
    
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::simple_component<datastructure_type>,
    distributed_datastructure_type);
HPX_DEFINE_GET_COMPONENT_TYPE(datastructure_type);


/*HPX_REGISTER_ACTION_EX(
    hpx::lcos::base_lco_with_value<distributed::config_comp>::set_result_action, 
    set_result_action_distributed_config_comp);
HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    hpx::lcos::base_lco_with_value<distributed::config_comp>,
    hpx::components::component_base_lco_with_value);
*/
