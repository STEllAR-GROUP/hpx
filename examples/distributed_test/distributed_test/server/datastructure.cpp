

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
    datastructure::datastructure(){}
    datastructure::~datastructure(){}
    void datastructure::data_init(std::string const& symbolic_name
        , std::size_t num_instances, std::size_t my_cardinality
        , std::size_t init_length, std::size_t init_value)
    {
        BOOST_ASSERT(data_.size() == 0);
        while(data_.size() != init_length)
        {
            data_.push_back(init_value+my_cardinality);
        }
        
        std::cout << "My Cardinality:" << my_cardinality << std::endl;
        std::cout << "Num Instances:" << num_instances << std::endl;
        BOOST_FOREACH(std::size_t temp, data_)
        {
            std::cout << temp << std::endl;
        } 
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

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::simple_component<datastructure_type>,
    distributed_datastructure_type);
HPX_DEFINE_GET_COMPONENT_TYPE(datastructure_type);

//what doesthis do?
/*HPX_REGISTER_ACTION_EX(
    hpx::lcos::base_lco_with_value<distributed::config_comp>::set_result_action, 
    set_result_action_distributed_config_comp);
HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    hpx::lcos::base_lco_with_value<distributed::config_comp>,
    hpx::components::component_base_lco_with_value);
*/