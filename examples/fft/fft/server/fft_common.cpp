#include <hpx/hpx_fwd.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/actions.hpp>
//#include <hpx/lcos/future.hpp>
//#include <hpx/runtime/actions/component_action.hpp>
//#include <hpx/runtime/components/server/simple_component_base.hpp>
//#include <hpx/include/async.hpp>
//#include <vector>

#include "fft_common.hpp"

/*namespace fft{ namespace server{
    fft_common::fft_common(){}
    fft_common::~fft_common(){}

    void fft_common::init(std::string const& filename
        , std::size_t const& num_workers, std::size_t const& num_localities)
    {
        data_.data_filename_ = filename;
        data_.num_workers_ = num_workers;
        data_.num_localities_ = num_localities;
    }

    //bool fft_common::is_instantiated()

    fft::config_data fft_common::config_read()
    {
        return data_;
    }
}}*/

////////////////////////////////////////////////////////////////////////////////
namespace boost { namespace serialization                                        
{                                                                                
    //Implement serialization functions. 
    template <typename Archive>                                                  
    void serialize(Archive& ar, fft::complex_type& c_type,                       
        unsigned int const)                                                      
    {                                                                            
        ar & c_type.re;                                                          
        ar & c_type.im;                                                          
    }

    template <typename Archive>                                                  
    void serialize(Archive& ar, fft::config_data& config_type,                       
        unsigned int const)                                                      
    {                                                                            
        ar & config_type.data_filename_;                                                          
        ar & config_type.symbolic_name_;
        ar & config_type.num_workers_;
        ar & config_type.num_localities_;
        ar & config_type.comp_cardinality_;

    }                                                                            
    //////////////////////////////////////////////////////////////               
    // Explicit instantiation for the correct archive types.                     
#if HPX_USE_PORTABLE_ARCHIVES != 0                                               
    template HPX_COMPONENT_EXPORT void                                           
        serialize(hpx::util::portable_binary_iarchive&, fft::complex_type&,      
        unsigned int const);                                                     
    template HPX_COMPONENT_EXPORT void                                           
        serialize(hpx::util::portable_binary_oarchive&, fft::complex_type&,      
        unsigned int const);                                                     
    template HPX_COMPONENT_EXPORT void                                           
        serialize(hpx::util::portable_binary_iarchive&, fft::config_data&,      
        unsigned int const);                                                     
    template HPX_COMPONENT_EXPORT void                                           
        serialize(hpx::util::portable_binary_oarchive&, fft::config_data&,      
        unsigned int const);                                                     
#else                                                                            
    template HPX_COMPONENT_EXPORT void                                           
        serialize(boost::archive::binary_iarchive&, fft::complex_type&,          
        unsigned int const);                                                     
    template HPX_COMPONENT_EXPORT void                                           
        serialize(boost::archive::binary_oarchive&, fft::complex_type&,          
        unsigned int const);                                                     
    template HPX_COMPONENT_EXPORT void                                           
        serialize(boost::archive::binary_iarchive&, fft::config_data&,          
        unsigned int const);                                                     
    template HPX_COMPONENT_EXPORT void                                           
        serialize(boost::archive::binary_oarchive&, fft::config_data&,          
        unsigned int const);                                                     
#endif
}}

//////////////////////////////////////////////////////////////////////////////
/*typedef hpx::components::simple_component<
    fft::server::fft_common
> fft_common_component_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(                                          
    fft_common_component_type, fft_common_type);

HPX_REGISTER_ACTION_EX(fft::server::fft_common::init_action,  
    fft_fft_common_init_action);
HPX_REGISTER_ACTION_EX(fft::server::fft_common::config_read_action,  
    fft_fft_common_config_read_action);
    */
