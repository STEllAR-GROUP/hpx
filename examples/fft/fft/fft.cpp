
#include <hpx/hpx.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/serialization.hpp>

#include "server/fft.hpp"

//Add factory registration functionality.
//
HPX_REGISTER_COMPONENT_MODULE();
//////////////////////////////////////////////////////////////////////
typedef hpx::components::simple_component<
    fft::server::fourier_xform
> fourier_xform_component_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(                                          
    fourier_xform_component_type, fourier_xform_type);                           
//////////////////////////////////////////////////////////////////////////////// 
// Serialization support  
HPX_REGISTER_ACTION(                                              
    fft::server::fourier_xform::magnitude_from_complex_action                     
    , cc_magnitude_from_complex_action);                                         
HPX_REGISTER_ACTION(                                              
    fft::server::fourier_xform::complex_from_exponential_action                   
    , cc_complex_from_exponential_action);                                       
//HPX_REGISTER_ACTION_EX(                                            
//    fft::server::fourier_xform::complex_from_polar_action                       
//    , cc_complex_from_polar_action);                                           
HPX_REGISTER_ACTION(                                              
    fft::server::fourier_xform::complex_mult_action                               
    , cc_complex_mult_action);                                                   
HPX_REGISTER_ACTION(                                              
    fft::server::fourier_xform::complex_add_action                                
    , cc_complex_add_action);                                                    
HPX_REGISTER_ACTION(                                              
    fft::server::fourier_xform::complex_sub_action                                
    , cc_complex_sub_action);
HPX_REGISTER_ACTION(                                                          
    fft::server::fourier_xform::sdft_action                                      
    , fourier_xform_sdft_action);
HPX_REGISTER_ACTION(                                                          
    fft::server::fourier_xform::r2ditfft_action                                   
    , fourier_xform_r2ditfft_action);
HPX_REGISTER_ACTION(                                                          
    fft::server::fourier_xform::r2ditfft_args_action                                   
    , fourier_xform_r2ditfft_args_action);
    
