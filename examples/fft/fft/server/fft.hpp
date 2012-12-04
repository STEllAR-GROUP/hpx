#if !defined(HPX_JpLW7ZrPxNxTYtpYkwqERHBvBhmY6p0CvFMQAPwE) 
#define HPX_JpLW7ZrPxNxTYtpYkwqERHBvBhmY6p0CvFMQAPwE

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/actions.hpp>
//#include <hpx/lcos/future.hpp>
//#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>
#include <hpx/include/async.hpp>

#include <vector>

#include "fft_common.hpp"

#define default_size 1024                                                        
#define pi 3.1415926535897932

namespace fft { namespace server
{
    class HPX_COMPONENT_EXPORT fourier_xform
        : public hpx::components::simple_component_base<fourier_xform>
    {
    public:
        typedef fft::complex_type complex_type;
        typedef std::vector<complex_type> complex_vector;
        fourier_xform();
        ~fourier_xform();

        
      
        double magnitude_from_complex(complex_type const& a);
        complex_type complex_from_exponential(double const& m, double const& exp);
        //complex_type complex_from_polar(double const& r, double const& radian);
        complex_type complex_mult(complex_type const& a, complex_type const& b);
        complex_type complex_add(complex_type const& a, complex_type const& b);
        complex_type complex_sub(complex_type const& a, complex_type const& b);
        complex_vector sdft(complex_vector const& x);
        complex_vector r2ditfft(complex_vector const& x);
        complex_vector r2ditfft_args(complex_vector const& even, complex_vector const& odd);

        HPX_DEFINE_COMPONENT_ACTION(fourier_xform, magnitude_from_complex
            , magnitude_from_complex_action);
        HPX_DEFINE_COMPONENT_ACTION(fourier_xform, complex_from_exponential 
            , complex_from_exponential_action);
        //HPX_DEFINE_COMPONENT_ACTION(fourier_xform, complex_from_polar
        //    , complex_from_polar_action);
        HPX_DEFINE_COMPONENT_ACTION(fourier_xform, complex_mult, complex_mult_action);
        HPX_DEFINE_COMPONENT_ACTION(fourier_xform, complex_add, complex_add_action);
        HPX_DEFINE_COMPONENT_ACTION(fourier_xform, complex_sub, complex_sub_action);
        HPX_DEFINE_COMPONENT_ACTION(fourier_xform, sdft, sdft_action);
        HPX_DEFINE_COMPONENT_ACTION(fourier_xform, r2ditfft, r2ditfft_action);
        HPX_DEFINE_COMPONENT_ACTION(fourier_xform, r2ditfft_args, r2ditfft_args_action);
    };
}}
////////////////////////////////////////////////////////////////////////////////

HPX_REGISTER_ACTION_DECLARATION(
    fft::server::fourier_xform::magnitude_from_complex_action
    , cc_magnitude_from_complex_action);
HPX_REGISTER_ACTION_DECLARATION(
    fft::server::fourier_xform::complex_from_exponential_action
    , cc_complex_from_exponential_action);
//HPX_REGISTER_ACTION_DECLARATION_EX(
//    fft::server::fourier_xform::complex_from_polar_action
//    , cc_complex_from_polar_action);
HPX_REGISTER_ACTION_DECLARATION(
    fft::server::fourier_xform::complex_mult_action
    , cc_complex_mult_action);
HPX_REGISTER_ACTION_DECLARATION(
    fft::server::fourier_xform::complex_add_action
    , cc_complex_add_action);
HPX_REGISTER_ACTION_DECLARATION(
    fft::server::fourier_xform::complex_sub_action
    , cc_complex_sub_action);
HPX_REGISTER_ACTION_DECLARATION(
    fft::server::fourier_xform::sdft_action
    , fourier_xform_sdft_action);
HPX_REGISTER_ACTION_DECLARATION(
    fft::server::fourier_xform::r2ditfft_action
    , fourier_xform_r2ditfft_action);
HPX_REGISTER_ACTION_DECLARATION(
    fft::server::fourier_xform::r2ditfft_args_action
    , fourier_xform_r2ditfft_args_action);
#endif //
