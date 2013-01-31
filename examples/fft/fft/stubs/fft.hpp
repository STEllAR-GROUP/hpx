#if !defined(HPX_8kpSZmgLBu9DfSbKYoVbf1YUDH00Vb1G2iCALrJJ)
#define HPX_8kpSZmgLBu9DfSbKYoVbf1YUDH00Vb1G2iCALrJJ
#include <hpx/hpx_fwd.hpp>
//#include <hpx/lcos/future.hpp>
//#include <hpx/runtime/components/stubs/stub_base.hpp>
#include "../server/fft.hpp"

#include <vector>
////////////////////////////////////////////////////////////////////////////////
namespace fft { namespace stubs
{
    ////////////////////////////////////////////////////////////////////////////
    struct fourier_xform
        : hpx::components::stub_base<fft::server::fourier_xform>
    {
        typedef server::fourier_xform::complex_vector complex_vector;
    ////////////////////////////////////////////////////////////////////////////
   
        static hpx::lcos::future<complex_vector>
        sdft_async(hpx::naming::id_type const& gid, complex_vector const& x)
        {
            typedef server::fourier_xform::sdft_action action_type;
            return hpx::async<action_type>(gid, x);
        }

        static complex_vector sdft(hpx::naming::id_type const& gid,
            complex_vector const& x)
        {
            return sdft_async(gid, x).get();
        }
    ////////////////////////////////////////////////////////////////////////////

        static hpx::lcos::future<complex_vector>
        r2ditfft_async(hpx::naming::id_type const& gid, complex_vector const& x)
        {
            typedef server::fourier_xform::r2ditfft_action action_type;
            return hpx::async<action_type>(gid, x);
        }

        static complex_vector r2ditfft(hpx::naming::id_type const& gid, 
            complex_vector const& x)
        {
            return r2ditfft_async(gid, x).get();
        }
    ////////////////////////////////////////////////////////////////////////////

        static hpx::lcos::future<complex_vector>
            r2ditfft_args_async(hpx::naming::id_type const& gid
            , complex_vector const& even, complex_vector const& odd)
        {
            typedef server::fourier_xform::r2ditfft_args_action action_type;
            return hpx::async<action_type>(gid, even, odd);
        }

        static complex_vector r2ditfft_args(hpx::naming::id_type const& gid, 
            complex_vector const& even, complex_vector const& odd)
        {
            return r2ditfft_args_async(gid, even, odd).get();
        }
        ////////////////////////////////////////////////////////////////////////////
    };
}}
#endif
