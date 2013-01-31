#if !defined(HPX_NbzU3Ec9ZIDaLEo3JAF7bh8h5QFTeisEJ8wfO5wh)
#define HPX_NbzU3Ec9ZIDaLEo3JAF7bh8h5QFTeisEJ8wfO5wh
//#include <hpx/hpx_fwd.hpp>
//#include <hpx/lcos/future.hpp>
#include <hpx/include/components.hpp>
//#include <hpx/runtime/components/client_base.hpp>
#include "stubs/fft.hpp"
#include <vector>
namespace fft
{
    ////////////////////////////////////////////////////////////////////////////
    class fourier_xform
        : public hpx::components::client_base<
            fourier_xform, stubs::fourier_xform
    >
    {
        typedef hpx::components::client_base<
            fourier_xform, stubs::fourier_xform
        > base_type;
        typedef std::vector<fft::complex_type> complex_vector;
    public:
        fourier_xform()
        {}
        //fourier_xform(hpx::naming::id_type const& target_gid)
        //: base_type(base_type::stub_type::create_sync(target_gid))
        //{}

    ////////////////////////////////////////////////////////////////////////////
        //complex_vector sdft(complex_vector const& x)
        //double sdft(double x, double y)
        //{
            //BOOST_ASSERT(this->gid_);
            //return this->base_type::sdft(this->gid_, x);
            //this->base_type::sdft(this->gid_, x, y);
        //}
    ////////////////////////////////////////////////////////////////////////////
        //complex_vector r2ditfft(complex_vector const& x)
        //{
            //return this->base_type::r2ditfft(this->gid_, x);
        //}
    ////////////////////////////////////////////////////////////////////////////
    };
}
#endif
