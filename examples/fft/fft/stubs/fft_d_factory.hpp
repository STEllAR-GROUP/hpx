#if !defined(HPX_HhLFWilVnNcq2zaQaiAB86RYBnXzRz7mnITzdwH8)
#define HPX_HhLFWilVnNcq2zaQaiAB86RYBnXzRz7mnITzdwH8
#include <hpx/hpx_fwd.hpp>                                          
#include <hpx/lcos/future.hpp>                                      
#include <hpx/runtime/components/stubs/stub_base.hpp>                            
#include "../server/fft_d_factory.hpp"

namespace fft { namespace stubs{
    struct d_factory : stub_base<server::d_factory>
    {
        //////////////////////////////////////////////////////////////////////
        typedef server::d_factory::result_type result_type;
        typedef server::d_factory::remote_result_type remote_result_type;
        
    };
}}
#endif //HPX_HhLFWilVnNcq2zaQaiAB86RYBnXzRz7mnITzdwH8
