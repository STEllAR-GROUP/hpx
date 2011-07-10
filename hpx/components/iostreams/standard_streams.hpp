////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_8F5A7F0B_E4CE_422C_B58A_2AEC43AD2057)
#define HPX_8F5A7F0B_E4CE_422C_B58A_2AEC43AD2057

#include <hpx/config.hpp>
#include <hpx/components/iostreams/lazy_ostream.hpp> 

namespace hpx 
{ 
    namespace iostreams
    {
        HPX_COMPONENT_EXPORT void create_cout();
        HPX_COMPONENT_EXPORT void create_cerr();

        HPX_COMPONENT_EXPORT lazy_ostream& cout();
        HPX_COMPONENT_EXPORT lazy_ostream& cerr();
    }

    using iostreams::cout;
    using iostreams::cerr;
}

#endif // HPX_8F5A7F0B_E4CE_422C_B58A_2AEC43AD2057

