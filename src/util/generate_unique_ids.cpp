////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/config.hpp>

#if   HPX_AGAS_VERSION < 0x20 
    #include "generate_unique_ids_v1.ipp"
#elif HPX_AGAS_VERSION < 0x30 
    #include "generate_unique_ids_v2.ipp"
#else
    #error Unknown AGAS version 
#endif

