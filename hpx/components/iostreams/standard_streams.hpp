//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2011-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#if !defined(HPX_8F5A7F0B_E4CE_422C_B58A_2AEC43AD2057)
#define HPX_8F5A7F0B_E4CE_422C_B58A_2AEC43AD2057

#include <hpx/config.hpp>
#include <hpx/components/iostreams/export_definitions.hpp>
#include <hpx/components/iostreams/ostream.hpp>

namespace hpx
{
    HPX_IOSTREAMS_EXPORT extern iostreams::ostream<> cout;
    HPX_IOSTREAMS_EXPORT extern iostreams::ostream<> cerr;

    // special stream which writes to a predefine stringstream on the console
    HPX_IOSTREAMS_EXPORT extern iostreams::ostream<> consolestream;
    HPX_IOSTREAMS_EXPORT std::stringstream const& get_consolestream();
}

#endif // HPX_8F5A7F0B_E4CE_422C_B58A_2AEC43AD2057

