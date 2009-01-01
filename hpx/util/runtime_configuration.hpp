//  Copyright (c) 2007-2009 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_RUNTIME_CONFIGURATION_OCT_02_2008_0530PM)
#define HPX_UTIL_RUNTIME_CONFIGURATION_OCT_02_2008_0530PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/naming/locality.hpp>
#include <hpx/util/ini.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{
    // The runtime_configuration class is a wrapper for the runtime 
    // configuration data allowing to extract configuration information in a 
    // more convenient way
    class runtime_configuration : public section
    {
    public:
        // initialize and load configuration information
        runtime_configuration();
        runtime_configuration(std::vector<std::string> const& prefill);

        // Get the AGAS locality to use 
        naming::locality get_agas_locality();

        // Get the AGAS locality to use (default_address/default_port are 
        // the default values describing the locality to use if no 
        // configuration info can be found).
        naming::locality get_agas_locality(
            std::string default_address, boost::uint16_t default_port);
    };

}}

#endif
