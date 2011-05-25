////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_03FA32FA_F3AF_42EC_AAA0_F158404425A3)
#define HPX_03FA32FA_F3AF_42EC_AAA0_F158404425A3

#include <hpx/config.hpp>

namespace hpx { namespace util
{

#if !defined(BOOST_WINDOWS)
    std::string mangle_component_name (std::string const& name)
    {
        return std::string(BOOST_PP_STRINGIZE(HPX_MANGLE_COMPONENT_NAME_PREFIX))
             + name + HPX_SHARED_LIB_EXTENSION; 
    }

    std::string mangle_name (std::string const& name)
    { return std::string("lib") + name + HPX_SHARED_LIB_EXTENSION; }
#elif(HPX_DEBUG)
    std::string mangle_component_name (std::string const& name)
    { return name + "d"; }

    std::string mangle_name (std::string const& name)
    { return name + "d"; }
#else
    std::string const& mangle_component_name (std::string const& name)
    { return name; }

    std::string const& mangle_name (std::string const& name)
    { return name; }
#endif

}}

#endif // HPX_03FA32FA_F3AF_42EC_AAA0_F158404425A3

