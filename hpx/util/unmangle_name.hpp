////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_705C4C2C_8C5F_40F1_9D4B_D6627BAC04DC)
#define HPX_705C4C2C_8C5F_40F1_9D4B_D6627BAC04DC

#include <string>

#include <boost/config.hpp>

#include <hpx/config.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    // reverse the effect of the HPX_MANGLE_... macros in config.hpp
#if !defined(BOOST_WINDOWS)
    inline std::string unmangle_name(std::string const& name)
    {
        // remove the 'libhpx_component_' prefix
        std::string::size_type p = name.find
            (BOOST_PP_STRINGIZE(HPX_MANGLE_COMPONENT_NAME_PREFIX));
        if (p == 0)
            return name.substr(sizeof
                (BOOST_PP_STRINGIZE(HPX_MANGLE_COMPONENT_NAME_PREFIX))-1);
        return name;
    }
#elif defined(HPX_DEBUG)
    inline std::string unmangle_name(std::string const& name)
    {
        // remove the 'd' suffix 
        if (name[name.size()-1] == 'd')
            return name.substr(0, name.size()-1);
        return name;
    }
#else
    inline std::string const& unmangle_name(std::string const& name)
    {
        return name;    // nothing to do here
    }
#endif
}}

#endif // HPX_705C4C2C_8C5F_40F1_9D4B_D6627BAC04DC

