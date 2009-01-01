//  Copyright (c) 2007-2009 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_UTIL_MAR_24_2008_1009AM)
#define HPX_UTIL_UTIL_MAR_24_2008_1009AM

#include <boost/config.hpp>
#include <boost/version.hpp>
#include <boost/filesystem/path.hpp>

///////////////////////////////////////////////////////////////////////////////
// Helper macros/functions to overcome the lack of stringstream on certain 
// architectures
#ifdef BOOST_NO_STRINGSTREAM
# include <strstream>
inline std::string HPX_OSSTREAM_GETSTRING (std::ostrstream& ss)
{
    ss << std::ends;
    std::string rval = ss.str ();
    ss.freeze (false);
    return (rval);
}
# define HPX_OSSTREAM std::ostrstream
# define HPX_ISSTREAM std::istrstream
#else
# include <sstream>
# define HPX_OSSTREAM_GETSTRING(ss) ss.str()
# define HPX_OSSTREAM std::ostringstream
# define HPX_ISSTREAM std::istringstream
#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    // reverse the effect of the HPX_MANGLE_... macros in config.hpp
#if !defined(BOOST_WINDOWS)
    inline std::string unmangle_name(std::string const& name)
    {
        // remove the 'libhpx_component_' prefix
        std::string::size_type p = name.find(BOOST_PP_STRINGIZE(HPX_MANGLE_COMPONENT_NAME_PREFIX));
        if (p == 0)
            return name.substr(sizeof(BOOST_PP_STRINGIZE(HPX_MANGLE_COMPONENT_NAME_PREFIX))-1);
        return name;
    }
#elif defined(_DEBUG)
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

    ///////////////////////////////////////////////////////////////////////////
    inline std::string leaf(boost::filesystem::path const& p)
    {
#if BOOST_VERSION >= 103600
        return p.empty() ? std::string() : *--p.end();
#else
        return p.leaf();
#endif
    }

}}

#endif


