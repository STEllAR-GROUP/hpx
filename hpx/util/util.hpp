//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_UTIL_MAR_24_2008_1009AM)
#define HPX_UTIL_UTIL_MAR_24_2008_1009AM

#include <boost/config.hpp>

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

#endif


