////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_63EF92C7_B763_4D30_9188_D296BD91AFB0)
#define HPX_63EF92C7_B763_4D30_9188_D296BD91AFB0

#include <boost/config.hpp>

///////////////////////////////////////////////////////////////////////////////
// Helper macros/functions to overcome the lack of stringstream on certain
// architectures
#if defined(BOOST_NO_STRINGSTREAM)
#include <strstream>

namespace hpx { namespace util
{
    typedef std::ostrstream osstream;
    typedef std::istrstream isstream;

    inline std::string osstream_get_string (osstream& ss)
    {
        std::string rval = ss.str();
        ss.freeze(false);
        return rval;
    }

    inline std::string isstream_get_string (isstream& ss)
    {
        std::string rval = ss.str();
        ss.freeze(false);
        return rval;
    }

}}
#else
#include <sstream>

namespace hpx { namespace util
{
    typedef std::ostringstream osstream;
    typedef std::istringstream isstream;

    inline std::string osstream_get_string (osstream& ss)
    { return ss.str(); }

    inline std::string isstream_get_string (isstream& ss)
    { return ss.str(); }

}}
#endif

#endif // HPX_63EF92C7_B763_4D30_9188_D296BD91AFB0

