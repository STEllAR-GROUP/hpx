////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/version.hpp>

namespace hpx
{

boost::uint8_t major_version()
{
    return HPX_VERSION_MAJOR;
}

boost::uint8_t minor_version()
{
    return HPX_VERSION_MINOR;
}

boost::uint8_t subminor_version()
{
    return HPX_VERSION_SUBMINOR;
}

boost::uint32_t full_version()
{
    return HPX_VERSION_FULL;
}

boost::uint8_t agas_version()
{
    return HPX_AGAS_VERSION;
}

}

