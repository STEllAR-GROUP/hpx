//  Copyright (c) 2019 The STE||AR GROUP
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_CONFIG_FORCE_LINKING_HPP)
#define HPX_CONFIG_FORCE_LINKING_HPP

namespace hpx { namespace config
{
    struct force_linking_helper
    {
        const char* const hpx_version;
        const char* const boost_version;
    };

    force_linking_helper& force_linking();
}}

#endif
