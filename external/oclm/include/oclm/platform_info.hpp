
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef OCLM_PLATFORM_INFO_HPP
#define OCLM_PLATFORM_INFO_HPP

#include <oclm/info.hpp>

#include <string>

#include <boost/mpl/bool.hpp>

namespace oclm
{
    template <int Name>
    struct platform_info
        : info<
            ::cl_platform_id
          , std::string
          , ::clGetPlatformInfo
          , Name
        >
    {};
    
    template <typename>
    struct is_platform_info
        : boost::mpl::false_
    {};

    template <int Name>
    struct is_platform_info<platform_info<Name> >
        : boost::mpl::true_
    {};

    extern const platform_info<CL_PLATFORM_PROFILE>    platform_profile;
    extern const platform_info<CL_PLATFORM_VERSION>    platform_version;
    extern const platform_info<CL_PLATFORM_NAME>       platform_name;
    extern const platform_info<CL_PLATFORM_VENDOR>     platform_vendor;
    extern const platform_info<CL_PLATFORM_EXTENSIONS> platform_extensions;
}

#endif
