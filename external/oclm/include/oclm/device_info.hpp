
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef OCLM_DEVICE_INFO_HPP
#define OCLM_DEVICE_INFO_HPP

#include <oclm/info.hpp>

#include <string>
#include <vector>

#include <boost/mpl/bool.hpp>

namespace oclm
{
    template <int Name>
    struct device_info
        : info<
            ::cl_device_id
          , std::string
          , ::clGetDeviceInfo
          , Name
        >
    {};
    
    template <typename>
    struct is_device_info
        : boost::mpl::false_
    {};

    template <int Name>
    struct is_device_info<device_info<Name> >
        : boost::mpl::true_
    {};

    extern const device_info<CL_DEVICE_PROFILE>    device_profile;
    extern const device_info<CL_DEVICE_VERSION>    device_version;
    extern const device_info<CL_DEVICE_NAME>       device_name;
    extern const device_info<CL_DEVICE_VENDOR>     device_vendor;
    extern const device_info<CL_DEVICE_EXTENSIONS> device_extensions;

    template <cl_device_type Type>
    struct device_type
        : info<
            ::cl_device_id
          , cl_device_type
          , ::clGetDeviceInfo
          , CL_DEVICE_TYPE
        >
    {
        template <typename T>
        bool operator()(T const & d, std::vector<T> &) const
        {
            return d.get(device_type<Type>()) == Type;
        }
    };

    template <cl_device_type Type>
    struct is_device_info<device_type<Type> >
        : boost::mpl::true_
    {};
    
}

#endif
