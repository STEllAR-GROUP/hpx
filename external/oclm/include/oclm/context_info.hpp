
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef OCLM_CONTEXT_INFO_HPP
#define OCLM_CONTEXT_INFO_HPP

#include <oclm/info.hpp>

#include <string>
#include <vector>

#include <boost/mpl/bool.hpp>

namespace oclm
{
    template <int Name, typename T>
    struct context_info
        : info<
            ::cl_context
          , T
          , ::clGetContextInfo
          , Name
        >
    {};
    
    template <typename>
    struct is_context_info
        : boost::mpl::false_
    {};

    template <int Name, typename T>
    struct is_context_info<context_info<Name, T> >
        : boost::mpl::true_
    {};

    extern const context_info<CL_CONTEXT_DEVICES, std::vector<cl_device_id> >  context_devices;
    // TODO: add remaining context infos ...
}

#endif
