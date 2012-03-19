//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_MAP_HOSTNAMES_AUG_29_2011_1257PM)
#define HPX_UTIL_MAP_HOSTNAMES_AUG_29_2011_1257PM

#include <hpx/hpx_fwd.hpp>

#include <map>
#include <iostream>
#include <fstream>

#if defined(BOOST_MSVC)
#pragma warning(push)
#pragma warning(disable:4251)
#endif

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    // Try to map a given host name based on the list of mappings read from a
    // file
    struct HPX_EXPORT map_hostnames
    {
        typedef HPX_STD_FUNCTION<std::string(std::string const&)>
            transform_function_type;

        map_hostnames(bool debug = false)
          : debug_(debug)
        {}

        void use_suffix(std::string const& suffix)
        {
            suffix_ = suffix;
        }

        void use_prefix(std::string const& prefix)
        {
            prefix_ = prefix;
        }

        void use_transform(transform_function_type f)
        {
            transform_ = f;
        }

        std::string map(std::string host_name, boost::uint16_t port) const;

      private:
        transform_function_type transform_;
        std::string suffix_;
        std::string prefix_;
        bool debug_;
    };
}}

#if defined(BOOST_MSVC)
#pragma warning(pop)
#endif

#endif
