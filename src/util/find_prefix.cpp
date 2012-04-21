////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2012 Bryce Adelstein-Lelbach
//  Copyright (c) 2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/config.hpp>

#include <boost/plugin/dll.hpp>
#include <boost/filesystem/path.hpp>

namespace hpx { namespace util
{
    std::string find_prefix(
        std::string library
        )
    {
        try {
            boost::plugin::dll dll(
                HPX_MANGLE_NAME_STR(library) + HPX_SHARED_LIB_EXTENSION);

            using boost::filesystem::path;

            std::string const prefix =
                path(dll.get_directory()).parent_path().parent_path().string();

            if (prefix.empty())
                return HPX_PREFIX;

            return prefix;
        }
        catch (std::logic_error const&) {
            ;   // just ignore loader problems
        }
        return HPX_PREFIX;
    }
}}

