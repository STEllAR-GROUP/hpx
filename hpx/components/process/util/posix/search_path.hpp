// Copyright (c) 2006, 2007 Julio M. Merino Vidal
// Copyright (c) 2008 Ilya Sokolov, Boris Schaeling
// Copyright (c) 2009 Boris Schaeling
// Copyright (c) 2010 Felipe Tanus, Boris Schaeling
// Copyright (c) 2011, 2012 Jeff Flinn, Boris Schaeling
// Copyright (c) 2016 Hartmut Kaiser
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PROCESS_POSIX_SEARCH_PATH_HPP
#define HPX_PROCESS_POSIX_SEARCH_PATH_HPP

#include <hpx/config.hpp>

#if !defined(HPX_WINDOWS)
#include <hpx/components/process/export_definitions.hpp>

#include <string>

namespace hpx { namespace components { namespace process { namespace posix
{
    HPX_PROCESS_EXPORT std::string search_path(const std::string &filename,
        std::string path = "");
}}}}

#endif
#endif
