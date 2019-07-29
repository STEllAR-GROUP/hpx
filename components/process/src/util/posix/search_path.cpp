// Copyright (c) 2006, 2007 Julio M. Merino Vidal
// Copyright (c) 2008 Ilya Sokolov, Boris Schaeling
// Copyright (c) 2009 Boris Schaeling
// Copyright (c) 2010 Felipe Tanus, Boris Schaeling
// Copyright (c) 2011, 2012 Jeff Flinn, Boris Schaeling
// Copyright (c) 2016 Hartmut Kaiser
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if !defined(HPX_WINDOWS)
#include <hpx/exception.hpp>
#include <hpx/components/process/util/search_path.hpp>

#include <boost/filesystem.hpp>
#include <boost/tokenizer.hpp>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace hpx { namespace components { namespace process { namespace posix
{
    std::string search_path(const std::string &filename, std::string path)
    {
        if (path.empty())
        {
            path = ::getenv("PATH");
            if (path.empty())
            {
                HPX_THROW_EXCEPTION(invalid_status,
                    "process::search_path",
                    "Environment variable PATH not found");
            }
        }

        std::string result;
        typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
        boost::char_separator<char> sep(":");
        tokenizer tok(path, sep);
        for (tokenizer::iterator it = tok.begin(); it != tok.end(); ++it)
        {
            boost::filesystem::path p = *it;
            p /= filename;
            if (!::access(p.c_str(), X_OK))
            {
                result = p.string();
                break;
            }
        }
        return result;
    }
}}}}

#endif
