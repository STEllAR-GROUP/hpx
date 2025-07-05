//  path_name_check implementation  ------------------------------------------//

//  Copyright Beman Dawes 2002.
//  Copyright Gennaro Prota 2006.
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/modules/filesystem.hpp>
#include <hpx/util/to_string.hpp>

#include "function_hyper.hpp"
#include "path_name_check.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <string>

using std::string;

namespace {
    const char allowable[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-.";
    const char initial_char[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.";
}    // namespace

namespace boost { namespace inspect {

    file_name_check::file_name_check()
      : m_name_errors(0)
    {
    }

    void file_name_check::inspect(
        const string& library_name, const path& full_path)
    {
        //  called for each file and directory, so only the leaf need be tested
        string const leaf(full_path.filename().string());
        string::size_type pos = leaf.find_first_not_of(allowable);

        //  includes only allowable characters
        if (pos != string::npos)
        {
            ++m_name_errors;
            error(library_name, full_path,
                loclink(full_path, string(name())) +
                    " file or directory name contains unacceptable character "
                    "'" +
                    leaf[pos] + "'");
        }

        //  allowable initial character
        if (std::strchr(initial_char, leaf[0]) == nullptr)
        {
            ++m_name_errors;
            error(library_name, full_path,
                loclink(full_path, string(name())) +
                    " file or directory name begins with an unacceptable "
                    "character");
        }

        // We want to inspect some of the hidden dirs

        ////  rules for dot characters differ slightly for directories and files
        //if ( hpx::filesystem::is_directory( full_path ) )
        //{
        //  if ( std::strchr( leaf.c_str(), '.' ) )
        //  {
        //    ++m_name_errors;
        //    error( library_name, full_path, loclink(full_path, string(name()))
        //        + " directory name contains a dot character ('.')" );
        //  }
        //}
        //else // not a directory
        //{
        //  //  includes at most one dot character
        //  const char * first_dot = std::strchr( leaf.c_str(), '.' );
        //  if ( first_dot && std::strchr( first_dot+1, '.' ) )
        //  {
        //    ++m_name_errors;
        //    error( library_name, full_path, string(name())
        //        + " file name with more than one dot character ('.')" );
        //  }
        //}

        //  the path, including a presumed root, does not exceed the maximum size
        path const relative_path(relative_to(full_path, search_root_path()));
        const unsigned max_relative_path =
            207;    // ISO 9660:1999 sets this limit
        const string generic_root("boost_X_XX_X/");
        if (relative_path.string().size() >
            (max_relative_path - generic_root.size()))
        {
            ++m_name_errors;
            error(library_name, full_path,
                loclink(full_path, string(name())) + " path will exceed " +
                    hpx::util::to_string(max_relative_path) +
                    " characters in a directory tree with a root in the form " +
                    generic_root +
                    ", and this exceeds ISO 9660:1999 limit of 207");
        }
    }
}}    // namespace boost::inspect
