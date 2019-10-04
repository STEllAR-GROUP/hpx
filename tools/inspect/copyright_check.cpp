//  copyright_check implementation  ------------------------------------------------//

//  Copyright Beman Dawes 2002.
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#include "copyright_check.hpp"
#include "function_hyper.hpp"

namespace boost
{
  namespace inspect
  {
   copyright_check::copyright_check() : m_files_with_errors(0)
   {
   }

   void copyright_check::inspect(
      const string & library_name,
      const path & full_path,   // example: c:/foo/boost/filesystem/path.hpp
      const string & contents )     // contents of file to be inspected
    {
      if (contents.find( "hpxinspect:" "nocopyright" ) != string::npos) return;

      if ( contents.find( "Copyright" ) == string::npos
        && contents.find( "copyright" ) == string::npos )
      {
        ++m_files_with_errors;
        std::string lineloc = loclink(full_path, name());
        error( library_name, full_path, lineloc );
      }
    }
  } // namespace inspect
} // namespace boost


