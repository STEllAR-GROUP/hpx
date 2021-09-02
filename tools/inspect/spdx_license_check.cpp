//  SPDX_CHECK implementation  --------------------------------------------//
//  Copyright Ste||ar Group
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/local/config.hpp>

#include "boost/regex.hpp"
#include "spdx_license_check.hpp"
#include "function_hyper.hpp"

namespace
{
  boost::regex spdx_license_regex(
    //~ The next two lines change the regex so that it detects when the license
    //~ doesn't follow the preferred statement.
    "spdx-license-identifier[\\s\\W]*:[\\s\\W]+bsl-",
    boost::regbase::normal | boost::regbase::icase);

} // unnamed namespace

namespace boost
{
  namespace inspect
  {
   spdx_license_check::spdx_license_check() : m_files_with_errors(0)
   {
   }

   void spdx_license_check::inspect(
      const string & library_name,
      const path & full_path,   // example: c:/foo/boost/filesystem/path.hpp
      const string & contents )     // contents of file to be inspected
    {
      if (contents.find( "hpxinspect:" "nospdxlicense" ) != string::npos) return;

      if ( !boost::regex_search( contents, spdx_license_regex ) )
      {
        ++m_files_with_errors;
        error( library_name, full_path, loclink(full_path, name()) );
      }
    }
  } // namespace inspect
} // namespace boost
