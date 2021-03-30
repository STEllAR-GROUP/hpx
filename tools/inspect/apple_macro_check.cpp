//  apple_macro_check implementation  ------------------------------------------------//

//  Copyright Marshall Clow 2007.
//  Based on the tab-check checker by Beman Dawes
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/modules/filesystem.hpp>

#include "apple_macro_check.hpp"
#include "function_hyper.hpp"
#include <functional>
#include <string>
#include "boost/regex.hpp"

namespace fs = hpx::filesystem;

namespace
{
  boost::regex apple_macro_regex(
    "("
    "^\\s*#\\s*undef\\s*" // # undef
    "\\b(check|verify|require|check_error)\\b"
      // followed by apple macro name, whole word
    ")"
    "|"                   // or (ignored)
    "("
    "//[^\\n]*"           // single line comments (//)
    "|"
    "/\\*.*?\\*/"         // multi line comments (/**/)
    "|"
    "\"(?:\\\\\\\\|\\\\\"|[^\"])*\"" // string literals
    ")"
    "|"                   // or
    "("
    "\\b(check|verify|require|check_error)\\b" // apple macro name, whole word
    "\\s*\\("         // followed by 0 or more spaces and an opening paren
    ")"
    , boost::regex::normal);

} // unnamed namespace


namespace boost
{
  namespace inspect
  {
   apple_macro_check::apple_macro_check() : m_files_with_errors(0)
   {
     register_signature( ".c" );
     register_signature( ".cpp" );
     register_signature( ".cu" );
     register_signature( ".cxx" );
     register_signature( ".h" );
     register_signature( ".hpp" );
     register_signature( ".hxx" );
     register_signature( ".ipp" );
   }

   void apple_macro_check::inspect(
      const string & library_name,
      const path & full_path,   // example: c:/foo/boost/filesystem/path.hpp
      const string & contents )     // contents of file to be inspected
    {
      std::string::size_type p = contents.find("hpxinspect:" "noapple_macros");
      if (p != string::npos)
      {
          // ignore this directive here (it is handled below) if it is followed
          // by a ':'
          if (p == contents.size() - 25 ||
              (contents.size() > p + 25 && contents[p + 25] != ':'))
          {
              return;
          }
      }

      boost::sregex_iterator cur(contents.begin(),
          contents.end(), apple_macro_regex), end;

      long errors = 0;

      for( ; cur != end; ++cur /*, ++m_files_with_errors*/ )
      {
        auto m = *cur;

        if(!m[3].matched)
        {
          std::string found_name(m[3].first, m[3].second);

          std::string tag("hpxinspect:" "noapple_macros:" + found_name);
          if (contents.find(tag) != string::npos)
              continue;

          string::const_iterator it = contents.begin();
          string::const_iterator match_it = m[0].first;

          string::const_iterator line_start = it;

          string::size_type line_number = 1;
          for ( ; it != match_it; ++it) {
              if (string::traits_type::eq(*it, '\n')) {
                  ++line_number;
                  line_start = it + 1; // could be end()
              }
          }
          ++errors;
          error( library_name, full_path,
            "Apple macro clash: " + std::string(m[0].first, m[0].second-1),
            line_number );
        }
      }
      if(errors > 0) {
        ++m_files_with_errors;
      }
    }
  } // namespace inspect
} // namespace boost


