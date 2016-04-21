//  deprecated_name_check implementation  -----------------------------------//

//  Copyright Beman Dawes   2002.
//  Copyright Gennaro Prota 2006.
//  Copyright Hartmut Kaiser 2016.
//
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <algorithm>

#include "deprecated_name_check.hpp"
#include "boost/regex.hpp"
#include "boost/lexical_cast.hpp"
#include "function_hyper.hpp"

#include <set>
#include <string>
#include <vector>

namespace boost
{
  namespace inspect
  {
    deprecated_names const names[] =
    {
      { "(\\bboost\\s*::\\s*move\\b)", "std::move" },
      { "(\\bboost\\s*::\\s*forward\\b)", "std::forward" },
      { "(\\bboost\\s*::\\s*noncopyable\\b)", "HPX_NON_COPYABLE" },
      { "(\\bboost\\s*::\\s*result_of\\b)", "std::result_of" },
      { "(\\bboost\\s*::\\s*decay\\b)", "std::decay" },
//       { "(\\bboost\\s*::\\s*(is_[^\\s]*?\\b))", "std::\\2" },
      { "(\\bboost\\s*::\\s*lock_guard\\b)", "std::lock_guard" },
      { 0, 0 }
    };

    //  deprecated_name_check constructor  --------------------------------- //

    deprecated_name_check::deprecated_name_check()
      : m_errors(0)
    {
      // C/C++ source code...
      register_signature( ".c" );
      register_signature( ".cpp" );
      register_signature( ".cxx" );
      register_signature( ".h" );
      register_signature( ".hpp" );
      register_signature( ".hxx" );
      register_signature( ".inc" );
      register_signature( ".ipp" );

      for (deprecated_names const* names_it = &names[0];
           names_it->name_regex != 0;
           ++names_it)
      {
        std::string rx(names_it->name_regex);
        rx +=
          "|"                   // or (ignored)
          "("
          "//[^\\n]*"           // single line comments (//)
          "|"
          "/\\*.*?\\*/"         // multi line comments (/**/)
          "|"
          "\"([^\"\\\\]|\\\\.)*\"" // string literals
          ")";
        regex_data.push_back(deprecated_names_regex_data(names_it, rx));
      }
    }

    //  inspect ( C++ source files )  ---------------------------------------//

    void deprecated_name_check::inspect(
      const string & library_name,
      const path & full_path,      // example: c:/foo/boost/filesystem/path.hpp
      const string & contents)     // contents of file to be inspected
    {
      if (contents.find( "hpxinspect:" "nodeprecatedname" ) != string::npos)
        return;

      std::set<std::string> found_names;

      // for all given names, check whether any is used
      for (deprecated_names_regex_data const& d : regex_data)
      {
        boost::sregex_iterator cur(contents.begin(), contents.end(), d.pattern), end;
        for(/**/; cur != end; ++cur)
        {
          auto m = *cur;
          if (m[1].matched)
          {
            // avoid errors to be reported twice
            std::string found_name(m[1].first, m[1].second);
            if (found_names.find(found_name) == found_names.end())
            {
              // name was found
              found_names.insert(found_name);

              auto it = contents.begin();
              auto match_it = m[1].first;
              auto line_start = it;

              string::size_type line_number = 1;
              for (/**/; it != match_it; ++it)
              {
                if (string::traits_type::eq(*it, '\n'))
                {
                  ++line_number;
                  line_start = it + 1; // could be end()
                }
              }

              ++m_errors;
              error(library_name, full_path, string(name())
                  + " deprecated name ("
                  + found_name
                  + ") on line "
                  + linelink(full_path, boost::lexical_cast<string>(line_number))
                  + ", use " + m.format(d.data->use_instead)
                  + " instead");
            }
          }
        }
      }
    }

  } // namespace inspect
} // namespace boost

