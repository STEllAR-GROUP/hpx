//  deprecated_include_check implementation  --------------------------------//

//  Copyright Beman Dawes   2002.
//  Copyright Gennaro Prota 2006.
//  Copyright Hartmut Kaiser 2016.
//
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <algorithm>

#include "deprecated_include_check.hpp"
#include "boost/regex.hpp"
#include "boost/lexical_cast.hpp"
#include "function_hyper.hpp"

namespace boost
{
  namespace inspect
  {
    deprecated_includes const names[] =
    {
      { "boost/move/move\\.hpp", "utility" },
      { "boost/atomic/atomic\\.hpp", "boost/atomic.hpp" },
      { "boost/thread/locks.hpp", "mutex" },
      { "boost/type_traits/.*?\\.hpp", "type_traits" },
      { 0, 0 }
    };

    //  deprecated_include_check constructor  -------------------------------//

    deprecated_include_check::deprecated_include_check()
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

      for (deprecated_includes const* includes_it = &names[0];
           includes_it->include_regex != 0;
           ++includes_it)
      {
        std::string rx =
            std::string("^\\s*#\\s*include\\s*<(")
          +   includes_it->include_regex
          + ")>\\s*$"
          + "|"
          + "^\\s*#\\s*include\\s*\"("
          +   includes_it->include_regex
          + ")\"\\s*$";

        regex_data.push_back(deprecated_includes_regex_data(includes_it, rx));
      }
    }

    //  inspect ( C++ source files )  ---------------------------------------//

    void deprecated_include_check::inspect(
      const string & library_name,
      const path & full_path,      // example: c:/foo/boost/filesystem/path.hpp
      const string & contents)     // contents of file to be inspected
    {
      if (contents.find( "hpxinspect:" "nodeprecatedinclude" ) != string::npos)
        return;

      std::set<std::string> found_includes;

      // check for all given includes
      for (deprecated_includes_regex_data const& d : regex_data)
      {
        boost::sregex_iterator cur(contents.begin(), contents.end(), d.pattern), end;
        for(/**/; cur != end; ++cur)
        {
          auto m = *cur;
          if (m[1].matched || m[2].matched)
          {
            int idx = (m[1].matched ? 1 : 2);

            // avoid errors to be reported twice
            std::string found_include(m[1].first, m[1].second);
            if (found_includes.find(found_include) == found_includes.end())
            {
              // name was found
              found_includes.insert(found_include);

              // include is missing
              auto it = contents.begin();
              auto match_it = m[idx].first;
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
                  + " deprecated #include ("
                  + found_include
                  + ") on line "
                  + linelink(full_path, boost::lexical_cast<string>(line_number))
                  + " use " + m.format(d.data->use_instead) + " instead");
            }
          }
        }
      }
    }

  } // namespace inspect
} // namespace boost

