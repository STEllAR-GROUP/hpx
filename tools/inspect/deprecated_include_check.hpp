//  include_check header  -------------------------------------------------------//

//  Copyright Beman Dawes   2002
//  Copyright Rene Rivera   2004.
//  Copyright Gennaro Prota 2006.
//  Copyright Hartmut Kaiser 2016.
//
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_DEPRECATED_INCLUDE_CHECK_HPP
#define HPX_DEPRECATED_INCLUDE_CHECK_HPP

#include "inspector.hpp"

#include "boost/regex.hpp"

#include <vector>

namespace boost
{
  namespace inspect
  {

    struct deprecated_includes
    {
      char const* include_regex;
      char const* use_instead;
    };

    struct deprecated_includes_regex_data
    {
      deprecated_includes_regex_data(deprecated_includes const* d,
            std::string const& rx)
        : data(d), pattern(rx, boost::regex::normal)
      {}

      deprecated_includes const* data;
      boost::regex pattern;
    };

    class deprecated_include_check : public inspector
    {
      long m_errors;
      std::vector<deprecated_includes_regex_data> regex_data;

    public:

      deprecated_include_check();
      virtual const char * name() const { return "*DI*"; }
      virtual const char * desc() const { return "#include'ing deprecated header"; }

      virtual void inspect(
        const std::string & library_name,
        const path & full_path,
        const std::string & contents);

      virtual void print_summary(std::ostream& out)
      {
        out << "  " << m_errors << " deprecated #include's" << line_break();
      }

      virtual ~deprecated_include_check() {}
    };
  }
}

#endif // HPX_DEPRECATED_INCLUDE_CHECK_HPP
