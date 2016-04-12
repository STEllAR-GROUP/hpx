//  include_check header  -------------------------------------------------------//

//  Copyright Beman Dawes   2002
//  Copyright Rene Rivera   2004.
//  Copyright Gennaro Prota 2006.
//  Copyright Hartmut Kaiser 2016.
//
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_INCLUDE_CHECK_HPP
#define HPX_INCLUDE_CHECK_HPP

#include "inspector.hpp"

namespace boost
{
  namespace inspect
  {
    class include_check : public inspector
    {
      long m_errors;

    public:

      include_check();
      virtual const char * name() const { return "*I*"; }
      virtual const char * desc() const { return "uses of function without "
          "#include'ing corresponding header"; }

      virtual void inspect(
        const std::string & library_name,
        const path & full_path,
        const std::string & contents);

      virtual void print_summary(std::ostream& out)
      {
        out << "  " << m_errors << " missing #include's" << line_break();
      }

      virtual ~include_check() {}
    };
  }
}

#endif // HPX_INCLUDE_CHECK_HPP
