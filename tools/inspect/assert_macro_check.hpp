//  assert_macro_check header  --------------------------------------------------------//

//  Copyright Eric Niebler 2010.
//  Based on the apple_macro_check checker by Marshall Clow
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_ASSERT_MACRO_CHECK_HPP
#define BOOST_ASSERT_MACRO_CHECK_HPP

#include "inspector.hpp"


namespace boost
{
  namespace inspect
  {
    class assert_macro_check : public inspector
    {
      long m_files_with_errors;
    public:

      assert_macro_check();
      virtual const char * name() const { return "*ASSERT-MACROS*"; }
      virtual const char * desc() const
        { return "presence of C-style assert macro in file (use HPX_ASSERT instead)"; }

      virtual void inspect(
        const std::string & library_name,
        const path & full_path,
        const std::string & contents );

      virtual void print_summary(std::ostream& out)
        { out << "  " << m_files_with_errors << " files with a C-style assert macro" << line_break(); }

      virtual ~assert_macro_check() {}
    };
  }
}

#endif // BOOST_ASSERT_MACRO_CHECK_HPP
