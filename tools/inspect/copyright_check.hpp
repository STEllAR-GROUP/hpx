//  copyright_check header  --------------------------------------------------//

//  Copyright Beman Dawes 2002, 2003.
//  Copyright Rene Rivera 2004.
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_COPYRIGHT_CHECK_HPP
#define BOOST_COPYRIGHT_CHECK_HPP

#include "inspector.hpp"

namespace boost
{
  namespace inspect
  {
    class copyright_check : public source_inspector
    {
      long m_files_with_errors;
    public:

      copyright_check();
      virtual const char * name() const { return "*C*"; }
      virtual const char * desc() const { return "missing copyright notice"; }

      virtual void inspect(
        const std::string & library_name,
        const path & full_path,
        const std::string & contents );

      virtual void print_summary(std::ostream& out)
        { out << "  " << m_files_with_errors << " files " << desc() << line_break(); }

      virtual ~copyright_check() {}
    };
  }
}

#endif // BOOST_COPYRIGHT_CHECK_HPP
