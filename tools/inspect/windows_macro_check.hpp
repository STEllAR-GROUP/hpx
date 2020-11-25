//  windows_macro_check header  --------------------------------------------------------//
//  Copyright Ste||ar Group

//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include "inspector.hpp"


namespace boost
{
  namespace inspect
  {
    class windows_macro_check : public inspector
    {
      long m_files_with_errors;
    public:

      windows_macro_check();
      virtual const char * name() const { return "*WINDOWS-MACROS*"; }
      virtual const char * desc() const { return "calls to Windows macros in file"; }

      virtual void inspect(
        const std::string & library_name,
        const path & full_path,
        const std::string & contents );

      virtual void print_summary(std::ostream& out)
        { out << "  " << m_files_with_errors << " files with Windows macros" << line_break(); }

      virtual ~windows_macro_check() {}
    };
  }
}
