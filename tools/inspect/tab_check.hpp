//  tab_check header  --------------------------------------------------------//

//  Copyright Beman Dawes 2002.
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
    class tab_check : public inspector
    {
      long m_files_with_errors;
    public:

      tab_check();
      virtual const char * name() const { return "*Tabs*"; }
      virtual const char * desc() const { return "tabs in file"; }

      virtual void inspect(
        const std::string & library_name,
        const path & full_path,
        const std::string & contents );

      virtual void print_summary(std::ostream& out)
        { out << "  " << m_files_with_errors << " files with tabs"
              << line_break(); }

      virtual ~tab_check() {}
    };
  }
}

