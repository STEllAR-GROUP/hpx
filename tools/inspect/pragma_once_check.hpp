//  SPDX_CHECK header  ----------------------------------------------------//

//  Copyright Ste||ar Group
//
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
    class pragma_once_check : public header_inspector
    {
      long m_files_with_errors;
    public:

      pragma_once_check();
      virtual const char * name() const { return "*PRAGMA-ONCE*"; }
      virtual const char * desc() const { return "missing #pragma once"; }

      virtual void inspect(
        const std::string & library_name,
        const path & full_path,
        const std::string & contents );

      virtual void print_summary(std::ostream& out)
        { out << "  " << m_files_with_errors
              << " header files missing #pragma once"
              << line_break(); }

      virtual ~pragma_once_check() {}
    };
  }
}

