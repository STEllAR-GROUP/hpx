//  crfl_check header  --------------------------------------------------------//

//  Copyright Beman Dawes 2002.
//  Copyright Rene Rivera 2004.
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

//  Contributed by Joerg Walter

#pragma once

#include "inspector.hpp"

namespace boost
{
  namespace inspect
  {
    class crlf_check : public source_inspector
    {
      long m_files_with_errors;
    public:

      crlf_check();
      virtual const char * name() const { return "*EOL*"; }
      virtual const char * desc() const { return "invalid (cr only) line-ending"; }

      virtual void inspect(
        const std::string & library_name,
        const path & full_path,
        const std::string & contents );

      virtual void print_summary(std::ostream& out)
        { out << "  " << m_files_with_errors << " files with invalid line endings" << line_break(); }

      virtual ~crlf_check() {}
    };
  }
}

