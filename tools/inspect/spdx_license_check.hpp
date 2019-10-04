//  SPDX_CHECK header  ----------------------------------------------------//

//  Copyright Ste||ar Group
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_SPDX_LICENSE_CHECK_HPP
#define BOOST_SPDX_LICENSE_CHECK_HPP

#include "inspector.hpp"

namespace boost
{
  namespace inspect
  {
    class spdx_license_check : public source_inspector
    {
      long m_files_with_errors;
    public:

      spdx_license_check();
      virtual const char * name() const { return "*SPDX-Lic*"; }
      virtual const char * desc() const { return "missing SPDX license info, or wrong reference text"; }

      virtual void inspect(
        const std::string & library_name,
        const path & full_path,
        const std::string & contents );

      virtual void print_summary(std::ostream& out)
        { out << "  " << m_files_with_errors
              << " files missing SPDX license info or having wrong reference text"
              << line_break(); }

      virtual ~spdx_license_check() {}
    };
  }
}

#endif // BOOST_SPDX_LICENSE_CHECK_HPP
