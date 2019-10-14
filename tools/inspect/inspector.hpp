//  inspector header  --------------------------------------------------------//

//  Copyright Beman Dawes 2002.
//  Copyright Rene Rivera 2004.
//  Copyright Gennaro Prota 2006.
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_INSPECTOR_HPP
#define BOOST_INSPECTOR_HPP

#include <hpx/filesystem.hpp>

#include <cstddef>
#include <iostream>
#include <ostream>
#include <set>
#include <string>


using std::string;
using hpx::filesystem::path;

namespace boost
{
  namespace inspect
  {
    typedef std::set< string > string_set;

    const char * line_break();

    path search_root_path();

    class inspector
    {
    protected:
        inspector() {}

    public:
      virtual ~inspector() {}

      virtual const char * name() const = 0; // example: "tab-check"
      virtual const char * desc() const = 0; // example: "verify no tabs"

      // always called:
      virtual void inspect(
        const string & /*library_name*/, // "filesystem"
        const path & /*full_path*/ ) {}  // "c:/foo/boost/filesystem/path.hpp"

      // called only for registered leaf() signatures:
      virtual void inspect(
        const string & library_name, // "filesystem"
        const path & full_path,      // "c:/foo/boost/filesystem/path.hpp"
        const string & contents )    // contents of file
      = 0
      ;

      // called after all paths visited, but still in time to call error():
      virtual void close() {}

      virtual void print_summary(std::ostream& out) = 0;

      // callback used by constructor to register leaf() signature.
      // Signature can be a full file name (Jamfile) or partial (.cpp)
      void register_signature( const string & signature );
      const string_set & signatures() const { return m_signatures; }

      // report error callback (from inspect(), close() ):
      void error(
        const string & library_name,
        const path & full_path,
        const string & msg,
        std::size_t line_number = 0);  // 0 if not available or not applicable

    private:
      string_set m_signatures;
    };

    // for inspection of source code of one form or other
    class source_inspector : public inspector
    {
    public:
      // registers the basic set of known source signatures
      source_inspector();
    };

    // for inspection of hypertext, specifically html
    class hypertext_inspector : public inspector
    {
    public:
      // registers the set of known html source signatures
      hypertext_inspector();
    };

    inline string relative_to( const path & src_arg, const path & base_arg )
    {
      path base( base_arg );
      base.lexically_normal();
      string::size_type pos( base.string().size() );
      string src_arg_s(src_arg.string());
      path src;
      if (pos < src_arg_s.size())
        src = path(src_arg.string().substr(pos));
      else
        src = path(src_arg_s);
      src.lexically_normal();
      return src.string();
    }

    string impute_library( const path & full_dir_path );

  }
}

#endif // BOOST_INSPECTOR_HPP

