//  end_check implementation  -------------------------------------------------//

//  Copyright Beman Dawes 2002.
//  Copyright Daniel James 2009.
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/modules/string_util.hpp>

#include <boost/next_prior.hpp>

#include <iterator>
#include <string>
#include "end_check.hpp"
#include "function_hyper.hpp"

namespace boost { namespace inspect {
    end_check::end_check()
      : m_files_with_errors(0)
    {
        register_signature(".c");
        register_signature(".cpp");
        register_signature(".cu");
        register_signature(".cxx");
        register_signature(".h");
        register_signature(".hpp");
        register_signature(".hxx");
        register_signature(".ipp");
        register_signature(".ixx");
    }

    void end_check::inspect(const string& library_name,
        const path& full_path,     // example: c:/foo/boost/filesystem/path.hpp
        const string& contents)    // contents of file to be inspected
    {
        if (contents.find("hpxinspect:"
                          "noend") != string::npos)
            return;
        hpx::string_util::char_separator sep(
            "\n", "", hpx::string_util::empty_token_policy::keep);
        hpx::string_util::tokenizer tokens(contents, sep);
        const auto linenumb = std::distance(tokens.begin(), tokens.end());
        std::string lineloc = std::to_string(linenumb);
        // this file deliberately contains errors
        const char test_file_name[] = "wrong_line_ends_test.cpp";

        char final_char = contents.begin() == contents.end() ?
            '\0' :
            *(boost::prior(contents.end()));

        bool failed = final_char != '\n' && final_char != '\r';

        if (failed && full_path.filename() != test_file_name)
        {
            ++m_files_with_errors;
            error(library_name, full_path, string(name()) + ' ' + desc());
        }

        if (!failed && full_path.filename() == test_file_name)
        {
            ++m_files_with_errors;
            error(library_name, full_path,
                string(name()) +
                    wordlink(full_path, lineloc, " should end with a newline"));
        }
    }
}}    // namespace boost::inspect
