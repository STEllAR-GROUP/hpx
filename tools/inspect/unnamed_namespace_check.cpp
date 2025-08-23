//  unnamed_namespace_check -----------------------------------------//

//  Copyright Gennaro Prota 2006.
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#include "boost/regex.hpp"
#include "function_hyper.hpp"
#include "unnamed_namespace_check.hpp"

#include <string>

namespace {

    boost::regex unnamed_namespace_regex(
        "\\<namespace\\s*(\\?\\?<|\\{)"    // trigraph ??< or {
    );

}    // namespace

namespace boost { namespace inspect {
    unnamed_namespace_check::unnamed_namespace_check()
      : m_errors(0)
    {
        register_signature(".h");
        register_signature(".hh");    // just in case
        register_signature(".hpp");
        register_signature(".hxx");    // just in case
        register_signature(".inc");
        register_signature(".ipp");
        register_signature(".inl");
        register_signature(".ixx");
    }

    void unnamed_namespace_check::inspect(const string& library_name,
        const path& full_path,     // example: c:/foo/boost/filesystem/path.hpp
        const string& contents)    // contents of file to be inspected
    {
        if (contents.find("hpxinspect:"
                          "nounnamed") != string::npos)
            return;

        boost::sregex_iterator cur(
            contents.begin(), contents.end(), unnamed_namespace_regex),
            end;
        for (; cur != end; ++cur, ++m_errors)
        {
            const string::size_type ln =
                std::count(contents.begin(), (*cur)[0].first, '\n') + 1;

            error(library_name, full_path,
                "Unnamed namespace" + linelink(full_path, std::to_string(ln)));
        }
    }
}}    // namespace boost::inspect
