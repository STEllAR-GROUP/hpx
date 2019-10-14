// Copyright Vladimir Prus 2002-2004.
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// This example shows how a user-defined class can be parsed using
// specific mechanism -- not the iostream operations used by default.
//
// A new class 'magic_number' is defined and the 'validate' method is overloaded
// to validate the values of that class using Boost.Regex.
// To test, run
//
//    regex -m 123-456
//    regex -m 123-4567
//
// The first invocation should output:
//
//   The magic is "456"
//
// and the second invocation should issue an error message.

#include <hpx/datastructures/any.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/program_options.hpp>

#include <iostream>
#include <regex>
#include <string>
#include <vector>

#include <boost/lexical_cast.hpp>

using namespace hpx::program_options;
using namespace std;

/* Define a completely non-sensical class. */
struct magic_number
{
public:
    magic_number(int n)
      : n(n)
    {
    }
    int n;
};

bool operator==(magic_number const& lhs, magic_number const& rhs)
{
    return lhs.n == rhs.n;
}

/* Overload the 'validate' function for the user-defined class.
   It makes sure that value is of form XXX-XXX
   where X are digits and converts the second group to an integer.
   This has no practical meaning, meant only to show how
   regex can be used to validate values.
*/
void validate(
    any& v, const std::vector<std::string>& values, magic_number*, int)
{
    static regex r(R"(\d\d\d-(\d\d\d))");

    // Make sure no previous assignment to 'a' was made.
    validators::check_first_occurrence(v);

    // Extract the first string from 'values'. If there is more than
    // one string, it's an error, and exception will be thrown.
    const string& s = validators::get_single_string(values);

    // Do regex match and convert the interesting part to
    // int.
    smatch match;
    if (regex_match(s, match, r))
    {
        v = any(magic_number(boost::lexical_cast<int>(match[1])));
    }
    else
    {
        throw validation_error(validation_error::invalid_option_value);
    }
}

int main(int ac, char* av[])
{
    try
    {
        options_description desc("Allowed options");
        // clang-format off
        desc.add_options()
            ("help","produce a help screen")
            ("version,v", "print the version number")
            ("magic,m", value<magic_number>(),
                "magic value (in NNN-NNN format)")
            ;
        // clang-format on

        variables_map vm;
        store(parse_command_line(ac, av, desc), vm);

        if (vm.count("help"))
        {
            cout << "Usage: regex [options]\n";
            cout << desc;
            return 0;
        }
        if (vm.count("version"))
        {
            cout << "Version 1.\n";
            return 0;
        }
        if (vm.count("magic"))
        {
            cout << "The magic is \"" << vm["magic"].as<magic_number>().n
                 << "\"\n";
        }
    }
    catch (std::exception const& e)
    {
        cout << e.what() << "\n";
    }
    return 0;
}
