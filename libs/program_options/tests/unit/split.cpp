//  Copyright Sascha Ochsenknecht 2009.
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt
//  or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/modules/testing.hpp>

#include <hpx/program_options/cmdline.hpp>
#include <hpx/program_options/options_description.hpp>
#include <hpx/program_options/parsers.hpp>
#include <hpx/program_options/value_semantic.hpp>
#include <hpx/program_options/variables_map.hpp>

#include <cstddef>
#include <string>
#include <vector>

using namespace hpx::program_options;
using namespace std;

void check_value(const string& option, const string& value)
{
    HPX_TEST_EQ(option, value);
}

void split_whitespace(const options_description& description)
{
    const char* cmdline =
        "prg --input input.txt \r --optimization 4  \t  --opt \n  option";

    vector<string> tokens = split_unix(cmdline, " \t\n\r");

    HPX_TEST_EQ(tokens.size(), std::size_t(7));

    check_value(tokens[0], "prg");
    check_value(tokens[1], "--input");
    check_value(tokens[2], "input.txt");
    check_value(tokens[3], "--optimization");
    check_value(tokens[4], "4");
    check_value(tokens[5], "--opt");
    check_value(tokens[6], "option");

    variables_map vm;
    store(command_line_parser(tokens).options(description).run(), vm);
    notify(vm);
}

void split_equalsign(const options_description& description)
{
    const char* cmdline =
        "prg --input=input.txt  --optimization=4 --opt=option";

    vector<string> tokens = split_unix(cmdline, "= ");

    HPX_TEST_EQ(tokens.size(), std::size_t(7));
    check_value(tokens[0], "prg");
    check_value(tokens[1], "--input");
    check_value(tokens[2], "input.txt");
    check_value(tokens[3], "--optimization");
    check_value(tokens[4], "4");
    check_value(tokens[5], "--opt");
    check_value(tokens[6], "option");

    variables_map vm;
    store(command_line_parser(tokens).options(description).run(), vm);
    notify(vm);
}

void split_semi(const options_description& description)
{
    const char* cmdline = "prg;--input input.txt;--optimization 4;--opt option";

    vector<string> tokens = split_unix(cmdline, "; ");

    HPX_TEST_EQ(tokens.size(), std::size_t(7));
    check_value(tokens[0], "prg");
    check_value(tokens[1], "--input");
    check_value(tokens[2], "input.txt");
    check_value(tokens[3], "--optimization");
    check_value(tokens[4], "4");
    check_value(tokens[5], "--opt");
    check_value(tokens[6], "option");

    variables_map vm;
    store(command_line_parser(tokens).options(description).run(), vm);
    notify(vm);
}

void split_quotes(const options_description& description)
{
    const char* cmdline =
        R"(prg --input "input.txt input.txt" --optimization 4 --opt "option1 option2")";

    vector<string> tokens = split_unix(cmdline, " ");

    HPX_TEST_EQ(tokens.size(), std::size_t(7));
    check_value(tokens[0], "prg");
    check_value(tokens[1], "--input");
    check_value(tokens[2], "input.txt input.txt");
    check_value(tokens[3], "--optimization");
    check_value(tokens[4], "4");
    check_value(tokens[5], "--opt");
    check_value(tokens[6], "option1 option2");

    variables_map vm;
    store(command_line_parser(tokens).options(description).run(), vm);
    notify(vm);
}

void split_escape(const options_description& description)
{
    const char* cmdline =
        R"(prg --input \"input.txt\" --optimization 4 --opt \"option1\ option2\")";

    vector<string> tokens = split_unix(cmdline, " ");

    HPX_TEST_EQ(tokens.size(), std::size_t(7));
    check_value(tokens[0], "prg");
    check_value(tokens[1], "--input");
    check_value(tokens[2], "\"input.txt\"");
    check_value(tokens[3], "--optimization");
    check_value(tokens[4], "4");
    check_value(tokens[5], "--opt");
    check_value(tokens[6], "\"option1 option2\"");

    variables_map vm;
    store(command_line_parser(tokens).options(description).run(), vm);
    notify(vm);
}

void split_single_quote(const options_description& description)
{
    const char* cmdline = "prg --input 'input.txt input.txt' --optimization 4 "
                          "--opt 'option1 option2'";

    vector<string> tokens = split_unix(cmdline, " ", "'");

    HPX_TEST_EQ(tokens.size(), std::size_t(7));
    check_value(tokens[0], "prg");
    check_value(tokens[1], "--input");
    check_value(tokens[2], "input.txt input.txt");
    check_value(tokens[3], "--optimization");
    check_value(tokens[4], "4");
    check_value(tokens[5], "--opt");
    check_value(tokens[6], "option1 option2");

    variables_map vm;
    store(command_line_parser(tokens).options(description).run(), vm);
    notify(vm);
}

void split_defaults(const options_description& description)
{
    const char* cmdline = "prg --input \t \'input file.txt\' \t   "
                          "--optimization 4 --opt \\\"option1\\ option2\\\"";

    vector<string> tokens = split_unix(cmdline);

    HPX_TEST_EQ(tokens.size(), std::size_t(7));
    check_value(tokens[0], "prg");
    check_value(tokens[1], "--input");
    check_value(tokens[2], "input file.txt");
    check_value(tokens[3], "--optimization");
    check_value(tokens[4], "4");
    check_value(tokens[5], "--opt");
    check_value(tokens[6], "\"option1 option2\"");

    variables_map vm;
    store(command_line_parser(tokens).options(description).run(), vm);
    notify(vm);
}

int main(int /*ac*/, char** /*av*/)
{
    options_description desc;
    // clang-format off
    desc.add_options()
        ("input,i", value<string>(),"the input file")
        ("optimization,O", value<unsigned>(), "optimization level")
        ("opt,o", value<string>(), "misc option")
        ;
    // clang-format on

    split_whitespace(desc);
    split_equalsign(desc);
    split_semi(desc);
    split_quotes(desc);
    split_escape(desc);
    split_single_quote(desc);
    split_defaults(desc);

    return hpx::util::report_errors();
}
