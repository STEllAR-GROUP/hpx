// Copyright Vladimir Prus 2002-2004.
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/modules/testing.hpp>

#include <hpx/program_options/detail/utf8_codecvt_facet.hpp>
#include <hpx/program_options/option.hpp>
#include <hpx/program_options/options_description.hpp>
#include <hpx/program_options/parsers.hpp>
#include <hpx/program_options/value_semantic.hpp>
#include <hpx/program_options/variables_map.hpp>

#include <cstddef>
#include <locale>
#include <sstream>
#include <string>
#include <vector>

using namespace hpx::program_options;
using namespace std;

// Test that unicode input is forwarded to unicode option without
// problems.
void test_unicode_to_unicode()
{
    options_description desc;

    desc.add_options()("foo", wvalue<wstring>(), "unicode option");

    vector<wstring> args;
    args.push_back(L"--foo=\x044F");

    variables_map vm;
    basic_parsed_options<wchar_t> parsed =
        wcommand_line_parser(args).options(desc).run();
    store(parsed, vm);

    HPX_TEST(vm["foo"].as<wstring>() == L"\x044F");
    HPX_TEST_EQ(parsed.options[0].original_tokens.size(), std::size_t(1));
    HPX_TEST(parsed.options[0].original_tokens[0] == L"--foo=\x044F");
}

// Test that unicode input is property converted into
// local 8 bit string. To test this, make local 8 bit encoding
// be utf8.
void test_unicode_to_native()
{
    std::codecvt<wchar_t, char, mbstate_t>* facet =
        new hpx::program_options::detail::utf8_codecvt_facet;
    locale::global(locale(locale(), facet));

    options_description desc;

    desc.add_options()("foo", value<string>(), "unicode option");

    vector<wstring> args;
    args.push_back(L"--foo=\x044F");

    variables_map vm;
    store(wcommand_line_parser(args).options(desc).run(), vm);

    HPX_TEST_EQ(vm["foo"].as<string>(), "\xD1\x8F");
}

void test_native_to_unicode()
{
    std::codecvt<wchar_t, char, mbstate_t>* facet =
        new hpx::program_options::detail::utf8_codecvt_facet;
    locale::global(locale(locale(), facet));

    options_description desc;

    desc.add_options()("foo", wvalue<wstring>(), "unicode option");

    vector<string> args;
    args.push_back("--foo=\xD1\x8F");

    variables_map vm;
    store(command_line_parser(args).options(desc).run(), vm);

    HPX_TEST(vm["foo"].as<wstring>() == L"\x044F");
}

vector<wstring> sv(const wchar_t* array[], unsigned size)
{
    vector<wstring> r;
    for (unsigned i = 0; i < size; ++i)
        r.emplace_back(array[i]);
    return r;
}

void check_value(const woption& option, const char* name, const wchar_t* value)
{
    HPX_TEST_EQ(option.string_key, name);
    HPX_TEST_EQ(option.value.size(), std::size_t(1));
    HPX_TEST(option.value.front() == value);
}

void test_command_line()
{
#if !defined(HPX_PROGRAM_OPTIONS_HAVE_BOOST_PROGRAM_OPTIONS_COMPATIBILITY) ||  \
    (defined(BOOST_VERSION) && BOOST_VERSION >= 106800)
    // the long_names() API function was introduced in Boost V1.68
    options_description desc;
    desc.add_options()("foo,f", new untyped_value(), "")
        // Explicit qualification is a workaround for vc6
        ("bar,b", value<std::string>(), "")("baz", new untyped_value())(
            "qux,plug*", new untyped_value());

    const wchar_t* cmdline4_[] = {
        L"--foo=1\u0FF52", L"-f4", L"--bar=11", L"-b4", L"--plug3=10"};
    vector<wstring> cmdline4 =
        sv(cmdline4_, sizeof(cmdline4_) / sizeof(cmdline4_[0]));
    vector<woption> a4 =
        wcommand_line_parser(cmdline4).options(desc).run().options;

    HPX_TEST_EQ(a4.size(), std::size_t(5));

    check_value(a4[0], "foo", L"1\u0FF52");
    check_value(a4[1], "foo", L"4");
    check_value(a4[2], "bar", L"11");
    check_value(a4[4], "qux", L"10");
#endif
}

// Since we've already tested conversion between parser encoding and
// option encoding, all we need to check for config file is that
// when reading wistream, it generates proper UTF8 data.
void test_config_file()
{
    std::codecvt<wchar_t, char, mbstate_t>* facet =
        new hpx::program_options::detail::utf8_codecvt_facet;
    locale::global(locale(locale(), facet));

    options_description desc;

    desc.add_options()("foo", value<string>(), "unicode option");

    std::wstringstream stream(L"foo = \x044F");

    variables_map vm;
    store(parse_config_file(stream, desc), vm);

    HPX_TEST_EQ(vm["foo"].as<string>(), "\xD1\x8F");
}

int main(int, char*[])
{
    test_unicode_to_unicode();
    test_unicode_to_native();
    test_native_to_unicode();
    test_command_line();
    test_config_file();

    return hpx::util::report_errors();
}
