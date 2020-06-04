//  Copyright Vladimir Prus 2002-2004.
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt
//  or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/modules/testing.hpp>

#include <hpx/program_options/options_description.hpp>
#include <hpx/program_options/value_semantic.hpp>

#include <cstddef>
#include <functional>
#include <sstream>
#include <string>
#include <utility>

using namespace hpx::program_options;
using namespace std;

void test_type()
{
    options_description desc;
    // clang-format off
    desc.add_options()
        ("foo", value<int>(), "")
        ("bar", value<string>(), "")
        ;
    //clang-format on

    const typed_value_base* b = dynamic_cast<const typed_value_base*>(
        desc.find("foo", false).semantic().get());
    HPX_TEST(b);
    HPX_TEST(b->value_type() == typeid(int));

    const typed_value_base* b2 = dynamic_cast<const typed_value_base*>(
        desc.find("bar", false).semantic().get());
    HPX_TEST(b2);
    HPX_TEST(b2->value_type() == typeid(string));
}

void test_approximation()
{
    options_description desc;
    // clang-format off
    desc.add_options()
        ("foo", new untyped_value())
        ("fee", new untyped_value())
        ("baz", new untyped_value())
        ("all-chroots", new untyped_value())
        ("all-sessions", new untyped_value())
        ("all", new untyped_value())
        ;
    // clang-format on

    HPX_TEST_EQ(desc.find("fo", true).long_name(), "foo");

    HPX_TEST_EQ(desc.find("all", true).long_name(), "all");
    HPX_TEST_EQ(desc.find("all-ch", true).long_name(), "all-chroots");

    options_description desc2;
    // clang-format off
    desc2.add_options()
        ("help", "display this message")
        ("config", value<string>(), "config file name")
        ("config-value", value<string>(), "single config value")
        ;
    // clang-format on

    HPX_TEST_EQ(desc2.find("config", true).long_name(), "config");
    HPX_TEST_EQ(desc2.find("config-value", true).long_name(), "config-value");
}

void test_approximation_with_multiname_options()
{
#if !defined(HPX_PROGRAM_OPTIONS_HAVE_BOOST_PROGRAM_OPTIONS_COMPATIBILITY) ||  \
    (defined(BOOST_VERSION) && BOOST_VERSION >= 106800)
    // the long_names() API function was introduced in Boost V1.68
    options_description desc;
    // clang-format off
    desc.add_options()
        ("foo", new untyped_value())
        ("fee", new untyped_value())
        ("fe,baz", new untyped_value())
        ("chroots,all-chroots", new untyped_value())
        ("sessions,all-sessions", new untyped_value())
        ("everything,all", new untyped_value())
        ("qux,fo", new untyped_value())
        ;
    // clang-format on

    HPX_TEST_EQ(desc.find("fo", true).long_name(), "qux");

    HPX_TEST_EQ(desc.find("all", true).long_name(), "everything");
    HPX_TEST_EQ(desc.find("all-ch", true).long_name(), "chroots");

    HPX_TEST_EQ(desc.find("foo", false, false, false).long_names().second,
        std::size_t(1));
    HPX_TEST_EQ(
        desc.find("foo", false, false, false).long_names().first[0], "foo");

    HPX_TEST_EQ(desc.find("fe", false, false, false).long_names().second,
        std::size_t(2));
    HPX_TEST_EQ(
        desc.find("fe", false, false, false).long_names().first[0], "fe");
    HPX_TEST_EQ(
        desc.find("baz", false, false, false).long_names().first[1], "baz");

    HPX_TEST_EQ(desc.find("baz", false, false, false).long_names().second,
        std::size_t(2));
    HPX_TEST_EQ(
        desc.find("baz", false, false, false).long_names().first[0], "fizbaz");
    HPX_TEST_EQ(
        desc.find("baz", false, false, false).long_names().first[1], "baz");
#endif
}

void test_long_names_for_option_description()
{
#if !defined(HPX_PROGRAM_OPTIONS_HAVE_BOOST_PROGRAM_OPTIONS_COMPATIBILITY) ||  \
    (defined(BOOST_VERSION) && BOOST_VERSION >= 106800)
    // the long_names() API function was introduced in Boost V1.68
    options_description desc;
    // clang-format off
    desc.add_options()
        ("foo", new untyped_value())
        ("fe,baz", new untyped_value())
        ("chroots,all-chroots", new untyped_value())
        ("sessions,all-sessions", new untyped_value())
        ("everything,all", new untyped_value())
        ("qux,fo,q", new untyped_value())
        ;
    // clang-format on

    HPX_TEST_EQ(desc.find("foo", false, false, false).long_names().second,
        std::size_t(1));
    HPX_TEST_EQ(
        desc.find("foo", false, false, false).long_names().first[0], "foo");

    HPX_TEST_EQ(desc.find("fe", false, false, false).long_names().second,
        std::size_t(2));
    HPX_TEST_EQ(
        desc.find("fe", false, false, false).long_names().first[0], "fe");
    HPX_TEST_EQ(
        desc.find("baz", false, false, false).long_names().first[1], "baz");

    HPX_TEST_EQ(desc.find("qux", false, false, false).long_names().second,
        std::size_t(2));
    HPX_TEST_EQ(
        desc.find("qux", false, false, false).long_names().first[0], "qux");
    HPX_TEST_EQ(
        desc.find("qux", false, false, false).long_names().first[1], "fo");
#endif
}

void test_formatting()
{
    // Long option descriptions used to crash on MSVC-8.0.
    options_description desc;
    // clang-format off
    desc.add_options()("test", new untyped_value(),
        "foo foo foo foo foo foo foo foo foo foo foo foo foo foo"
        "foo foo foo foo foo foo foo foo foo foo foo foo foo foo"
        "foo foo foo foo foo foo foo foo foo foo foo foo foo foo"
        "foo foo foo foo foo foo foo foo foo foo foo foo foo foo")("list",
        new untyped_value(),
        "a list:\n      \t"
        "item1, item2, item3, item4, item5, item6, item7, item8, item9, "
        "item10, item11, item12, item13, item14, item15, item16, item17, "
        "item18")("well_formated", new untyped_value(),
        "As you can see this is a very well formatted option description.\n"
        "You can do this for example:\n\n"
        "Values:\n"
        "  Value1: \tdoes this and that, bla bla bla bla bla bla bla bla bla "
        "bla bla bla bla bla bla\n"
        "  Value2: \tdoes something else, bla bla bla bla bla bla bla bla bla "
        "bla bla bla bla bla bla\n\n"
        "    This paragraph has a first line indent only, bla bla bla bla bla "
        "bla bla bla bla bla bla bla bla bla bla");

    stringstream ss;
    ss << desc;
    HPX_TEST_EQ(ss.str(),
"  --test arg            foo foo foo foo foo foo foo foo foo foo foo foo foo \n"
"                        foofoo foo foo foo foo foo foo foo foo foo foo foo foo \n"
"                        foofoo foo foo foo foo foo foo foo foo foo foo foo foo \n"
"                        foofoo foo foo foo foo foo foo foo foo foo foo foo foo \n"
"                        foo\n"
"  --list arg            a list:\n"
"                              item1, item2, item3, item4, item5, item6, item7, \n"
"                              item8, item9, item10, item11, item12, item13, \n"
"                              item14, item15, item16, item17, item18\n"
"  --well_formated arg   As you can see this is a very well formatted option \n"
"                        description.\n"
"                        You can do this for example:\n"
"                        \n"
"                        Values:\n"
"                          Value1: does this and that, bla bla bla bla bla bla \n"
"                                  bla bla bla bla bla bla bla bla bla\n"
"                          Value2: does something else, bla bla bla bla bla bla \n"
"                                  bla bla bla bla bla bla bla bla bla\n"
"                        \n"
"                            This paragraph has a first line indent only, bla \n"
"                        bla bla bla bla bla bla bla bla bla bla bla bla bla bla\n"
    );
    // clang-format on
}

void test_multiname_option_formatting()
{
#if !defined(HPX_PROGRAM_OPTIONS_HAVE_BOOST_PROGRAM_OPTIONS_COMPATIBILITY) ||  \
    (defined(BOOST_VERSION) && BOOST_VERSION >= 106800)
    // the long_names() API function was introduced in Boost V1.68
    options_description desc;
    desc.add_options()(
        "foo,bar", new untyped_value(), "a multiple-name option");

    stringstream ss;
    ss << desc;
    HPX_TEST_EQ(ss.str(), "  --foo arg             a multiple-name option\n");
#endif
}

void test_formatting_description_length()
{
    {
        options_description desc("", options_description::m_default_line_length,
            options_description::m_default_line_length / 2U);
        // clang-format off
        desc.add_options()
            // 40 available for desc
            ("an-option-that-sets-the-max", new untyped_value(),
            "this description sits on the same line, but wrapping should still "
             "work correctly")
            ("a-long-option-that-would-leave-very-little-space-for-description",
            new untyped_value(),
            "the description of the long opt, but placed on the next line\n"
            "    \talso ensure that the tabulation works correctly when a"
            " description size has been set");

        stringstream ss;
        ss << desc;
        HPX_TEST_EQ(ss.str(),
            "  --an-option-that-sets-the-max arg     this description sits on "
            "the same line,\n"
            "                                        but wrapping should still "
            "work \n"
            "                                        correctly\n"
            "  --a-long-option-that-would-leave-very-little-space-for-"
            "description arg\n"
            "                                        the description of the "
            "long opt, but \n"
            "                                        placed on the next line\n"
            "                                            also ensure that the "
            "tabulation \n"
            "                                            works correctly when "
            "a description \n"
            "                                            size has been set\n");
        // clang-format on
    }
    {
        // the default behavior reserves 23 (+1 space) characters for the
        // option column; this shows that the min_description_length does not
        // breach that.
        options_description desc("", options_description::m_default_line_length,
            options_description::m_default_line_length -
                10U);    // leaves < 23 (default option space)
        desc.add_options()("an-option-that-encroaches-description",
            new untyped_value(),
            "this description should always be placed on the next line, and "
            "wrapping should continue as normal");

        stringstream ss;
        ss << desc;
        HPX_TEST_EQ(ss.str(),
            "  --an-option-that-encroaches-description arg\n"
            //123456789_123456789_
            "          this description should always be placed on the next "
            "line, and \n"
            "          wrapping should continue as normal\n");
    }
}

void test_long_default_value()
{
    options_description desc;
    desc.add_options()("cfgfile,c",
        value<string>()->default_value(
            "/usr/local/etc/myprogramXXXXXXXXX/configuration.conf"),
        "the configfile");

    stringstream ss;
    ss << desc;
    HPX_TEST_EQ(ss.str(),
        "  -c [ --cfgfile ] arg "
        "(=/usr/local/etc/myprogramXXXXXXXXX/configuration.conf)\n"
        "                                        the configfile\n");
}

void test_word_wrapping()
{
    options_description desc("Supported options");
    desc.add_options()(
        "help", "this is a sufficiently long text to require word-wrapping");
    desc.add_options()("prefix",
        value<string>()->default_value("/h/proj/tmp/dispatch"),
        "root path of the dispatch installation");
    desc.add_options()("opt1",
        "this_is_a_sufficiently_long_text_to_require_word-wrapping_but_cannot_"
        "be_wrapped");
    desc.add_options()(
        "opt2", "this_is_a_sufficiently long_text_to_require_word-wrapping");
    desc.add_options()("opt3",
        "this_is_a "
        "sufficiently_long_text_to_require_word-wrapping_but_will_not_be_"
        "wrapped");

    stringstream ss;
    ss << desc;
    HPX_TEST_EQ(ss.str(),
        "Supported options:\n"
        "  --help                               this is a sufficiently long "
        "text to \n"
        "                                       require word-wrapping\n"
        "  --prefix arg (=/h/proj/tmp/dispatch) root path of the dispatch "
        "installation\n"
        "  --opt1                               "
        "this_is_a_sufficiently_long_text_to_requ\n"
        "                                       "
        "ire_word-wrapping_but_cannot_be_wrapped\n"
        "  --opt2                               this_is_a_sufficiently \n"
        "                                       "
        "long_text_to_require_word-wrapping\n"
        "  --opt3                               this_is_a "
        "sufficiently_long_text_to_requ\n"
        "                                       "
        "ire_word-wrapping_but_will_not_be_wrappe\n"
        "                                       d\n");
}

void test_default_values()
{
    options_description desc("Supported options");
    desc.add_options()("maxlength", value<double>()->default_value(.1, "0.1"),
        "Maximum edge length to keep.");
    stringstream ss;
    ss << desc;
    HPX_TEST_EQ(ss.str(),
        "Supported options:\n"
        "  --maxlength arg (=0.1) Maximum edge length to keep.\n");
}

void test_value_name()
{
    options_description desc("Supported options");
    desc.add_options()("include", value<string>()->value_name("directory"),
        "Search for headers in 'directory'.");

    stringstream ss;
    ss << desc;
    HPX_TEST_EQ(ss.str(),
        "Supported options:\n"
        "  --include directory   Search for headers in 'directory'.\n");
}

int main(int, char*[])
{
    test_type();
    test_approximation();
    test_long_names_for_option_description();
    test_formatting();
    test_multiname_option_formatting();
    test_formatting_description_length();
    test_long_default_value();
    test_word_wrapping();
    test_default_values();
    test_value_name();

    return hpx::util::report_errors();
}
