//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// (c) Copyright John R. Bandela 2001.

// See http://www.boost.org for updates, documentation, and revision history.

#include <hpx/modules/string_util.hpp>
#include <hpx/modules/testing.hpp>

#include <algorithm>
#include <array>
#include <iterator>
#include <string>

int main()
{
    using namespace hpx::string_util;

    // Use tokenizer
    {
        std::string const test_string = ";;Hello|world||-foo--bar;yow;baz|";
        std::string answer[] = {"Hello", "world", "foo", "bar", "yow", "baz"};
        using Tok = tokenizer<char_separator<char>>;
        char_separator<char> sep("-;|");
        Tok t(test_string, sep);
        HPX_TEST(std::equal(t.begin(), t.end(), answer));
    }

    {
        std::string const test_string = ";;Hello|world||-foo--bar;yow;baz|";
        std::string answer[] = {"", "", "Hello", "|", "world", "|", "", "|", "",
            "foo", "", "bar", "yow", "baz", "|", ""};
        using Tok = tokenizer<char_separator<char>>;
        char_separator<char> sep("-;", "|", empty_token_policy::keep);
        Tok t(test_string, sep);
        HPX_TEST(std::equal(t.begin(), t.end(), answer));
    }

    {
        std::string const test_string = "This,,is, a.test..";
        std::string answer[] = {"This", "is", "a", "test"};
        using Tok = tokenizer<char_delimiters_separator<char>>;
        Tok t(test_string);
        HPX_TEST(std::equal(t.begin(), t.end(), answer));
    }

    {
        std::string const test_string =
            "Field 1,\"embedded,comma\",quote \\\", escape \\\\";
        std::string answer[] = {
            "Field 1", "embedded,comma", "quote \"", " escape \\"};
        using Tok = tokenizer<escaped_list_separator<char>>;
        Tok t(test_string);
        HPX_TEST(std::equal(t.begin(), t.end(), answer));
    }

    {
        std::string const test_string = ",1,;2\\\";3\\;,4,5^\\,\'6,7\';";
        std::string answer[] = {
            "", "1", "", "2\"", "3;", "4", "5\\", "6,7", ""};
        using Tok = tokenizer<escaped_list_separator<char>>;
        escaped_list_separator<char> sep("\\^", ",;", "\"\'");
        Tok t(test_string, sep);
        HPX_TEST(std::equal(t.begin(), t.end(), answer));
    }

    {
        std::string const test_string = "12252001";
        std::string answer[] = {"12", "25", "2001"};
        using Tok = tokenizer<offset_separator>;
        std::array<int, 3> offsets = {{2, 2, 4}};
        offset_separator func(offsets.begin(), offsets.end());
        Tok t(test_string, func);
        HPX_TEST(std::equal(t.begin(), t.end(), answer));
    }

    // Use token_iterator_generator
    {
        std::string const test_string = "This,,is, a.test..";
        std::string answer[] = {"This", "is", "a", "test"};
        using Iter =
            token_iterator_generator<char_delimiters_separator<char>>::type;
        Iter begin = make_token_iterator<std::string>(test_string.begin(),
            test_string.end(), char_delimiters_separator<char>());
        Iter end;
        HPX_TEST(std::equal(begin, end, answer));
    }

    {
        std::string const test_string =
            "Field 1,\"embedded,comma\",quote \\\", escape \\\\";
        std::string answer[] = {
            "Field 1", "embedded,comma", "quote \"", " escape \\"};
        using Iter =
            token_iterator_generator<escaped_list_separator<char>>::type;
        Iter begin = make_token_iterator<std::string>(test_string.begin(),
            test_string.end(), escaped_list_separator<char>());
        Iter begin_c(begin);
        Iter end;
        HPX_TEST(std::equal(begin, end, answer));

        while (begin_c != end)
        {
            HPX_TEST(begin_c.at_end() == 0);
            ++begin_c;
        }
        HPX_TEST(begin_c.at_end());
    }

    {
        std::string const test_string = "12252001";
        std::string answer[] = {"12", "25", "2001"};
        using Iter = token_iterator_generator<offset_separator>::type;
        std::array<int, 3> offsets = {{2, 2, 4}};
        offset_separator func(offsets.begin(), offsets.end());
        Iter begin = make_token_iterator<std::string>(
            test_string.begin(), test_string.end(), func);
        Iter end = make_token_iterator<std::string>(
            test_string.end(), test_string.end(), func);
        HPX_TEST(std::equal(begin, end, answer));
    }

    // Test copying
    {
        std::string const test_string = "abcdef";
        token_iterator_generator<offset_separator>::type beg, end, other;
        std::array<int, 3> ar = {{1, 2, 3}};
        offset_separator f(ar.begin(), ar.end());
        beg = make_token_iterator<std::string>(
            test_string.begin(), test_string.end(), f);

        ++beg;
        other = beg;
        ++other;

        HPX_TEST(*beg == "bc");
        HPX_TEST(*other == "def");

        other = make_token_iterator<std::string>(
            test_string.begin(), test_string.end(), f);

        HPX_TEST(*other == "a");
    }

    // Test non-default constructed char_separator
    {
        std::string const test_string = "how,are you, doing";
        std::string answer[] = {"how", ",", "are you", ",", " doing"};
        tokenizer t(
            test_string, char_delimiters_separator<char>(true, ",", ""));
        HPX_TEST(std::equal(t.begin(), t.end(), answer));
    }

    return hpx::util::report_errors();
}
