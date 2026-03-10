//  Copyright (c) 2026 Pratyksh Gupta
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <cstddef>
#include <string>
#include <string_view>
#include <unordered_map>

struct transparent_hash
{
    using is_transparent = void;
    std::size_t operator()(std::string_view sv) const
    {
        return std::hash<std::string_view>{}(sv);
    }
    std::size_t operator()(std::string const& s) const
    {
        return std::hash<std::string>{}(s);
    }
};

struct transparent_equal
{
    using is_transparent = void;
    bool operator()(std::string_view sv, std::string const& s) const
    {
        return sv == s;
    }
    bool operator()(std::string const& s, std::string_view sv) const
    {
        return s == sv;
    }
    bool operator()(std::string const& s1, std::string const& s2) const
    {
        return s1 == s2;
    }
};

int main()
{
    std::unordered_map<std::string, int, transparent_hash, transparent_equal> m;
    m["hello"] = 1;

    // Test operator[] with transparent key (C++26)
    m[std::string_view("hello")] = 42;

    // Test at() with transparent key (C++26)
    if (m.at(std::string_view("hello")) != 42)
        return 1;

    return 0;
}
