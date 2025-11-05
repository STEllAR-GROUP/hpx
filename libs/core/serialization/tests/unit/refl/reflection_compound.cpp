//  Copyright (c) 2025 Ujjwal Shekhar
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/modules/serialization.hpp>
#include <hpx/modules/testing.hpp>

#include <optional>
#include <string>
#include <tuple>
#include <variant>

// A simple class
class ReflStruct
{
private:
    int i;
    std::string s;

public:
    ReflStruct() = default;
    ReflStruct(int i, std::string s) : i(i), s(std::move(s)) {}

    bool operator==(ReflStruct const& rhs) const
    {
        return i == rhs.i && s == rhs.s;
    }
};

// A host struct containing compound types
// This struct has NO serialize() function.
struct CompoundHost
{
    std::optional<ReflStruct> opt_full;
    std::optional<ReflStruct> opt_empty;
    std::variant<int, ReflStruct> var_int;
    std::variant<int, ReflStruct> var_struct;
    std::tuple<int, std::string, ReflStruct> tup;

    bool operator==(CompoundHost const& rhs) const
    {
        return opt_full == rhs.opt_full && opt_empty == rhs.opt_empty &&
            var_int == rhs.var_int && var_struct == rhs.var_struct &&
            tup == rhs.tup;
    }
};

int main()
{
    std::vector<char> buffer;
    hpx::serialization::output_archive oarchive(buffer);

    CompoundHost input_data;
    input_data.opt_full = {1, "one"};
    input_data.opt_empty = std::nullopt;
    input_data.var_int = 10;
    input_data.var_struct = ReflStruct(2, "two");
    input_data.tup = {3, "three", {4, "four"}};

    oarchive << input_data;

    hpx::serialization::input_archive iarchive(buffer);
    CompoundHost output_data;
    iarchive >> output_data;

    HPX_TEST(input_data == output_data);

    return hpx::util::report_errors();
}