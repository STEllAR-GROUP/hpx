//  Copyright (c) 2025 Ujjwal Shekhar
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/modules/serialization.hpp>
#include <hpx/modules/testing.hpp>

#include <map>
#include <string>
#include <vector>

// A simple class
class ReflStruct
{
private:
    int i;
    std::string s;

public: // <-- Added public
    ReflStruct() = default;
    ReflStruct(int i, std::string s) : i(i), s(std::move(s)) {}

    bool operator==(ReflStruct const& rhs) const
    {
        return i == rhs.i && s == rhs.s;
    }
};

// A host class containing containers of the reflection-only class
// This class has NO serialize() function.
class ContainerHost
{
    std::vector<ReflStruct> vec;
    std::map<int, ReflStruct> map;

public:
    ContainerHost() = default;
    ContainerHost(std::vector<ReflStruct> vec,
        std::map<int, ReflStruct> map)
      : vec(std::move(vec))
      , map(std::move(map))
    {
    }

    bool operator==(ContainerHost const& rhs) const
    {
        return vec == rhs.vec && map == rhs.map;
    }
};

int main()
{
    std::vector<char> buffer;
    hpx::serialization::output_archive oarchive(buffer);

    ContainerHost input_data;
    input_data = ContainerHost({{1, "one"}, {2, "two"}},
        {
            {3, {3, "three"}},
            {4, {4, "four"}},
        });

    oarchive << input_data;

    hpx::serialization::input_archive iarchive(buffer);
    ContainerHost output_data;
    iarchive >> output_data;

    HPX_TEST(input_data == output_data);

    return hpx::util::report_errors();
}
