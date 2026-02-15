//  Copyright (c) 2026 Ujjwal Shekhar
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/datastructures/optional.hpp>
#include <hpx/datastructures/serialization/optional.hpp>
#include <hpx/modules/serialization.hpp>
#include <hpx/modules/testing.hpp>

#include <array>
#include <deque>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include <experimental/meta>
#include <iostream>

struct Person
{
    std::string name;
    int age;
};

struct Coordinate
{
    double x;
    double y;
};

struct Base2
{
    int a = 42;
    std::unique_ptr<int> q = std::make_unique<int>(99);
    virtual ~Base2() = default;
};
HPX_POLYMORPHIC_AUTO_REGISTER(Base2)

namespace Deeply { namespace Nested {
    struct Base3
    {
        std::deque<int> e = {1, 2, 3};
        hpx::optional<int> p = 7;
        virtual ~Base3() = default;
    };
}}    // namespace Deeply::Nested
HPX_POLYMORPHIC_AUTO_REGISTER(Deeply::Nested::Base3)

struct Derived2_2
  : public Base2
  , public Deeply::Nested::Base3
{
    std::vector<Person> c = {{"Alice", 30}};
    std::list<std::string> d = {"hpx", "serialization"};
    std::set<std::string> h = {"alpha", "omega"};
};
HPX_POLYMORPHIC_AUTO_REGISTER(Derived2_2)

namespace Outer {
    struct Derived3_1 : public Deeply::Nested::Base3
    {
    protected:
        std::vector<std::map<int, std::vector<Person>>> registry_history;
        std::multiset<int> i = {1, 1, 2, 3};
    };
}    // namespace Outer
HPX_POLYMORPHIC_AUTO_REGISTER(Outer::Derived3_1)

struct Derived3_2 : public Deeply::Nested::Base3
{
private:
    std::unordered_multimap<int, int> k = {{1, 10}, {1, 20}};
    std::pair<std::string, Person> o = {"manager", {"Bob", 45}};

public:
    struct internal_stats
    {
        double score = 0.0;
        std::vector<std::string> tags;
    };
    internal_stats stats;
};
HPX_POLYMORPHIC_AUTO_REGISTER(Derived3_2)

struct Derived2_2_1 : private Derived2_2
{
    std::array<Person, 2> m = {{{"X", 1}, {"Y", 2}}};
};
HPX_POLYMORPHIC_AUTO_REGISTER(Derived2_2_1)

struct Derived3_1_1 : public Outer::Derived3_1
{
    hpx::optional<std::list<std::pair<std::string, Person>>> metadata;
};
HPX_POLYMORPHIC_AUTO_REGISTER(Derived3_1_1)

namespace Templates {
    template <typename T>
    struct Derived3_2_1 : public Derived3_2
    {
        std::array<Coordinate, 4> fixed_points;
        T template_member;
    };

    struct Derived3_2_2 : protected Derived3_2
    {
        std::unordered_map<int, Person> j;
    };
}    // namespace Templates

HPX_POLYMORPHIC_AUTO_REGISTER(Templates::Derived3_2_1<std::vector<int>>)
HPX_POLYMORPHIC_AUTO_REGISTER(Templates::Derived3_2_2)

int main()
{
    std::vector<char> buffer;

    using TargetType = Templates::Derived3_2_1<std::vector<int>>;
    std::unique_ptr<Deeply::Nested::Base3> input_ptr =
        std::make_unique<TargetType>();

    auto* raw = static_cast<TargetType*>(input_ptr.get());
    raw->fixed_points[0] = {5.5, 6.6};
    raw->template_member = {1, 2, 3};

    {
        hpx::serialization::output_archive oarchive(buffer);
        oarchive << input_ptr;
    }

    std::unique_ptr<Deeply::Nested::Base3> output_ptr;
    {
        hpx::serialization::input_archive iarchive(buffer);
        iarchive >> output_ptr;
    }

    HPX_TEST(output_ptr != nullptr);

    auto* final_check = dynamic_cast<TargetType*>(output_ptr.get());
    HPX_TEST(final_check != nullptr);
    HPX_TEST(final_check->fixed_points[0].y == 6.6);
    HPX_TEST(final_check->template_member.size() == 3);

    return hpx::util::report_errors();
}
