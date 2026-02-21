//  Copyright (c) 2026 Ujjwal Shekhar
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/modules/serialization.hpp>
#include <hpx/modules/testing.hpp>

#include <string>
#include <vector>

namespace empty_space {
    template <typename... Args>
    struct void_void
    {
    };
}    // namespace empty_space

namespace a::b::c::d {
    template <typename T1, typename T2, typename T3, typename T4, typename T5>
    struct pentagon
    {
    };
}    // namespace a::b::c::d

namespace world { namespace continent { namespace country {
    template <typename T>
    struct city
    {
    };
}}}    // namespace world::continent::country

namespace local {
    struct person
    {
    };
}    // namespace local

int main()
{
    using hpx::serialization::detail::qualified_name_of;

    // Empty/Variadic Template Test
    {
        using type = empty_space::void_void<>;
        char const* name = qualified_name_of<type>::get();
        HPX_TEST(name != nullptr);
        HPX_TEST_EQ(std::string(name), std::string("empty_space::void_void<>"));
    }

    // Deep Namespace + High Arity
    {
        using type = a::b::c::d::pentagon<int, char, double, float, long>;
        char const* name = qualified_name_of<type>::get();
        HPX_TEST_EQ(std::string(name),
            std::string("a::b::c::d::pentagon<int,char,double,float,long>"));
    }

    // Deeply Nested Custom Types as Template Args
    {
        // City containing a Person from a different namespace
        using nested_type = world::continent::country::city<local::person>;

        // Wrap that in ANOTHER layer
        using deeper_nested_type = world::continent::country::city<nested_type>;

        char const* name = qualified_name_of<deeper_nested_type>::get();

        // Expected
        std::string expected =
            "world::continent::country::city<"
            "world::continent::country::city<local::person>>";

        HPX_TEST_EQ(std::string(name), expected);
    }

    return hpx::util::report_errors();
}
