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

namespace server {
    struct request
    {
    };
}    // namespace server

namespace client {
    struct request
    {
    };
}    // namespace client

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
        using type =
            a::b::c::d::pentagon<int, char, double, float, unsigned int>;
        char const* name = qualified_name_of<type>::get();
        HPX_TEST_EQ(std::string(name),
            std::string(
                "a::b::c::d::pentagon<int,char,double,float,unsigned int>"));
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

    // Same Name in Different Namespaces
    {
        std::string server_req = qualified_name_of<server::request>::get();
        std::string client_req = qualified_name_of<client::request>::get();

        HPX_TEST_NEQ(server_req, client_req);
        HPX_TEST_EQ(server_req, std::string("server::request"));
        HPX_TEST_EQ(client_req, std::string("client::request"));
    }

    return hpx::util::report_errors();
}
