//  Copyright (c) 2022 Joseph Kleinhenz
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test illustrates the problem reported by #5964: component with multiple
// inheritance

#include <hpx/config.hpp>

#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/runtime.hpp>
#include <hpx/include/util.hpp>
#include <hpx/modules/testing.hpp>

#include <atomic>
#include <cstddef>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
struct foo_t
{
    int a;
    int b;

    foo_t() = default;
    foo_t(int a, int b)
      : a(a)
      , b(b)
    {
    }

    friend class hpx::serialization::access;
    template <typename Archive>
    inline void serialize(Archive& ar, unsigned int const)
    {
        // clang-format off
        ar & a & b;
        // clang-format on
    }
};

struct component_server
  : hpx::components::component_base<component_server>
  , foo_t
{
    explicit component_server(foo_t foo)
      : foo_t(std::move(foo))
    {
    }
};

HPX_REGISTER_COMPONENT(
    hpx::components::component<component_server>, component_server_component)

struct component_server2
  : foo_t
  , hpx::components::component_base<component_server2>

{
    explicit component_server2(foo_t foo)
      : foo_t(std::move(foo))
    {
    }
};

HPX_REGISTER_COMPONENT(
    hpx::components::component<component_server2>, component_server2_component)

int hpx_main()
{
    foo_t in{1, 2};

    {
        hpx::id_type id =
            hpx::new_<component_server>(hpx::find_here(), in).get();
        auto out = hpx::get_ptr<component_server>(hpx::launch::sync, id);

        HPX_TEST_EQ(out->a, in.a);
        HPX_TEST_EQ(out->b, in.b);
    }

    {
        hpx::id_type id =
            hpx::new_<component_server2>(hpx::find_here(), in).get();
        auto out = hpx::get_ptr<component_server2>(hpx::launch::sync, id);

        HPX_TEST_EQ(out->a, in.a);
        HPX_TEST_EQ(out->b, in.b);
    }

    return hpx::finalize();
}

int main(int argc, char** argv)
{
    HPX_TEST_EQ(hpx::init(argc, argv), 0);
    return hpx::util::report_errors();
}

#endif
