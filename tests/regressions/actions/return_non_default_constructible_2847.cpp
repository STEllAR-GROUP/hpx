//  Copyright (c) 2014-2017 Hartmut Kaiser
//  Copyright (c) 2017 Igor Krivenko
//  Copyright (c) 2022 Joseph Kleinhenz
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx_main.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/runtime.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <vector>

struct non_default_ctor
{
    int i;
    int j;

    non_default_ctor() = delete;

    non_default_ctor(int i)
      : i(i), j(0)
    {
    }

    non_default_ctor(int i, int j)
      : i(i), j(j)
    {
    }

    template <typename Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        ar & j;
    }

    template<typename Archive>
    void friend save_construct_data(Archive& ar, const non_default_ctor* p, const unsigned int) {
        ar << p->i;
    }

    template <typename Archive>
    void friend load_construct_data(
        Archive& ar, non_default_ctor* p, const unsigned int)
    {
        int i;
        ar >> i;
        ::new (p) non_default_ctor(i);
    }
};

non_default_ctor plain_non_default_ctor()
{
    return non_default_ctor(42, 11);
}

HPX_PLAIN_ACTION(plain_non_default_ctor, plain_non_default_ctor_action);

void test_plain_call_non_default_ctor(hpx::id_type id)
{
    // test apply
    for (std::size_t i = 0; i != 100; ++i)
    {
        hpx::apply<plain_non_default_ctor_action>(id);
    }

    // test async
    std::vector<hpx::future<non_default_ctor>> calls;
    for (std::size_t i = 0; i != 100; ++i)
    {
        calls.push_back(hpx::async<plain_non_default_ctor_action>(id));
    }
    hpx::wait_all(calls);

    for (auto && f : calls)
    {
        const auto val = f.get();
        HPX_TEST_EQ(val.i, 42);
        HPX_TEST_EQ(val.j, 11);
    }
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    for (hpx::id_type id : hpx::find_all_localities())
    {
        test_plain_call_non_default_ctor(id);
    }
    return hpx::util::report_errors();
}
#endif
