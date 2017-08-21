//  Copyright (c) 2014-2017 Hartmut Kaiser
//  Copyright (c) 2017 Igor Krivenko
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <cstddef>
#include <vector>

struct non_default_ctor
{
    int i;

    non_default_ctor() = delete;
    non_default_ctor(int i)
      : i(i)
    {
    }

    template <typename Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        ar & i;
    }

    template <typename Archive>
    void friend load_construct_data(
        Archive& ar, non_default_ctor* p, const unsigned int)
    {
        ::new (p) non_default_ctor(0);
    }
};

non_default_ctor plain_non_default_ctor()
{
    return non_default_ctor(42);
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
        HPX_TEST_EQ(f.get().i, 42);
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
