//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test case demonstrates the issue described in #588: Continuations do not
// keep object alive

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <vector>

#define NUM_INSTANCES 100

///////////////////////////////////////////////////////////////////////////////
boost::detail::atomic_count count_(0);

long get_count() { return count_; }

HPX_PLAIN_ACTION(get_count, get_count_action)

///////////////////////////////////////////////////////////////////////////////
struct foo
  : public hpx::components::simple_component_base<foo>
{
    foo()
    {
        ++count_;
    }

    int bar() { return 42; }

    HPX_DEFINE_COMPONENT_ACTION(foo, bar, bar_action);
};

HPX_REGISTER_ACTION(foo::bar_action, foo_bar_action);

HPX_REGISTER_COMPONENT(hpx::components::simple_component<foo>, foo);

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    std::vector<hpx::id_type> const localities =
        hpx::find_all_localities(foo::get_component_type());

    std::vector<hpx::future<hpx::id_type> > components;
    for (int i = 0; i != NUM_INSTANCES; ++i)
    {
        for (std::size_t j = 0; j != localities.size(); ++j)
        {
            components.push_back(hpx::new_<foo>(localities[j]));
        }
    }
    hpx::wait_all(components);

    for (std::size_t j = 0; j != localities.size(); ++j)
    {
        HPX_TEST_EQ(NUM_INSTANCES, get_count_action()(localities[j]));
    }

    return hpx::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    HPX_TEST_EQ(0, hpx::init(argc, argv));
    return hpx::util::report_errors();
}

