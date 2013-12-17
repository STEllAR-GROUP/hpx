//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Demonstrating #565

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/components.hpp>
#include <hpx/util/lightweight_test.hpp>

void func(hpx::id_type id);

struct test_server1
  : hpx::components::simple_component_base<test_server1>
{
    test_server1()
    {
        BOOST_ASSERT(!alive);
        alive=true;
    }
    test_server1(hpx::id_type o)
      : other(o)
    {
        BOOST_ASSERT(!alive);
        alive=true;
    }

    ~test_server1()
    {
        hpx::apply(hpx::util::bind(func, other));
        alive = false;
    }

    void test();
    
    HPX_DEFINE_COMPONENT_ACTION(test_server1, test);

    hpx::id_type other;
    static bool alive;
};

bool test_server1::alive = false;

typedef hpx::components::simple_component<test_server1> server1_type;
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(server1_type, test_server1);

struct test_server2
  : hpx::components::simple_component_base<test_server2>
{
    test_server2()
    {
        BOOST_ASSERT(!alive);
        alive=true;
    }
    ~test_server2()
    {
        alive = false;
    }

    hpx::id_type create_test_server1()
    {
        return hpx::new_<test_server1>(hpx::find_here(), this->get_gid()).get();
    }

    HPX_DEFINE_COMPONENT_ACTION(test_server2, create_test_server1);

    static bool alive;
};

bool test_server2::alive = false;

typedef hpx::components::simple_component<test_server2> server2_type;
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(server2_type, test_server2);

void test_server1::test()
{
    HPX_TEST(test_server2::alive);
    HPX_TEST(other);
}

void func(hpx::id_type id)
{
    HPX_TEST(test_server2::alive);
    HPX_TEST(id);
}

int hpx_main()
{
    {
        HPX_TEST(!test_server1::alive);
        HPX_TEST(!test_server2::alive);

        // creating test_server2 instance
        hpx::id_type server2 = hpx::new_<test_server2>(hpx::find_here()).get();
        HPX_TEST(!test_server1::alive);
        HPX_TEST(test_server2::alive);

        // creating test_server1 instance
        hpx::id_type server1 = hpx::async(test_server2::create_test_server1_action(), server2).get();
        server2 = hpx::id_type();
        hpx::agas::garbage_collect();
        hpx::agas::garbage_collect();
        hpx::agas::garbage_collect();

        HPX_TEST(test_server1::alive);
        HPX_TEST(test_server2::alive);

        test_server1::test_action()(server1);
        server1 = hpx::id_type();
        hpx::agas::garbage_collect();
        hpx::agas::garbage_collect();
        hpx::agas::garbage_collect();

        HPX_TEST(!test_server1::alive);
        HPX_TEST(!test_server2::alive);
    }

    HPX_TEST(!test_server1::alive);
    HPX_TEST(!test_server2::alive);
    
    return hpx::finalize();
}

int main(int argc, char **argv)
{
    hpx::init(argc, argv);

    HPX_TEST(!test_server1::alive);
    HPX_TEST(!test_server2::alive);

    return hpx::util::report_errors();
}
