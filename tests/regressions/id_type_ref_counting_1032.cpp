//  Copyright (c) 2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Demonstrating #1032: id_type local reference counting is wrong

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/components.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <chrono>

template <typename T>
struct simple_base
{
    typedef hpx::components::simple_component_base<T> type;
};

template <typename T>
struct managed_base
{
    typedef hpx::components::simple_component_base<T> type;
};

template <template <typename> class ComponentBase>
void func(hpx::id_type id);

template <template <typename> class ComponentBase>
struct test_server1
  : ComponentBase<test_server1<ComponentBase> >::type
{
    test_server1()
    {
        HPX_ASSERT(!alive);
        alive = true;
    }
    test_server1(hpx::id_type o)
      : other(o)
    {
        HPX_ASSERT(!alive);
        alive = true;
    }

    ~test_server1()
    {
        HPX_ASSERT(alive);
        void (*f)(hpx::id_type) = func<ComponentBase>;
        hpx::apply(f, other);
        alive = false;
        other = hpx::invalid_id;
    }

    void test();

    HPX_DEFINE_COMPONENT_ACTION(test_server1, test, test_action);

    hpx::id_type other;
    static bool alive;
};

template <template <typename> class ComponentBase>
struct test_server2
  : ComponentBase<test_server2<ComponentBase> >::type
{
    test_server2()
    {
        HPX_ASSERT(!alive);
        alive = true;
    }
    ~test_server2()
    {
        HPX_ASSERT(alive);
        alive = false;
    }

    hpx::id_type create_test_server1()
    {
        return hpx::new_<test_server1<ComponentBase> >(
            hpx::find_here(), this->get_id()).get();
    }

    HPX_DEFINE_COMPONENT_ACTION(test_server2, create_test_server1,
        create_test_server1_action);

    static bool alive;
};

template <template <typename> class ComponentBase>
bool test_server1<ComponentBase>::alive = false;

template <template <typename> class ComponentBase>
bool test_server2<ComponentBase>::alive = false;

typedef test_server1<simple_base> test_simple_server1;
typedef test_server2<simple_base> test_simple_server2;

typedef test_server1<managed_base> test_managed_server1;
typedef test_server2<managed_base> test_managed_server2;

typedef hpx::components::simple_component<test_simple_server1> simple_server1_type;
HPX_REGISTER_COMPONENT(simple_server1_type, test_simple_server1);

typedef hpx::components::simple_component<test_simple_server2> simple_server2_type;
HPX_REGISTER_COMPONENT(simple_server2_type, test_simple_server2);

typedef hpx::components::simple_component<test_managed_server1> managed_server1_type;
HPX_REGISTER_COMPONENT(managed_server1_type, test_managed_server1);

typedef hpx::components::simple_component<test_managed_server2> managed_server2_type;
HPX_REGISTER_COMPONENT(managed_server2_type, test_managed_server2);

template <template <typename> class ComponentBase>
void test_server1<ComponentBase>::test()
{
    HPX_TEST(test_server2<ComponentBase>::alive);
    HPX_TEST(other);
}

void ensure_garbage_collect()
{
    hpx::this_thread::sleep_for(std::chrono::milliseconds(500));
    hpx::agas::garbage_collect();
    hpx::this_thread::sleep_for(std::chrono::milliseconds(500));
    hpx::agas::garbage_collect();
    hpx::this_thread::sleep_for(std::chrono::milliseconds(500));
    hpx::agas::garbage_collect();
}

template <template <typename> class ComponentBase>
void func(hpx::id_type id)
{
    ensure_garbage_collect();

    HPX_TEST(!test_server1<ComponentBase>::alive);
    HPX_TEST(test_server2<ComponentBase>::alive);
    HPX_TEST(id);
}

int hpx_main()
{
    {
        HPX_TEST(!test_simple_server1::alive);
        HPX_TEST(!test_simple_server2::alive);

        // creating test_server2 instance
        hpx::id_type server2 = hpx::new_<test_simple_server2>(hpx::find_here()).get();
        HPX_TEST(!test_simple_server1::alive);
        HPX_TEST(test_simple_server2::alive);

        // creating test_server1 instance
        hpx::id_type server1 = hpx::async(
            test_simple_server2::create_test_server1_action(), server2).get();
        server2 = hpx::invalid_id;
        ensure_garbage_collect();

        HPX_TEST(test_simple_server1::alive);
        HPX_TEST(test_simple_server2::alive);

        test_simple_server1::test_action()(server1);
        server1 = hpx::invalid_id;
        ensure_garbage_collect();
    }
    {
        HPX_TEST(!test_managed_server1::alive);
        HPX_TEST(!test_managed_server2::alive);

        // creating test_server2 instance
        hpx::id_type server2 = hpx::new_<test_managed_server2>(hpx::find_here()).get();
        HPX_TEST(!test_managed_server1::alive);
        HPX_TEST(test_managed_server2::alive);

        // creating test_server1 instance
        hpx::id_type server1 = hpx::async(
            test_managed_server2::create_test_server1_action(), server2).get();
        server2 = hpx::invalid_id;
        ensure_garbage_collect();

        HPX_TEST(test_managed_server1::alive);
        HPX_TEST(test_managed_server2::alive);

        test_managed_server1::test_action()(server1);
        server1 = hpx::invalid_id;
        ensure_garbage_collect();
    }

    return hpx::finalize();
}

int main(int argc, char **argv)
{
    HPX_TEST_EQ(hpx::init(argc, argv), 0);

    HPX_TEST(!test_simple_server1::alive);
    HPX_TEST(!test_simple_server2::alive);

    HPX_TEST(!test_managed_server1::alive);
    HPX_TEST(!test_managed_server2::alive);

    return hpx::util::report_errors();
}
