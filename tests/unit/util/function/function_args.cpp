//  Copyright 2013 (c) Agustin Berge
//
//  Use, modification and
//  distribution is subject to the Boost Software License, Version
//  1.0. (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

// For more information, see http://www.boost.org

#include <hpx/hpx_main.hpp>
#include <hpx/include/util.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <utility>

struct counter
{
    static int default_constructions;
    static int copy_constructions;
    static int move_constructions;

    static void reset()
    {
        default_constructions = 0;
        copy_constructions = 0;
        move_constructions = 0;
    }

    static void print()
    {
        std::cout << "default-constructions: " << default_constructions << "\n";
        std::cout << "copy-constructions: " << copy_constructions << "\n";
        std::cout << "move-constructions: " << move_constructions << "\n" << std::endl;
    }

    counter(){ ++default_constructions; }
    counter(counter const&){ ++copy_constructions; }
    counter(counter &&){ ++move_constructions; }

private:
    ;

    counter& operator=(counter const&);
    counter& operator=(counter &&);
};

int counter::default_constructions = 0;
int counter::copy_constructions = 0;
int counter::move_constructions = 0;

////////////////////////////////////////////////////////////////////////////////
void f_value(counter){}

void test_by_value()
{
    hpx::util::function_nonser<void(counter)> f = f_value;

    counter::reset();

    counter c;
    f(c);
    f(std::move(c));

    HPX_TEST(counter::default_constructions == 1);
    HPX_TEST(counter::copy_constructions <= 1);
    HPX_TEST(counter::move_constructions <= 3);

    counter::print();
}

void f_lvalue_ref(counter&){}

void test_by_lvalue_ref()
{
    hpx::util::function_nonser<void(counter&)> f = f_lvalue_ref;

    counter::reset();

    counter c;
    f(c);
    //f(std::move(c)); // cannot bind rvalue to lvalue-ref (except MSVC)

    HPX_TEST(counter::default_constructions == 1);
    HPX_TEST(counter::copy_constructions == 0);
    HPX_TEST(counter::move_constructions == 0);

    counter::print();
}

void f_const_lvalue_ref(counter const&){}

void test_by_const_lvalue_ref()
{
    hpx::util::function_nonser<void(counter const&)> f = f_const_lvalue_ref;

    counter::reset();

    counter c;
    f(c);
    f(std::move(c));

    HPX_TEST(counter::default_constructions == 1);
    HPX_TEST(counter::copy_constructions == 0);
    HPX_TEST(counter::move_constructions == 0);

    counter::print();
}

void f_rvalue_ref(counter &&){}

void test_by_rvalue_ref()
{
    hpx::util::function_nonser<void(counter &&)> f = f_rvalue_ref;

    counter::reset();

    counter c;
    //f(c); // cannot bind lvalue to rvalue-ref
    f(std::move(c));

    HPX_TEST(counter::default_constructions == 1);
    HPX_TEST(counter::copy_constructions == 0);
    HPX_TEST(counter::move_constructions == 0);

    counter::print();
}

int main(int, char* [])
{
    test_by_value();
    test_by_lvalue_ref();
    test_by_const_lvalue_ref();
    test_by_rvalue_ref();

    return hpx::util::report_errors();
}
