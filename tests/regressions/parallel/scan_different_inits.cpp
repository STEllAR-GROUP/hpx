//  Copyright (c) 2015 Daniel Bourgeois
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_scan.hpp>
#include <hpx/include/parallel_transform_scan.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

// FIXME: Intel 15 currently can not compile this code. This needs to be fixed. See #1408
#if !(defined(HPX_INTEL_VERSION) && HPX_INTEL_VERSION == 1500)
void test_zero()
{
    using namespace hpx::parallel;
    typedef std::vector<int>::iterator Iter;
    std::vector<int> a;
    std::vector<int> b, c, d, e, f, g;

    Iter i_inc_add =
        inclusive_scan(par, a.begin(), a.end(), b.begin(), 100,
        [](int bar, int baz){ return bar+baz; });
    Iter i_inc_mult =
        inclusive_scan(par, a.begin(), a.end(), c.begin(), 10,
        [](int bar, int baz){ return bar*baz; });
    Iter i_exc_add =
        exclusive_scan(par, a.begin(), a.end(), d.begin(), 100,
        [](int bar, int baz){ return bar+baz; });
    Iter i_exc_mult =
        exclusive_scan(par, a.begin(), a.end(), e.begin(), 10,
        [](int bar, int baz){ return bar*baz; });
    Iter i_transform_inc =
        transform_inclusive_scan(par, a.begin(), a.end(), f.begin(),
        [](int foo){ return foo - 3; }, 10,
        [](int bar, int baz){ return 2*bar+2*baz; });
    Iter i_transform_exc =
        transform_exclusive_scan(par, a.begin(), a.end(), g.begin(),
        [](int foo){ return foo - 3; }, 10,
        [](int bar, int baz){ return 2*bar+2*baz; });

    HPX_TEST(i_inc_add == b.begin());
    HPX_TEST(i_inc_mult == c.begin());
    HPX_TEST(i_exc_add == d.begin());
    HPX_TEST(i_exc_mult == e.begin());
    HPX_TEST(i_transform_inc == f.begin());
    HPX_TEST(i_transform_exc == g.begin());
}
void test_async_zero()
{
    using namespace hpx::parallel;
    typedef std::vector<int>::iterator Iter;
    typedef hpx::future<Iter> Fut_Iter;
    std::vector<int> a;
    std::vector<int> b, c, d, e, f, g;

    Fut_Iter f_inc_add =
        inclusive_scan(par(task), a.begin(), a.end(), b.begin(), 100,
        [](int bar, int baz){ return bar+baz; });
    Fut_Iter f_inc_mult =
        inclusive_scan(par(task), a.begin(), a.end(), c.begin(), 10,
        [](int bar, int baz){ return bar*baz; });
    Fut_Iter f_exc_add =
        exclusive_scan(par(task), a.begin(), a.end(), d.begin(), 100,
        [](int bar, int baz){ return bar+baz; });
    Fut_Iter f_exc_mult =
        exclusive_scan(par(task), a.begin(), a.end(), e.begin(), 10,
        [](int bar, int baz){ return bar*baz; });
    Fut_Iter f_transform_inc =
        transform_inclusive_scan(par(task), a.begin(), a.end(), f.begin(),
        [](int foo){ return foo - 3; }, 10,
        [](int bar, int baz){ return 2*bar+2*baz; });
    Fut_Iter f_transform_exc =
        transform_exclusive_scan(par(task), a.begin(), a.end(), g.begin(),
        [](int foo){ return foo - 3; }, 10,
        [](int bar, int baz){ return 2*bar+2*baz; });


    HPX_TEST(f_inc_add.get() == b.begin());
    HPX_TEST(f_inc_mult.get() == c.begin());
    HPX_TEST(f_exc_add.get() == d.begin());
    HPX_TEST(f_exc_mult.get() == e.begin());
    HPX_TEST(f_transform_inc.get() == f.begin());
    HPX_TEST(f_transform_exc.get() == g.begin());
}
void test_one(std::vector<int> a)
{
    using namespace hpx::parallel;
    typedef std::vector<int>::iterator Iter;
    std::size_t n = a.size();
    std::vector<int> b(n), c(n), d(n), e(n), f(n), g(n);
    std::vector<int> b_ans(n), c_ans(n), d_ans(n), e_ans(n), f_ans(n), g_ans(n);

    //  fun_add and fun_mult must be mathematically associative functions otherwise
    //  output is non-deterministic
    auto fun_add = [](int bar, int baz){ return bar+baz+1; };
    auto fun_mult = [](int bar, int baz){ return 2*bar*baz; };
    auto fun_conv = [](int foo){ return foo - 3; };

    Iter f_inc_add =
        inclusive_scan(par, a.begin(), a.end(), b.begin(), 10, fun_add);
    Iter f_inc_mult =
        inclusive_scan(par, a.begin(), a.end(), c.begin(), 10, fun_mult);
    Iter f_exc_add =
        exclusive_scan(par, a.begin(), a.end(), d.begin(), 10, fun_add);
    Iter f_exc_mult =
        exclusive_scan(par, a.begin(), a.end(), e.begin(), 10, fun_mult);
    Iter f_transform_inc =
        transform_inclusive_scan(par, a.begin(), a.end(), f.begin(),
        fun_conv, 10, fun_add);
    Iter f_transform_exc =
        transform_exclusive_scan(par, a.begin(), a.end(), g.begin(),
        fun_conv, 10, fun_add);

    HPX_UNUSED(f_inc_add);
    HPX_UNUSED(f_inc_mult);
    HPX_UNUSED(f_exc_add);
    HPX_UNUSED(f_exc_mult);
    HPX_UNUSED(f_transform_inc);
    HPX_UNUSED(f_transform_exc);

    hpx::parallel::v1::detail::sequential_inclusive_scan(
        a.begin(), a.end(), b_ans.begin(), 10, fun_add);
    hpx::parallel::v1::detail::sequential_inclusive_scan(
        a.begin(), a.end(), c_ans.begin(), 10, fun_mult);
    hpx::parallel::v1::detail::sequential_exclusive_scan(
        a.begin(), a.end(), d_ans.begin(), 10, fun_add);
    hpx::parallel::v1::detail::sequential_exclusive_scan(
        a.begin(), a.end(), e_ans.begin(), 10, fun_mult);
    hpx::parallel::v1::detail::sequential_transform_inclusive_scan(
        a.begin(), a.end(), f_ans.begin(), fun_conv, 10, fun_add);
    hpx::parallel::v1::detail::sequential_transform_exclusive_scan(
        a.begin(), a.end(), g_ans.begin(), fun_conv, 10, fun_add);

    HPX_TEST(std::equal(b.begin(), b.end(), b_ans.begin()));
    HPX_TEST(std::equal(c.begin(), c.end(), c_ans.begin()));
    HPX_TEST(std::equal(d.begin(), d.end(), d_ans.begin()));
    HPX_TEST(std::equal(e.begin(), e.end(), e_ans.begin()));
    HPX_TEST(std::equal(f.begin(), f.end(), f_ans.begin()));
    HPX_TEST(std::equal(g.begin(), g.end(), g_ans.begin()));
}

void test_async_one(std::vector<int> a)
{
    using namespace hpx::parallel;
    typedef std::vector<int>::iterator Iter;
    typedef hpx::future<Iter> Fut_Iter;
    std::size_t n = a.size();
    std::vector<int> b(n), c(n), d(n), e(n), f(n), g(n);
    std::vector<int> b_ans(n), c_ans(n), d_ans(n), e_ans(n), f_ans(n), g_ans(n);

    //  fun_add and fun_mult must be mathematically associative functions otherwise
    //  output is non-deterministic
    auto fun_add = [](int bar, int baz){ return bar+baz+1; };
    auto fun_mult = [](int bar, int baz){ return 2*bar*baz; };
    auto fun_conv = [](int foo){ return foo - 3; };

    Fut_Iter f_inc_add =
        inclusive_scan(par(task), a.begin(), a.end(), b.begin(), 10, fun_add);
    Fut_Iter f_inc_mult =
        inclusive_scan(par(task), a.begin(), a.end(), c.begin(), 10, fun_mult);
    Fut_Iter f_exc_add =
        exclusive_scan(par(task), a.begin(), a.end(), d.begin(), 10, fun_add);
    Fut_Iter f_exc_mult =
        exclusive_scan(par(task), a.begin(), a.end(), e.begin(), 10, fun_mult);
    Fut_Iter f_transform_inc =
        transform_inclusive_scan(par(task), a.begin(), a.end(), f.begin(),
        fun_conv, 10, fun_add);
    Fut_Iter f_transform_exc =
        transform_exclusive_scan(par(task), a.begin(), a.end(), g.begin(),
        fun_conv, 10, fun_add);

    hpx::parallel::v1::detail::sequential_inclusive_scan(
        a.begin(), a.end(), b_ans.begin(), 10, fun_add);
    hpx::parallel::v1::detail::sequential_inclusive_scan(
        a.begin(), a.end(), c_ans.begin(), 10, fun_mult);
    hpx::parallel::v1::detail::sequential_exclusive_scan(
        a.begin(), a.end(), d_ans.begin(), 10, fun_add);
    hpx::parallel::v1::detail::sequential_exclusive_scan(
        a.begin(), a.end(), e_ans.begin(), 10, fun_mult);
    hpx::parallel::v1::detail::sequential_transform_inclusive_scan(
        a.begin(), a.end(), f_ans.begin(), fun_conv, 10, fun_add);
    hpx::parallel::v1::detail::sequential_transform_exclusive_scan(
        a.begin(), a.end(), g_ans.begin(), fun_conv, 10, fun_add);

    f_inc_add.wait();
    f_inc_mult.wait();
    f_exc_add.wait();
    f_exc_mult.wait();
    f_transform_inc.wait();
    f_transform_exc.wait();

    HPX_TEST(std::equal(b.begin(), b.end(), b_ans.begin()));
    HPX_TEST(std::equal(c.begin(), c.end(), c_ans.begin()));
    HPX_TEST(std::equal(d.begin(), d.end(), d_ans.begin()));
    HPX_TEST(std::equal(e.begin(), e.end(), e_ans.begin()));
    HPX_TEST(std::equal(f.begin(), f.end(), f_ans.begin()));
    HPX_TEST(std::equal(g.begin(), g.end(), g_ans.begin()));
}

int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    auto get_next_num = [](int& num){ num = std::rand() % 100; };

    std::vector<int> a1(8); std::for_each(a1.begin(), a1.end(), get_next_num);
    test_one(a1);
    test_async_one(a1);

    std::vector<int> a2(2); std::for_each(a2.begin(), a2.end(), get_next_num);
    test_one(a2);
    test_async_one(a2);

    std::vector<int> a3(1); std::for_each(a3.begin(), a3.end(), get_next_num);
    test_one(a3);
    test_async_one(a3);

    test_zero();
    test_async_zero();

    return hpx::finalize();
}
#else
int hpx_main(boost::program_options::variables_map& vm)
{
    return hpx::finalize();
}
#endif

int main(int argc, char* argv[])
{
    // add command line option which controls the random number generator seed
    using namespace boost::program_options;
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ("seed,s", value<unsigned int>(),
        "the random number generator seed to use for this run")
        ;

    // By default this test should run on all available cores
    std::vector<std::string> const cfg = {
        "hpx.os_threads=all"
    };

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
