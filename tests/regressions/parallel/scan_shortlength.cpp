//  Copyright (c) 2015 Daniel Bourgeois
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_copy.hpp>
#include <hpx/include/parallel_remove_copy.hpp>
#include <hpx/util/lightweight_test.hpp>

// FIXME: Intel 15 currently can not compile this code. This needs to be fixed. See #1408
#if !(defined(HPX_INTEL_VERSION) && HPX_INTEL_VERSION == 1500)
void test_zero()
{
    using namespace hpx::parallel;
    typedef std::vector<int>::iterator Iter;
    std::vector<int> a;
    std::vector<int> b, c, d;

    Iter i_copy_if =
        copy_if(par, a.begin(), a.end(), b.begin(),
        [](int bar){ return bar % 2 == 1; });
    Iter i_remove_copy_if =
        remove_copy_if(par, a.begin(), a.end(), c.begin(),
        [](int bar){ return bar % 2 != 1; });
    Iter i_remove_copy =
        remove_copy(par, a.begin(), a.end(), d.begin(), 0);

    Iter ans_copy_if = std::copy_if(a.begin(), a.end(), b.begin(),
        [](int bar){ return bar % 2 == 1; });
    Iter ans_remove_copy_if = std::remove_copy_if(a.begin(), a.end(), c.begin(),
        [](int bar){ return bar % 2 != 1; });
    Iter ans_remove_copy = std::remove_copy(a.begin(), a.end(), d.begin(), 0);

    HPX_TEST(i_copy_if == ans_copy_if);
    HPX_TEST(i_remove_copy_if == ans_remove_copy_if);
    HPX_TEST(i_remove_copy == ans_remove_copy);
}

void test_async_zero()
{
    using namespace hpx::parallel;
    typedef std::vector<int>::iterator Iter;
    typedef hpx::future<Iter> Fut_Iter;
    std::vector<int> a;
    std::vector<int> b, c, d;

    Fut_Iter f_copy_if =
        copy_if(par(task), a.begin(), a.end(), b.begin(),
        [](int bar){ return bar % 2 == 1; });
    Fut_Iter f_remove_copy_if =
        remove_copy_if(par(task), a.begin(), a.end(), c.begin(),
        [](int bar){ return bar % 2 != 1; });
    Fut_Iter f_remove_copy =
        remove_copy(par(task), a.begin(), a.end(), d.begin(), 0);

    Iter ans_copy_if = std::copy_if(a.begin(), a.end(), b.begin(),
        [](int bar){ return bar % 2 == 1; });
    Iter ans_remove_copy_if = std::remove_copy_if(a.begin(), a.end(), c.begin(),
        [](int bar){ return bar % 2 != 1; });
    Iter ans_remove_copy = std::remove_copy(a.begin(), a.end(), d.begin(), 0);

    HPX_TEST(f_copy_if.get() == ans_copy_if);
    HPX_TEST(f_remove_copy_if.get() == ans_remove_copy_if);
    HPX_TEST(f_remove_copy.get() == ans_remove_copy);
}

void test_one(std::vector<int> a)
{
    using namespace hpx::parallel;
    typedef std::vector<int>::iterator Iter;
    std::size_t n = a.size();
    std::vector<int> b(n), c(n), d(n);
    std::vector<int> b_ans(n), c_ans(n), d_ans(n);

    Iter i_copy_if =
        copy_if(par, a.begin(), a.end(), b.begin(),
        [](int bar){ return bar % 2 == 1; });
    Iter i_remove_copy_if =
        remove_copy_if(par, a.begin(), a.end(), c.begin(),
        [](int bar){ return bar % 2 != 1; });
    Iter i_remove_copy =
        remove_copy(par, a.begin(), a.end(), d.begin(), 0);

    HPX_UNUSED(i_copy_if);
    HPX_UNUSED(i_remove_copy_if);
    HPX_UNUSED(i_remove_copy);

    std::copy_if(a.begin(), a.end(), b_ans.begin(),
        [](int bar){ return bar % 2 == 1; });
    std::remove_copy_if(a.begin(), a.end(), c_ans.begin(),
        [](int bar){ return bar % 2 != 1; });
    std::remove_copy(a.begin(), a.end(), d_ans.begin(), 0);

    HPX_TEST(std::equal(b.begin(), b.end(), b_ans.begin()));
    HPX_TEST(std::equal(c.begin(), c.end(), c_ans.begin()));
    HPX_TEST(std::equal(d.begin(), d.end(), d_ans.begin()));

}

void print(std::vector<int> const& result, std::vector<int> const& correct)
{
    HPX_TEST_EQ(result.size(), correct.size());
    for (std::size_t i = 0; i != result.size(); ++i)
        std::cout << i << ": " << result[i] << " == " << correct[i] << std::endl;
}

void test_async_one(std::vector<int> const& a)
{
    using namespace hpx::parallel;
    typedef std::vector<int>::iterator Iter;
    typedef hpx::future<Iter> Fut_Iter;
    std::size_t n = a.size();
    std::vector<int> b(n), c(n), d(n);
    std::vector<int> b_ans(n), c_ans(n), d_ans(n);

    Fut_Iter f_copy_if =
        copy_if(par(task), a.begin(), a.end(), b.begin(),
            [](int bar){ return bar % 2 == 1; });
    Fut_Iter f_remove_copy_if =
        remove_copy_if(par(task), a.begin(), a.end(), c.begin(),
            [](int bar){ return bar % 2 != 1; });
    Fut_Iter f_remove_copy =
        remove_copy(par(task), a.begin(), a.end(), d.begin(), 0);

    std::copy_if(a.begin(), a.end(), b_ans.begin(),
        [](int bar){ return bar % 2 == 1; });
    std::remove_copy_if(a.begin(), a.end(), c_ans.begin(),
        [](int bar){ return bar % 2 != 1; });
    std::remove_copy(a.begin(), a.end(), d_ans.begin(), 0);

    f_copy_if.wait();
    f_remove_copy_if.wait();
    f_remove_copy.wait();

    bool copy_if_eq = std::equal(b.begin(), b.end(), b_ans.begin());
    HPX_TEST(copy_if_eq);
    if (!copy_if_eq)
        print(b, b_ans);

    bool remove_copy_if_eq = std::equal(c.begin(), c.end(), c_ans.begin());
    HPX_TEST(remove_copy_if_eq);
    if (!remove_copy_if_eq)
        print(c, c_ans);

    bool remove_copy_eq = std::equal(d.begin(), d.end(), d_ans.begin());
    HPX_TEST(remove_copy_eq);
    if (!remove_copy_eq)
        print(d, d_ans);
}

int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(0);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    std::vector<int> a1(8);
    std::fill(a1.begin(), a1.end(), std::rand() % 1000);
    test_one(a1);
    test_async_one(a1);

    std::vector<int> a2(2);
    std::fill(a1.begin(), a1.end(), std::rand() % 1000);
    test_one(a2);
    test_async_one(a2);

    std::vector<int> a3(1);
    std::fill(a1.begin(), a1.end(), std::rand() % 1000);
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
    std::vector<std::string> cfg;
    cfg.push_back("hpx.os_threads=" +
        boost::lexical_cast<std::string>(hpx::threads::hardware_concurrency()));

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
