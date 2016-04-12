//  Copyright (c) 2014 Grant Mercer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_copy.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/range/functions.hpp>

#include <string>

#include "test_utils.hpp"

////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy>
void test_copy_if(ExPolicy policy)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, std::input_iterator_tag> iterator;

    std::vector<int> c(10007);
    std::vector<int> d(c.size());
    auto middle = boost::begin(c) + c.size()/2;
    std::iota(boost::begin(c), middle, std::rand());
    std::fill(middle, boost::end(c), -1);

    hpx::parallel::copy_if(policy,
        iterator(boost::begin(c)), iterator(boost::end(c)),
        boost::begin(d), [](int i){ return !(i < 0); });

    std::size_t count = 0;
    HPX_TEST(std::equal(boost::begin(c), middle, boost::begin(d),
        [&count](int v1, int v2) -> bool {
            HPX_TEST_EQ(v1, v2);
            ++count;
            return v1 == v2;
        }));

    HPX_TEST(std::equal(middle, boost::end(c),
        boost::begin(d) + d.size()/2,
        [&count](int v1, int v2) -> bool {
            HPX_TEST_NEQ(v1, v2);
            ++count;
            return v1!=v2;
    }));

    HPX_TEST_EQ(count, d.size());
}

template <typename ExPolicy>
void test_copy_if_async(ExPolicy p)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, std::input_iterator_tag> iterator;

    std::vector<int> c(10007);
    std::vector<int> d(c.size());
    auto middle = boost::begin(c) + c.size()/2;
    std::iota(boost::begin(c), middle, std::rand());
    std::fill(middle, boost::end(c), -1);

    auto f =
        hpx::parallel::copy_if(p,
            iterator(boost::begin(c)), iterator(boost::end(c)),
            boost::begin(d), [](int i){ return !(i < 0); });
    f.wait();

    std::size_t count = 0;
    HPX_TEST(std::equal(boost::begin(c), middle, boost::begin(d),
        [&count](int v1, int v2) -> bool {
            HPX_TEST_EQ(v1, v2);
            ++count;
            return v1 == v2;
        }));

    HPX_TEST(std::equal(middle, boost::end(c),
        boost::begin(d) + d.size()/2,
        [&count](int v1, int v2) -> bool {
            HPX_TEST_NEQ(v1, v2);
            ++count;
            return v1!=v2;
    }));

    HPX_TEST_EQ(count, d.size());
}

template <typename ExPolicy>
void test_copy_if_outiter(ExPolicy policy)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, std::input_iterator_tag> iterator;

    std::vector<int> c(10007);
    std::vector<int> d(0);
    auto middle = boost::begin(c) + c.size()/2;
    std::iota(boost::begin(c), middle, std::rand());
    std::fill(middle, boost::end(c), -1);

    hpx::parallel::copy_if(policy,
        iterator(boost::begin(c)), iterator(boost::end(c)),
        std::back_inserter(d), [](int i){ return !(i < 0); });

    HPX_TEST(std::equal(boost::begin(c), middle, boost::begin(d),
        [](int v1, int v2) -> bool {
            HPX_TEST_EQ(v1, v2);
            return v1 == v2;
        }));

    // assure D is half the size of C
    HPX_TEST_EQ(c.size()/2, d.size());
}

template <typename ExPolicy>
void test_copy_if_outiter_async(ExPolicy p)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, std::input_iterator_tag> iterator;

    std::vector<int> c(10007);
    std::vector<int> d(0);
    auto middle = boost::begin(c) + c.size()/2;
    std::iota(boost::begin(c), middle, std::rand());
    std::fill(middle, boost::end(c), -1);

    auto f =
        hpx::parallel::copy_if(p,
            iterator(boost::begin(c)), iterator(boost::end(c)),
            std::back_inserter(d), [](int i){ return !(i < 0); });
    f.wait();

    HPX_TEST(std::equal(boost::begin(c), middle, boost::begin(d),
        [](int v1, int v2) -> bool {
            HPX_TEST_EQ(v1, v2);
            return v1 == v2;
        }));

    HPX_TEST_EQ(c.size()/2, d.size());
}

void test_copy_if()
{
    using namespace hpx::parallel;

    test_copy_if(seq);
    test_copy_if(par);
    test_copy_if(par_vec);

    test_copy_if_async(seq(task));
    test_copy_if_async(par(task));

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_copy_if(execution_policy(seq));
    test_copy_if(execution_policy(par));
    test_copy_if(execution_policy(par_vec));

    test_copy_if(execution_policy(seq(task)));
    test_copy_if(execution_policy(par(task)));
#endif

    test_copy_if_outiter(seq);
    test_copy_if_outiter(par);
    test_copy_if_outiter(par_vec);

    test_copy_if_outiter_async(seq(task));
    test_copy_if_outiter_async(par(task));

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_copy_if_outiter(execution_policy(seq));
    test_copy_if_outiter(execution_policy(par));
    test_copy_if_outiter(execution_policy(par_vec));

    test_copy_if_outiter(execution_policy(seq(task)));
    test_copy_if_outiter(execution_policy(par(task)));
#endif
}

int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(0);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    test_copy_if();
    return hpx::finalize();
}

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
        std::to_string(hpx::threads::hardware_concurrency()));

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
