//  Copyright (c) 2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_mismatch.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/range/functions.hpp>

#include "test_utils.hpp"

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_mismatch_binary1(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    typedef std::pair<base_iterator, base_iterator> base_return_type;
    typedef std::pair<iterator, base_iterator> return_type;

    std::vector<std::size_t> c1(10007);
    std::vector<std::size_t> c2(c1.size());

    std::size_t first_value = std::rand(); //-V101
    std::iota(boost::begin(c1), boost::end(c1), first_value);
    std::iota(boost::begin(c2), boost::end(c2), first_value);

    iterator begin1 = iterator(boost::begin(c1));
    iterator end1 = iterator(boost::end(c1));

    {
        return_type result = hpx::parallel::mismatch(policy,
            begin1, end1, boost::begin(c2), boost::end(c2));

        // verify values
        HPX_TEST_EQ(std::size_t(std::distance(begin1, result.first)), c1.size());
        HPX_TEST_EQ(std::size_t(std::distance(boost::begin(c2), result.second)),
            c2.size());
    }

    {
        std::size_t changed_idx = std::rand() % c1.size(); //-V104
        ++c1[changed_idx];

        return_type result = hpx::parallel::mismatch(policy,
            begin1, end1, boost::begin(c2), boost::end(c2));

        // verify values
        HPX_TEST_EQ(std::size_t(std::distance(begin1, result.first)), changed_idx);
        HPX_TEST_EQ(std::size_t(std::distance(boost::begin(c2), result.second)),
            changed_idx);
    }
}

template <typename ExPolicy, typename IteratorTag>
void test_mismatch_binary1_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    typedef std::pair<base_iterator, base_iterator> base_return_type;
    typedef std::pair<iterator, base_iterator> return_type;

    std::vector<std::size_t> c1(10007);
    std::vector<std::size_t> c2(c1.size());

    std::size_t first_value = std::rand(); //-V101
    std::iota(boost::begin(c1), boost::end(c1), first_value);
    std::iota(boost::begin(c2), boost::end(c2), first_value);

    iterator begin1 = iterator(boost::begin(c1));
    iterator end1 = iterator(boost::end(c1));

    {
        hpx::future<return_type> f =
            hpx::parallel::mismatch(p,
                begin1, end1, boost::begin(c2), boost::end(c2));
        f.wait();

        // verify values
        return_type result = f.get();
        HPX_TEST_EQ(std::size_t(std::distance(begin1, result.first)),
            c1.size());
        HPX_TEST_EQ(std::size_t(std::distance(boost::begin(c2),
            result.second)), c2.size());
    }

    {
        std::size_t changed_idx = std::rand() % c1.size(); //-V104
        ++c1[changed_idx];

        hpx::future<return_type> f =
            hpx::parallel::mismatch(p,
                begin1, end1, boost::begin(c2), boost::end(c2));
        f.wait();

        // verify values
        return_type result = f.get();
        HPX_TEST_EQ(std::size_t(std::distance(begin1, result.first)),
            changed_idx);
        HPX_TEST_EQ(std::size_t(std::distance(boost::begin(c2),
            result.second)), changed_idx);
    }
}

template <typename IteratorTag>
void test_mismatch_binary1()
{
    using namespace hpx::parallel;

    test_mismatch_binary1(seq, IteratorTag());
    test_mismatch_binary1(par, IteratorTag());
    test_mismatch_binary1(par_vec, IteratorTag());

    test_mismatch_binary1_async(seq(task), IteratorTag());
    test_mismatch_binary1_async(par(task), IteratorTag());

    test_mismatch_binary1(execution_policy(seq), IteratorTag());
    test_mismatch_binary1(execution_policy(par), IteratorTag());
    test_mismatch_binary1(execution_policy(par_vec), IteratorTag());

    test_mismatch_binary1(execution_policy(seq(task)), IteratorTag());
    test_mismatch_binary1(execution_policy(par(task)), IteratorTag());
}

void mismatch_binary_test1()
{
    test_mismatch_binary1<std::random_access_iterator_tag>();
    test_mismatch_binary1<std::forward_iterator_tag>();
    test_mismatch_binary1<std::input_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_mismatch_binary2(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    typedef std::pair<base_iterator, base_iterator> base_return_type;
    typedef std::pair<iterator, base_iterator> return_type;

    std::vector<std::size_t> c1(10007);
    std::vector<std::size_t> c2(c1.size());

    std::size_t first_value = std::rand(); //-V101
    std::iota(boost::begin(c1), boost::end(c1), first_value);
    std::iota(boost::begin(c2), boost::end(c2), first_value);

    iterator begin1 = iterator(boost::begin(c1));
    iterator end1 = iterator(boost::end(c1));

    {
        return_type result = hpx::parallel::mismatch(policy,
            begin1, end1, boost::begin(c2), boost::end(c2),
            std::equal_to<std::size_t>());

        // verify values
        HPX_TEST_EQ(std::size_t(std::distance(begin1, result.first)),
            c1.size());
        HPX_TEST_EQ(std::size_t(std::distance(boost::begin(c2),
            result.second)), c2.size());
    }

    {
        std::size_t changed_idx = std::rand() % c1.size(); //-V104
        ++c1[changed_idx];

        return_type result = hpx::parallel::mismatch(policy,
            begin1, end1, boost::begin(c2), boost::end(c2),
            std::equal_to<std::size_t>());

        // verify values
        HPX_TEST_EQ(std::size_t(std::distance(begin1, result.first)), changed_idx);
        HPX_TEST_EQ(std::size_t(std::distance(boost::begin(c2), result.second)),
            changed_idx);
    }
}

template <typename ExPolicy, typename IteratorTag>
void test_mismatch_binary2_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    typedef std::pair<base_iterator, base_iterator> base_return_type;
    typedef std::pair<iterator, base_iterator> return_type;

    std::vector<std::size_t> c1(10007);
    std::vector<std::size_t> c2(c1.size());

    std::size_t first_value = std::rand(); //-V101
    std::iota(boost::begin(c1), boost::end(c1), first_value);
    std::iota(boost::begin(c2), boost::end(c2), first_value);

    iterator begin1 = iterator(boost::begin(c1));
    iterator end1 = iterator(boost::end(c1));

    {
        hpx::future<return_type> f =
            hpx::parallel::mismatch(p,
                begin1, end1, boost::begin(c2), boost::end(c2),
                std::equal_to<std::size_t>());
        f.wait();

        // verify values
        return_type result = f.get();
        HPX_TEST_EQ(std::size_t(std::distance(begin1, result.first)), c1.size());
        HPX_TEST_EQ(std::size_t(std::distance(boost::begin(c2),
            result.second)), c2.size());
    }

    {
        std::size_t changed_idx = std::rand() % c1.size(); //-V104
        ++c1[changed_idx];

        hpx::future<return_type> f =
            hpx::parallel::mismatch(p,
                begin1, end1, boost::begin(c2), boost::end(c2),
                std::equal_to<std::size_t>());
        f.wait();

        // verify values
        return_type result = f.get();
        HPX_TEST_EQ(std::size_t(std::distance(begin1, result.first)), changed_idx);
        HPX_TEST_EQ(std::size_t(std::distance(boost::begin(c2),
            result.second)), changed_idx);
    }
}

template <typename IteratorTag>
void test_mismatch_binary2()
{
    using namespace hpx::parallel;

    test_mismatch_binary2(seq, IteratorTag());
    test_mismatch_binary2(par, IteratorTag());
    test_mismatch_binary2(par_vec, IteratorTag());

    test_mismatch_binary2_async(seq(task), IteratorTag());
    test_mismatch_binary2_async(par(task), IteratorTag());

    test_mismatch_binary2(execution_policy(seq), IteratorTag());
    test_mismatch_binary2(execution_policy(par), IteratorTag());
    test_mismatch_binary2(execution_policy(par_vec), IteratorTag());

    test_mismatch_binary2(execution_policy(seq(task)), IteratorTag());
    test_mismatch_binary2(execution_policy(par(task)), IteratorTag());
}

void mismatch_binary_test2()
{
    test_mismatch_binary2<std::random_access_iterator_tag>();
    test_mismatch_binary2<std::forward_iterator_tag>();
    test_mismatch_binary2<std::input_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_mismatch_binary_exception(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    typedef std::pair<base_iterator, base_iterator> base_return_type;
    typedef std::pair<iterator, base_iterator> return_type;

    std::vector<std::size_t> c1(10007);
    std::vector<std::size_t> c2(c1.size());

    std::size_t first_value = std::rand(); //-V101
    std::iota(boost::begin(c1), boost::end(c1), first_value);
    std::iota(boost::begin(c2), boost::end(c2), first_value);

    bool caught_exception = false;
    try {
        hpx::parallel::mismatch(policy,
            iterator(boost::begin(c1)), iterator(boost::end(c1)),
            boost::begin(c2), boost::end(c2),
            [](std::size_t v1, std::size_t v2) {
                return throw std::runtime_error("test"), true;
            });

        HPX_TEST(false);
    }
    catch(hpx::exception_list const& e) {
        caught_exception = true;
        test::test_num_exceptions<ExPolicy, IteratorTag>::call(policy, e);
    }
    catch(...) {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
}

template <typename ExPolicy, typename IteratorTag>
void test_mismatch_binary_exception_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    typedef std::pair<base_iterator, base_iterator> base_return_type;
    typedef std::pair<iterator, base_iterator> return_type;

    std::vector<std::size_t> c1(10007);
    std::vector<std::size_t> c2(c1.size());

    std::size_t first_value = std::rand(); //-V101
    std::iota(boost::begin(c1), boost::end(c1), first_value);
    std::iota(boost::begin(c2), boost::end(c2), first_value);

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try {
        hpx::future<return_type> f =
            hpx::parallel::mismatch(p,
                iterator(boost::begin(c1)), iterator(boost::end(c1)),
                boost::begin(c2), boost::end(c2),
                [](std::size_t v1, std::size_t v2) {
                    return throw std::runtime_error("test"), true;
                });
        returned_from_algorithm = true;
        f.get();

        HPX_TEST(false);
    }
    catch(hpx::exception_list const& e) {
        caught_exception = true;
        test::test_num_exceptions<ExPolicy, IteratorTag>::call(p, e);
    }
    catch(...) {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
    HPX_TEST(returned_from_algorithm);
}

template <typename IteratorTag>
void test_mismatch_binary_exception()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_mismatch_binary_exception(seq, IteratorTag());
    test_mismatch_binary_exception(par, IteratorTag());

    test_mismatch_binary_exception_async(seq(task), IteratorTag());
    test_mismatch_binary_exception_async(par(task), IteratorTag());

    test_mismatch_binary_exception(execution_policy(seq), IteratorTag());
    test_mismatch_binary_exception(execution_policy(par), IteratorTag());

    test_mismatch_binary_exception(execution_policy(seq(task)), IteratorTag());
    test_mismatch_binary_exception(execution_policy(par(task)), IteratorTag());
}

void mismatch_binary_exception_test()
{
    test_mismatch_binary_exception<std::random_access_iterator_tag>();
    test_mismatch_binary_exception<std::forward_iterator_tag>();
    test_mismatch_binary_exception<std::input_iterator_tag>();
}

/////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_mismatch_binary_bad_alloc(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    typedef std::pair<base_iterator, base_iterator> base_return_type;
    typedef std::pair<iterator, base_iterator> return_type;

    std::vector<std::size_t> c1(10007);
    std::vector<std::size_t> c2(c1.size());

    std::size_t first_value = std::rand(); //-V101
    std::iota(boost::begin(c1), boost::end(c1), first_value);
    std::iota(boost::begin(c2), boost::end(c2), first_value);

    bool caught_bad_alloc = false;
    try {
        hpx::parallel::mismatch(policy,
            iterator(boost::begin(c1)), iterator(boost::end(c1)),
            boost::begin(c2), boost::end(c2),
            [](std::size_t v1, std::size_t v2) {
                return throw std::bad_alloc(), true;
            });

        HPX_TEST(false);
    }
    catch(std::bad_alloc const&) {
        caught_bad_alloc = true;
    }
    catch(...) {
        HPX_TEST(false);
    }

    HPX_TEST(caught_bad_alloc);
}

template <typename ExPolicy, typename IteratorTag>
void test_mismatch_binary_bad_alloc_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    typedef std::pair<base_iterator, base_iterator> base_return_type;
    typedef std::pair<iterator, base_iterator> return_type;

    std::vector<std::size_t> c1(10007);
    std::vector<std::size_t> c2(c1.size());

    std::size_t first_value = std::rand(); //-V101
    std::iota(boost::begin(c1), boost::end(c1), first_value);
    std::iota(boost::begin(c2), boost::end(c2), first_value);

    bool caught_bad_alloc = false;
    bool returned_from_algorithm = false;
    try {
        hpx::future<return_type> f =
            hpx::parallel::mismatch(p,
                iterator(boost::begin(c1)), iterator(boost::end(c1)),
                boost::begin(c2), boost::end(c2),
                [](std::size_t v1, std::size_t v2) {
                    return throw std::bad_alloc(), true;
                });
        returned_from_algorithm = true;
        f.get();

        HPX_TEST(false);
    }
    catch(std::bad_alloc const&) {
        caught_bad_alloc = true;
    }
    catch(...) {
        HPX_TEST(false);
    }

    HPX_TEST(caught_bad_alloc);
    HPX_TEST(returned_from_algorithm);
}

template <typename IteratorTag>
void test_mismatch_binary_bad_alloc()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_mismatch_binary_bad_alloc(seq, IteratorTag());
    test_mismatch_binary_bad_alloc(par, IteratorTag());

    test_mismatch_binary_bad_alloc_async(seq(task), IteratorTag());
    test_mismatch_binary_bad_alloc_async(par(task), IteratorTag());

    test_mismatch_binary_bad_alloc(execution_policy(seq), IteratorTag());
    test_mismatch_binary_bad_alloc(execution_policy(par), IteratorTag());

    test_mismatch_binary_bad_alloc(execution_policy(seq(task)), IteratorTag());
    test_mismatch_binary_bad_alloc(execution_policy(par(task)), IteratorTag());
}

void mismatch_binary_bad_alloc_test()
{
    test_mismatch_binary_bad_alloc<std::random_access_iterator_tag>();
    test_mismatch_binary_bad_alloc<std::forward_iterator_tag>();
    test_mismatch_binary_bad_alloc<std::input_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(0);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    mismatch_binary_test1();
    mismatch_binary_test2();
    mismatch_binary_exception_test();
    mismatch_binary_bad_alloc_test();
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
        boost::lexical_cast<std::string>(hpx::threads::hardware_concurrency()));

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}


