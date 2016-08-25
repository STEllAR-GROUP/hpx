//  copyright (c) 2014 Grant Mercer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_generate.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/range/functions.hpp>

#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

#include "test_utils.hpp"

// FIXME: Intel 15 currently can not compile this code. This needs to be fixed. See #1408
#if !(defined(HPX_INTEL_VERSION) && HPX_INTEL_VERSION == 1500)

////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_generate(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);

    auto gen = [](){ return std::size_t(10); };

    hpx::parallel::generate(policy,
        iterator(boost::begin(c)), iterator(boost::end(c)), gen);

    // verify values
    std::size_t count = 0;
    std::for_each(boost::begin(c), boost::end(c),
        [&count](std::size_t v) -> void {
            HPX_TEST_EQ(v, std::size_t(10));
            ++count;
        });
    HPX_TEST_EQ(count, c.size());
}

template <typename ExPolicy, typename IteratorTag>
void test_generate_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);

    auto gen = [](){ return std::size_t(10); };

    hpx::future<void> f =
        hpx::parallel::generate(p,
            iterator(boost::begin(c)), iterator(boost::end(c)),
            gen);
    f.wait();

    std::size_t count =0;
    std::for_each(boost::begin(c), boost::end(c),
        [&count](std::size_t v) -> void {
            HPX_TEST_EQ(v, std::size_t(10));
            ++count;
    });
    HPX_TEST_EQ(count, c.size());
}

template <typename IteratorTag>
void test_generate()
{
    using namespace hpx::parallel;
    test_generate(seq, IteratorTag());
    test_generate(par, IteratorTag());
    test_generate(par_vec, IteratorTag());

    test_generate_async(seq(task), IteratorTag());
    test_generate_async(par(task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_generate(execution_policy(seq), IteratorTag());
    test_generate(execution_policy(par), IteratorTag());
    test_generate(execution_policy(par_vec), IteratorTag());

    test_generate(execution_policy(seq(task)), IteratorTag());
    test_generate(execution_policy(par(task)), IteratorTag());
#endif
}

void generate_test()
{
    test_generate<std::random_access_iterator_tag>();
    test_generate<std::forward_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_generate_exception(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;
    std::vector<std::size_t> c(10007);

    auto gen = [](){return std::size_t(10);};

    bool caught_exception = false;
    try {
        hpx::parallel::generate(policy,
            decorated_iterator(
                boost::begin(c),
                [](){ throw std::runtime_error("test"); }),
            decorated_iterator(boost::end(c)),
            gen);
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
void test_generate_exception_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);

    auto gen = [](){return std::size_t(10);};

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try {
        hpx::future<void> f =
            hpx::parallel::generate(p,
                decorated_iterator(
                    boost::begin(c),
                    [](){ throw std::runtime_error("test"); }),
                decorated_iterator(boost::end(c)),
                gen);
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
void test_generate_exception()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_generate_exception(seq, IteratorTag());
    test_generate_exception(par, IteratorTag());

    test_generate_exception_async(seq(task), IteratorTag());
    test_generate_exception_async(par(task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_generate_exception(execution_policy(seq), IteratorTag());
    test_generate_exception(execution_policy(par), IteratorTag());

    test_generate_exception(execution_policy(seq(task)), IteratorTag());
    test_generate_exception(execution_policy(par(task)), IteratorTag());
#endif
}

void generate_exception_test()
{
    test_generate_exception<std::random_access_iterator_tag>();
    test_generate_exception<std::forward_iterator_tag>();
}

//////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_generate_bad_alloc(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(100007);

    auto gen = [](){return 10;};

    bool caught_bad_alloc = false;
    try {
        hpx::parallel::generate(policy,
            decorated_iterator(
                boost::begin(c),
                [](){ throw std::bad_alloc(); }),
            decorated_iterator(boost::end(c)),
            gen);
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
void test_generate_bad_alloc_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);

    auto gen = [](){return std::size_t(10);};

    bool caught_bad_alloc = false;
    bool returned_from_algorithm = false;
    try {
        hpx::future<void> f =
            hpx::parallel::generate(p,
                decorated_iterator(
                    boost::begin(c),
                    [](){ throw std::bad_alloc(); }),
                decorated_iterator(boost::end(c)),
                gen);
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
void test_generate_bad_alloc()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_generate_bad_alloc(seq, IteratorTag());
    test_generate_bad_alloc(par, IteratorTag());

    test_generate_bad_alloc_async(seq(task), IteratorTag());
    test_generate_bad_alloc_async(par(task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_generate_bad_alloc(execution_policy(seq), IteratorTag());
    test_generate_bad_alloc(execution_policy(par), IteratorTag());

    test_generate_bad_alloc(execution_policy(seq(task)), IteratorTag());
    test_generate_bad_alloc(execution_policy(par(task)), IteratorTag());
#endif
}

void generate_bad_alloc_test()
{
    test_generate_bad_alloc<std::random_access_iterator_tag>();
    test_generate_bad_alloc<std::forward_iterator_tag>();
}

int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    generate_test();
    generate_exception_test();
    generate_bad_alloc_test();
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
