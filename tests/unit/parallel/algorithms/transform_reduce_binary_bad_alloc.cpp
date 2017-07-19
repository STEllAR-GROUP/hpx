//  Copyright (c) 2015 Daniel Bourgeois
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_transform_reduce.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <cstddef>
#include <ctime>
#include <iostream>
#include <string>
#include <vector>

#include "test_utils.hpp"

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_transform_reduce_binary_bad_alloc(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c = test::random_iota(10007);
    std::vector<std::size_t> d = test::random_iota(10007);
    std::size_t init = std::rand() % 1007; //-V101

    bool caught_exception = false;
    try {
        hpx::parallel::transform_reduce(policy,
        decorated_iterator(
                std::begin(c),
                [](){ throw std::bad_alloc(); }),
            decorated_iterator(std::end(c)),
            std::begin(d), init);

        HPX_TEST(false);
    }
    catch(std::bad_alloc const&) {
        caught_exception = true;
    }
    catch(...) {
        HPX_TEST(false);
    }
    HPX_TEST(caught_exception);
}

template <typename ExPolicy, typename IteratorTag>
void test_transform_reduce_binary_bad_alloc_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c = test::random_iota(10007);
    std::vector<std::size_t> d = test::random_iota(10007);
    std::size_t init = std::rand() % 1007; //-V101

    bool returned_from_algorithm = false;
    bool caught_exception = false;

    try {
        hpx::future<std::size_t> f =
            hpx::parallel::transform_reduce(p,
                decorated_iterator(
                    std::begin(c),
                    [](){ throw std::bad_alloc(); }),
                decorated_iterator(std::end(c)),
                std::begin(d), init);

        returned_from_algorithm = true;

        f.get();

        HPX_TEST(false);
    }
    catch(std::bad_alloc const&) {
        caught_exception = true;
    }
    catch(...) {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
    HPX_TEST(returned_from_algorithm);
}

template <typename IteratorTag>
void test_transform_reduce_binary_bad_alloc()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_transform_reduce_binary_bad_alloc(execution::seq, IteratorTag());
    test_transform_reduce_binary_bad_alloc(execution::par, IteratorTag());

    test_transform_reduce_binary_bad_alloc_async(
        execution::seq(execution::task), IteratorTag());
    test_transform_reduce_binary_bad_alloc_async(
            execution::par(execution::task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_transform_reduce_binary_bad_alloc(execution_policy(execution::seq),
        IteratorTag());
    test_transform_reduce_binary_bad_alloc(execution_policy(execution::par),
        IteratorTag());

    test_transform_reduce_binary_bad_alloc(
        execution_policy(execution::seq(execution::task)),
        IteratorTag());
    test_transform_reduce_binary_bad_alloc(
        execution_policy(execution::par(execution::task)),
        IteratorTag());
#endif
}

void transform_reduce_binary_bad_alloc_test()
{
    test_transform_reduce_binary_bad_alloc<std::random_access_iterator_tag>();
    test_transform_reduce_binary_bad_alloc<std::forward_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    transform_reduce_binary_bad_alloc_test();

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
    std::vector<std::string> const cfg = {
        "hpx.os_threads=all"
    };

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
