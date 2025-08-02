#include <hpx/algorithm.hpp>
#include <hpx/init.hpp>
#include <hpx/memory.hpp>
#include <hpx/modules/testing.hpp>

#include <ctime>
#include <iostream>
#include <string>
#include <vector>

#include "uninitialized_default_construct_tests.hpp"

template <typename IteratorTag>
void uninitialized_default_construct_sender_test()
{
    using namespace hpx::execution;
    test_uninitialized_default_construct_sender(
        hpx::launch::sync, seq(task), IteratorTag());
    test_uninitialized_default_construct_sender(
        hpx::launch::sync, unseq(task), IteratorTag());

    test_uninitialized_default_construct_sender(
        hpx::launch::async, par(task), IteratorTag());
    test_uninitialized_default_construct_sender(
        hpx::launch::async, par_unseq(task), IteratorTag());
}

template <typename IteratorTag>
void uninitialized_default_construct_sender_test2()
{
    using namespace hpx::execution;
    test_uninitialized_default_construct_sender2(
        hpx::launch::sync, seq(task), IteratorTag());
    test_uninitialized_default_construct_sender2(
        hpx::launch::sync, unseq(task), IteratorTag());

    test_uninitialized_default_construct_sender2(
        hpx::launch::async, par(task), IteratorTag());
    test_uninitialized_default_construct_sender2(
        hpx::launch::async, par_unseq(task), IteratorTag());
}

template <typename IteratorTag>
void uninitialized_default_construct_exception_sender_test()
{
    using namespace hpx::execution;
    test_uninitialized_default_construct_exception_sender(
        hpx::launch::sync, seq(task), IteratorTag());
    test_uninitialized_default_construct_exception_sender(
        hpx::launch::sync, unseq(task), IteratorTag());

    test_uninitialized_default_construct_exception_sender(
        hpx::launch::async, par(task), IteratorTag());
    test_uninitialized_default_construct_exception_sender(
        hpx::launch::async, par_unseq(task), IteratorTag());
}

template <typename IteratorTag>
void uninitialized_default_construct_bad_alloc_sender_test()
{
    using namespace hpx::execution;
    test_uninitialized_default_construct_bad_alloc_sender(
        hpx::launch::sync, seq(task), IteratorTag());
    test_uninitialized_default_construct_bad_alloc_sender(
        hpx::launch::sync, unseq(task), IteratorTag());

    test_uninitialized_default_construct_bad_alloc_sender(
        hpx::launch::async, par(task), IteratorTag());
    test_uninitialized_default_construct_bad_alloc_sender(
        hpx::launch::async, par_unseq(task), IteratorTag());
}

int hpx_main(hpx::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int) std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    uninitialized_default_construct_sender_test<std::forward_iterator_tag>();
    uninitialized_default_construct_sender_test<
        std::random_access_iterator_tag>();

    uninitialized_default_construct_sender_test2<std::forward_iterator_tag>();
    uninitialized_default_construct_sender_test2<
        std::random_access_iterator_tag>();

    uninitialized_default_construct_exception_sender_test<
        std::forward_iterator_tag>();
    uninitialized_default_construct_exception_sender_test<
        std::random_access_iterator_tag>();

    uninitialized_default_construct_bad_alloc_sender_test<
        std::forward_iterator_tag>();
    uninitialized_default_construct_bad_alloc_sender_test<
        std::random_access_iterator_tag>();

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    // add command line option which controls the random number generator seed
    using namespace hpx::program_options;
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()("seed,s", value<unsigned int>(),
        "the random number generator seed to use for this run");

    // By default this test should run on all available cores
    std::vector<std::string> const cfg = {"hpx.os_threads=all"};

    // Initialize and run HPX
    hpx::local::init_params init_args;
    init_args.desc_cmdline = desc_commandline;
    init_args.cfg = cfg;

    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv, init_args), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
