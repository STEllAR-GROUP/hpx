//  copyright (c) 2014 Grant Mercer
//                2018 Bruno Pitrus
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_find.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <cstddef>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "test_utils.hpp"

////////////////////////////////////////////////////////////////////////////
unsigned int seed = std::random_device{}();
std::mt19937 gen(seed);
std::uniform_int_distribution<> dis(0,10006);
std::uniform_int_distribution<> dist(0,2);

template <typename ExPolicy, typename IteratorTag>
void test_find_first_of(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;

    int find_first_of_pos = dis(gen);
    int random_sub_seq_pos = dist(gen);

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), gen() + 19);
    std::size_t h[] = {1, 7, 18, 3};
    c[find_first_of_pos] = h[random_sub_seq_pos]; //-V108

    base_iterator index = hpx::parallel::find_first_of(policy, c, h);

    base_iterator test_index = std::begin(c) + find_first_of_pos;

    HPX_TEST(index == test_index);
}
template <typename ExPolicy, typename IteratorTag>
void test_find_first_of_proj(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;

    int find_first_of_pos = dis(gen);
    int random_sub_seq_pos = dist(gen);

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), (gen() % 32768) + 19);
    std::size_t h[] = {1+65536, 7+65536, 18+65536, 3+65536};
    c[find_first_of_pos] = h[random_sub_seq_pos];    //-V108

    base_iterator index =
        hpx::parallel::find_first_of(policy, c, h,
            std::equal_to<std::size_t>(),
        [](std::size_t x){ return x % 65536;},
        [](std::size_t x) { return x % 65536; });

    base_iterator test_index = std::begin(c) + find_first_of_pos;

    HPX_TEST(index == test_index);
}

template <typename ExPolicy, typename IteratorTag>
void test_find_first_of_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;

    int find_first_of_pos = dis(gen);
    int random_sub_seq_pos = dist(gen);

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), gen() + 19);
    std::size_t h[] = {1, 7, 18, 3};
    c[find_first_of_pos] = h[random_sub_seq_pos]; //-V108

    hpx::future<base_iterator> f = hpx::parallel::find_first_of(p, c, h);
    f.wait();

    // create iterator at position of value to be found
    base_iterator test_index = std::begin(c) + find_first_of_pos;

    HPX_TEST(f.get() == test_index);
}
template <typename ExPolicy, typename IteratorTag>
void test_find_first_of_async_proj(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;

    int find_first_of_pos = dis(gen);
    int random_sub_seq_pos = dist(gen);

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c),( gen() % 32768)  + 19);
    std::size_t h[] = {1+65536, 7+65536, 18+65536, 3+65536};
    c[find_first_of_pos] = h[random_sub_seq_pos];    //-V108

    hpx::future<base_iterator> f =
        hpx::parallel::find_first_of(p, c,h,
            std::equal_to<std::size_t>(),
        [](std::size_t x){ return x % 65536;},
        [](std::size_t x) { return x % 65536; });
    f.wait();

    // create iterator at position of value to be found
    base_iterator test_index = std::begin(c) + find_first_of_pos;

    HPX_TEST(f.get() == test_index);
}

template <typename IteratorTag>
void test_find_first_of()
{
    using namespace hpx::parallel;
    test_find_first_of(execution::seq, IteratorTag());
    test_find_first_of(execution::par, IteratorTag());
    test_find_first_of(execution::par_unseq, IteratorTag());
    test_find_first_of_proj(execution::seq, IteratorTag());
    test_find_first_of_proj(execution::par, IteratorTag());
    test_find_first_of_proj(execution::par_unseq, IteratorTag());

    test_find_first_of_async(execution::seq(execution::task), IteratorTag());
    test_find_first_of_async(execution::par(execution::task), IteratorTag());
    test_find_first_of_async_proj(execution::seq(execution::task), IteratorTag());
    test_find_first_of_async_proj(execution::par(execution::task), IteratorTag());
}

void find_first_of_test()
{
    test_find_first_of<std::random_access_iterator_tag>();
    test_find_first_of<std::forward_iterator_tag>();
#if defined(HPX_HAVE_ALGORITHM_INPUT_ITERATOR_SUPPORT)
    test_find_first_of<std::input_iterator_tag>();
#endif
}

///////////////////////////////////////////////////////////////////////////////

int hpx_main(boost::program_options::variables_map& vm)
{
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    gen.seed(seed);

    find_first_of_test();
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
