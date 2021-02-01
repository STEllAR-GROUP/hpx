///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2017 Taeguk Kwon
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/parallel_copy.hpp>
#include <hpx/include/parallel_generate.hpp>
#include <hpx/include/parallel_remove.hpp>
#include <hpx/modules/format.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/modules/timing.hpp>

#include <hpx/modules/program_options.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "utils.hpp"

///////////////////////////////////////////////////////////////////////////////
unsigned int seed = std::random_device{}();
///////////////////////////////////////////////////////////////////////////////
struct random_fill
{
    random_fill(std::size_t random_range)
      : gen(seed)
      , dist(0, static_cast<int>(random_range - 1))
    {
    }

    int operator()()
    {
        return dist(gen);
    }

    std::mt19937 gen;
    std::uniform_int_distribution<> dist;
};

///////////////////////////////////////////////////////////////////////////////
struct vector_type
{
    vector_type() = default;
    vector_type(int rand_no)
    {
        vec_.reserve(vec_size_);
        for (std::size_t i = 0u; i < vec_size_; ++i)
            vec_.push_back(rand_no);
    }

    bool operator==(vector_type const& t) const
    {
        return vec_ == t.vec_;
    }

    std::vector<int> vec_;
    static const std::size_t vec_size_{30};
};

struct array_type
{
    array_type() = default;
    array_type(int rand_no)
    {
        for (std::size_t i = 0u; i < arr_size_; ++i)
            arr_[i] = rand_no;
    }

    bool operator==(array_type const& t) const
    {
        return arr_ == t.arr_;
    }

    static const std::size_t arr_size_{30};
    std::array<int, arr_size_> arr_;
};

///////////////////////////////////////////////////////////////////////////////
template <typename OrgIter, typename InIter, typename ValueType>
double run_remove_benchmark_std(int test_count, OrgIter org_first,
    OrgIter org_last, InIter first, InIter last, ValueType value)
{
    std::uint64_t time = std::uint64_t(0);

    for (int i = 0; i < test_count; ++i)
    {
        // Restore [first, last) with original data.
        hpx::copy(hpx::execution::par, org_first, org_last, first);

        std::uint64_t elapsed = hpx::chrono::high_resolution_clock::now();
        (void) std::remove(first, last, value);
        time += hpx::chrono::high_resolution_clock::now() - elapsed;
    }

    return (time * 1e-9) / test_count;
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename OrgIter, typename FwdIter,
    typename ValueType>
double run_remove_benchmark_hpx(int test_count, ExPolicy policy,
    OrgIter org_first, OrgIter org_last, FwdIter first, FwdIter last,
    ValueType value)
{
    std::uint64_t time = std::uint64_t(0);

    for (int i = 0; i < test_count; ++i)
    {
        // Restore [first, last) with original data.
        hpx::copy(hpx::execution::par, org_first, org_last, first);

        std::uint64_t elapsed = hpx::chrono::high_resolution_clock::now();
        hpx::remove(policy, first, last, value);
        time += hpx::chrono::high_resolution_clock::now() - elapsed;
    }

    return (time * 1e-9) / test_count;
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag, typename DataType>
void run_benchmark(std::size_t vector_size, int test_count,
    std::size_t random_range, IteratorTag, DataType)
{
    std::cout << "* Preparing Benchmark..." << std::endl;

    typedef test_container<IteratorTag, DataType> test_container;
    typedef typename test_container::type container;

    container v = test_container::get_container(vector_size);
    container org_v;

    auto first = std::begin(v);
    auto last = std::end(v);

    // initialize data
    using namespace hpx::execution;
    hpx::generate(par, std::begin(v), std::end(v), random_fill(random_range));
    org_v = v;

    auto value = DataType(static_cast<int>(random_range / 2));

    auto dest_dist = std::distance(first, std::remove(first, last, value));

    auto org_first = std::begin(org_v);
    auto org_last = std::end(org_v);

    std::cout << "*** Distance of new range after performing the algorithm : "
              << dest_dist << std::endl
              << std::endl;

    std::cout << "* Running Benchmark..." << std::endl;
    std::cout << "--- run_remove_benchmark_std ---" << std::endl;
    double time_std = run_remove_benchmark_std(
        test_count, org_first, org_last, first, last, value);

    std::cout << "--- run_remove_benchmark_seq ---" << std::endl;
    double time_seq = run_remove_benchmark_hpx(
        test_count, seq, org_first, org_last, first, last, value);

    std::cout << "--- run_remove_benchmark_par ---" << std::endl;
    double time_par = run_remove_benchmark_hpx(
        test_count, par, org_first, org_last, first, last, value);

    std::cout << "--- run_remove_benchmark_par_unseq ---" << std::endl;
    double time_par_unseq = run_remove_benchmark_hpx(
        test_count, par_unseq, org_first, org_last, first, last, value);

    std::cout << "\n-------------- Benchmark Result --------------"
              << std::endl;
    auto fmt = "remove ({1}) : {2}(sec)";
    hpx::util::format_to(std::cout, fmt, "std", time_std) << std::endl;
    hpx::util::format_to(std::cout, fmt, "seq", time_seq) << std::endl;
    hpx::util::format_to(std::cout, fmt, "par", time_par) << std::endl;
    hpx::util::format_to(std::cout, fmt, "par_unseq", time_par_unseq)
        << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
}

template <typename IteratorTag>
void run_benchmark(std::size_t vector_size, int test_count,
    std::size_t random_range, IteratorTag iterator_tag,
    std::string const& data_type_str)
{
    if (data_type_str == "int")
        run_benchmark(
            vector_size, test_count, random_range, iterator_tag, int());
    else if (data_type_str == "vector")
        run_benchmark(
            vector_size, test_count, random_range, iterator_tag, vector_type());
    else    // array
        run_benchmark(
            vector_size, test_count, random_range, iterator_tag, array_type());
}

void run_benchmark(std::size_t vector_size, int test_count,
    std::size_t random_range, std::string const& iterator_tag_str,
    std::string const& data_type_str)
{
    if (iterator_tag_str == "random")
        run_benchmark(vector_size, test_count, random_range,
            std::random_access_iterator_tag(), data_type_str);
    else if (iterator_tag_str == "bidirectional")
        run_benchmark(vector_size, test_count, random_range,
            std::bidirectional_iterator_tag(), data_type_str);
    else    // forward
        run_benchmark(vector_size, test_count, random_range,
            std::forward_iterator_tag(), data_type_str);
}

///////////////////////////////////////////////////////////////////////////////
std::string correct_iterator_tag_str(std::string const& iterator_tag)
{
    if (iterator_tag != "random" && iterator_tag != "bidirectional" &&
        iterator_tag != "forward")
        return "random";
    else
        return iterator_tag;
}

std::string correct_data_type_str(std::string const& data_type)
{
    if (data_type != "int" && data_type != "vector" && data_type != "array")
        return "int";
    else
        return data_type;
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(hpx::program_options::variables_map& vm)
{
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    // pull values from cmd
    std::size_t vector_size = vm["vector_size"].as<std::size_t>();
    std::size_t random_range = vm["random_range"].as<std::size_t>();
    int test_count = vm["test_count"].as<int>();
    std::string iterator_tag_str =
        correct_iterator_tag_str(vm["iterator_tag"].as<std::string>());
    std::string data_type_str =
        correct_data_type_str(vm["data_type"].as<std::string>());

    std::size_t const os_threads = hpx::get_os_thread_count();

    if (random_range < 1)
        random_range = 1;

    std::cout << "-------------- Benchmark Config --------------" << std::endl;
    std::cout << "seed         : " << seed << std::endl;
    std::cout << "vector_size  : " << vector_size << std::endl;
    std::cout << "random_range : " << random_range << std::endl;
    std::cout << "iterator_tag : " << iterator_tag_str << std::endl;
    std::cout << "data_type    : " << data_type_str << std::endl;
    std::cout << "test_count   : " << test_count << std::endl;
    std::cout << "os threads   : " << os_threads << std::endl;
    std::cout << "----------------------------------------------\n"
              << std::endl;

    run_benchmark(
        vector_size, test_count, random_range, iterator_tag_str, data_type_str);

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    using namespace hpx::program_options;
    options_description desc_commandline(
        "usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()("vector_size",
        hpx::program_options::value<std::size_t>()->default_value(1000000),
        "size of vector (default: 1000000)")("random_range",
        hpx::program_options::value<std::size_t>()->default_value(6),
        "range of random numbers [0, x) (default: 6)")("iterator_tag",
        hpx::program_options::value<std::string>()->default_value("random"),
        "the kind of iterator tag (random/bidirectional/forward)")("data_type",
        hpx::program_options::value<std::string>()->default_value("int"),
        "the kind of data type (int/vector/array)")("test_count",
        hpx::program_options::value<int>()->default_value(10),
        "number of tests to be averaged (default: 10)")("seed,s",
        hpx::program_options::value<unsigned int>(),
        "the random number generator seed to use for this run");

    // initialize program
    std::vector<std::string> const cfg = {"hpx.os_threads=all"};

    // Initialize and run HPX
    hpx::init_params init_args;
    init_args.desc_cmdline = desc_commandline;
    init_args.cfg = cfg;

    HPX_TEST_EQ_MSG(hpx::init(argc, argv, init_args), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
