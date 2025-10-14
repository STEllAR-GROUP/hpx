///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2017 Taeguk Kwon
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#include <hpx/algorithm.hpp>
#include <hpx/assert.hpp>
#include <hpx/chrono.hpp>
#include <hpx/format.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/itt_notify.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/program_options.hpp>
#include <hpx/schedulers/local_priority_queue_scheduler.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "utils.hpp"

///////////////////////////////////////////////////////////////////////////////
unsigned int seed = std::random_device{}();

///////////////////////////////////////////////////////////////////////////////
template <typename InIter1, typename InIter2, typename OutIter>
double run_merge_benchmark_std(int const test_count, InIter1 first1,
    InIter1 last1, InIter2 first2, InIter2 last2, OutIter dest)
{
    std::uint64_t time = hpx::chrono::high_resolution_clock::now();

    for (int i = 0; i < test_count; ++i)
    {
        std::merge(first1, last1, first2, last2, dest);
    }

    time = hpx::chrono::high_resolution_clock::now() - time;

    return (static_cast<double>(time) * 1e-9) / test_count;
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
    typename FwdIter3>
double run_merge_benchmark_hpx(int const test_count, ExPolicy policy,
    FwdIter1 first1, FwdIter1 last1, FwdIter2 first2, FwdIter2 last2,
    FwdIter3 dest)
{
    std::uint64_t time = hpx::chrono::high_resolution_clock::now();

    for (int i = 0; i < test_count; ++i)
    {
        hpx::merge(policy, first1, last1, first2, last2, dest);
    }

    time = hpx::chrono::high_resolution_clock::now() - time;

    return (static_cast<double>(time) * 1e-9) / test_count;
}

///////////////////////////////////////////////////////////////////////////////
struct compute_chunk_size
{
    explicit constexpr compute_chunk_size(std::size_t times_cores = 4)
      : times_cores_(times_cores)
    {
    }

    //template <typename Executor>
    //friend std::size_t tag_override_invoke(
    //    hpx::execution::experimental::maximal_number_of_chunks_t,
    //    compute_chunk_size& this_, Executor&, std::size_t const cores,
    //    std::size_t const)
    //{
    //    return this_.times_cores_ * cores;
    //}

    template <typename Executor>
    friend std::size_t tag_override_invoke(
        hpx::execution::experimental::get_chunk_size_t,
        compute_chunk_size& this_, Executor&,
        hpx::chrono::steady_duration const&, std::size_t const cores,
        std::size_t const num_iterations)
    {
        if (cores == 1)
        {
            return num_iterations;
        }

        // Return a chunk size that ensures that each core ends up with the same
        // number of chunks the sizes of which are equal (except for the last
        // chunk, which may be smaller by not more than the number of chunks in
        // terms of elements).
        std::size_t const num_chunks = this_.times_cores_ * cores;
        std::size_t chunk_size = (num_iterations + num_chunks - 1) / num_chunks;

        // we should not consider more chunks than we have elements
        auto const max_chunks = (std::min) (num_chunks, num_iterations);

        // we should not make chunks smaller than what's determined by the max
        // chunk size
        chunk_size = (std::max) (chunk_size,
            (num_iterations + max_chunks - 1) / max_chunks);

        HPX_ASSERT(chunk_size * num_chunks >= num_iterations);

        return chunk_size;
    }

    std::size_t times_cores_;
};

template <>
struct hpx::execution::experimental::is_executor_parameters<compute_chunk_size>
  : std::true_type
{
};

///////////////////////////////////////////////////////////////////////////////
template <typename T>
struct random_to_item_t
{
    double m_min;
    double m_max;

    random_to_item_t(T min, T max)
      : m_min(static_cast<double>(min))
      , m_max(static_cast<double>(max))
    {
    }

    T operator()(double random_value) const
    {
        if constexpr (std::is_floating_point_v<T>)
        {
            return static_cast<T>((m_max - m_min) * random_value + m_min);
        }
        else
        {
            return static_cast<T>(
                std::floor((m_max - m_min + 1) * random_value + m_min));
        }
    }
};

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag, typename Allocator>
void run_benchmark(std::size_t vector_size1, std::size_t vector_size2,
    int test_count, IteratorTag, Allocator const& alloc,
    std::string const& type, int entropy, int num_chunks)
{
    std::cout << "* Preparing Benchmark... (" << type << ")" << std::endl;

    // initialize data
    using namespace hpx::execution;

    std::vector<double> uniform_distribution(vector_size1 + vector_size2);

    std::default_random_engine re(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    hpx::generate(par, std::begin(uniform_distribution),
        std::end(uniform_distribution), [&] { return dist(re); });

    using test_container = test_container<IteratorTag, int, Allocator>;
    using container = typename test_container::type;
    using T = typename container::value_type;

    container src =
        test_container::get_container(vector_size1 + vector_size2, alloc);

    T const min = (std::numeric_limits<T>::min)();
    T const max = (std::numeric_limits<T>::max)();

    hpx::transform(par, std::begin(uniform_distribution),
        std::end(uniform_distribution), src.data(),
        random_to_item_t<T>(min, max));

    // entropy: 0 -> 1, 4 -> 0.201
    for (int i = 0; i < entropy; ++i, ++seed)
    {
        hpx::generate(par, std::begin(uniform_distribution),
            std::end(uniform_distribution), [&] { return dist(re); });

        container tmp_vec(src.size(), alloc);
        hpx::transform(par, std::begin(uniform_distribution),
            std::end(uniform_distribution), tmp_vec.data(),
            random_to_item_t<T>(min, max));

        hpx::transform(par, src.data(), src.data() + src.size(), tmp_vec.data(),
            src.data(), std::bit_and{});
    }

    hpx::sort(par, src.begin(), src.begin() + vector_size1);
    hpx::sort(par, src.begin() + vector_size1, src.end());

    auto first1 = std::begin(src);
    auto last1 = std::begin(src) + vector_size1;
    auto first2 = std::begin(src) + vector_size1;
    auto last2 = std::end(src);

    container result =
        test_container::get_container(vector_size1 + vector_size2, alloc);

    auto dest = std::begin(result);

    std::cout << "* Running Benchmark... (" << type << ")" << std::endl;
    std::cout << "--- run_merge_benchmark_std ---" << std::endl;

    double const time_std =
        run_merge_benchmark_std(test_count, first1, last1, first2, last2, dest);

    hpx::this_thread::sleep_for(std::chrono::seconds(1));

    std::cout << "--- run_merge_benchmark_seq ---" << std::endl;

    double const time_seq = run_merge_benchmark_hpx(
        test_count, seq, first1, last1, first2, last2, dest);

    hpx::this_thread::sleep_for(std::chrono::seconds(1));

    std::cout << "--- run_merge_benchmark_par ---" << std::endl;

    HPX_ITT_RESUME();

    //hpx::threads::thread_schedule_hint hint(
    //    hpx::threads::thread_sharing_hint::do_not_share_function);
    //auto policy = hpx::execution::experimental::with_hint(par, hint);

    auto const policy = hpx::execution::experimental::with_priority(
        par, hpx::threads::thread_priority::initially_bound);

    compute_chunk_size ccs(num_chunks);
    double const time_par = run_merge_benchmark_hpx(
        test_count, policy.with(ccs), first1, last1, first2, last2, dest);

    HPX_ITT_PAUSE();

    std::cout << "--- run_merge_benchmark_par_fork_join ---" << std::endl;
    double time_par_fork_join = 0;
    {
        hpx::execution::experimental::fork_join_executor exec;
        time_par_fork_join = run_merge_benchmark_hpx(
            test_count, par.on(exec), first1, last1, first2, last2, dest);
    }

    std::cout << "--- run_merge_benchmark_par_unseq ---" << std::endl;
    double const time_par_unseq = run_merge_benchmark_hpx(
        test_count, par_unseq, first1, last1, first2, last2, dest);

    std::cout << "\n-------------- Benchmark Result --------------"
              << std::endl;
    auto const fmt = "merge ({1}) : {2}(sec)";
    hpx::util::format_to(std::cout, fmt, "std", time_std) << std::endl;
    hpx::util::format_to(std::cout, fmt, "seq", time_seq) << std::endl;
    hpx::util::format_to(std::cout, fmt, "par", time_par) << std::endl;
    hpx::util::format_to(std::cout, fmt, "par_fork_join", time_par_fork_join)
        << std::endl;
    hpx::util::format_to(std::cout, fmt, "par_unseq", time_par_unseq)
        << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(hpx::program_options::variables_map& vm)
{
    HPX_ITT_PAUSE();

    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    // pull values from cmd
    std::size_t const vector_size = vm["vector_size"].as<std::size_t>();
    double const vector_ratio = vm["vector_ratio"].as<double>();
    int const test_count = vm["test_count"].as<int>();
    int const entropy = vm["entropy"].as<int>();
    int const num_chunks = vm["num_chunks"].as<int>();

    std::size_t const os_threads = hpx::get_os_thread_count();

    std::size_t const vector_size1 = static_cast<std::size_t>(
        static_cast<double>(vector_size) * vector_ratio);
    std::size_t const vector_size2 = vector_size - vector_size1;

    std::cout << "-------------- Benchmark Config --------------" << std::endl;
    std::cout << "seed         : " << seed << std::endl;
    std::cout << "entropy      : " << entropy << std::endl;
    std::cout << "vector_size1 : " << vector_size1 << std::endl;
    std::cout << "vector_size2 : " << vector_size2 << std::endl;
    std::cout << "test_count   : " << test_count << std::endl;
    std::cout << "os threads   : " << os_threads << std::endl;
    std::cout << "num chunks   : " << num_chunks * os_threads << std::endl;
    std::cout << "----------------------------------------------\n"
              << std::endl;

    {
        using allocator_type = std::allocator<int>;
        allocator_type alloc;

        run_benchmark(vector_size1, vector_size2, test_count,
            std::random_access_iterator_tag(), alloc, "std::vector", entropy,
            num_chunks);
    }

    {
        auto policy = hpx::execution::par;
        using allocator_type =
            hpx::compute::host::detail::policy_allocator<int, decltype(policy)>;
        allocator_type alloc(policy);

        run_benchmark(vector_size1, vector_size2, test_count,
            std::random_access_iterator_tag(), alloc, "hpx::compute::vector",
            entropy, num_chunks);
    }

    return hpx::local::finalize();
}

int main(int const argc, char* argv[])
{
    using namespace hpx::program_options;
    options_description desc_commandline(
        "usage: " HPX_APPLICATION_STRING " [options]");

#if defined(HPX_DEBUG)
    constexpr std::size_t vector_size = 268435;
#else
    constexpr std::size_t vector_size = 268435456;
#endif

    std::string const vector_size_help =
        "sum of sizes of two vectors (default: " + std::to_string(vector_size) +
        ")";

    // clang-format off
    desc_commandline.add_options()
        ("vector_size", value<std::size_t>()->default_value(vector_size),
         vector_size_help.c_str())
        ("vector_ratio", value<double>()->default_value(0.7),
         "ratio of two vector sizes (default: 0.7)")
        ("entropy", value<int>()->default_value(1),
         "entropy value: 0 -> 1, 4 -> 0.201 (default: 0)")
        ("num_chunks", value<int>()->default_value(8),
         "number of chunks (times number of cores) (default: 8)")
        ("test_count", value<int>()->default_value(10),
         "number of tests to be averaged (default: 10)")
        ("seed,s", value<unsigned int>(),
         "the random number generator seed to use for this run")
    ;
    // clang-format on

    // initialize program
    std::vector<std::string> const cfg = {"hpx.os_threads=all"};

    // Initialize and run HPX
    hpx::local::init_params init_args;
    init_args.desc_cmdline = desc_commandline;
    init_args.cfg = cfg;

    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv, init_args), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
