//  Copyright (c) 2020-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/local/execution.hpp>
#include <hpx/local/init.hpp>
#include <hpx/modules/testing.hpp>

#include <atomic>
#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

// This test verifies that all parameters customization points dispatch
// through the executor before potentially being handled by the parameters
// object.

std::atomic<std::size_t> params_count(0);
std::atomic<std::size_t> exec_count(0);

///////////////////////////////////////////////////////////////////////////////
// get_chunks_size

struct test_executor_get_chunk_size : hpx::execution::parallel_executor
{
    test_executor_get_chunk_size()
      : hpx::execution::parallel_executor()
    {
    }

    template <typename Parameters, typename F>
    static std::size_t get_chunk_size(Parameters&& /* params */, F&& /* f */,
        std::size_t cores, std::size_t count)
    {
        ++exec_count;
        return (count + cores - 1) / cores;
    }
};

namespace hpx { namespace parallel { namespace execution {
    template <>
    struct is_two_way_executor<test_executor_get_chunk_size> : std::true_type
    {
    };
}}}    // namespace hpx::parallel::execution

struct test_chunk_size
{
    template <typename Executor, typename F>
    static std::size_t get_chunk_size(Executor&& /* exec */, F&& /* f */,
        std::size_t cores, std::size_t count)
    {
        ++params_count;
        return (count + cores - 1) / cores;
    }
};

namespace hpx { namespace parallel { namespace execution {
    /// \cond NOINTERNAL
    template <>
    struct is_executor_parameters<test_chunk_size> : std::true_type
    {
    };
    /// \endcond
}}}    // namespace hpx::parallel::execution

///////////////////////////////////////////////////////////////////////////////
void test_get_chunk_size()
{
    {
        params_count = 0;
        exec_count = 0;

        hpx::parallel::execution::get_chunk_size(
            test_chunk_size{}, hpx::execution::par.executor(),
            [](std::size_t) { return 0; }, 1, 1);

        HPX_TEST_EQ(params_count, std::size_t(1));
        HPX_TEST_EQ(exec_count, std::size_t(0));
    }

    {
        params_count = 0;
        exec_count = 0;

        hpx::parallel::execution::get_chunk_size(
            test_chunk_size{}, test_executor_get_chunk_size{},
            [](std::size_t) { return 0; }, 1, 1);

        HPX_TEST_EQ(params_count, std::size_t(0));
        HPX_TEST_EQ(exec_count, std::size_t(1));
    }
}

///////////////////////////////////////////////////////////////////////////////
// maximal_number_of_chunks

struct test_executor_maximal_number_of_chunks
  : hpx::execution::parallel_executor
{
    test_executor_maximal_number_of_chunks()
      : hpx::execution::parallel_executor()
    {
    }

    template <typename Parameters>
    static std::size_t maximal_number_of_chunks(
        Parameters&&, std::size_t, std::size_t num_tasks)
    {
        ++exec_count;
        return num_tasks;
    }
};

namespace hpx { namespace parallel { namespace execution {
    template <>
    struct is_two_way_executor<test_executor_maximal_number_of_chunks>
      : std::true_type
    {
    };
}}}    // namespace hpx::parallel::execution

struct test_number_of_chunks
{
    template <typename Executor>
    std::size_t maximal_number_of_chunks(
        Executor&&, std::size_t, std::size_t num_tasks)
    {
        ++params_count;
        return num_tasks;
    }
};

namespace hpx { namespace parallel { namespace execution {
    template <>
    struct is_executor_parameters<test_number_of_chunks> : std::true_type
    {
    };
}}}    // namespace hpx::parallel::execution

///////////////////////////////////////////////////////////////////////////////
void test_maximal_number_of_chunks()
{
    {
        params_count = 0;
        exec_count = 0;

        hpx::parallel::execution::maximal_number_of_chunks(
            test_number_of_chunks{}, hpx::execution::par.executor(), 1, 1);

        HPX_TEST_EQ(params_count, std::size_t(1));
        HPX_TEST_EQ(exec_count, std::size_t(0));
    }

    {
        params_count = 0;
        exec_count = 0;

        hpx::parallel::execution::maximal_number_of_chunks(
            test_number_of_chunks{}, test_executor_maximal_number_of_chunks{},
            1, 1);

        HPX_TEST_EQ(params_count, std::size_t(0));
        HPX_TEST_EQ(exec_count, std::size_t(1));
    }
}

///////////////////////////////////////////////////////////////////////////////
// reset_thread_distribution

struct test_executor_reset_thread_distribution
  : hpx::execution::parallel_executor
{
    test_executor_reset_thread_distribution()
      : hpx::execution::parallel_executor()
    {
    }

    template <typename Parameters>
    static void reset_thread_distribution(Parameters&&)
    {
        ++exec_count;
    }
};

namespace hpx { namespace parallel { namespace execution {
    template <>
    struct is_two_way_executor<test_executor_reset_thread_distribution>
      : std::true_type
    {
    };
}}}    // namespace hpx::parallel::execution

struct test_thread_distribution
{
    template <typename Executor>
    void reset_thread_distribution(Executor&&)
    {
        ++params_count;
    }
};

namespace hpx { namespace parallel { namespace execution {
    template <>
    struct is_executor_parameters<test_thread_distribution> : std::true_type
    {
    };
}}}    // namespace hpx::parallel::execution

///////////////////////////////////////////////////////////////////////////////
void test_reset_thread_distribution()
{
    {
        params_count = 0;
        exec_count = 0;

        hpx::parallel::execution::reset_thread_distribution(
            test_thread_distribution{}, hpx::execution::par.executor());

        HPX_TEST_EQ(params_count, std::size_t(1));
        HPX_TEST_EQ(exec_count, std::size_t(0));
    }

    {
        params_count = 0;
        exec_count = 0;

        hpx::parallel::execution::reset_thread_distribution(
            test_thread_distribution{},
            test_executor_reset_thread_distribution{});

        HPX_TEST_EQ(params_count, std::size_t(0));
        HPX_TEST_EQ(exec_count, std::size_t(1));
    }
}

///////////////////////////////////////////////////////////////////////////////
// processing_units_count

struct test_executor_processing_units_count
  : hpx::parallel::execution::parallel_executor_aggregated
{
    test_executor_processing_units_count()
      : hpx::parallel::execution::parallel_executor_aggregated()
    {
    }

    template <typename Parameters>
    static std::size_t processing_units_count(Parameters&&)
    {
        ++exec_count;
        return 1;
    }
};

namespace hpx { namespace parallel { namespace execution {
    template <>
    struct is_two_way_executor<test_executor_processing_units_count>
      : std::true_type
    {
    };
}}}    // namespace hpx::parallel::execution

struct test_processing_units
{
    template <typename Executor>
    static std::size_t processing_units_count(Executor&&)
    {
        ++params_count;
        return 1;
    }
};

namespace hpx { namespace parallel { namespace execution {
    template <>
    struct is_executor_parameters<test_processing_units> : std::true_type
    {
    };
}}}    // namespace hpx::parallel::execution

///////////////////////////////////////////////////////////////////////////////
void test_processing_units_count()
{
    {
        params_count = 0;
        exec_count = 0;

        hpx::parallel::execution::processing_units_count(
            test_processing_units{},
            hpx::parallel::execution::parallel_executor_aggregated());

        HPX_TEST_EQ(params_count, std::size_t(1));
        HPX_TEST_EQ(exec_count, std::size_t(0));
    }

    {
        params_count = 0;
        exec_count = 0;

        hpx::parallel::execution::processing_units_count(
            test_processing_units{}, test_executor_processing_units_count{});

        HPX_TEST_EQ(params_count, std::size_t(0));
        HPX_TEST_EQ(exec_count, std::size_t(1));
    }

    {
        params_count = 0;

        auto p = hpx::parallel::execution::with_processing_units_count(
            hpx::execution::par, 2);

        std::size_t num_cores =
            hpx::parallel::execution::processing_units_count(
                test_processing_units{}, p.executor());

        HPX_TEST_EQ(num_cores, std::size_t(2));
        HPX_TEST_EQ(params_count, std::size_t(0));
    }

    {
        params_count = 0;

        hpx::execution::num_cores nc(2);
        auto p = hpx::parallel::execution::with_processing_units_count(
            hpx::execution::par, nc);

        std::size_t num_cores =
            hpx::parallel::execution::processing_units_count(
                test_processing_units{}, p.executor());

        HPX_TEST_EQ(num_cores, std::size_t(2));
        HPX_TEST_EQ(params_count, std::size_t(0));
    }
}

///////////////////////////////////////////////////////////////////////////////
// mark_begin_execution, mark_end_of_scheduling, mark_end_execution

struct test_executor_begin_end : hpx::execution::parallel_executor
{
    test_executor_begin_end()
      : hpx::execution::parallel_executor()
    {
    }

    template <typename Parameters>
    void mark_begin_execution(Parameters&&)
    {
        ++exec_count;
    }

    template <typename Parameters>
    void mark_end_of_scheduling(Parameters&&)
    {
        ++exec_count;
    }

    template <typename Parameters>
    void mark_end_execution(Parameters&&)
    {
        ++exec_count;
    }
};

namespace hpx { namespace parallel { namespace execution {
    template <>
    struct is_two_way_executor<test_executor_begin_end> : std::true_type
    {
    };
}}}    // namespace hpx::parallel::execution

struct test_begin_end
{
    template <typename Executor>
    void mark_begin_execution(Executor&&)
    {
        ++params_count;
    }

    template <typename Executor>
    void mark_end_of_scheduling(Executor&&)
    {
        ++params_count;
    }

    template <typename Executor>
    void mark_end_execution(Executor&&)
    {
        ++params_count;
    }
};

namespace hpx { namespace parallel { namespace execution {
    template <>
    struct is_executor_parameters<test_begin_end> : std::true_type
    {
    };
}}}    // namespace hpx::parallel::execution

///////////////////////////////////////////////////////////////////////////////
void test_mark_begin_execution()
{
    {
        params_count = 0;
        exec_count = 0;

        hpx::parallel::execution::mark_begin_execution(
            test_begin_end{}, hpx::execution::par.executor());

        HPX_TEST_EQ(params_count, std::size_t(1));
        HPX_TEST_EQ(exec_count, std::size_t(0));
    }

    {
        params_count = 0;
        exec_count = 0;

        hpx::parallel::execution::mark_begin_execution(
            test_begin_end{}, test_executor_begin_end{});

        HPX_TEST_EQ(params_count, std::size_t(0));
        HPX_TEST_EQ(exec_count, std::size_t(1));
    }
}

void test_mark_end_of_scheduling()
{
    {
        params_count = 0;
        exec_count = 0;

        hpx::parallel::execution::mark_end_of_scheduling(
            test_begin_end{}, hpx::execution::par.executor());

        HPX_TEST_EQ(params_count, std::size_t(1));
        HPX_TEST_EQ(exec_count, std::size_t(0));
    }

    {
        params_count = 0;
        exec_count = 0;

        hpx::parallel::execution::mark_end_of_scheduling(
            test_begin_end{}, test_executor_begin_end{});

        HPX_TEST_EQ(params_count, std::size_t(0));
        HPX_TEST_EQ(exec_count, std::size_t(1));
    }
}

void test_mark_end_execution()
{
    {
        params_count = 0;
        exec_count = 0;

        hpx::parallel::execution::mark_end_execution(
            test_begin_end{}, hpx::execution::par.executor());

        HPX_TEST_EQ(params_count, std::size_t(1));
        HPX_TEST_EQ(exec_count, std::size_t(0));
    }

    {
        params_count = 0;
        exec_count = 0;

        hpx::parallel::execution::mark_end_execution(
            test_begin_end{}, test_executor_begin_end{});

        HPX_TEST_EQ(params_count, std::size_t(0));
        HPX_TEST_EQ(exec_count, std::size_t(1));
    }
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    test_get_chunk_size();
    test_maximal_number_of_chunks();
    test_reset_thread_distribution();
    test_processing_units_count();
    test_mark_begin_execution();
    test_mark_end_of_scheduling();
    test_mark_end_execution();

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    // By default this test should run on all available cores
    std::vector<std::string> const cfg = {"hpx.os_threads=all"};

    // Initialize and run HPX
    hpx::local::init_params init_args;
    init_args.cfg = cfg;

    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv, init_args), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
