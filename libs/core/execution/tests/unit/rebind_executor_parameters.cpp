//  Copyright (c) 2007-2026 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/execution.hpp>
#include <hpx/future.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/tag_invoke.hpp>
#include <hpx/modules/testing.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "foreach_tests.hpp"

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy>
void parameters_test(ExPolicy&& policy)
{
    static_assert(hpx::is_execution_policy_v<ExPolicy>,
        "hpx::is_execution_policy_v<Parameters>");

    using iterator_tag = std::random_access_iterator_tag;
    test_for_each(policy, iterator_tag());

    //test_for_each_async(policy(hpx::execution::task), iterator_tag());

    //hpx::execution::sequenced_executor seq_exec;
    //test_for_each(to_non_par(policy).on(seq_exec), iterator_tag());
    //test_for_each_async(
    //    to_non_par(policy)(hpx::execution::task).on(seq_exec), iterator_tag());

    //hpx::execution::parallel_executor par_exec;
    //test_for_each(to_par(policy).on(par_exec), iterator_tag());
    //test_for_each_async(
    //    to_par(policy)(hpx::execution::task).on(par_exec), iterator_tag());
}

////////////////////////////////////////////////////////////////////////////////
// test parameters object with get_chunk_size
struct test_replaced_get_chunk_size
{
    explicit test_replaced_get_chunk_size(std::atomic<bool>& invoked) noexcept
      : invoked(&invoked)
    {
    }

    template <typename Executor>
    friend std::size_t tag_override_invoke(
        hpx::execution::experimental::get_chunk_size_t,
        test_replaced_get_chunk_size& self, Executor&&,
        hpx::chrono::steady_duration const&, std::size_t, std::size_t) noexcept
    {
        *self.invoked = true;
        return 0;
    }

    std::atomic<bool>* invoked;
};

struct test_wrapping_replaced_get_chunk_size
{
    explicit test_wrapping_replaced_get_chunk_size(
        std::atomic<bool>& invoked) noexcept
      : invoked(&invoked)
    {
    }

    template <typename InnerParams, typename Executor>
    friend std::size_t tag_override_invoke(
        hpx::execution::experimental::get_chunk_size_t,
        test_wrapping_replaced_get_chunk_size& self, InnerParams&& inner,
        Executor&& exec, hpx::chrono::steady_duration const& duration,
        std::size_t cores, std::size_t num_tasks) noexcept
    {
        std::size_t result = hpx::execution::experimental::get_chunk_size(
            HPX_FORWARD(InnerParams, inner), HPX_FORWARD(Executor, exec),
            duration, cores, num_tasks);

        *self.invoked = true;
        return result;
    }

    std::atomic<bool>* invoked;
};

namespace hpx::execution::experimental {

    template <>
    struct is_executor_parameters<test_replaced_get_chunk_size> : std::true_type
    {
    };

    template <>
    struct is_executor_parameters<test_wrapping_replaced_get_chunk_size>
      : std::true_type
    {
    };
}    // namespace hpx::execution::experimental

void replace_chunk_size()
{
    using namespace hpx::execution;
    using namespace hpx::execution::experimental;

    // replace chunk size with another parameters object exposing get_chunk_size
    // as well
    {
        std::atomic<bool> invoked_replaced(false);

        auto params =
            join_executor_parameters(experimental::static_chunk_size());
        auto rebound_params = rebind_executor_parameters(
            params, test_replaced_get_chunk_size(invoked_replaced));
        auto policy = create_rebound_policy(par, rebound_params);
        parameters_test(policy);

        HPX_TEST(invoked_replaced);
    }

    // replace a parameters object that doesn't expose get_chunk_size with
    // a parameters object that does expose it
    {
        std::atomic<bool> invoked_replaced(false);

        auto params = join_executor_parameters(experimental::max_num_chunks());
        auto rebound_params = rebind_executor_parameters(
            params, test_replaced_get_chunk_size(invoked_replaced));
        auto policy = create_rebound_policy(par, rebound_params);
        parameters_test(policy);

        HPX_TEST(invoked_replaced);
    }

    // replace chunk size with another parameters object not exposing it
    {
        std::atomic<bool> invoked_replaced(false);

        auto params = join_executor_parameters(
            test_replaced_get_chunk_size(invoked_replaced));
        auto rebound_params =
            rebind_executor_parameters(params, experimental::num_cores(4));
        auto policy = create_rebound_policy(par, rebound_params);
        parameters_test(policy);

        HPX_TEST(invoked_replaced);
    }

    // replace combined parameters object that exposes get_chunk_size with
    // another parameters object that exposes it
    {
        std::atomic<bool> invoked_replaced(false);

        auto params = join_executor_parameters(
            experimental::static_chunk_size(), experimental::num_cores(4));
        auto rebound_params = rebind_executor_parameters(
            params, test_replaced_get_chunk_size(invoked_replaced));
        auto policy = create_rebound_policy(par, rebound_params);
        parameters_test(policy);

        HPX_TEST(invoked_replaced);
    }

    // test wrapped get_chunk_size
    {
        std::atomic<bool> invoked_replaced(false);
        std::atomic<bool> invoked_inner_replaced(false);

        auto params = join_executor_parameters(
            test_replaced_get_chunk_size(invoked_inner_replaced));
        auto rebound_params = rebind_executor_parameters(
            params, test_wrapping_replaced_get_chunk_size(invoked_replaced));
        auto policy = create_rebound_policy(par, rebound_params);
        parameters_test(policy);

        HPX_TEST(invoked_replaced);
        HPX_TEST(invoked_inner_replaced);
    }
}

////////////////////////////////////////////////////////////////////////////////
// test parameters object with measure_iteration
struct base_measure_iteration
{
    using invokes_testing_function = void;

    template <typename Executor, typename F>
    friend hpx::chrono::steady_duration tag_override_invoke(
        hpx::execution::experimental::measure_iteration_t,
        base_measure_iteration, Executor&&, F&&, std::size_t) noexcept
    {
        return hpx::chrono::null_duration;
    }
};

struct test_replaced_measure_iteration
{
    using invokes_testing_function = void;

    explicit test_replaced_measure_iteration(
        std::atomic<bool>& invoked) noexcept
      : invoked(&invoked)
    {
    }

    template <typename Executor, typename F>
    friend hpx::chrono::steady_duration tag_override_invoke(
        hpx::execution::experimental::measure_iteration_t,
        test_replaced_measure_iteration& self, Executor&&, F&&,
        std::size_t) noexcept
    {
        *self.invoked = true;
        return hpx::chrono::null_duration;
    }

    std::atomic<bool>* invoked;
};

namespace hpx::execution::experimental {

    template <>
    struct is_executor_parameters<base_measure_iteration> : std::true_type
    {
    };

    template <>
    struct is_executor_parameters<test_replaced_measure_iteration>
      : std::true_type
    {
    };
}    // namespace hpx::execution::experimental

void replace_measure_iteration()
{
    using namespace hpx::execution;
    using namespace hpx::execution::experimental;

    // replace measure_iteration with another parameters object exposing
    // measure_iteration as well
    {
        std::atomic<bool> invoked_replaced(false);

        auto params = join_executor_parameters(base_measure_iteration());
        auto bound_params = rebind_executor_parameters(
            params, test_replaced_measure_iteration(invoked_replaced));
        static_assert(
            extract_invokes_testing_function_v<decltype(bound_params)>,
            "extract_invokes_testing_function_v<decltype(bound_params)>");
        auto policy = create_rebound_policy(par, bound_params);
        parameters_test(policy);

        HPX_TEST(invoked_replaced);
    }

    // replace a parameters object that doesn't expose measure_iteration with a
    // parameters object that does expose it
    {
        std::atomic<bool> invoked_replaced(false);

        auto params = join_executor_parameters(experimental::max_num_chunks());
        auto bound_params = rebind_executor_parameters(
            params, test_replaced_measure_iteration(invoked_replaced));
        static_assert(
            extract_invokes_testing_function_v<decltype(bound_params)>,
            "extract_invokes_testing_function_v<decltype(bound_params)>");
        auto policy = create_rebound_policy(par, bound_params);
        parameters_test(policy);

        HPX_TEST(invoked_replaced);
    }

    // replace measure_iteration with another parameters object not exposing it
    {
        std::atomic<bool> invoked_replaced(false);

        auto params = join_executor_parameters(
            test_replaced_measure_iteration(invoked_replaced));
        auto bound_params =
            rebind_executor_parameters(params, experimental::num_cores(4));
        static_assert(
            extract_invokes_testing_function_v<decltype(bound_params)>,
            "extract_invokes_testing_function_v<decltype(bound_params)>");
        auto policy = create_rebound_policy(par, bound_params);
        parameters_test(policy);

        HPX_TEST(invoked_replaced);
    }

    // replace combined parameters object that exposes measure_iteration with
    // another parameters object that exposes it
    {
        std::atomic<bool> invoked_replaced(false);

        auto params = join_executor_parameters(
            base_measure_iteration(), experimental::num_cores(4));
        auto bound_params = rebind_executor_parameters(
            params, test_replaced_measure_iteration(invoked_replaced));
        static_assert(
            extract_invokes_testing_function_v<decltype(bound_params)>,
            "extract_invokes_testing_function_v<decltype(bound_params)>");
        auto policy = create_rebound_policy(par, bound_params);
        parameters_test(policy);

        HPX_TEST(invoked_replaced);
    }
}

////////////////////////////////////////////////////////////////////////////////
// test parameters object with maximal_number_of_chunks
struct test_replaced_maximal_number_of_chunks
{
    explicit test_replaced_maximal_number_of_chunks(
        std::atomic<bool>& invoked) noexcept
      : invoked(&invoked)
    {
    }

    template <typename Executor>
    friend std::size_t tag_override_invoke(
        hpx::execution::experimental::maximal_number_of_chunks_t,
        test_replaced_maximal_number_of_chunks& self, Executor&&, std::size_t,
        std::size_t) noexcept
    {
        *self.invoked = true;
        return 0;
    }

    std::atomic<bool>* invoked;
};

namespace hpx::execution::experimental {

    template <>
    struct is_executor_parameters<test_replaced_maximal_number_of_chunks>
      : std::true_type
    {
    };
}    // namespace hpx::execution::experimental

void replace_maximal_number_of_chunks()
{
    using namespace hpx::execution;
    using namespace hpx::execution::experimental;

    // replace maximal_number_of_chunks with another parameters object exposing
    // maximal_number_of_chunks as well
    {
        std::atomic<bool> invoked_replaced(false);

        auto params = join_executor_parameters(experimental::max_num_chunks());
        auto bound_params = rebind_executor_parameters(
            params, test_replaced_maximal_number_of_chunks(invoked_replaced));
        auto policy = create_rebound_policy(par, bound_params);
        parameters_test(policy);

        HPX_TEST(invoked_replaced);
    }

    // replace a parameters object that doesn't expose maximal_number_of_chunks
    // with a parameters object that does expose it
    {
        std::atomic<bool> invoked_replaced(false);

        auto params =
            join_executor_parameters(experimental::static_chunk_size());
        auto bound_params = rebind_executor_parameters(
            params, test_replaced_maximal_number_of_chunks(invoked_replaced));
        auto policy = create_rebound_policy(par, bound_params);
        parameters_test(policy);

        HPX_TEST(invoked_replaced);
    }

    // replace maximal_number_of_chunks with another parameters object not
    // exposing it
    {
        std::atomic<bool> invoked_replaced(false);

        auto params = join_executor_parameters(
            test_replaced_maximal_number_of_chunks(invoked_replaced));
        auto bound_params =
            rebind_executor_parameters(params, experimental::num_cores(4));
        auto policy = create_rebound_policy(par, bound_params);
        parameters_test(policy);

        HPX_TEST(invoked_replaced);
    }

    // replace combined parameters object that exposes maximal_number_of_chunks
    // with another parameters object that exposes it
    {
        std::atomic<bool> invoked_replaced(false);

        auto params = join_executor_parameters(
            experimental::max_num_chunks(), experimental::num_cores(4));
        auto bound_params = rebind_executor_parameters(
            params, test_replaced_maximal_number_of_chunks(invoked_replaced));
        auto policy = create_rebound_policy(par, bound_params);
        parameters_test(policy);

        HPX_TEST(invoked_replaced);
    }
}

///////////////////////////////////////////////////////////////////////////////
// test parameters object with mark_begin_execution, mark_end_of_scheduling, and
// mark_end_execution
struct base_execution_markers
{
    template <typename Executor>
    friend constexpr void tag_override_invoke(
        hpx::execution::experimental::mark_begin_execution_t,
        base_execution_markers, Executor&&) noexcept
    {
    }

    template <typename Executor>
    friend constexpr void tag_override_invoke(
        hpx::execution::experimental::mark_end_of_scheduling_t,
        base_execution_markers, Executor&&) noexcept
    {
    }

    template <typename Executor>
    friend constexpr void tag_override_invoke(
        hpx::execution::experimental::mark_end_execution_t,
        base_execution_markers, Executor&&) noexcept
    {
    }
};

struct test_replaced_execution_markers
{
    explicit test_replaced_execution_markers(std::atomic<bool>& invoked_begin,
        std::atomic<bool>& invoked_end,
        std::atomic<bool>& invoked_end_execution) noexcept
      : invoked_begin(&invoked_begin)
      , invoked_end(&invoked_end)
      , invoked_end_execution(&invoked_end_execution)
    {
    }

    template <typename Executor>
    friend void tag_override_invoke(
        hpx::execution::experimental::mark_begin_execution_t,
        test_replaced_execution_markers self, Executor&&) noexcept
    {
        *self.invoked_begin = true;
    }

    template <typename Executor>
    friend void tag_override_invoke(
        hpx::execution::experimental::mark_end_of_scheduling_t,
        test_replaced_execution_markers self, Executor&&) noexcept
    {
        *self.invoked_end = true;
    }

    template <typename Executor>
    friend void tag_override_invoke(
        hpx::execution::experimental::mark_end_execution_t,
        test_replaced_execution_markers self, Executor&&) noexcept
    {
        *self.invoked_end_execution = true;
    }

    std::atomic<bool>* invoked_begin;
    std::atomic<bool>* invoked_end;
    std::atomic<bool>* invoked_end_execution;
};

namespace hpx::execution::experimental {

    template <>
    struct is_executor_parameters<base_execution_markers> : std::true_type
    {
    };

    template <>
    struct is_executor_parameters<test_replaced_execution_markers>
      : std::true_type
    {
    };
}    // namespace hpx::execution::experimental

void replace_execution_markers()
{
    using namespace hpx::execution;
    using namespace hpx::execution::experimental;

    // replace mark_begin_execution, mark_end_of_scheduling, and
    // mark_end_execution with another parameters object exposing them as well
    {
        std::atomic<bool> invoked_begin(false);
        std::atomic<bool> invoked_end(false);
        std::atomic<bool> invoked_end_execution(false);

        auto params = join_executor_parameters(base_execution_markers());
        auto bound_params = rebind_executor_parameters(params,
            test_replaced_execution_markers(
                invoked_begin, invoked_end, invoked_end_execution));
        auto policy = create_rebound_policy(par, bound_params);
        parameters_test(policy);

        HPX_TEST(invoked_begin);
        HPX_TEST(invoked_end);
        HPX_TEST(invoked_end_execution);
    }

    // replace a parameters object that doesn't expose mark_begin_execution,
    // mark_end_of_scheduling, and mark_end_execution with a parameters object
    // that does expose them
    {
        std::atomic<bool> invoked_begin(false);
        std::atomic<bool> invoked_end(false);
        std::atomic<bool> invoked_end_execution(false);

        auto params = join_executor_parameters(experimental::max_num_chunks());
        auto bound_params = rebind_executor_parameters(params,
            test_replaced_execution_markers(
                invoked_begin, invoked_end, invoked_end_execution));
        auto policy = create_rebound_policy(par, bound_params);
        parameters_test(policy);

        HPX_TEST(invoked_begin);
        HPX_TEST(invoked_end);
        HPX_TEST(invoked_end_execution);
    }

    // replace mark_begin_execution, mark_end_of_scheduling, and
    // mark_end_execution with another parameters object not exposing them
    {
        std::atomic<bool> invoked_begin(false);
        std::atomic<bool> invoked_end(false);
        std::atomic<bool> invoked_end_execution(false);

        auto params = join_executor_parameters(test_replaced_execution_markers(
            invoked_begin, invoked_end, invoked_end_execution));
        auto bound_params =
            rebind_executor_parameters(params, experimental::num_cores(4));
        auto policy = create_rebound_policy(par, bound_params);
        parameters_test(policy);

        HPX_TEST(invoked_begin);
        HPX_TEST(invoked_end);
        HPX_TEST(invoked_end_execution);
    }

    // replace combined parameters object that exposes mark_begin_execution,
    // mark_end_of_scheduling, and mark_end_execution with another parameters
    // object that exposes them
    {
        std::atomic<bool> invoked_begin(false);
        std::atomic<bool> invoked_end(false);
        std::atomic<bool> invoked_end_execution(false);

        auto params = join_executor_parameters(
            base_execution_markers(), experimental::num_cores(4));
        auto bound_params = rebind_executor_parameters(params,
            test_replaced_execution_markers(
                invoked_begin, invoked_end, invoked_end_execution));
        auto policy = create_rebound_policy(par, bound_params);
        parameters_test(policy);

        HPX_TEST(invoked_begin);
        HPX_TEST(invoked_end);
        HPX_TEST(invoked_end_execution);
    }
}

///////////////////////////////////////////////////////////////////////////////
// test parameters object with processing_units_count
struct base_processing_units_count
{
    template <typename Executor>
    friend constexpr std::size_t tag_override_invoke(
        hpx::execution::experimental::processing_units_count_t,
        base_processing_units_count const&, Executor&&,
        hpx::chrono::steady_duration const&, std::size_t) noexcept
    {
        return 1;
    }
};

struct test_replaced_processing_units_count
{
    explicit test_replaced_processing_units_count(
        std::atomic<bool>& invoked) noexcept
      : invoked(&invoked)
    {
    }

    template <typename Executor>
    friend std::size_t tag_override_invoke(
        hpx::execution::experimental::maximal_number_of_chunks_t,
        test_replaced_processing_units_count& self, Executor&&, std::size_t,
        std::size_t) noexcept
    {
        *self.invoked = true;
        return 1;
    }

    std::atomic<bool>* invoked;
};

namespace hpx::execution::experimental {

    template <>
    struct is_executor_parameters<base_processing_units_count> : std::true_type
    {
    };

    template <>
    struct is_executor_parameters<test_replaced_processing_units_count>
      : std::true_type
    {
    };
}    // namespace hpx::execution::experimental

void replace_processing_units_count()
{
    using namespace hpx::execution;
    using namespace hpx::execution::experimental;

    // replace processing_units_count with another parameters object exposing
    // processing_units_count as well
    {
        std::atomic<bool> invoked_replaced(false);

        auto params = join_executor_parameters(base_processing_units_count());
        auto bound_params = rebind_executor_parameters(
            params, test_replaced_processing_units_count(invoked_replaced));
        auto policy = create_rebound_policy(par, bound_params);
        parameters_test(policy);

        HPX_TEST(invoked_replaced);
    }

    // replace a parameters object that doesn't expose processing_units_count
    // with a parameters object that does expose it
    {
        std::atomic<bool> invoked_replaced(false);

        auto params =
            join_executor_parameters(experimental::static_chunk_size());
        auto bound_params = rebind_executor_parameters(
            params, test_replaced_processing_units_count(invoked_replaced));
        auto policy = create_rebound_policy(par, bound_params);
        parameters_test(policy);

        HPX_TEST(invoked_replaced);
    }

    // replace processing_units_count with another parameters object not
    // exposing it
    {
        std::atomic<bool> invoked_replaced(false);

        auto params = join_executor_parameters(
            test_replaced_processing_units_count(invoked_replaced));
        auto bound_params = rebind_executor_parameters(
            params, experimental::static_chunk_size());
        auto policy = create_rebound_policy(par, bound_params);
        parameters_test(policy);

        HPX_TEST(invoked_replaced);
    }

    // replace combined parameters object that exposes processing_units_count
    // with another parameters object that exposes it
    {
        std::atomic<bool> invoked_replaced(false);
        auto params = join_executor_parameters(
            base_processing_units_count(), experimental::static_chunk_size());
        auto bound_params = rebind_executor_parameters(
            params, test_replaced_processing_units_count(invoked_replaced));
        auto policy = create_rebound_policy(par, bound_params);
        parameters_test(policy);

        HPX_TEST(invoked_replaced);
    }
}

#if 0
///////////////////////////////////////////////////////////////////////////////
// test parameters object with reset_thread_distribution
void replace_reset_thread_distribution()
{
    using namespace hpx::execution;
    using namespace hpx::execution::experimental;

    // replace maximal_number_of_chunks with another parameters object exposing
    // reset_thread_distribution as well
    {
        std::atomic<bool> invoked_replaced(false);

        auto params = join_executor_parameters(experimental::static_chunk_size());
        auto bound_params = rebind_executor_parameters(
            params, test_replaced_reset_thread_distribution(invoked_replaced));
        auto policy = create_rebound_policy(par, bound_params);
        parameters_test(policy);

        HPX_TEST(invoked_replaced);
    }

    // replace a parameters object that doesn't expose reset_thread_distribution
    // with a parameters object that does expose it
    {
        std::atomic<bool> invoked_replaced(false);

        auto params = join_executor_parameters(experimental::max_num_chunks());
        auto bound_params = rebind_executor_parameters(
            params, test_replaced_reset_thread_distribution(invoked_replaced));
        auto policy = create_rebound_policy(par, bound_params);
        parameters_test(policy);

        HPX_TEST(invoked_replaced);
    }

    // replace reset_thread_distribution with another parameters object not
    // exposing it
    {
        std::atomic<bool> invoked_replaced(false);

        auto params = join_executor_parameters(
            test_replaced_reset_thread_distribution(invoked_replaced));
        auto bound_params =
            rebind_executor_parameters(params, experimental::num_cores(4));
        auto policy = create_rebound_policy(par, bound_params);
        parameters_test(policy);

        HPX_TEST(invoked_replaced);
    }

    // replace combined parameters object that exposes reset_thread_distribution
    // with another parameters object that exposes it
    {
        std::atomic<bool> invoked_replaced(false);
        auto params = join_executor_parameters(reset_thread_distribution());
        auto bound_params = rebind_executor_parameters(
            params, test_replaced_reset_thread_distribution(invoked_replaced));
        auto policy = create_rebound_policy(par, bound_params);
        parameters_test(policy);

        HPX_TEST(invoked_replaced);
    }
}

///////////////////////////////////////////////////////////////////////////////
// test parameters object with collect_execution_parameters
void replace_collect_execution_parameters()
{
    using namespace hpx::execution;
    using namespace hpx::execution::experimental;

    // replace maximal_number_of_chunks with another parameters object exposing
    // collect_execution_parameters as well
    {
        std::atomic<bool> invoked_replaced(false);

        auto params = join_executor_parameters(experimental::static_chunk_size());
        auto bound_params = rebind_executor_parameters(params,
            test_replaced_collect_execution_parameters(invoked_replaced));
        auto policy = create_rebound_policy(par, bound_params);
        parameters_test(policy);

        HPX_TEST(invoked_replaced);
    }

    // replace a parameters object that doesn't expose
    // collect_execution_parameters with a parameters object that does expose it
    {
        std::atomic<bool> invoked_replaced(false);

        auto params = join_executor_parameters(experimental::max_num_chunks());
        auto bound_params = rebind_executor_parameters(params,
            test_replaced_collect_execution_parameters(invoked_replaced));
        auto policy = create_rebound_policy(par, bound_params);
        parameters_test(policy);

        HPX_TEST(invoked_replaced);
    }

    // replace collect_execution_parameters with another parameters object not
    // exposing it
    {
        std::atomic<bool> invoked_replaced(false);

        auto params = join_executor_parameters(
            test_replaced_collect_execution_parameters(invoked_replaced));
        auto bound_params =
            rebind_executor_parameters(params, experimental::num_cores(4));
        auto policy = create_rebound_policy(par, bound_params);
        parameters_test(policy);

        HPX_TEST(invoked_replaced);
    }

    // replace combined parameters object that exposes
    // collect_execution_parameters with another parameters object that exposes
    // it
    {
        std::atomic<bool> invoked_replaced(false);
        auto params = join_executor_parameters(collect_execution_parameters());
        auto bound_params = rebind_executor_parameters(params,
            test_replaced_collect_execution_parameters(invoked_replaced));
        auto policy = create_rebound_policy(par, bound_params);
        parameters_test(policy);

        HPX_TEST(invoked_replaced);
    }
}
#endif

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    replace_chunk_size();
    replace_measure_iteration();
    replace_maximal_number_of_chunks();
    replace_execution_markers();
    replace_processing_units_count();
    //test_replace_reset_thread_distribution();
    //test_replace_collect_execution_parameters();

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    // By default, this test should run on all available cores
    std::vector<std::string> const cfg = {"hpx.os_threads=all"};

    // Initialize and run HPX
    hpx::local::init_params init_args;
    init_args.cfg = cfg;

    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv, init_args), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
