//  Copyright (c) 2022-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/datapar.hpp>
#include <hpx/execution.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/program_options.hpp>

#include <type_traits>

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy>
void test_mappings_to_par(ExPolicy&& policy)
{
    using policy_t = std::decay_t<ExPolicy>;
    using mapped_policy_t =
        decltype((hpx::execution::experimental::to_par(policy)));

    static_assert(hpx::is_execution_policy_v<mapped_policy_t>,
        "hpx::is_execution_policy_v<mapped_policy_t>");

    static_assert(hpx::is_parallel_execution_policy_v<mapped_policy_t>,
        "must now be parallel");

    static_assert(hpx::is_async_execution_policy_v<policy_t> ==
            hpx::is_async_execution_policy_v<mapped_policy_t>,
        "must maintain task type");
    static_assert(hpx::is_unsequenced_execution_policy_v<policy_t> ==
            hpx::is_unsequenced_execution_policy_v<mapped_policy_t>,
        "must maintain unsequenced type");
    static_assert(hpx::is_vectorpack_execution_policy_v<policy_t> ==
            hpx::is_vectorpack_execution_policy_v<mapped_policy_t>,
        "must maintain vectorpack type");

    auto mapped_policy = hpx::execution::experimental::to_par(policy);
    (void) mapped_policy;
}

void test_mappings_to_par()
{
    using namespace hpx::execution;

    test_mappings_to_par(seq);
    test_mappings_to_par(par);
    test_mappings_to_par(unseq);
    test_mappings_to_par(par_unseq);
    test_mappings_to_par(seq(task));
    test_mappings_to_par(par(task));
    test_mappings_to_par(unseq(task));
    test_mappings_to_par(par_unseq(task));
#if defined(HPX_HAVE_DATAPAR)
    test_mappings_to_par(simd);
    test_mappings_to_par(par_simd);
    test_mappings_to_par(simd(task));
    test_mappings_to_par(par_simd(task));
#endif
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy>
void test_mappings_to_non_par(ExPolicy&& policy)
{
    using policy_t = std::decay_t<ExPolicy>;
    using mapped_policy_t =
        decltype((hpx::execution::experimental::to_non_par(policy)));

    static_assert(hpx::is_execution_policy_v<mapped_policy_t>,
        "hpx::is_execution_policy_v<mapped_policy_t>");

    static_assert(!hpx::is_parallel_execution_policy_v<mapped_policy_t>,
        "must now be non-parallel");

    static_assert(hpx::is_async_execution_policy_v<policy_t> ==
            hpx::is_async_execution_policy_v<mapped_policy_t>,
        "must maintain task type");
    static_assert(hpx::is_unsequenced_execution_policy_v<policy_t> ==
            hpx::is_unsequenced_execution_policy_v<mapped_policy_t>,
        "must maintain unsequenced type");
    static_assert(hpx::is_vectorpack_execution_policy_v<policy_t> ==
            hpx::is_vectorpack_execution_policy_v<mapped_policy_t>,
        "must maintain vectorpack type");

    auto mapped_policy = hpx::execution::experimental::to_non_par(policy);
    (void) mapped_policy;
}

void test_mappings_to_non_par()
{
    using namespace hpx::execution;

    test_mappings_to_non_par(seq);
    test_mappings_to_non_par(par);
    test_mappings_to_non_par(unseq);
    test_mappings_to_non_par(par_unseq);
    test_mappings_to_non_par(seq(task));
    test_mappings_to_non_par(par(task));
    test_mappings_to_non_par(unseq(task));
    test_mappings_to_non_par(par_unseq(task));
#if defined(HPX_HAVE_DATAPAR)
    test_mappings_to_non_par(simd);
    test_mappings_to_non_par(par_simd);
    test_mappings_to_non_par(simd(task));
    test_mappings_to_non_par(par_simd(task));
#endif
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy>
void test_mappings_to_task(ExPolicy&& policy)
{
    using policy_t = std::decay_t<ExPolicy>;
    using mapped_policy_t =
        decltype((hpx::execution::experimental::to_task(policy)));

    static_assert(hpx::is_execution_policy_v<mapped_policy_t>,
        "hpx::is_execution_policy_v<mapped_policy_t>");

    static_assert(hpx::is_async_execution_policy_v<mapped_policy_t>,
        "must now be task based");

    static_assert(hpx::is_sequenced_execution_policy_v<policy_t> ==
            hpx::is_sequenced_execution_policy_v<mapped_policy_t>,
        "must maintain sequenced type");
    static_assert(hpx::is_parallel_execution_policy_v<policy_t> ==
            hpx::is_parallel_execution_policy_v<mapped_policy_t>,
        "must maintain parallel type");
    static_assert(hpx::is_unsequenced_execution_policy_v<policy_t> ==
            hpx::is_unsequenced_execution_policy_v<mapped_policy_t>,
        "must maintain unsequenced type");
    static_assert(hpx::is_vectorpack_execution_policy_v<policy_t> ==
            hpx::is_vectorpack_execution_policy_v<mapped_policy_t>,
        "must maintain vectorpack type");

    auto mapped_policy = hpx::execution::experimental::to_task(policy);
    (void) mapped_policy;
}

void test_mappings_to_task()
{
    using namespace hpx::execution;

    test_mappings_to_task(seq);
    test_mappings_to_task(par);
    test_mappings_to_task(unseq);
    test_mappings_to_task(par_unseq);
    test_mappings_to_task(seq(task));
    test_mappings_to_task(par(task));
    test_mappings_to_task(unseq(task));
    test_mappings_to_task(par_unseq(task));
#if defined(HPX_HAVE_DATAPAR)
    test_mappings_to_task(simd);
    test_mappings_to_task(par_simd);
    test_mappings_to_task(simd(task));
    test_mappings_to_task(par_simd(task));
#endif
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy>
void test_mappings_to_non_task(ExPolicy&& policy)
{
    using policy_t = std::decay_t<ExPolicy>;
    using mapped_policy_t =
        decltype((hpx::execution::experimental::to_non_task(policy)));

    static_assert(hpx::is_execution_policy_v<mapped_policy_t>,
        "hpx::is_execution_policy_v<mapped_policy_t>");

    static_assert(!hpx::is_async_execution_policy_v<mapped_policy_t>,
        "must now be non-task based");

    static_assert(hpx::is_sequenced_execution_policy_v<policy_t> ==
            hpx::is_sequenced_execution_policy_v<mapped_policy_t>,
        "must maintain sequenced type");
    static_assert(hpx::is_parallel_execution_policy_v<policy_t> ==
            hpx::is_parallel_execution_policy_v<mapped_policy_t>,
        "must maintain parallel type");
    static_assert(hpx::is_unsequenced_execution_policy_v<policy_t> ==
            hpx::is_unsequenced_execution_policy_v<mapped_policy_t>,
        "must maintain unsequenced type");
    static_assert(hpx::is_vectorpack_execution_policy_v<policy_t> ==
            hpx::is_vectorpack_execution_policy_v<mapped_policy_t>,
        "must maintain vectorpack type");

    auto mapped_policy = hpx::execution::experimental::to_non_task(policy);
    (void) mapped_policy;
}

void test_mappings_to_non_task()
{
    using namespace hpx::execution;

    test_mappings_to_non_task(seq);
    test_mappings_to_non_task(par);
    test_mappings_to_non_task(unseq);
    test_mappings_to_non_task(par_unseq);
    test_mappings_to_non_task(seq(task));
    test_mappings_to_non_task(par(task));
    test_mappings_to_non_task(unseq(task));
    test_mappings_to_non_task(par_unseq(task));
#if defined(HPX_HAVE_DATAPAR)
    test_mappings_to_non_task(simd);
    test_mappings_to_non_task(par_simd);
    test_mappings_to_non_task(simd(task));
    test_mappings_to_non_task(par_simd(task));
#endif
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy>
void test_mappings_to_unseq(ExPolicy&& policy)
{
    using policy_t = std::decay_t<ExPolicy>;
    using mapped_policy_t =
        decltype((hpx::execution::experimental::to_unseq(policy)));

    static_assert(hpx::is_execution_policy_v<mapped_policy_t>,
        "hpx::is_execution_policy_v<mapped_policy_t>");

    static_assert(hpx::is_unsequenced_execution_policy_v<mapped_policy_t>,
        "must now be unsequenced");

    static_assert(hpx::is_sequenced_execution_policy_v<policy_t> ==
            hpx::is_sequenced_execution_policy_v<mapped_policy_t>,
        "must maintain sequenced type");
    static_assert(hpx::is_parallel_execution_policy_v<policy_t> ==
            hpx::is_parallel_execution_policy_v<mapped_policy_t>,
        "must maintain parallel type");
    static_assert(hpx::is_async_execution_policy_v<policy_t> ==
            hpx::is_async_execution_policy_v<mapped_policy_t>,
        "must maintain task-type type");
    static_assert(hpx::is_vectorpack_execution_policy_v<policy_t> ==
            hpx::is_vectorpack_execution_policy_v<mapped_policy_t>,
        "must maintain vectorpack type");

    auto mapped_policy = hpx::execution::experimental::to_unseq(policy);
    (void) mapped_policy;
}

void test_mappings_to_unseq()
{
    using namespace hpx::execution;

    test_mappings_to_unseq(seq);
    test_mappings_to_unseq(par);
    test_mappings_to_unseq(unseq);
    test_mappings_to_unseq(par_unseq);
    test_mappings_to_unseq(seq(task));
    test_mappings_to_unseq(par(task));
    test_mappings_to_unseq(unseq(task));
    test_mappings_to_unseq(par_unseq(task));
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy>
void test_mappings_to_non_unseq(ExPolicy&& policy)
{
    using policy_t = std::decay_t<ExPolicy>;
    using mapped_policy_t =
        decltype((hpx::execution::experimental::to_non_unseq(policy)));

    static_assert(hpx::is_execution_policy_v<mapped_policy_t>,
        "hpx::is_execution_policy_v<mapped_policy_t>");

    static_assert(!hpx::is_unsequenced_execution_policy_v<mapped_policy_t>,
        "must now be non-unsequenced");

    static_assert(hpx::is_sequenced_execution_policy_v<policy_t> ==
            hpx::is_sequenced_execution_policy_v<mapped_policy_t>,
        "must maintain sequenced type");
    static_assert(hpx::is_parallel_execution_policy_v<policy_t> ==
            hpx::is_parallel_execution_policy_v<mapped_policy_t>,
        "must maintain parallel type");
    static_assert(hpx::is_async_execution_policy_v<policy_t> ==
            hpx::is_async_execution_policy_v<mapped_policy_t>,
        "must maintain task-type type");
    static_assert(hpx::is_vectorpack_execution_policy_v<policy_t> ==
            hpx::is_vectorpack_execution_policy_v<mapped_policy_t>,
        "must maintain vectorpack type");

    auto mapped_policy = hpx::execution::experimental::to_non_unseq(policy);
    (void) mapped_policy;
}

void test_mappings_to_non_unseq()
{
    using namespace hpx::execution;

    test_mappings_to_non_unseq(seq);
    test_mappings_to_non_unseq(par);
    test_mappings_to_non_unseq(unseq);
    test_mappings_to_non_unseq(par_unseq);
    test_mappings_to_non_unseq(seq(task));
    test_mappings_to_non_unseq(par(task));
    test_mappings_to_non_unseq(unseq(task));
    test_mappings_to_non_unseq(par_unseq(task));
}

#if defined(HPX_HAVE_DATAPAR)
///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy>
void test_mappings_to_simd(ExPolicy&& policy)
{
    using policy_t = std::decay_t<ExPolicy>;
    using mapped_policy_t =
        decltype((hpx::execution::experimental::to_simd(policy)));

    static_assert(hpx::is_execution_policy_v<mapped_policy_t>,
        "hpx::is_execution_policy_v<mapped_policy_t>");

    static_assert(hpx::is_vectorpack_execution_policy_v<mapped_policy_t>,
        "must now be vectorpack based");

    static_assert(hpx::is_sequenced_execution_policy_v<policy_t> ==
            hpx::is_sequenced_execution_policy_v<mapped_policy_t>,
        "must maintain sequenced type");
    static_assert(hpx::is_parallel_execution_policy_v<policy_t> ==
            hpx::is_parallel_execution_policy_v<mapped_policy_t>,
        "must maintain parallel type");
    static_assert(hpx::is_async_execution_policy_v<policy_t> ==
            hpx::is_async_execution_policy_v<mapped_policy_t>,
        "must maintain task type");
    static_assert(hpx::is_unsequenced_execution_policy_v<policy_t> ==
            hpx::is_unsequenced_execution_policy_v<mapped_policy_t>,
        "must maintain unsequenced type");

    auto mapped_policy = hpx::execution::experimental::to_simd(policy);
    (void) mapped_policy;
}

void test_mappings_to_simd()
{
    using namespace hpx::execution;

    test_mappings_to_simd(seq);
    test_mappings_to_simd(par);
    test_mappings_to_simd(simd);
    test_mappings_to_simd(par_simd);
    test_mappings_to_simd(seq(task));
    test_mappings_to_simd(par(task));
    test_mappings_to_simd(simd(task));
    test_mappings_to_simd(par_simd(task));
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy>
void test_mappings_to_non_simd(ExPolicy&& policy)
{
    using policy_t = std::decay_t<ExPolicy>;
    using mapped_policy_t =
        decltype((hpx::execution::experimental::to_non_simd(policy)));

    static_assert(hpx::is_execution_policy_v<mapped_policy_t>,
        "hpx::is_execution_policy_v<mapped_policy_t>");

    static_assert(!hpx::is_vectorpack_execution_policy_v<mapped_policy_t>,
        "must now be non-simd based");

    static_assert(hpx::is_sequenced_execution_policy_v<policy_t> ==
            hpx::is_sequenced_execution_policy_v<mapped_policy_t>,
        "must maintain sequenced type");
    static_assert(hpx::is_parallel_execution_policy_v<policy_t> ==
            hpx::is_parallel_execution_policy_v<mapped_policy_t>,
        "must maintain parallel type");
    static_assert(hpx::is_async_execution_policy_v<policy_t> ==
            hpx::is_async_execution_policy_v<mapped_policy_t>,
        "must maintain task type");
    static_assert(hpx::is_unsequenced_execution_policy_v<policy_t> ==
            hpx::is_unsequenced_execution_policy_v<mapped_policy_t>,
        "must maintain unsequenced type");

    auto mapped_policy = hpx::execution::experimental::to_non_simd(policy);
    (void) mapped_policy;
}

void test_mappings_to_non_simd()
{
    using namespace hpx::execution;

    test_mappings_to_non_simd(seq);
    test_mappings_to_non_simd(par);
    test_mappings_to_non_simd(simd);
    test_mappings_to_non_simd(par_simd);
    test_mappings_to_non_simd(seq(task));
    test_mappings_to_non_simd(par(task));
    test_mappings_to_non_simd(simd(task));
    test_mappings_to_non_simd(par_simd(task));
}
#endif

///////////////////////////////////////////////////////////////////////////////
int hpx_main(hpx::program_options::variables_map&)
{
    test_mappings_to_par();
    test_mappings_to_non_par();

    test_mappings_to_task();
    test_mappings_to_non_task();

    test_mappings_to_unseq();
    test_mappings_to_non_unseq();

#if defined(HPX_HAVE_DATAPAR)
    test_mappings_to_simd();
    test_mappings_to_non_simd();
#endif

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
