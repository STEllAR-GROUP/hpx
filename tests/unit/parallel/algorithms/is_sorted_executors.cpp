//  Copyright (c) 2015 Daniel Bourgeois
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>

#include <string>
#include <utility>
#include <vector>

#include "is_sorted_tests.hpp"

////////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy>
void test_executors(ExPolicy && policy)
{
    typedef std::random_access_iterator_tag iterator_tag;

    test_sorted2(policy, iterator_tag());
    test_sorted3(policy, iterator_tag());
    test_sorted_exception(policy, iterator_tag());
    test_sorted_bad_alloc(policy, iterator_tag());
    test_sorted1(std::forward<ExPolicy>(policy), iterator_tag());
}

template <typename ExPolicy>
void test_executors_async(ExPolicy && p)
{
    typedef std::random_access_iterator_tag iterator_tag;

    test_sorted2_async(p, iterator_tag());
    test_sorted3_async(p, iterator_tag());
    test_sorted_exception_async(p, iterator_tag());
    test_sorted_bad_alloc_async(p, iterator_tag());
    test_sorted1_async(std::forward<ExPolicy>(p), iterator_tag());
}

void sorted_executors_test()
{
    using namespace hpx::parallel;

    {
        parallel_executor exec;

        test_executors(par.on(exec));
        test_executors_async(par(task).on(exec));
    }

    {
        sequential_executor exec;

        test_executors(seq.on(exec));
        test_executors_async(seq(task).on(exec));

        test_executors(par.on(exec));
        test_executors_async(par(task).on(exec));
    }
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    sorted_executors_test();
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    using namespace boost::program_options;
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    std::vector<std::string> const cfg = {
        "hpx.os_threads=all"
    };

    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
