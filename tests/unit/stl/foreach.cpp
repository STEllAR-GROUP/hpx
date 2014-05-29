//  Copyright (c) 2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/algorithm.hpp>
#include <hpx/util/lightweight_test.hpp>

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy>
void test_foreach(ExPolicy const& policy)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

    std::vector<std::size_t> c(10000);
    hpx::parallel::for_each(policy, boost::begin(c), boost::end(c),
        [](std::size_t& v) {
            v = 42;
        });

    // verify values
    std::for_each(boost::begin(c), boost::end(c),
        [](std::size_t v) {
            HPX_TEST_EQ(v, std::size_t(42));
        });
}

void test_foreach(hpx::parallel::task_execution_policy const& policy)
{
    std::vector<std::size_t> c(10000);

    hpx::future<void> f =
        hpx::parallel::for_each(policy, boost::begin(c), boost::end(c),
            [](std::size_t& v) {
                v = 42;
            });
    f.wait();

    // verify values
    std::for_each(boost::begin(c), boost::end(c),
        [](std::size_t v) {
            HPX_TEST_EQ(v, std::size_t(42));
        });
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    test_foreach(hpx::parallel::seq);
    test_foreach(hpx::parallel::par);
    test_foreach(hpx::parallel::vec);
    test_foreach(hpx::parallel::task);

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(argc, argv), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}


