//  Copyright (c) 2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_for_each.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <algorithm>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    std::vector<int> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand());

    hpx::parallel::for_each(
        hpx::parallel::execution::par, c.begin(), c.end(),
        hpx::util::annotated_function(
            [](int i) -> void
            {
                hpx::util::thread_description desc(
                    hpx::threads::get_thread_description(hpx::threads::get_self_id())
                );
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
                HPX_TEST_EQ(std::string(desc.get_description()),
                        "annotated_function");
#else
                HPX_TEST_EQ(std::string(desc.get_description()),
                        "<unknown>");
#endif
            },
            "annotated_function"
        )
    );

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    std::vector<std::string> const cfg = {
        "hpx.os_threads=4"
    };

    HPX_TEST_EQ_MSG(hpx::init(argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
