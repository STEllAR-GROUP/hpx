//  Copyright (c) 2016 John Biddiscombe
//  Copyright (c) 2016 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_scan.hpp>
#include <hpx/util/lightweight_test.hpp>

////////////////////////////////////////////////////////////////////////////////
void test_scan_non_commutative() {
    std::vector<std::string> const vs = {
        "a", "b", "c", "d", "e",
        "f", "g", "h", "i", "j",
        "k", "l", "m", "n", "o",
        "p", "q", "r", "s", "t",
        "u", "v", "w", "x", "y", "z" };

    for (unsigned int i = 0; i < vs.size(); ++i) {
        std::vector<std::string> rs(vs.size());
        hpx::parallel::inclusive_scan(
            hpx::parallel::par.with(hpx::parallel::static_chunk_size(i)),
            vs.cbegin(), vs.cend(),
            rs.begin()
        );
        std::cout << rs.back() << "\n";
        bool is_equal = (std::string("abcdefghijklmnopqrstuvwxyz") == rs.back());
        HPX_TEST(is_equal);
    }
    //
    for (unsigned int i = 0; i < vs.size(); ++i) {
        std::vector<std::string> rs(vs.size());
        hpx::parallel::exclusive_scan(
            hpx::parallel::par.with(hpx::parallel::static_chunk_size(i)),
            vs.cbegin(), vs.cend(),
            rs.begin(), std::string("0")
        );
        std::cout << rs.back() << "\n";
        bool is_equal = (std::string("0abcdefghijklmnopqrstuvwxy")==rs.back());
        HPX_TEST(is_equal);

    }
}

////////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    test_scan_non_commutative();

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // add command line option which controls the random number generator seed
    using namespace boost::program_options;
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    // By default this test should run on all available cores
    std::vector<std::string> cfg;
    cfg.push_back("hpx.os_threads=" +
        std::to_string(hpx::threads::hardware_concurrency()));

    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
