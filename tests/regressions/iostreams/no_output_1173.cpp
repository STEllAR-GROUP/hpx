//  Copyright (c) 2014 Luis Ayuso
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/util/lightweight_test.hpp>

///////////////////////////////////////////////////////////////////////////////
bool on_shutdown_executed = false;
boost::uint32_t locality_id = boost::uint32_t(-1);

void worker()
{
    locality_id = hpx::get_locality_id();
    hpx::consolestream << "hello!" << hpx::endl;
}
HPX_PLAIN_ACTION(worker, worker_action);

///////////////////////////////////////////////////////////////////////////////
void on_shutdown(std::string const& expected)
{
    std::stringstream const& console_strm = hpx::get_consolestream();
    HPX_TEST_EQ(console_strm.str(), expected);
    on_shutdown_executed = true;
}

int hpx_main(boost::program_options::variables_map& vm)
{
    typedef hpx::future<void> wait_for_worker;
    std::vector<wait_for_worker> futures;

    // get locations and start workers
    std::string expected;

    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    for (hpx::id_type const& l : localities)
    {
        futures.push_back(hpx::async(worker_action(), l));
        expected += "hello!\n";
    }

    hpx::register_shutdown_function(hpx::util::bind(&on_shutdown, expected));
    hpx::wait_all(futures);

    HPX_TEST_EQ(hpx::finalize(), 0);

    return 0;
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ_MSG(hpx::init(argc, argv), 0,
        "HPX main exited with non-zero status");

    HPX_TEST_NEQ(boost::uint32_t(-1), locality_id);
    HPX_TEST(on_shutdown_executed || 0 != locality_id);

    return hpx::util::report_errors();
}
