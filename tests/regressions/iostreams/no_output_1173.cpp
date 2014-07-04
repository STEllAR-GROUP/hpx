//  Copyright (c) 2014 Luis Ayuso
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/foreach.hpp>

///////////////////////////////////////////////////////////////////////////////
void worker()
{
    hpx::consolestream << "hello!" << hpx::endl;
}
HPX_PLAIN_ACTION(worker, worker_action);

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    typedef hpx::future<void> wait_for_worker;
    std::vector<wait_for_worker> futures;

    // get locations and start workers
    std::string expected;
    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    BOOST_FOREACH(hpx::id_type l, localities)
    {
        futures.push_back(hpx::async(worker_action(), l));
        expected += "hello!\n";
    }
    hpx::wait_all(futures);

    std::stringstream const& console_strm = hpx::get_consolestream();

    HPX_TEST_EQ(hpx::finalize(), 0);
    HPX_TEST_EQ(console_strm.str(), expected);

    return 0;
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ_MSG(hpx::init(argc, argv), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
