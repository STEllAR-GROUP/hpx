// Copyright (c) 2014 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/util/lightweight_test.hpp>

///////////////////////////////////////////////////////////////////////////////
void print_stuff()
{
    std::string expected;
    std::stringstream const& console_strm = hpx::get_consolestream();

    char const* const s1 = "Adding workloads to queue ...\n";
    char const* const s2 = "\tAdding workload 10:10 ...\n";

    hpx::this_thread::sleep_for(boost::posix_time::milliseconds(1000));

    hpx::consolestream << "Adding workloads to queue ..." << hpx::endl;
    expected += s1;
    HPX_TEST_EQ(expected, console_strm.str());

    hpx::consolestream << "\tAdding workload " << 10 << ":" << 10 << " ..." << hpx::endl;
    expected += s2;
    HPX_TEST_EQ(expected, console_strm.str());

    hpx::consolestream << "Adding workloads to queue ..." << hpx::endl;
    expected += s1;
    HPX_TEST_EQ(expected, console_strm.str());

    hpx::consolestream << "\tAdding workload " << 10 << ":" << 10 << " ..." << hpx::endl;
    expected += s2;
    HPX_TEST_EQ(expected, console_strm.str());

    hpx::consolestream << "Adding workloads to queue ..." << hpx::endl;
    expected += s1;
    HPX_TEST_EQ(expected, console_strm.str());

    hpx::consolestream << "\tAdding workload " << 10 << ":" << 10 << " ..." << hpx::endl;
    expected += s2;
    HPX_TEST_EQ(expected, console_strm.str());

    hpx::consolestream << "Adding workloads to queue ..." << hpx::endl;
    expected += s1;
    HPX_TEST_EQ(expected, console_strm.str());

    hpx::consolestream << "\tAdding workload " << 10 << ":" << 10 << " ..." << hpx::endl;
    expected += s2;
    HPX_TEST_EQ(expected, console_strm.str());

    hpx::this_thread::sleep_for(boost::posix_time::milliseconds(1000));

    hpx::consolestream << "\tAdding workload " << 10 << ":" << 10 << " ..." << hpx::endl;
    expected += s2;
    HPX_TEST_EQ(expected, console_strm.str());
}

int hpx_main(boost::program_options::variables_map & vm)
{
    print_stuff();
    return hpx::finalize();
}

//////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    HPX_TEST_EQ_MSG(hpx::init(argc, argv), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
