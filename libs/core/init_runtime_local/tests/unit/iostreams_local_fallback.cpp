//  Copyright (c) 2026 Abhishek Bansal
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/init.hpp>
#include <hpx/iostream.hpp>
#include <hpx/modules/format.hpp>
#include <hpx/modules/testing.hpp>

#include <iostream>
#include <map>
#include <ostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

    constexpr int num_threads = 4;
    constexpr int iterations = 32;
    constexpr int fixed_console_lines = 5;

    std::string make_line(int tid, int iteration)
    {
        return "thread " + std::to_string(tid) + " line " +
            std::to_string(iteration);
    }

    void test_low_level_iostream_module()
    {
        std::vector<char> dest;

        hpx::iostream::filtering_ostream out;
        out.push(hpx::iostream::back_inserter(dest));
        out << "low-level iostream module";
        out.flush();

        std::string const output(dest.begin(), dest.end());
        HPX_TEST_EQ(output, "low-level iostream module");
    }

    struct stream_redirect
    {
        explicit stream_redirect(std::ostream& stream)
          : stream_(stream)
          , previous_(stream.rdbuf(captured_.rdbuf()))
        {
        }

        ~stream_redirect()
        {
            stream_.rdbuf(previous_);
        }

        std::string str() const
        {
            return captured_.str();
        }

    private:
        std::ostream& stream_;
        std::streambuf* previous_;
        std::ostringstream captured_;
    };
}    // namespace

int hpx_main()
{
    hpx::cout << "local cout smoke test" << hpx::endl;
    hpx::cerr << "local cerr smoke test" << hpx::flush;

    hpx::consolestream << "hello" << ' ' << "local" << hpx::endl;
    hpx::util::format_to(hpx::consolestream, "{} {}", "format", 42)
        << hpx::endl;

    std::ostream& out = hpx::consolestream;
    out << "base-sync" << hpx::endl;
    out << "base-async" << hpx::async_endl;
    out << "base-flush\n" << hpx::async_flush;

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for (int tid = 0; tid != num_threads; ++tid)
    {
        threads.emplace_back([tid]() {
            for (int i = 0; i != iterations; ++i)
            {
                hpx::consolestream << make_line(tid, i) << '\n' << hpx::flush;
            }
        });
    }

    for (auto& thread : threads)
    {
        thread.join();
    }

    return hpx::local::finalize();
}

int main(int argc, char** argv)
{
    test_low_level_iostream_module();

    stream_redirect captured_cout(std::cout);
    stream_redirect captured_cerr(std::cerr);

    HPX_TEST_EQ(hpx::local::init(hpx_main, argc, argv), 0);

    std::string const output = hpx::get_consolestream().str();
    std::string const cout_output = captured_cout.str();
    std::string const cerr_output = captured_cerr.str();

    HPX_TEST(cout_output.find("local cout smoke test\n") != std::string::npos);
    HPX_TEST(cerr_output.find("local cerr smoke test") != std::string::npos);

    HPX_TEST(output.find("hello local\n") != std::string::npos);
    HPX_TEST(output.find("format 42\n") != std::string::npos);
    HPX_TEST(output.find("base-sync\n") != std::string::npos);
    HPX_TEST(output.find("base-async\n") != std::string::npos);
    HPX_TEST(output.find("base-flush\n") != std::string::npos);

    std::map<std::string, int> seen_lines;
    std::istringstream input(output);
    std::string line;

    while (std::getline(input, line))
    {
        ++seen_lines[line];
    }

    for (int tid = 0; tid != num_threads; ++tid)
    {
        for (int i = 0; i != iterations; ++i)
        {
            HPX_TEST_EQ(seen_lines[make_line(tid, i)], 1);
        }
    }

    HPX_TEST_EQ(static_cast<int>(seen_lines.size()),
        fixed_console_lines + num_threads * iterations);

    return hpx::util::report_errors();
}
