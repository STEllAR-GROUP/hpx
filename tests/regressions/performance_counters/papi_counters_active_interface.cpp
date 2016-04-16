//  Copyright (c) 2013 Maciej Brodowicz
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <stdio.h>
#include <unistd.h>
#include <string>
#include <vector>

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/hpx_start.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>

///////////////////////////////////////////////////////////////////////////////
const char *counter_name = "/papi{locality#0/worker-thread#0}/PAPI_SR_INS";
const size_t nstores = 1000000;

///////////////////////////////////////////////////////////////////////////////
inline bool close_enough(double m, double ex, double perc)
{
    return 100.0*fabs(m-ex)/ex <= perc;
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map&)
{
    hpx::start_active_counters();

    // perform n stores, active counter
    volatile size_t i;
    for (i = 0; i < nstores; i++) ;

    hpx::evaluate_active_counters();
    // stop the counter w/o resetting
    hpx::stop_active_counters();

    // perform n stores (should be uncounted)
    for (i = 0; i < nstores; i++) ;
    // get value and reset, and start again
    hpx::evaluate_active_counters(true);
    hpx::start_active_counters();

    // perform 2*n stores, counted from 0 (or close to it)
    for (i = 0; i < 2*nstores; i++) ;
    hpx::evaluate_active_counters();

    // perform another n stores, counted from 0 (or close to it)
    hpx::reset_active_counters();
    for (i = 0; i < nstores; i++) ;
    hpx::evaluate_active_counters();

    return hpx::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int check(int fd)
{
    std::string out;
    std::vector<double> cnt;
    while (true)
    {
        char buf[1024];
        ssize_t n = read(fd, buf, 1024);
        if (n > 0)
        {
            out += buf;
            size_t pos = out.find(counter_name);
            if (pos != out.npos)
            { // counter name found
                out.erase(0, pos);
                pos = out.find('\n');
                if (pos != out.npos)
                { // this is complete line; extract counter value
                    size_t cpos = out.rfind(',', pos);
                    std::cerr << out.substr(0, pos+1);
                    cnt.push_back(boost::lexical_cast<double>
                        (out.substr(cpos+1, pos-cpos-1)));
                    if (cnt.size() == 5) break;
                    out.erase(0, pos+1);
                }
            }
        }
        else
            throw std::runtime_error("truncated input; didn't get all counter values");
    }

    // since printing affects the counts, the relative error bounds need to be
    // increased compared to the "basic_functions" test
    bool pass = close_enough(cnt[0], nstores, 5.0) &&
        (cnt[1] >= cnt[0]) && close_enough(cnt[0], cnt[1], 5.0) &&
        close_enough(cnt[2], 2.0*cnt[0], 5.0) &&
        close_enough(cnt[3], cnt[0], 5.0);

    std::cerr << (pass? "PASSED": "FAILED") << ".\n";

    return pass? 0: 1;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Prepare to grab the output stream.
    int pipefd[2];
    if (pipe(pipefd) != 0 || dup2(pipefd[1], STDOUT_FILENO) < 0)
        throw std::runtime_error("could not create pipe to stdout");

    // Configure application-specific options.
    boost::program_options::options_description cmdline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    // Add the required counter command line option.
    char **opt = new char *[argc+2];
    for (int i = 0; i < argc; i++) opt[i] = argv[i];
    char copt[256];
    snprintf(copt, 256, "--hpx:print-counter=%s", counter_name);
    opt[argc] = copt;
    opt[argc+1] = 0;

    // Run test in HPX domain.
    hpx::start(argc+1, opt);

    // Collect and process the output.
    int rc = check(pipefd[0]);

    hpx::stop();
    return rc;
}
