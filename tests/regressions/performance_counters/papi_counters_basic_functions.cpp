//  Copyright (c) 2013 Maciej Brodowicz
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include <boost/program_options.hpp>
#include <boost/format.hpp>

///////////////////////////////////////////////////////////////////////////////
inline bool close_enough(double m, double ex, double perc)
{
    return 100.0*fabs(m-ex)/ex <= perc;
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map&)
{
    const size_t n = 1000000;

    boost::uint32_t const prefix = hpx::get_locality_id();
    boost::format cnt_name("/papi{locality#%d/worker-thread#0}/PAPI_SR_INS");

    using hpx::naming::id_type;
    using hpx::performance_counters::get_counter;
    using hpx::performance_counters::stubs::performance_counter;
    using hpx::performance_counters::counter_value;

    id_type id = get_counter(boost::str(cnt_name % prefix));

    performance_counter::start(id);

    // perform n stores, active counter
    volatile size_t i;
    for (i = 0; i < n; i++) ;

    counter_value value1 = performance_counter::get_value(id);
    // stop the counter w/o resetting
    performance_counter::stop(id);

    // perform n stores (should be uncounted)
    for (i = 0; i < n; i++) ;
    // get value and reset and start
    counter_value value2 = performance_counter::get_value(id, true);
    performance_counter::start(id);

    // perform 2*n stores, counted from 0 (or close to it)
    for (i = 0; i < 2*n; i++) ;
    counter_value value3 = performance_counter::get_value(id);

    bool pass = status_is_valid(value1.status_) &&
                status_is_valid(value2.status_) &&
                status_is_valid(value3.status_);
    if (pass)
    {
        boost::uint64_t cnt1 = value1.get_value<double>();
        boost::uint64_t cnt2 = value2.get_value<double>();
        boost::uint64_t cnt3 = value3.get_value<double>();
        // assume cache line on any architecture doesn't exceed 1KB
        std::cout << n << " counted store instructions, result: " << cnt1 << std::endl
                  << n << " uncounted store instructions, result: " << cnt2-cnt1 << std::endl
                  << 2*n << " store instructions, count after reset: " << cnt3 << std::endl;

        pass &= close_enough(cnt1, n, 1.0);
        pass &= (cnt2 >= cnt1) && close_enough(cnt1, cnt2, 1.0);
        pass &= close_enough(cnt3, 2.0*cnt1, 1.0);
    }
    std::cout << (pass? "PASSED": "FAILED") << ".\n";

    hpx::finalize();
    return pass? 0: 1;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options.
    boost::program_options::options_description cmdline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    // Initialize and run HPX.
    return hpx::init(cmdline, argc, argv);
}
