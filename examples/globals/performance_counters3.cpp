///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach 
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#include <boost/format.hpp>

#include <hpx/hpx_init.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/performance_counters/manage_counter_type.hpp>
#include <hpx/performance_counters/manage_counter.hpp>
#include <hpx/performance_counters/stubs/performance_counter.hpp>
#include <hpx/util/static.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;

using hpx::cout;
using hpx::endl;

using hpx::init;
using hpx::finalize;

using hpx::applier::get_applier;

using hpx::performance_counters::manage_counter_type;
using hpx::performance_counters::manage_counter;
using hpx::performance_counters::counter_raw;
using hpx::performance_counters::counter_value;
using hpx::performance_counters::stubs::performance_counter;

using hpx::util::static_;

using hpx::naming::resolver_client;
using hpx::naming::gid_type;

struct parcels_total_type {};
struct parcels_receives_started {};
struct parcels_receives_completed {};

///////////////////////////////////////////////////////////////////////////////
void startup_()
{
    static_<manage_counter_type*, parcels_total_type>
      type_(new manage_counter_type);
    static_<manage_counter*, parcels_receives_started>
      receives_started_(new manage_counter);
    static_<manage_counter*, parcels_receives_completed>
      receives_completed_(new manage_counter);

    type_.get()->install("/parcels/total", counter_raw);  
    receives_started_.get()->install("/parcels(receives_started)/total",
        boost::bind(&hpx::parcelset::parcelport::total_receives_started, 
                    &get_applier().get_parcel_handler().get_parcelport())); 
    receives_completed_.get()->install("/parcels(receives_completed)/total",
        boost::bind(&hpx::parcelset::parcelport::total_receives_completed, 
                    &get_applier().get_parcel_handler().get_parcelport())); 
}

///////////////////////////////////////////////////////////////////////////////
void shutdown_()
{
    static_<manage_counter_type*, parcels_total_type>
      type_;
    static_<manage_counter*, parcels_receives_started>
      receives_started_;
    static_<manage_counter*, parcels_receives_completed>
      receives_completed_;

    delete type_.get();
    delete receives_started_.get();
    delete receives_completed_.get();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    {
        resolver_client& agas = get_applier().get_agas_client();

        boost::uint32_t here = get_applier().get_prefix_id();

        boost::format fmter("/parcels([L%d]/%s)/total");

        // Build full performance counter names.
        std::string receives_started
            = boost::str(fmter % here % "receives_started");

        std::string receives_completed
            = boost::str(fmter % here % "receives_completed");

        // Get GIDs of the performance counters.
        gid_type receives_started_gid, receives_completed_gid;

        agas.queryid(receives_started, receives_started_gid);
        agas.queryid(receives_completed, receives_completed_gid);

        cout << receives_started << " -> " << receives_started_gid << "\n"
             << receives_completed << " -> " << receives_completed_gid << endl; 

        counter_value total_started
            = performance_counter::get_value(receives_started_gid);

        counter_value total_completed
            = performance_counter::get_value(receives_completed_gid);

        cout << receives_started << " -> " << total_started.value_ << "\n"
             << receives_completed << " -> " << total_completed.value_ << endl; 
    }

    // Do nothing.
    finalize();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description
       desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    // Initialize and run HPX
    return init(desc_commandline, argc, argv, startup_, shutdown_);
}

