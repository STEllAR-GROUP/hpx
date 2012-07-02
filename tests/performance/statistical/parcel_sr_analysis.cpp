//  Copyright (c)      2012 Daniel Kogler
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "parcelsender/sender.hpp"
#include "parcelreceiver/receiver.hpp"

using hpx::components::server::parcelsender;
using hpx::components::server::parcelreceiver;

typedef hpx::lcos::future<bool> bf;

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm){
    uint64_t num = vm["number-sent"].as<uint64_t>();
    csv = (vm.count("csv") ? true : false);

    double ot = timer_overhead(num);
    hpx::naming::id_type sendid, receiveid;

    sendid = (hpx::applier::get_applier().get_runtime_support_gid());
    receiveid = hpx::applier::get_applier().get_runtime_support_gid();

    parcelsender sender(num, ot, sendid, receiveid);
    parcelreceiver receiver(num, ot);

    bf sent = hpx::async<
        hpx::components::server::parcelsender::send_all_action>(sender.get_gid());
    
    if(!vm.count("simultaneous")) sent.get();
    
    bf received = hpx::async<
        hpx::components::server::parcelreceiver::receive_all_action>(
        receiver.get_gid());

    if(vm.count("simultaneous")) sent.get();

    received.get();

    return hpx::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[]){
    // Configure application-specific options.
    options_description
        desc_commandline("usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ("number-sent,N",
            boost::program_options::value<uint64_t>()
                ->default_value(500000),
            "number of parcels sent")
        ("simultaneous,S",
            "send an receive simultaneously instead of sending"
            "all parcels then receiving all")
        ("csv",
            "output results as csv "
            "(format:count,mean,accurate mean,variance,min,max)");

    // Initialize and run HPX
    vector<string> cfg;
    cfg.push_back("hpx.components.parcelsender.enabled = 1");
    cfg.push_back("hpx.components.parcelreceiver.enabled = 1");
    return hpx::init(desc_commandline, argc, argv, cfg);
}

///////////////////////////////////////////////////////////////////////////////

