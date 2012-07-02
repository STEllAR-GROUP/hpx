//  Copyright (c)      2012 Daniel Kogler
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/actions/component_action.hpp> 
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>

#include <vector>
#include <string>
#include "../statstd.hpp"

namespace hpx { namespace components { namespace server
{

using hpx::applier::get_applier;
using hpx::parcelset::parcel;

class HPX_COMPONENT_EXPORT parcelreceiver : 
    public simple_component_base<parcelreceiver>
{
public:
    parcelreceiver(){}
    parcelreceiver(uint64_t num, double ot_):number(num), ot(ot_){
    }

    void set_source(hpx::naming::id_type sgid){
        source = sgid;
    }

    bool receive_all(){
        uint64_t i = 0;
        double mean1;
//        double mean2;
        string message = "Measuring time required to get parcels from queue:";
        vector<double> time;
        time.reserve(number);
        parcel p;
        hpx::parcelset::parcelhandler& ph = get_applier().get_parcel_handler();

        high_resolution_timer t;
        for(; i < number; ++i)
            ph.get_parcel(p);
        mean1 = t.elapsed()/number;
/*
        t.restart();
        for(i = 0; i < number; ++i){}
        mean2 = t.elapsed()/number;
*/
        for(i = 0; i < number; i++){
            high_resolution_timer t1;
            ph.get_parcel(p);
            time.push_back(t1.elapsed());
        }
        printout(time, ot, mean1, message);
/*
        message = "Measuring time required to process parcels:";
        for(i = 0; i < number; i++){
            high_resolution_timer t1;
            //packages[i]->get_gid();
            time.push_back(t1.elapsed());
        }
        printout(time, ot, mean2, message);
*/
        return true;
    }

    typedef hpx::actions::result_action0<parcelreceiver, bool, 0, 
        &parcelreceiver::receive_all> receive_all_action;
private:
    hpx::naming::id_type source;
    uint64_t number;
    double ot;
};

}}}


