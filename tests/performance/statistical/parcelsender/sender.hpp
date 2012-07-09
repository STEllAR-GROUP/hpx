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


void void_thread(){
}
typedef hpx::actions::plain_action0<void_thread> void_action;
typedef hpx::lcos::packaged_action<void_action> Package;
typedef hpx::actions::extract_action<void_action>::type action_type;

HPX_REGISTER_PLAIN_ACTION(void_action);

namespace hpx { namespace components { namespace server
{

using hpx::parcelset::parcel;
using hpx::applier::get_applier;
using hpx::naming::id_type;

class HPX_COMPONENT_EXPORT parcelsender : 
    public simple_component_base<parcelsender>
{
public:
    parcelsender(){}
    parcelsender(uint64_t num, double ot_, id_type s, id_type d) : 
        source(s), destination(d), number(num), ot(ot_){
        create_packages();
        create_parcels();
    }
    ~parcelsender(){
        for(unsigned int i = 0; i < number; i++){
            packages[i]->~Package();
            parcels[i]->~parcel();
        }
        free(packages);
        free(parcels);
    }

    void set_source(id_type sgid){
        source = sgid;
    }
    void set_destination(id_type dgid){
        destination = dgid;
    }

    bool send_all(){
        uint64_t i = 0;
        double mean;
        string message = "Measuring time required to send parcels:";
        vector<double> time;
        time.reserve(number);
        hpx::parcelset::parcelhandler& ph = get_applier().get_parcel_handler();

        init_parcels();
        high_resolution_timer t;
        for(; i < number; ++i){
            ph.put_parcel(*parcels[i]);
         }
        mean = t.elapsed()/number;
        for(i = 0; i < number; i++){
            parcels[i]->~parcel();
            hpx::actions::transfer_action<action_type>* action = 
                new hpx::actions::transfer_action<action_type>(
                hpx::threads::thread_priority_normal);
            parcels[i] = new parcel(destination.get_gid(), action);
        }
        
        init_parcels();
        for(i = 0; i < number; i++){
            high_resolution_timer t1;
            ph.put_parcel(*parcels[i]);
            time.push_back(t1.elapsed());
        }
        printout(time, ot, mean, message);
        return true;
    }

    typedef hpx::actions::result_action0<parcelsender, bool, 0, 
        &parcelsender::send_all> send_all_action;
    
private:
    //create the packages
    void create_packages(){
        uint64_t i = 0;
        packages = new Package*[number];
        for(; i < number; ++i)
            packages[i] = new Package();
    }
    //create the parcels
    void create_parcels(){
        uint64_t i = 0;
        parcels = new parcel*[number];
        hpx::naming::gid_type gid = destination.get_gid();
        for(; i < number; ++i){
            hpx::actions::transfer_action<action_type>* action = 
                new hpx::actions::transfer_action<action_type>(
                hpx::threads::thread_priority_normal);
            parcels[i] = new parcel(gid, action);
        }
    }

    void init_parcels(){
        hpx::naming::address addr;
        hpx::agas::is_local_address(destination, addr);
        for(uint64_t i = 0; i < number; i++){
            parcels[i]->set_destination_addr(addr);
        }
    }
    Package** packages;
    parcel** parcels;
    id_type source, destination;
    uint64_t number;
    double ot;
};

}}}


