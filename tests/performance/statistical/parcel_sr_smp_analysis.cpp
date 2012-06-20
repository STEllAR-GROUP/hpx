//  Copyright (c)      2012 Daniel Kogler
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "statstd.cpp"

///////////////////////////////////////////////////////////////////////////////
void void_thread(){
}
typedef hpx::actions::plain_action0<void_thread> void_action;
typedef hpx::lcos::packaged_action<void_action> Package;
typedef typename hpx::actions::extract_action<void_action>::type action_type;
HPX_REGISTER_PLAIN_ACTION(void_action);

using hpx::parcelset::parcel;
using hpx::applier::get_applier;
using hpx::naming::id_type;

class parcelsender{
public:
    parcelsender(){}
    parcelsender(uint64_t num, double ot_, id_type s, id_type d) : 
        number(num), ot(ot_), source(s), destination(d){
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

    void send_all(){
        uint64_t i = 0;
        double mean;
        string message = "Measuring time required to send parcels:";
        vector<double> time;
        time.reserve(number);

        init_parcels();
        high_resolution_timer t;
        for(; i < number; ++i){
            get_applier().get_parcel_handler();
            //.put_parcel(*parcels[i]);
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
            get_applier().get_parcel_handler().put_parcel(*parcels[i]);
            time.push_back(t1.elapsed());
        }
        printout(time, ot, mean, message);
    }

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
//                new hpx::actions::transfer_action<action_type>(
//                hpx::threads::thread_priority_normal));
        }
    }

    void init_parcels(){
        hpx::naming::address addr;
        hpx::agas::is_local_address(destination, addr);
        for(uint64_t i = 0; i < number; i++){
            parcels[i]->set_destination_addr(addr);
//            parcels[i]->set_source(source);
        }
    }

    Package** packages;
    parcel** parcels;
    id_type source, destination;
    uint64_t number;
    double ot;
};
///////////////////////////////////////////////////////////////////////////////
class parcelreceiver{
public:
    parcelreceiver(){}
    parcelreceiver(uint64_t num, double ot_):number(num), ot(ot_){
    }

    void set_source(id_type sgid){
        source = sgid;
    }
    void set_destination(id_type dgid){
        destination = dgid;
    }

    void receive_all(){
/*
        uint64_t i = 0;
        double mean;
        string message = "Measuring time required to receive parcels:";
        vector<double> time;
        time.reserve(number);

        high_resolution_timer t;
        for(; i < number; ++i)
            packages[i]->get_gid();
        mean = t.elapsed()/number;
        for(i = 0; i < number; i++){
            high_resolution_timer t1;
            packages[i]->get_gid();
            time.push_back(t1.elapsed());
        }
        printout(time, ot, mean, message);
*/
    }

private:
    id_type source, destination;
    uint64_t number;
    double ot;
};
///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm){
    uint64_t num = vm["number-spawned"].as<uint64_t>();
    csv = (vm.count("csv") ? true : false);

    double ot = timer_overhead(num);
    id_type sendid, receiveid;

    sendid = (hpx::applier::get_applier().get_runtime_support_gid());
    receiveid = hpx::applier::get_applier().get_runtime_support_gid();

    parcelsender sender(num, ot, sendid, receiveid);
    parcelreceiver receiver(num, ot);

    sender.send_all();
    //receiver.receive_all();
    return hpx::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[]){
    // Configure application-specific options.
    options_description
        desc_commandline("usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ("number-spawned,N",
            boost::program_options::value<uint64_t>()
                ->default_value(500000),
            "number of packaged_actions created and run")
        ("csv",
            "output results as csv "
            "(format:count,mean,accurate mean,variance,min,max)");

    // Initialize and run HPX
    return hpx::init(desc_commandline, argc, argv);
}

///////////////////////////////////////////////////////////////////////////////

