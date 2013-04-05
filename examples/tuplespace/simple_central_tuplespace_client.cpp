//  Copyright (c) 2013 Shuangyang Yang
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/actions.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/foreach.hpp>

#include "central_tuplespace/simple_central_tuplespace.hpp"
#include "small_big_object.hpp"

typedef examples::server::simple_central_tuplespace central_tuplespace_type;
typedef central_tuplespace_type::tuple_type tuple_type;
typedef central_tuplespace_type::key_type key_type;
typedef central_tuplespace_type::elem_type elem_type;


void print_tuple(const tuple_type& tuple)
{
    if(tuple.empty())
    {
        std::cout<<"()";
        return;
    }

    tuple_type::const_iterator it = tuple.begin();
    std::cout<<"("<<*it;
    for(++it; it != tuple.end(); ++it)
    {
        std::cout<<", "<<*it;
    }
    std::cout<<")";
}



void simple_central_tuplespace_test(const std::string& tuplespace_symbol_name, const key_type& key, const tuple_type& tuple)
{
   hpx::naming::id_type ts_gid;                                     
   hpx::agas::resolve_name(tuplespace_symbol_name, ts_gid);  

   examples::simple_central_tuplespace central_tuplespace(ts_gid);

   int ret = central_tuplespace.write_sync(tuple);
   std::cout << "locality " << hpx::get_locality_id() << ": " << "write_sync ";
   print_tuple(tuple);
   std::cout<<" returns " << ret << std::endl;

   tuple_type return_tuple = central_tuplespace.read_sync(key, 0);
   std::cout<< "locality " << hpx::get_locality_id() << ": " <<"read_sync tuple with key="<<key<<" returns ";
   print_tuple(return_tuple);
   std::cout<<std::endl;

   return_tuple = central_tuplespace.take_sync(key, 0);
   std::cout<< "locality " << hpx::get_locality_id() << ": " <<"take_sync tuple with key="<<key<<" (1st) returns ";
   print_tuple(return_tuple);
   std::cout<<std::endl;

   return_tuple = central_tuplespace.take_sync(key, 0);
   std::cout<< "locality " << hpx::get_locality_id() << ": " <<"take_sync tuple with key="<<key<<" (2nd) returns ";
   print_tuple(return_tuple);
   std::cout<<std::endl;
}

HPX_PLAIN_ACTION(simple_central_tuplespace_test, simple_central_tuplespace_test_action);

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    {
        // Find the localities connected to this application.
        std::vector<hpx::id_type> localities = hpx::find_all_localities();

        // Create an central_tuplespace component either on this locality (if the
        // example is executed on one locality only) or on any of the remote
        // localities (otherwise).
        examples::simple_central_tuplespace central_tuplespace(
            hpx::components::new_<central_tuplespace_type>(localities.back()));

        // register central_tuplespace component in agas
        const std::string tuplespace_symbol_name = "/tuplespace";
        hpx::agas::register_name(tuplespace_symbol_name, central_tuplespace.get_gid());


        tuple_type tuple1;
        key_type key1="first";
        tuple1.push_back(key1)
            .push_back(10) // first elem: int
            .push_back(small_object(20)) // second elem: small_object
            .push_back(big_object(30, 40)); // third elem: big_object

        tuple_type tuple2;
        key_type key2="second";
        tuple2.push_back(key2)
            .push_back(std::string("string")) // first elem: string
            .push_back(small_object(50)) // second elem: small_object
            .push_back(big_object(60, 70)); // third elem: big_object

        std::vector<hpx::lcos::future<void> > futures;

        BOOST_FOREACH(hpx::naming::id_type const& node, localities)
        {
            // Asynchronously start a new task. The task is encapsulated in a
            // future, which we can query to determine if the task has
            // completed.
            typedef simple_central_tuplespace_test_action action_type;
            futures.push_back(hpx::async<action_type>
                    (node, tuplespace_symbol_name, key1, tuple1));
            futures.push_back(hpx::async<action_type>
                    (node, tuplespace_symbol_name, key2, tuple2));
        }
        hpx::lcos::wait(futures);
    }

    // Initiate shutdown of the runtime systems on all localities.
    return hpx::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // We force this example to use 2 threads by default as one of the threads
    // will be sitting most of the time in the kernel waiting for user input.
    std::vector<std::string> cfg;
    cfg.push_back("hpx.os_threads=2");

    // Initialize and run HPX.
    return hpx::init(argc, argv, cfg);
}

