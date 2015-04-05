//  Copyright (c) 2013 Shuangyang Yang
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/iostreams.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/foreach.hpp>

#include "central_tuplespace/simple_central_tuplespace.hpp"
#include "small_big_object.hpp"

typedef examples::server::simple_central_tuplespace central_tuplespace_type;
typedef central_tuplespace_type::tuple_type tuple_type;
typedef central_tuplespace_type::elem_type elem_type;


void print_tuple(const tuple_type& tuple)
{
    if(tuple.empty())
    {
        hpx::cout<<"()";
        return;
    }

    tuple_type::const_iterator it = tuple.begin();
    hpx::cout<<"("<<*it;
    for(++it; it != tuple.end(); ++it)
    {
        hpx::cout<<", "<<*it;
    }
    hpx::cout<<")";
}

void simple_central_tuplespace_store_load_test(
        const std::vector<tuple_type>& tuples) {
    // Find the localities connected to this application.
    std::vector<hpx::id_type> localities = hpx::find_all_localities();

    const std::string tuplespace_symbol_name = "/tuplespace_load_store_test_" +
        boost::lexical_cast<std::string>(hpx::get_locality_id());
    examples::simple_central_tuplespace central_tuplespace;

    if(!central_tuplespace.create(tuplespace_symbol_name, localities.back()))
    {
        hpx::cerr << "locality " << hpx::get_locality_id() << ": "
            << "FAIL to create " << tuplespace_symbol_name << hpx::endl;

        return;
    }

   // insert tuples
   for (std::vector<tuple_type>::const_iterator it = tuples.begin();
           it != tuples.end(); ++it) {
       central_tuplespace.write_sync(*it);
   }

   boost::posix_time::ptime now =
       boost::posix_time::second_clock::local_time();
   std::string file_name = std::string("TupleSpace") +
       std::string("_") + boost::posix_time::to_iso_string(now);

   central_tuplespace.store_sync(file_name);

   hpx::cout<<"Original Tuple Space Content:\n"<<central_tuplespace.print()<<"\n";
   central_tuplespace.clear_sync();

   examples::simple_central_tuplespace copy_central_tuplespace;
   copy_central_tuplespace.create(tuplespace_symbol_name + "_copy", localities.back());

   copy_central_tuplespace.load_sync(file_name);

   hpx::cout<<"Copy Tuple Space Content:\n"<<copy_central_tuplespace.print()<<"\n";
}

HPX_PLAIN_ACTION(simple_central_tuplespace_store_load_test, simple_central_tuplespace_store_load_test_action);

void simple_central_tuplespace_test(
    const std::string& tuplespace_symbol_name, const tuple_type tuple)
{
   examples::simple_central_tuplespace central_tuplespace;

   if(!central_tuplespace.connect(tuplespace_symbol_name))
   {
       hpx::cerr << "locality " << hpx::get_locality_id() << ": " 
           << "FAIL to connect " << tuplespace_symbol_name << hpx::endl;
       return;
   }

   int ret = central_tuplespace.write_sync(tuple);
   hpx::cout << "locality " << hpx::get_locality_id() << ": " << "write_sync ";
   print_tuple(tuple);
   hpx::cout<<" returns " << ret << hpx::endl;

   tuple_type partial_tuple;

   if(tuple.size() > 1) // use second field
   {
       partial_tuple.push_back_empty()
           .push_back(*(tuple.begin() + 1));
   }
   else
   {
       partial_tuple.push_back(*(tuple.begin()));
   }

   tuple_type return_tuple = central_tuplespace.read_sync(partial_tuple, 0);
   hpx::cout<< "locality " << hpx::get_locality_id() << ": " <<"read_sync tuple with ";
   print_tuple(partial_tuple);
   hpx::cout<<" returns ";
   print_tuple(return_tuple);
   hpx::cout<<hpx::endl;

   return_tuple = central_tuplespace.take_sync(partial_tuple, 0);
   hpx::cout<< "locality " << hpx::get_locality_id() << ": " <<"take_sync tuple with ";
   print_tuple(partial_tuple);
   hpx::cout<<" (1st) returns ";
   print_tuple(return_tuple);
   hpx::cout<<hpx::endl;

   return_tuple = central_tuplespace.take_sync(partial_tuple, 0);
   hpx::cout<< "locality " << hpx::get_locality_id() << ": " <<"take_sync tuple with ";
   print_tuple(partial_tuple);
   hpx::cout<<" (2nd) returns ";
   print_tuple(return_tuple);
   hpx::cout<<hpx::endl<<hpx::flush;
}

HPX_PLAIN_ACTION(simple_central_tuplespace_test, simple_central_tuplespace_test_action);

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    {
        // Find the localities connected to this application.
        std::vector<hpx::id_type> localities = hpx::find_all_localities();

        hpx::cout << "Total " << localities.size() << " localities." << hpx::endl;

        tuple_type tuple1;
        tuple1.push_back(std::string("first"))
            .push_back(10) // first elem: int
            .push_back(small_object(20)) // second elem: small_object
            .push_back(big_object(30, 40)); // third elem: big_object

        hpx::cout << "locality " << hpx::get_locality_id() << ": " << "created tuple1: ";
        print_tuple(tuple1);
        hpx::cout<< hpx::endl;

        tuple_type tuple2;
        tuple2.push_back(std::string("second"))
            .push_back(std::string("string")) // first elem: string
            .push_back(small_object(50)) // second elem: small_object
            .push_back(big_object(60, 70)); // third elem: big_object

        hpx::cout << "locality " << hpx::get_locality_id() << ": " << "created tuple2: ";
        print_tuple(tuple2);
        hpx::cout<< hpx::endl;

        {
            // store and load tests

            std::vector<hpx::lcos::future<void> > futures;
            std::vector<tuple_type> tuples;
            tuples.push_back(tuple1);
            tuples.push_back(tuple2);

            BOOST_FOREACH(hpx::naming::id_type const& node, localities)
            {
                // Asynchronously start a new task. The task is encapsulated in a
                // future, which we can query to determine if the task has
                // completed.
                typedef simple_central_tuplespace_store_load_test_action action_type1;
                futures.push_back(hpx::async<action_type1>
                        (node, tuples));
            }
            hpx::wait_all(futures);
        }

        {
            // basic operations test

            std::vector<hpx::lcos::future<void> > futures;
            int id = 0;
            BOOST_FOREACH(hpx::naming::id_type const& node, localities)
            {
                const std::string tuplespace_symbol_name = "/tuplespace_test_" +
                    boost::lexical_cast<std::string>(id++);
                examples::simple_central_tuplespace central_tuplespace;

                if(!central_tuplespace.create(tuplespace_symbol_name, localities.back()))
                {
                    hpx::cerr << "locality " << hpx::get_locality_id() << ": "
                        << "FAIL to create " << tuplespace_symbol_name << hpx::endl;

                    return hpx::finalize();
                }

                // Asynchronously start a new task. The task is encapsulated in a
                // future, which we can query to determine if the task has
                // completed.
                typedef simple_central_tuplespace_test_action action_type;
                futures.push_back(hpx::async<action_type>
                        (node, tuplespace_symbol_name, tuple1));
                futures.push_back(hpx::async<action_type>
                        (node, tuplespace_symbol_name, tuple2));
            }
            hpx::wait_all(futures);
        }
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

