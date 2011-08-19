//  Copyright (c) 2007-2011 Hartmut Kaiser, Richard D Guidry Jr.
//  Copyright (c) 2011 Vinay C Amatya
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//  Parts of this nqueen_client.cpp has been taken from the accumulator example
//  by Hartmut Kaiser.

//#include <cstring>
//#include <iostream>

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

//#include <boost/lexical_cast.hpp>
//#include <boost/program_options.hpp>

#include "nqueen.hpp"

//using namespace hpx;
//using namespace std;
//using namespace boost;

//namespace po = boost::program_options;

int hpx_main(boost::program_options::variables_map&)
{
    //boost::atomic<int> soln_count_total = 0;
    std::size_t soln_count_total;
    std::vector<hpx::naming::id_type> prefixes;
    hpx::naming::id_type prefix;
    hpx::applier::applier& appl = hpx::applier::get_applier();
    if(appl.get_remote_prefixes(prefixes)) {
        prefix = prefixes[0];
    }
    else {
        prefix = appl.get_runtime_support_gid();
    }

    std::cout << "Enter size of board. Default size is 8." << std::endl;
    std::cout << "Command Options: size[value] | default | print | quit" 
              << std::endl;
    std::string cmd;
    std::cin >> cmd;

    while (std::cin.good())
    {
        if(cmd == "size")
        {   
            //using hpx::components::board;
            soln_count_total = 0;
            std::string arg;
            std::cin >> arg;
            std::size_t sz = boost::lexical_cast<std::size_t>(arg);
            
            std::size_t i = 0;
            std::list<hpx::components::board> b;
            hpx::components::board bi;
            while(i != sz)
            {
                b.push_back(bi);
                ++i;
            }
    
            i=0;
            for(std::list<hpx::components::board>::iterator iter = b.begin();
                iter != b.end(); ++iter)
            {  
                iter->create(prefix); 
                iter->init_board(sz); 
                soln_count_total+= iter->solve_board(iter->access_board(), 
                                                     sz, 0, i);
                ++i;
            }
            std::cout << "soln_count:" << soln_count_total << std::endl;
            b.clear();
        }
        else if(cmd == "default")
        {
            //using hpx::components::board;
            soln_count_total = 0;
            hpx::components::board a;
            std::size_t i = 0;
            std::vector<hpx::components::board> b;
            while(i != DS)
            {
                b.push_back(a);
                ++i;
            }
            i = 0;
            for(std::vector<hpx::components::board>::iterator iter = b.begin();
                iter != b.end(); ++iter)
            {  
                iter->create(prefix); 
                iter->init_board(DS); 
                soln_count_total+= iter->solve_board(iter->access_board(), 
                                                     DS, 0, i);
                ++i;
            }
            std::cout << "soln_count:" << soln_count_total << std::endl;
            b.clear();
        }
        else if(cmd == "print")
        {
            std::cout << "soln_count : " << soln_count_total << std::endl;
        }
        else if (cmd == "quit"){
            //std::cout << "soln_count : " << soln_count_total << std::endl;
            break;
        }
        else 
        {
            std::cout << "Invalid Command." << std::endl;
            std::cout << "Options: size[value] | default | print "<< 
            "| quit" << std::endl;
        }
        std::cin >> cmd;
    }
    //b.free();

    //components::stubs::runtime_support::shutdown_all();

    //return threads::terminated;
    hpx::finalize();
    
    return 0;
}

int main(int argc, char* argv[])
{
    boost::program_options::options_description 
        desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    return hpx::init(desc_commandline, argc, argv);
}
