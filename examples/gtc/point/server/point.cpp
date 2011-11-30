//  Copyright (c) 2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/lcos/eager_future.hpp>
#include <hpx/lcos/async_future_wait.hpp>
#include <hpx/include/iostreams.hpp>

#include <boost/bind.hpp>
#include <boost/ref.hpp>
#include <boost/lexical_cast.hpp>

#include "../../particle/stubs/particle.hpp"
#include "point.hpp"

#include <string>
#include <sstream>
#include <fstream>

///////////////////////////////////////////////////////////////////////////////
namespace gtc { namespace server
{
    void point::init(std::size_t objectid,std::size_t max_num_neighbors,
        std::string const& meshfile)
    {
        idx_ = objectid;
        neighbors_.reserve(max_num_neighbors);

        std::string line;
        std::string val1,val2,val3,val4;
        std::ifstream myfile;
        myfile.open(meshfile);
        if (myfile.is_open() ) {
            while (myfile.good()) { 
                while(std::getline(myfile,line)) {
                    std::istringstream isstream(line);
                    std::getline(isstream,val1,' ');
                    std::getline(isstream,val2,' ');
                    std::getline(isstream,val3,' ');
                    std::getline(isstream,val4,' ');
                    std::size_t node = boost::lexical_cast<std::size_t>(val1);  
                    double posx = boost::lexical_cast<double>(val2);   
                    double posy = boost::lexical_cast<double>(val3);
                    double posz = boost::lexical_cast<double>(val4);   
                    if ( node == objectid ) {
                        posx_ = posx;
                        posy_ = posy;
                        posz_ = posz;
                    }
                }
            }
            myfile.close(); 
        } 
    }

    bool search_callback(std::list<std::size_t>& deposits,
        std::size_t i,double const& distance)
    {
        double neighbor_distance = 0.1;
        if ( distance < neighbor_distance ) {
            // deposit the charge of this particle on the gridpoint
            deposits.push_back(i); 
        }
        return true; 
    }

    void point::search(std::vector<hpx::naming::id_type> const& particle_components)
    {
        // For demonstration, a simple search strategy: we check if the
        // particle is within a certain distance of the gridpoint.  If so, then
        // get its charge
        typedef std::vector<hpx::lcos::promise<double> > lazy_results_type;

        lazy_results_type lazy_results;

        BOOST_FOREACH(hpx::naming::id_type const& gid, particle_components)
        {
            lazy_results.push_back( stubs::particle::distance_async( gid,posx_,posy_,posz_ ) );
        }

        // List of particles whose charge should deposited on this gridpoint. 
        std::list<std::size_t> deposits;

        // Wait on the results, and invoke a callback when each result is ready.
        hpx::lcos::wait(lazy_results,
            boost::bind(&search_callback, boost::ref(deposits), _1, _2));

        // Print out the particles whose charge should be deposited on this
        // point.
        BOOST_FOREACH(std::size_t i, deposits)
        {
            hpx::cout << ( boost::format("deposit particle %1% on point %2%\n")
                         % idx_ % stubs::particle::get_index(particle_components.at(i)))
                      << hpx::flush; 
        }
    }
}}

