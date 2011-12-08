//  Copyright (c) 2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/include/iostreams.hpp>

#include "../stubs/point.hpp"
#include "point.hpp"

#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>

#include <iostream>
#include <fstream>

///////////////////////////////////////////////////////////////////////////////
namespace bfs { namespace server
{
    void point::read(std::size_t objectid,std::size_t grainsize,
        std::size_t max_num_neighbors,std::string const& graphfile)
    {
        idx_ = objectid;
        grainsize_ = grainsize;
        neighbors_.resize(grainsize);
        for (std::size_t i=0;i<grainsize;i++) {
          neighbors_[i].reserve(max_num_neighbors);
        }

        // Read in the graph file
        std::string line;
        std::string val1,val2;
        std::ifstream myfile;
        myfile.open(graphfile);
        if (myfile.is_open()) {
            while (myfile.good()) { 
                while (std::getline(myfile,line)) {
                    std::istringstream isstream(line);
                    std::getline(isstream,val1,' ');
                    std::getline(isstream,val2,' ');
                    std::size_t node = boost::lexical_cast<std::size_t>(val1);   
                    std::size_t neighbor = boost::lexical_cast<std::size_t>(val2);   
                    if ( node >= idx_*grainsize && node < (idx_+1)*grainsize && node != neighbor ) {
                        neighbors_[node-idx_*grainsize_].push_back(neighbor); 
                    }
                }
            }
        } 
    }

    void point::init(std::size_t objectid,std::size_t max_num_neighbors,
        std::string const& graphfile)
    {
         // make this a vector
        //visited_ = false;
    }

    std::vector<std::size_t> point::traverse(std::size_t level,std::size_t parent)
    {
            std::vector<std::size_t> tmp;
            return tmp;
#if 0
        if ( visited_ == false ) {
            visited_ = true;
            parent_ = parent;
            level_ = level; 

            hpx::cout << ( boost::format("node id %1%, parent id %2%, level %3%\n")
                         % idx_ % parent_ % level_) << hpx::flush; 

            // Return the neighbors.
            return neighbors_;
        } else {
            // Don't return neighbors.
            std::vector<std::size_t> tmp;
            return tmp;
        }
#endif
    }
}}

