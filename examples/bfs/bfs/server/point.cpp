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
    void point::init(std::size_t objectid,std::size_t max_num_neighbors,
        std::string const& graphfile)
    {
        idx_ = objectid;
        visited_ = false;
        neighbors_.reserve(max_num_neighbors);

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
                    if ( node == objectid ) {
                        neighbors_.push_back(neighbor); 
                    }
                }
            }
        } 
    }

    std::vector<std::size_t> point::traverse(std::size_t level,std::size_t parent)
    {
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
    }
}}

