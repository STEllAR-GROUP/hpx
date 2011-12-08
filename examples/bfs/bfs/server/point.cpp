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
    void point::init(std::size_t objectid,std::size_t grainsize,
        std::size_t max_num_neighbors,std::vector<std::size_t> const& nodelist,
        std::vector<std::size_t> const& neighborlist)
    {
        idx_ = objectid;
        grainsize_ = grainsize;
        neighbors_.resize(grainsize);
        visited_.resize(grainsize);
        for (std::size_t i=0;i<grainsize;i++) {
          neighbors_[i].reserve(max_num_neighbors);
          visited_[i] = false;
        }

        for (std::size_t i=0;i<nodelist.size();i++) {
          std::size_t node = nodelist[i];
          std::size_t neighbor = neighborlist[i];
          if ( node >= idx_*grainsize && node < (idx_+1)*grainsize && node != neighbor ) {
            neighbors_[node-idx_*grainsize_].push_back(neighbor); 
          }
          // symmetrize
          if ( neighbor >= idx_*grainsize_ && neighbor < (idx_+1)*grainsize && node != neighbor ) {
            neighbors_[neighbor-idx_*grainsize_].push_back(node); 
          }
        } 
    }

    std::vector<std::size_t> point::traverse(std::size_t level,std::size_t parent,std::size_t edge)
    {

        if ( visited_[edge-idx_*grainsize_] == false ) {
            visited_[edge-idx_*grainsize_] = true;
            parent_ = parent;
            level_ = level; 

            //hpx::cout << ( boost::format("node id %1%, parent id %2%, level %3%\n")
            //             % idx_ % parent_ % level_) << hpx::flush; 

            // Return the neighbors.
            return neighbors_[edge-idx_*grainsize_];
        } else {
            // Don't return neighbors.
            std::vector<std::size_t> tmp;
            return tmp;
        }
    }
}}

