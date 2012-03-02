//  Copyright (c) 2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
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
        std::vector<std::size_t> const& neighborlist,
        boost::numeric::ublas::mapped_vector<std::size_t> const& index,std::size_t max_levels)
    {
        hpx::lcos::local::mutex::scoped_lock l(mtx_);
        idx_ = objectid;
        grainsize_ = grainsize; 
        max_levels_ = max_levels;
        neighbors_.resize(grainsize_);
        visited_.resize(grainsize_);
        parent_.resize(grainsize_);
        level_.resize(grainsize_);
        for (std::size_t i=0;i<grainsize_;i++) {
          neighbors_[i].reserve(max_num_neighbors);
          visited_[i] = false;
        }

        index_ = index;
 
        boost::numeric::ublas::mapped_vector<bool> initialized;
        // initialize the mapping
        for (std::size_t i=0;i<nodelist.size();i++) {
          std::size_t node = nodelist[i];
          std::size_t neighbor = neighborlist[i];
          if ( index_(node) == idx_ && node != neighbor ) {
            if ( mapping_.find_element(node) == 0 ) mapping_.insert_element(node,0);
            if ( initialized.find_element(node) == 0 ) initialized.insert_element(node,false);
          }
          if ( index_(neighbor) == idx_ && node != neighbor ) {
            if ( mapping_.find_element(neighbor) == 0 ) mapping_.insert_element(neighbor,0);
            if ( initialized.find_element(neighbor) == 0 ) initialized.insert_element(neighbor,false);
          }
        }
        mapping_.resize(nodelist.size());
        initialized.resize(nodelist.size());

        std::size_t count = 0;
        for (std::size_t i=0;i<nodelist.size();i++) {
          std::size_t node = nodelist[i];
          std::size_t neighbor = neighborlist[i];
          BOOST_ASSERT(count < grainsize_);
          if ( index_(node) == idx_ && node != neighbor ) {
            if ( initialized(node) == false ) {
              mapping_(node) = count;
              initialized(node) = true;
              count++;
            } 
            neighbors_[ mapping_(node) ].push_back(neighbor);
          }

          BOOST_ASSERT(count < grainsize_);
          // symmetrize
          if ( index_(neighbor) == idx_ && node != neighbor ) {
            if ( initialized(neighbor) == false ) {
              mapping_(neighbor) = count;
              initialized(neighbor) = true;
              count++;
            }
            neighbors_[ mapping_(neighbor) ].push_back(node);
    
          }
        } 
    }

    void point::reset_visited(std::size_t id) {
      hpx::lcos::local::mutex::scoped_lock l(mtx_);
      std::fill( visited_.begin(),visited_.end(),false);
    }

    // tradional traverse
    std::vector<std::size_t> point::traverse(std::size_t level,std::size_t parent,std::size_t edge)
    {
        hpx::lcos::local::mutex::scoped_lock l(mtx_);
        if ( visited_[mapping_(edge)] == false ) {
          visited_[mapping_(edge)] = true;
          parent_[mapping_(edge)] = parent;
          level_[mapping_(edge)] = level; 
          return neighbors_[mapping_(edge)];
        } else {
          std::vector<std::size_t> tmp;
          return tmp;
        }
    }

    // eliminate a lock
    std::vector<nodedata> point::unlocked_depth_traverse(std::size_t level,std::size_t parent,std::size_t edge)
    {
        std::vector<nodedata> result,lresult;
        //result.reserve(10000);
        //lresult.reserve(10000);

        // verify the edge is local first
        if ( index_(edge) == idx_ ) { 
          std::size_t mapping = mapping_(edge);
          if ( visited_[mapping] == false || level_[mapping] > level ) {
            visited_[mapping] = true;
            parent_[mapping] = parent;
            level_[mapping] = level; 
            // search all neighbors local to this component to the max_levels depth
            if ( level < max_levels_ ) {
              for (std::size_t i=0;i<neighbors_[mapping].size();i++) {
                std::size_t neighbor = neighbors_[mapping][i];
                lresult = unlocked_depth_traverse(level+1,edge,neighbor);
                result.insert(result.end(),lresult.begin(),lresult.end());
              }
            }
          }
        } else {
          nodedata nonlocal;
          nonlocal.neighbor = edge; 
          nonlocal.parent = parent; 
          nonlocal.level = level;
          result.push_back(nonlocal);
        }

        return result;
    }

    // depth traverse
    std::vector<nodedata> point::depth_traverse(std::size_t level,std::size_t parent,std::size_t edge)
    {
        hpx::lcos::local::mutex::scoped_lock l(mtx_);
        std::vector<nodedata> result,lresult;
        //result.reserve(10000);
        //lresult.reserve(10000);

        // verify the edge is local first
        if ( index_(edge) == idx_ ) { 
          std::size_t mapping = mapping_(edge);
          if ( visited_[mapping] == false || level_[mapping] > level ) {
            visited_[mapping] = true;
            parent_[mapping] = parent;
            level_[mapping] = level; 
            // search all neighbors local to this component to the max_levels depth
            if ( level < max_levels_ ) {
              for (std::size_t i=0;i<neighbors_[mapping].size();i++) {
                std::size_t neighbor = neighbors_[mapping][i];
                //hpx::util::unlock_the_lock<hpx::lcos::local::mutex::scoped_lock> ul(l); 
                //lresult = depth_traverse(level+1,edge,neighbor);
                lresult = unlocked_depth_traverse(level+1,edge,neighbor);
                result.insert(result.end(),lresult.begin(),lresult.end());
              }
            }
          }
        } else {
          nodedata nonlocal;
          nonlocal.neighbor = edge; 
          nonlocal.parent = parent; 
          nonlocal.level = level;
          result.push_back(nonlocal);
        }

        return result;
    }
 
    std::size_t point::get_parent(std::size_t edge)
    {
      hpx::lcos::local::mutex::scoped_lock l(mtx_);
      if ( visited_[mapping_(edge)] == false ) {
        return 0;
      } else {
        return parent_[mapping_(edge)];
      }
    }

    std::size_t point::get_level(std::size_t edge)
    {
      hpx::lcos::local::mutex::scoped_lock l(mtx_);
      if ( visited_[mapping_(edge)] == false ) {
        return 0;
      } else {
        return level_[mapping_(edge)];
      }
    }
}}

