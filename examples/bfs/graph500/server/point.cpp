//  Copyright (c) 2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/lcos/eager_future.hpp>
#include <hpx/lcos/async_future_wait.hpp>

#include "../make_graph.hpp"
#include "../stubs/point.hpp"
#include "point.hpp"

#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>

#include <iostream>
#include <fstream>

/* Spread the two 64-bit numbers into five nonzero values in the correct
 * range. */
void make_mrg_seed(uint64_t userseed1, uint64_t userseed2, uint_fast32_t* seed) {
  seed[0] = (userseed1 & 0x3FFFFFFF) + 1;
  seed[1] = ((userseed1 >> 30) & 0x3FFFFFFF) + 1;
  seed[2] = (userseed2 & 0x3FFFFFFF) + 1;
  seed[3] = ((userseed2 >> 30) & 0x3FFFFFFF) + 1;
  seed[4] = ((userseed2 >> 60) << 4) + (userseed1 >> 60) + 1;
}

static void compute_edge_range(int rank, int size, int64_t M, int64_t* start_idx, int64_t* end_idx) {
  int64_t rankc = (int64_t)(rank);
  int64_t sizec = (int64_t)(size);
  *start_idx = rankc * (M / sizec) + (rankc < (M % sizec) ? rankc : (M % sizec));
  *end_idx = (rankc + 1) * (M / sizec) + (rankc + 1 < (M % sizec) ? rankc + 1 : (M % sizec));
}

///////////////////////////////////////////////////////////////////////////////
namespace graph500 { namespace server
{
    void point::init(std::size_t objectid,std::size_t log_numverts,std::size_t number_partitions,
                     double overlap)
    {
      idx_ = objectid;
      // Spread the two 64-bit numbers into five nonzero values in the correct
      //  range.
      uint_fast32_t seed[5];
      uint64_t userseed1 = 1;
      uint64_t userseed2 = 1;
      make_mrg_seed(userseed1, userseed2, seed);

      int64_t M = INT64_C(16) << log_numverts;

      if ( objectid < number_partitions ) {
        int64_t start_idx, end_idx;
        compute_edge_range(objectid, number_partitions, M, &start_idx, &end_idx);
        int64_t nedges = end_idx - start_idx;

        local_edges_.resize(nedges);

        generate_kronecker_range(seed, log_numverts, start_idx, end_idx, &*local_edges_.begin());

        for (std::size_t i=0;i<local_edges_.size();i++) {
          // the smallest node is 1 ( 0 is reserved for unvisited edges )
          // increment everyone by 1
          local_edges_[i].v0 += 1;
          local_edges_[i].v1 += 1;
        }

      } else {
        int64_t start_idx, end_idx;
        compute_edge_range(number_partitions-1, number_partitions, M, &start_idx, &end_idx);
        int64_t nedges = end_idx - start_idx;

        std::size_t size = (std::size_t) floor(overlap*nedges);
        local_edges_.resize(size);

        std::vector<int64_t> edges;
        edges.resize(size);

        bool found;
        int64_t edge;
        for ( std::size_t i=0;i<edges.size();i++) {
          while(1) {
            edge = rand() % end_idx;
            found = false;
            // make sure it's not a duplicate
            for (std::size_t j=0;j<i;j++) {
              if ( edges[j] == edge ) {
                //duplicate
                found = true;
                break;
              }
            }
            if ( !found ) break;
          }    
          edges[i] = edge;
        }

        packed_edge tmp[1];
        for (std::size_t i=0;i<local_edges_.size();i++) {
          generate_kronecker_range(seed, log_numverts, edges[i], edges[i]+1, tmp);

          local_edges_[i] = tmp[0];

          // the smallest node is 1 ( 0 is reserved for unvisited edges )
          // increment everyone by 1
          local_edges_[i].v0 += 1;
          local_edges_[i].v1 += 1;
        }
      }

      // find the biggest node or neighbor
      int64_t maxnode = 0;
      minnode_ = 99999;
      for (std::size_t i=0;i<local_edges_.size();i++) {
        if ( local_edges_[i].v0 > maxnode ) maxnode = local_edges_[i].v0;
        if ( local_edges_[i].v1 > maxnode ) maxnode = local_edges_[i].v1;

        if ( local_edges_[i].v0 < minnode_ ) minnode_ = local_edges_[i].v0;
        if ( local_edges_[i].v1 < minnode_ ) minnode_ = local_edges_[i].v1;
      }
      maxnode++;
      N_ = maxnode-minnode_;

      neighbors_.resize(N_);
      nedge_bins_.resize(N_);

      for (std::size_t i=0;i<local_edges_.size();i++) {
        int64_t node = local_edges_[i].v0;
        int64_t neighbor = local_edges_[i].v1;
        if ( node != neighbor ) {
          neighbors_[node-minnode_].push_back(neighbor);
          neighbors_[neighbor-minnode_].push_back(node);
        }
      }
    }

    void point::root(std::vector<int64_t> const& bfs_roots)
    {
      bfs_roots_ = bfs_roots;

      parent_.resize(N_,bfs_roots.size(),1);
      duplicates_.resize(N_);
      // initialize to 0 -- no edge is identified as 0
      for (std::size_t j=0;j<parent_.jsize();j++) {
        for (std::size_t i=0;i<parent_.isize();i++) {
          parent_(i,j,0).parent = 0;
          parent_(i,j,0).level = 0;
        }
      }
    }

    void point::receive_duplicates(int64_t j,
                std::vector<hpx::naming::id_type> const& duplicate_components)
    {
      hpx::lcos::local_mutex::scoped_lock l(mtx_);
      duplicates_[j-minnode_] = duplicate_components;
      return;
    }

    bool point::has_edge(int64_t edge)
    {
      hpx::lcos::local_mutex::scoped_lock l(mtx_);
      bool found = false;
      for (std::size_t i=0;i<local_edges_.size();i++) {
        if ( edge == local_edges_[i].v0 || 
             edge == local_edges_[i].v1 ) {
          found = true;
          break;
        }
      }
      return found;
    }

    void point::bfs()
    {
      for (std::size_t step=0;step<bfs_roots_.size();step++) {
        int64_t root_node = bfs_roots_[step];

        if ( root_node - minnode_ >= (int64_t) parent_.isize() || root_node < minnode_ ) return; // the root node is not on this partition

        std::queue<int64_t> q;
        parent_(root_node-minnode_,step,0).parent = root_node;
        parent_(root_node-minnode_,step,0).level = 0;
        q.push(root_node);

        while (!q.empty()) {
          int64_t node = q.front(); q.pop();
  
          std::vector<int64_t> const& node_neighbors = neighbors_[node-minnode_];
          std::vector<int64_t>::const_iterator end = node_neighbors.end();
          for (std::vector<int64_t>::const_iterator it = node_neighbors.begin();
                       it != end; ++it)
          {
            int64_t& node_parent = parent_(*it-minnode_,step,0).parent;
            if (!node_parent) {
              node_parent = node;
              parent_(*it-minnode_,step,0).level = parent_(node-minnode_,step,0).level + 1;
              q.push(*it);
            }
          }
        }
      }
    }

    bool point::resolve_conflict_callback(std::size_t i,resolvedata r)
    {
      // if there is a dispute about a parent, pick the edge with the lowest level
      for (std::size_t i=0;i<bfs_roots_.size();i++) {
        if ( r.level[i] != 0 && r.level[i] < parent_(r.edge-minnode_,i,0).level ) {
          parent_(r.edge-minnode_,i,0).level = r.level[i];
          parent_(r.edge-minnode_,i,0).parent = r.parent[i];
        }
      }
      return true;
    }

    void point::resolve_conflict()
    {
      // go through each particle on this component; if there are duplicates (i.e. the
      // same particle is on a different component as well), communicate with those components
      // to resolve the controversy over who is the real parent
      typedef std::vector<hpx::lcos::promise< resolvedata > > lazy_results_type;
      lazy_results_type lazy_results;
      hpx::naming::id_type this_gid = get_gid(); 
      for (int64_t i=0;i< (int64_t) duplicates_.size();i++) {
        if ( duplicates_[i].size() > 1 && duplicates_[i][0] == this_gid ) {  
          for (std::size_t j=1;j<duplicates_[i].size();j++) {
            lazy_results.push_back( stubs::point::get_parent_async(duplicates_[i][j],i+minnode_) ); 
          }
        }
      } 
      hpx::lcos::wait(lazy_results,
           boost::bind(&point::resolve_conflict_callback, this, _1, _2));

    }

    void point::distributed_validate()
    {
      // the parent of the root is always itself
      for (std::size_t step=0;step<bfs_roots_.size();step++) {
        int64_t root_node = bfs_roots_[step];
        if ( root_node - minnode_ >= (int64_t) parent_.isize() || root_node < minnode_ ) continue; // the root node is not on this partition
        else {
          if ( parent_(root_node-minnode_,step,0).parent != root_node ) {
            std::cerr << " Validation for root " << root_node << " false; bfs_root parent is "
                      << parent_(root_node-minnode_,step,0).parent << std::endl;
          } 
        } 
      }

      // The correct parent of any duplicate is identified in the component duplicates_[i][0]
      //typedef std::vector<hpx::lcos::promise< resolvedata > > lazy_results_type;
      //lazy_results_type lazy_results;
      //hpx::naming::id_type this_gid = get_gid(); 
      //for (int64_t i=0;i< (int64_t) duplicates_.size();i++) {
      //  if ( duplicates_[i].size() > 0 && duplicates_[i][0] != this_gid ) {  
      //  }
      //} 

    }

    resolvedata point::get_parent(int64_t edge)
    {
      hpx::lcos::local_mutex::scoped_lock l(mtx_);
      resolvedata result;
      result.level.resize(bfs_roots_.size());
      result.parent.resize(bfs_roots_.size());
      for (std::size_t i=0;i<bfs_roots_.size();i++) {
        result.level[i] = parent_(edge-minnode_,i,0).level;
        result.parent[i] = parent_(edge-minnode_,i,0).parent;
      }
      result.edge = edge; 
      return result;
    }
}}

