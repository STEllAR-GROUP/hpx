//  Copyright (c) 2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/lcos/eager_future.hpp>
#include <hpx/lcos/async_future_wait.hpp>

#include "../make_graph.h"
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

        std::vector<std::size_t> edges;
        edges.resize(size);

        bool found;
        std::size_t edge;
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
      std::size_t maxnode = 0;
      minnode_ = 99999;
      for (std::size_t i=0;i<local_edges_.size();i++) {
        if ( (std::size_t) local_edges_[i].v0 > maxnode ) maxnode = local_edges_[i].v0;
        if ( (std::size_t) local_edges_[i].v1 > maxnode ) maxnode = local_edges_[i].v1;

        if ( (std::size_t) local_edges_[i].v0 < minnode_ ) minnode_ = local_edges_[i].v0;
        if ( (std::size_t) local_edges_[i].v1 < minnode_ ) minnode_ = local_edges_[i].v1;
      }
      maxnode++;
      std::size_t N = maxnode-minnode_;

      neighbors_.resize(N);
      nedge_bins_.resize(N);

      for (std::size_t i=0;i<local_edges_.size();i++) {
        std::size_t node = local_edges_[i].v0;
        std::size_t neighbor = local_edges_[i].v1;
        if ( node != neighbor ) {
          neighbors_[node-minnode_].push_back(neighbor);
          neighbors_[neighbor-minnode_].push_back(node);
        }
      }

      parent_.resize(N);
      // initialize to 0 -- no edge is identified as 0
      for (std::size_t i=0;i<parent_.size();i++) {
        parent_[i].parent = 0;
        parent_[i].level = 0;
      }
    }

    bool point::has_edge(std::size_t edge)
    {
      bool found = false;
      for (std::size_t i=0;i<local_edges_.size();i++) {
        if ( edge == (std::size_t) local_edges_[i].v0 || 
             edge == (std::size_t) local_edges_[i].v1 ) {
          found = true;
          break;
        }
      }
      return found;
    }

    void point::bfs(std::size_t root_node)
    {
      if ( root_node - minnode_ >= parent_.size() ) return; // the root node is not on this partition

      std::queue<std::size_t> q;
      parent_[root_node-minnode_].parent = root_node;
      parent_[root_node-minnode_].level = 0;
      q.push(root_node);

      while (!q.empty()) {
        std::size_t node = q.front(); q.pop();

        std::vector<std::size_t> const& node_neighbors = neighbors_[node-minnode_];
        std::vector<std::size_t>::const_iterator end = node_neighbors.end();
        for (std::vector<std::size_t>::const_iterator it = node_neighbors.begin();
                     it != end; ++it)
        {
          std::size_t& node_parent = parent_[*it-minnode_].parent;
          if (!node_parent) {
            node_parent = node;
            parent_[*it-minnode_].level = parent_[node-minnode_].level + 1;
            q.push(*it);
          }
        }
      }
    }

    void point::reset()
    {
      for (std::size_t i=0;i<parent_.size();i++) {
        parent_[i].parent = 0;
        parent_[i].level = 0;
      }
    }

    std::vector<nodedata> point::validate()
    {
      std::vector<nodedata> result;
      nodedata tmp;
      for (std::size_t i=0;i<local_edges_.size();i++) {
        std::size_t node0 = local_edges_[i].v0;
        std::size_t node1 = local_edges_[i].v1;
        if ( parent_[node0-minnode_].parent != 0 ) {
          tmp.node = node0;
          tmp.parent = parent_[node0-minnode_].parent;
          tmp.level = parent_[node0-minnode_].level;

          result.push_back(tmp);
        }
        if ( parent_[node1-minnode_].parent != 0 ) {
          tmp.node = node1;
          tmp.parent = parent_[node1-minnode_].parent;
          tmp.level = parent_[node1-minnode_].level;

          result.push_back(tmp);
        }
      }
      return result;
    }

    validatedata point::scatter(std::vector<std::size_t> const& parent,std::size_t searchkey,
                                std::size_t scale)
    {
       validatedata result;
       // Get the number of edges for performance counting
       std::fill(nedge_bins_.begin(),nedge_bins_.end(),0);

       for (std::size_t i=0;i<local_edges_.size();i++) {
         std::size_t node = local_edges_[i].v0;
         std::size_t neighbor = local_edges_[i].v1;
         if ( node != neighbor ) {
           nedge_bins_[node-minnode_] += 1;
           nedge_bins_[neighbor-minnode_] += 1;
         }
       }

       std::size_t num_edges = 0;
       for (std::size_t i=0;i<nedge_bins_.size();i++) {
         if ( parent[i + minnode_] > 0 ) {
           num_edges += nedge_bins_[i];  
         }
       }

       // Volume/2
       num_edges = num_edges/2;
       
       result.num_edges = num_edges;

       // Find the indices of the nodeparents list that are nonzero
       // octave: level = zeros (size (parent));
       // octave: level (slice) = 1;
       std::vector<std::size_t> slice,level;
       level.resize( parent.size() );
       for (std::size_t i=0;i<parent.size();i++) {
         if ( parent[i] > 0 ) {
           slice.push_back(i);
           level[i] = 1;
         } else {
           level[i] = 0;
         }
       }

       // octave: P = parent (slice);
       std::vector<std::size_t> P;
       P.resize( slice.size() );
       for (std::size_t i=0;i<slice.size();i++) {
         P[i] = parent[ slice[i] ];
       }
  
       std::vector<bool> mask;
       mask.resize(slice.size());

       // fill the mask with zeros
       std::fill( mask.begin(),mask.end(),false);

       // Define a mask
       // octave:  mask = P != search_key;
       for (std::size_t i=0;i<slice.size();i++) {
         if ( P[i] != searchkey ) {
           mask[i] = true;
         }
       }

       std::size_t k = 0;

       int64_t N = INT64_C(16) << scale;
       N++; 

       // octave:  while any (mask)
       bool keep_going = false;
       while (1) {
         // check if there are any nonzero entries in mask  
         keep_going = false;
         for (std::size_t i=0;i<mask.size();i++) {
           if ( mask[i] == true ) {
             keep_going = true;
             break;
           }
         }
         if ( keep_going == false ) break;

         // octave:  level(slice(mask)) = level(slice(mask)) + 1;
         for (std::size_t i=0;i<slice.size();i++) {
           if ( mask[i] ) {
             level[ slice[i] ] += 1;
           }
         }

         // octave:  P = parent(P)
         for (std::size_t i=0;i<P.size();i++) {
           P[i] = parent[ P[i] ];
         }

         for (std::size_t i=0;i<P.size();i++) {
           if ( P[i] != searchkey ) mask[i] = true;
           else mask[i] = false;
         }

         k++;
         if ( k > (std::size_t) N ) {
           // there is a cycle in the tree -- something wrong
           result.rc = -3;
           return result;
         }
       }

       // octave: lij = level (ij);
       std::vector<std::size_t> li,lj;
       li.resize(local_edges_.size());
       lj.resize(local_edges_.size());
       for (std::size_t i=0;i<local_edges_.size();i++) {
         std::size_t node = local_edges_[i].v0;
         std::size_t neighbor = local_edges_[i].v1;
         li[i] = level[node];
         lj[i] = level[neighbor];
       }

       // octave: neither_in = lij(1,:) == 0 & lij(2,:) == 0;
       // both_in = lij(1,:) > 0 & lij(2,:) > 0;
       std::vector<bool> neither_in,both_in;
       neither_in.resize(local_edges_.size());
       both_in.resize(local_edges_.size());
       for (std::size_t i=0;i<local_edges_.size();i++) {
         if ( li[i] == 0 && lj[i] == 0 ) neither_in[i] = true;
         if ( li[i] > 0 && lj[i] > 0 ) both_in[i] = true;
       }

       // octave: 
       //  if any (not (neither_in | both_in)),
       //  out = -4;
       //  return
       //end
       for (std::size_t i=0;i<local_edges_.size();i++) {
         if ( !(neither_in[i] || both_in[i] ) ) {
#if 0
           std::cout << " TEST scatter " << li[i] << " " << lj[i] << std::endl;
           std::size_t node = local_edges_[i].v0;
           std::size_t neighbor = local_edges_[i].v1;
           std::cout << " TEST node " << node << " neighbor " << neighbor << std::endl;
           // let's examine the neighbors
           for ( std::size_t j=0;j<neighbors_[node-minnode_].size();j++) {
             std::cout << " TEST neighbors of node " << node << " : " << neighbors_[node-minnode_][j] << std::endl;
           }
           std::cout << " TEST B " << parent[neighbor]  << std::endl;
           std::cout << " TEST C " << parent_[neighbor-minnode_].parent  << std::endl;
           std::cout << " TEST D " << parent_[neighbor-minnode_].level  << std::endl;
           std::cout << " TEST E " << parent[node]  << std::endl;
           std::cout << " TEST F " << parent_[node-minnode_].parent  << std::endl;
           std::cout << " TEST G " << parent_[node-minnode_].level  << std::endl;
           std::cout << " TEST H " << parent[searchkey]  << std::endl;
           std::cout << " TEST I " << parent_[searchkey-minnode_].parent  << std::endl;
           std::cout << " TEST J " << parent_[searchkey-minnode_].level  << std::endl;
#endif
           result.rc = -4;
           return result;
         }
       }

       // octave: respects_tree_level = abs (lij(1,:) - lij(2,:)) <= 1;
       std::vector<bool> respects_tree_level;
       respects_tree_level.resize( local_edges_.size() );
       for (std::size_t i=0;i<local_edges_.size();i++) {
         if ( abs( (int) (li[i] - lj[i]) ) <= 1 ) respects_tree_level[i] = true;
         else respects_tree_level[i] = false;
       }

       // octave:
       // if any (not (neither_in | respects_tree_level)),
       //  out = -5;
       //  return
       for (std::size_t i=0;i<local_edges_.size();i++) {
         if ( !(neither_in[i] || respects_tree_level[i] ) ) {
           result.rc = -5;
           return result;
         }
       }

       result.rc = 0;
       return result;
    }

}}

