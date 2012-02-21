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

        if ( root_node - minnode_ >= (int64_t) parent_.isize() ) return; // the root node is not on this partition

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
      std::cout << " TEST edge " << r.edge << " root " << r.root << std::endl;
      // if there is a dispute about a parent, pick the edge with the lowest level
      if ( r.level != 0 && r.level < parent_(r.edge-minnode_,r.root,0).level ) {
        parent_(r.edge-minnode_,r.root,0).level = r.level;
        parent_(r.edge-minnode_,r.root,0).parent = r.parent;
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
      for (int64_t i=0;i< (int64_t) duplicates_.size();i++) {
        if ( duplicates_[i].size() > 1 ) {  
          for (int64_t j=0;j< (int64_t) bfs_roots_.size();j++) {
            BOOST_FOREACH(hpx::naming::id_type const& gid, duplicates_[i])   
            {
              lazy_results.push_back( stubs::point::get_parent_async( gid,i+minnode_,j ) ); 
            }
          }
        }
      } 
      hpx::lcos::wait(lazy_results,
           boost::bind(&point::resolve_conflict_callback, this, _1, _2));

    }

    resolvedata point::get_parent(int64_t edge,int64_t root)
    {
      hpx::lcos::local_mutex::scoped_lock l(mtx_);
      resolvedata result;
      result.level = parent_(edge,root,0).level;
      result.parent = parent_(edge,root,0).parent;
      result.edge = edge; 
      result.root = root; 
      return result;
    }

    std::vector<nodedata> point::validate()
    {
      std::vector<nodedata> result;
#if 0
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
#endif
      return result;
    }

    validatedata point::scatter(std::vector<std::size_t> const& parent,std::size_t searchkey,
                                std::size_t scale)
    {
       validatedata result;
#if 0
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
#endif
       return result;
    }

}}

