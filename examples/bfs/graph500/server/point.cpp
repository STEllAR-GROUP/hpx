//  Copyright (c) 2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/lcos/future_wait.hpp>
#include <hpx/include/async.hpp>

#include "../make_graph.hpp"
#include "../stubs/point.hpp"
#include "point.hpp"

#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/random/linear_congruential.hpp>

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

      boost::rand48 random_numbers;
      random_numbers.seed(boost::int64_t(objectid));

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
            edge = random_numbers() % end_idx;
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

    bool point::findwhohasthisedge_callback(int64_t j,std::vector<bool> const& has_edge,
                     std::vector<hpx::naming::id_type> const& point_components,
                     int64_t start)
    {
      // let the components which have duplicates know about each other
      std::vector<hpx::naming::id_type> duplicate;
      std::vector<std::size_t> duplicateid;
      for ( std::size_t i=0;i<has_edge.size();i++) {
        if ( has_edge[i] == true ) {
          duplicate.push_back(point_components[i]);
          duplicateid.push_back(i);
        }
      }
      
      std::vector<hpx::lcos::future<void> > send_duplicates_phase;
      for (std::size_t i=0;i<duplicate.size();i++) {
        // the edge number in question is j + start
        send_duplicates_phase.push_back(
               stubs::point::receive_duplicates_async(duplicate[i],j+start,
                                                     duplicate,duplicateid));
      }
      hpx::lcos::wait(send_duplicates_phase);

      return true;
    }

    std::vector<bool> point::findwhohasthisedge(int64_t edge,
           std::vector<hpx::naming::id_type> const& point_components)
    {
      std::vector<bool> search_vector;
      std::vector<hpx::lcos::future<bool> > has_edge_phase;
      for (std::size_t i=0;i<point_components.size();i++) {
        has_edge_phase.push_back(stubs::point::has_edge_async(point_components[i],edge));
      }
      hpx::lcos::wait(has_edge_phase,search_vector);

      return search_vector;
    }

    void point::ppedge(int64_t start, int64_t stop,
                  std::vector<hpx::naming::id_type> const& point_components)
    {
      typedef std::vector<hpx::lcos::future< std::vector<bool> > > lazy_results_type;
      lazy_results_type lazy_results;

      for (int64_t edge=start;edge<=stop;edge++) {
        lazy_results.push_back(stubs::point::findwhohasthisedge_async(point_components[idx_],edge,point_components));
      }
      // put a callback here instead of search_vector
      hpx::lcos::wait(lazy_results,boost::bind(&point::findwhohasthisedge_callback,
                              this,_1,_2,boost::ref(point_components),boost::ref(start)));
    }

    void point::root(std::vector<int64_t> const& bfs_roots)
    {
      bfs_roots_ = bfs_roots;

      parent_.resize(N_,bfs_roots.size(),1);
      duplicates_.resize(N_);
      duplicatesid_.resize(N_);
      // initialize to 0 -- no edge is identified as 0
      for (std::size_t j=0;j<parent_.jsize();j++) {
        for (std::size_t i=0;i<parent_.isize();i++) {
          parent_(i,j,0).parent = 0;
          parent_(i,j,0).level = 0;
        }
      }
    }

    void point::receive_duplicates(int64_t j,
                std::vector<hpx::naming::id_type> const& duplicate_components,
                std::vector<std::size_t> const& duplicateid)
    {
      hpx::lcos::local::mutex::scoped_lock l(mtx_);
      duplicates_[j-minnode_] = duplicate_components;
      duplicatesid_[j-minnode_] = duplicateid;
      return;
    }

    bool point::has_edge(int64_t edge)
    {
      hpx::lcos::local::mutex::scoped_lock l(mtx_);
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

        if ( root_node - minnode_ >= (int64_t) parent_.isize() || root_node < minnode_ ) continue; // the root node is not on this partition

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

    bool point::resolve_conflict_callback(std::size_t j,resolvedata r)
    {
      // if there is a dispute about a parent, pick the edge with the lowest level
      for (std::size_t i=0;i<bfs_roots_.size();i++) {
        if ( r.level[i] != 0 && 
             ( r.level[i] < parent_(r.edge-minnode_,i,0).level ||
               parent_(r.edge-minnode_,i,0).level == 0 )
           ) {
          parent_(r.edge-minnode_,i,0).level = r.level[i];
          parent_(r.edge-minnode_,i,0).parent = r.parent[i];
        }
      }
      return true;
    }

    void point::resolve_conflict()
    {
      typedef std::vector<hpx::lcos::future< resolvedata > > lazy_results_type;
      lazy_results_type lazy_results;
      // go through each particle on this component; if there are duplicates (i.e. the
      // same particle is on a different component as well), communicate with those components
      // to resolve the controversy over who is the real parent
      for (int64_t i=0;i< (int64_t) duplicates_.size();i++) {
        if ( duplicates_[i].size() > 1 && duplicatesid_[i][0] == idx_ ) {  
          for (std::size_t j=1;j<duplicates_[i].size();j++) {
            hpx::naming::id_type id = duplicates_[i][j];
            lazy_results.push_back( stubs::point::get_parent_async(id,i+minnode_) ); 
          }
        }
      }
      hpx::lcos::wait(lazy_results,
           boost::bind(&point::resolve_conflict_callback, this, _1, _2));

    }

    std::vector<int> point::distributed_validate(std::size_t scale)
    {
      // the parent of the root is always itself
      std::vector<int> rc;
      rc.resize(bfs_roots_.size());
      for (std::size_t step=0;step<bfs_roots_.size();step++) {
        // initialize the return code
        rc[step] = 1;

        int64_t root_node = bfs_roots_[step];
        if ( root_node - minnode_ >= (int64_t) parent_.isize() || root_node < minnode_ ) continue; // the root node is not on this partition
        else {
          if ( parent_(root_node-minnode_,step,0).parent != root_node ) {
            //std::cerr << " Validation for root " << root_node << " false; bfs_root parent is "
            //          << parent_(root_node-minnode_,step,0).parent << std::endl;
            if ( rc[step] == 1 ) rc[step] = -1;
          } 
        } 

        if ( rc[step] < 0 ) continue;

        // octave: lij = level (ij);
        std::vector<std::size_t> li,lj;
        li.resize(local_edges_.size());
        lj.resize(local_edges_.size());
        for (std::size_t i=0;i<local_edges_.size();i++) {
          int64_t node = local_edges_[i].v0;
          int64_t neighbor = local_edges_[i].v1;
          li[i] = parent_(node-minnode_,step,0).level;
          lj[i] = parent_(neighbor-minnode_,step,0).level;

          // validation assumes level >= 1 (it reserves level 0 for unvisited edges)
          // We have level 0 as the root_node also.  So change the level of any root
          // node to be 1

          if ( node == root_node ) li[i] = 1;
          if ( neighbor == root_node ) lj[i] = 1;

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
           // std::cerr << " Validation step " << step << " failed " << -4 << std::endl;
           // std::cerr << " li " << li[i] << " lj " << lj[i] << std::endl;
           // std::cerr << " v0 " << local_edges_[i].v0 << " v1 " << local_edges_[i].v1 << " root " << root_node << std::endl;
           // std::cerr << " duplicates v0 " << duplicates_[local_edges_[i].v0-minnode_].size() << std::endl;
           // std::cerr << " duplicates v1 " << duplicates_[local_edges_[i].v1-minnode_].size() << std::endl;
            if ( rc[step] == 1 ) rc[step] = -4;
          }
        }
 
        if ( rc[step] < 0 ) continue;

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
            //std::cerr << " Validation step " << step << " failed " << -5 << std::endl;
            //std::cerr << " li " << li[i] << " lj " << lj[i] << std::endl;
            //std::cerr << " v0 " << local_edges_[i].v0 << " v1 " << local_edges_[i].v1 << " root " << root_node << std::endl;
            //std::cerr << " duplicates v0 " << duplicates_[local_edges_[i].v0-minnode_].size() << std::endl;
            //std::cerr << " duplicates v1 " << duplicates_[local_edges_[i].v1-minnode_].size() << std::endl;
            if ( rc[step] == 1 ) rc[step] = -5;
          }
        }

        if ( rc[step] == 1 ) rc[step] = 0;
      }
      return rc;
    }

    bool point::get_numedges_callback(std::size_t j,resolvedata r)
    {
      // if there is a dispute about a parent, pick the edge with the lowest level
      for (std::size_t i=0;i<bfs_roots_.size();i++) {
        parent_(r.edge-minnode_,i,0).level = r.level[i];
        parent_(r.edge-minnode_,i,0).parent = r.parent[i];
      }
      return true;
    }

    std::vector<int64_t> point::get_numedges()
    {
      typedef std::vector<hpx::lcos::future< resolvedata > > lazy_results_type;
      lazy_results_type lazy_results;
      std::vector<int64_t> num_edges;
      // Get the number of edges for performance counting
      num_edges.resize(bfs_roots_.size());
      std::fill(num_edges.begin(),num_edges.end(),0);

      for (std::size_t i=0;i<local_edges_.size();i++) {
        std::size_t node = local_edges_[i].v0;
        std::size_t neighbor = local_edges_[i].v1;
        if ( node != neighbor ) {
          nedge_bins_[node-minnode_] += 1;
          nedge_bins_[neighbor-minnode_] += 1;
        }
      }

      for (std::size_t step=0;step<bfs_roots_.size();step++) {
        for (std::size_t i=0;i<nedge_bins_.size();i++) {
          if ( parent_(i,step,0).parent > 0 ) {
            num_edges[step] += nedge_bins_[i];
          }
        }
        // Volume/2
        num_edges[step] = num_edges[step]/2;
      }

      // This method also ensures the duplicate parent information is found
      // on the first number_partitions components
      for (int64_t i=0;i< (int64_t) duplicates_.size();i++) {
        if ( duplicates_[i].size() > 1 && duplicatesid_[i][0] != idx_ ) {
          hpx::naming::id_type id = duplicates_[i][0];
          lazy_results.push_back( stubs::point::get_parent_async(id,i+minnode_) );
        }
      }
      hpx::lcos::wait(lazy_results,
           boost::bind(&point::get_numedges_callback, this, _1, _2));

      return num_edges;
    }

    resolvedata point::get_parent(int64_t edge)
    {
      hpx::lcos::local::mutex::scoped_lock l(mtx_);
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

