//  Copyright (c) 2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <fstream>
#include <iostream>
#include <vector>
#include <queue>
#include <boost/lexical_cast.hpp>
#include <math.h>
#include <hpx/util/high_resolution_timer.hpp>

bool fexists(std::string const filename)
{
  std::ifstream ifile(filename.c_str());
  return ifile ? true : false;
}

void get_statistics(std::vector<double> const& x, double &minimum, double &mean,
                    double &stdev, double &firstquartile,
                    double &median, double &thirdquartile, double &maximum);

// this routine mirrors the matlab validation routine
int validate(std::vector<std::size_t> const& parent,
             std::vector<std::size_t> const& levels,
             std::vector<std::size_t> const& nodelist,
             std::vector<std::size_t> const& neighborlist,
             std::size_t searchkey,std::size_t &num_edges);

// void depth_traverse(std::size_t level,std::size_t parent,std::size_t edge,
//                     std::vector<char> & visited_,
//                     std::vector<std::size_t> & level_,
//                     std::vector<std::size_t> & parent_,
//                     std::vector<std::vector<std::size_t> > const& neighbors,
//                     std::size_t max_level)
// {
//   if ( visited_[edge] == false || level_[edge] > level ) {
//     visited_[edge] = true;
//     parent_[edge] = parent;
//     level_[edge] = level;
//     if ( level < max_level ) {
//       for (std::size_t i=0;i<neighbors[edge].size();i++) {
//         std::size_t neighbor = neighbors[edge][i];
//         depth_traverse(level+1,edge,neighbor,visited_,level_,parent_,neighbors,max_level);
//       }
//     }
//   }
//   return;
// }

void bfs(std::size_t root,
         std::vector<std::size_t> & parent,
         std::vector<std::vector<std::size_t> > const& neighbors)
{
  std::queue<std::size_t> q;

  parent[root] = root;
  q.push(root);

  while (!q.empty()) {
    std::size_t node = q.front(); q.pop();

    std::vector<std::size_t> const& node_neighbors = neighbors[node];
    std::vector<std::size_t>::const_iterator end = node_neighbors.end();
    for (std::vector<std::size_t>::const_iterator it = node_neighbors.begin();
         it != end; ++it)
    {
      std::size_t& node_parent = parent[*it];
      if (!node_parent) {
        node_parent = node;
        q.push(*it);
      }
    }
  }
}

int main() {

  // Read in the graph
  // Verify the input files are available for reading
  bool rc;
  std::string graphfile = "g10.txt";
  std::string searchfile = "g10_search.txt";

  rc = fexists(graphfile);
  if ( !rc ) {
    std::cerr << " File " << graphfile << " not found! Exiting... " << std::endl;
    return 0;
  }
  rc = fexists(searchfile);
  if ( !rc ) {
    std::cerr << " File " << searchfile << " not found! Exiting... " << std::endl;
    return 0;
  }

  std::vector<std::size_t> nodelist,neighborlist;
  {
    std::ifstream myfile(graphfile.c_str());
    if (myfile.is_open()) {
      std::size_t node, neighbor;
      while (myfile >> node >> neighbor) {
        // increment all nodes and neighbors by 1; the smallest edge number is 1
        // edge 0 is reserved for the parent of the root and for unvisited edges
        nodelist.push_back(node+1);
        neighborlist.push_back(neighbor+1);
      }
    }
  }

  // read in the searchfile containing the root vertices to search -- timing not reported
  std::vector<std::size_t> searchroot;
  {
    std::ifstream myfile(searchfile.c_str());
    if (myfile.is_open()) {
      std::size_t root;
      while (myfile >> root) {
          // increment all nodes and neighbors by 1; the smallest edge number is 1
          // edge 0 is reserved for the parent of the root and for unvisited edges
          searchroot.push_back(root+1);
      }
    }
  }

  // Find the biggest value
  std::size_t N = 0;
  for (std::size_t i=0;i<nodelist.size();i++) {
    if ( nodelist[i] > N ) N = nodelist[i];
    if ( neighborlist[i] > N ) N = neighborlist[i];
  }
  N++;

  std::vector< std::vector<std::size_t> > neighbors;
  std::vector<std::size_t> level,parent;
  std::vector<char> visited;
  neighbors.resize(N);

  for (std::size_t i=0;i<nodelist.size();i++) {
    std::size_t node = nodelist[i];
    std::size_t neighbor = neighborlist[i];
    if ( node != neighbor ) {
      neighbors[node].push_back(neighbor);
      neighbors[neighbor].push_back(node);
    }
  }

  level.resize(N);
  parent.resize(N);
  visited.resize(N);
  std::fill( visited.begin(),visited.end(),false);
  std::fill( level.begin(),level.end(),0);
  std::fill( parent.begin(),parent.end(),0);

  std::vector<double> kernel2_time;
  std::vector<double> kernel2_nedge;
  kernel2_time.resize(searchroot.size());
  kernel2_nedge.resize(searchroot.size());

  std::size_t max_level = 10;
  std::size_t zero = 0;
  // go through each root position
  for (std::size_t step=0;step<searchroot.size();step++) {
    // time this
    hpx::util::high_resolution_timer t1;
//     depth_traverse(zero,searchroot[step],searchroot[step],visited,level,parent,neighbors,max_level);
    bfs(searchroot[step], /*visited, */parent, neighbors);
    kernel2_time[step] = t1.elapsed();
    // end timing

    // Validate
    bool validation = true;
    std::size_t num_edges;
    int rc = validate(parent,level,nodelist,
                      neighborlist,searchroot[step],num_edges);
    kernel2_nedge[step] = (double) num_edges;

    if ( rc <= 0 ) {
      validation = false;
      std::cout << " Validation failed for searchroot " << searchroot[step] << " rc " << rc << std::endl;
      break;
    }

    // reset -- not timed
    std::fill( visited.begin(),visited.end(),false);
    std::fill( level.begin(),level.end(),0);
    std::fill( parent.begin(),parent.end(),0);
  }

  double minimum,mean,stdev,firstquartile,median,thirdquartile,maximum;
  get_statistics(kernel2_time,minimum,mean,stdev,firstquartile,
                         median,thirdquartile,maximum);

  double n_min,n_mean,n_stdev,n_firstquartile,n_median,n_thirdquartile,n_maximum;
  get_statistics(kernel2_nedge,n_min,n_mean,n_stdev,n_firstquartile,
                         n_median,n_thirdquartile,n_maximum);

  std::cout << " min_time:             " << minimum << std::endl;
  std::cout << " firstquartile_time:   " << firstquartile << std::endl;
  std::cout << " median_time:          " << median << std::endl;
  std::cout << " thirdquartile_time:   " << thirdquartile << std::endl;
  std::cout << " max_time:             " << maximum << std::endl;
  std::cout << " mean_time:            " << mean << std::endl;
  std::cout << " stddev_time:          " << stdev << std::endl;

  std::cout << " min_nedge:            " << n_min << std::endl;
  std::cout << " firstquartile_nedge:  " << n_firstquartile << std::endl;
  std::cout << " median_nedge:         " << n_median << std::endl;
  std::cout << " thirdquartile_nedge:  " << n_thirdquartile << std::endl;
  std::cout << " max_nedge:            " << n_maximum << std::endl;
  std::cout << " mean_nedge:           " << n_mean << std::endl;
  std::cout << " stddev_nedge:         " << n_stdev << std::endl;

  std::vector<double> TEPS;
  TEPS.resize(kernel2_nedge.size());
  for (std::size_t i=0;i<kernel2_nedge.size();i++) {
    TEPS[i] = kernel2_nedge[i]/kernel2_time[i];
  }
  std::size_t NN = TEPS.size();
  double TEPS_min,TEPS_mean,TEPS_stdev,TEPS_firstquartile,TEPS_median,TEPS_thirdquartile,TEPS_maximum;
  get_statistics(TEPS,TEPS_min,TEPS_mean,TEPS_stdev,TEPS_firstquartile,
                 TEPS_median,TEPS_thirdquartile,TEPS_maximum);

  // Harmonic standard deviation from:
  // Nilan Norris, The Standard Errors of the Geometric and Harmonic
  // Means and Their Application to Index Numbers, 1940.
  // http://www.jstor.org/stable/2235723
  TEPS_stdev = TEPS_stdev/(TEPS_mean*TEPS_mean*sqrt( (double) (NN-1) ) );

  std::cout << " min_TEPS:            " << TEPS_min << std::endl;
  std::cout << " firstquartile_TEPS:  " << TEPS_firstquartile << std::endl;
  std::cout << " median_TEPS:         " << TEPS_median << std::endl;
  std::cout << " thirdquartile_TEPS:  " << TEPS_thirdquartile << std::endl;
  std::cout << " max_TEPS:            " << TEPS_maximum << std::endl;
  std::cout << " harmonic_mean_TEPS:  " << TEPS_mean << std::endl;
  std::cout << " harmonic_stddev_TEPS:" << TEPS_stdev << std::endl;


  return 0;
}
