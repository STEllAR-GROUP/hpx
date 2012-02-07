//  Copyright (c) 2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include<iostream>
#include<vector>
#include<math.h>
#include "graph500/point.hpp"

static int compare_doubles(const void* a, const void* b) {
  double aa = *(const double*)a;
  double bb = *(const double*)b;
  return (aa < bb) ? -1 : (aa == bb) ? 0 : 1;
}

void get_statistics(std::vector<double> const& x, double &minimum, double &mean, double &stdev, double &firstquartile,
                                                  double &median, double &thirdquartile, double &maximum)
{
  // Compute mean
  double temp = 0.0;
  std::size_t n = x.size();
  for (std::size_t i=0;i<n;i++) temp += x[i];
  temp /= n;
  mean = temp;

  // Compute std dev
  temp = 0.0;
  for (std::size_t i=0;i<n;i++) temp += (x[i] - mean)*(x[i]-mean);
  temp /= n-1;
  stdev = sqrt(temp);

  // Sort x
  std::vector<double> xx;    
  xx.resize(n);
  for (std::size_t i=0;i<n;i++) {
    xx[i] = x[i];
  }
  qsort(&*xx.begin(),n,sizeof(double),compare_doubles);
  minimum = xx[0];
  firstquartile = (xx[(n - 1) / 4] + xx[n / 4]) * .5;
  median = (xx[(n - 1) / 2] + xx[n / 2]) * .5;
  thirdquartile = (xx[n - 1 - (n - 1) / 4] + xx[n - 1 - n / 4]) * .5; 
  maximum = xx[n - 1];
};

void clean_up(std::vector<nodedata> &full,std::size_t scale,std::vector<std::size_t> &parent) 
{
  int64_t nglobalverts = (int64_t)(1) << scale;

  // find duplicates and resolve discrepancies
  std::vector< std::vector<leveldata> >  nodes;
  // add one since we increment all edges by one so that zero remains special
  nodes.resize(nglobalverts+1);
  leveldata tmp;
  for (std::size_t i=0;i<full.size();i++) {
    std::size_t node = full[i].node;
    tmp.parent = full[i].parent;
    tmp.level = full[i].level;
    nodes[node].push_back(tmp); 
  }

  // some nodes will list multiple parents; resolve the discrepancy (if any) by
  // choosing the parent corresponding to the smallest level 
  parent.resize(nglobalverts+1);
  for (std::size_t i=0;i<nodes.size();i++) {
    if ( nodes[i].size() == 0 ) {
      // this node never visited
      parent[i] = 0;
    } else if ( nodes[i].size() == 1 ) {
      // just one answer
      parent[i] = nodes[i][0].parent;
    } else {
      // there are multiple responses -- check them
      std::size_t ld = 0;
      std::size_t ll = 0;
      for (std::size_t j=0;j<nodes[i].size();j++) {
        if ( nodes[i][j].parent != 0 && ld == 0 ) {
          ld = nodes[i][j].parent;
          ll = nodes[i][j].level;
        }
        if ( ld != 0 && ld != nodes[i][j].parent ) {
          // multiple opinions for the opinion -- distinguish them by level
          if ( ll > nodes[i][j].level ) {
            ld = nodes[i][j].parent;
            ll = nodes[i][j].level;
          }
          //std::cout << " Multiple opinion for:  i " << i << " ld " << ld << " " << nodes[i][j] << std::endl;
        }
      } 
      parent[i] = ld;
    }
  }
}
