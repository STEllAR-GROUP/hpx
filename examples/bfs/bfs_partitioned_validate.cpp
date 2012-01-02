//  Copyright (c) 2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include<iostream>
#include<vector>
#include<math.h>

static int compare_doubles(const void* a, const void* b) {
  double aa = *(const double*)a;
  double bb = *(const double*)b;
  return (aa < bb) ? -1 : (aa == bb) ? 0 : 1;
}

// this routine mirrors the matlab validation routine
int validate(std::vector<std::size_t> const& preorder_parent,
             std::vector<std::size_t> const& preorder_level,
             std::vector<std::size_t> const& preorder_parentindex,
             std::vector<std::size_t> const& nodelist,
             std::vector<std::size_t> const& neighborlist,
             std::size_t searchkey,std::size_t &num_edges) {
  // find the largest edge number
  // octave:  N = max (max (ij));
  std::size_t N = 0;
  for (std::size_t i=0;i<nodelist.size();i++) {
    if ( nodelist[i] > N ) N = nodelist[i];
    if ( neighborlist[i] > N ) N = neighborlist[i];
  }  
  N++;

  std::vector<std::size_t> parent;
  std::vector<std::size_t> levels;
  parent.resize( N );
  levels.resize( N );
  std::fill( parent.begin(),parent.end(),0 );
  //order the parent so that the indices correspond to the node parent it refers to
  for (std::size_t i=0;i<N;i++) {
    for (std::size_t j=0;j<preorder_parentindex.size();j++) {
      if ( i == preorder_parentindex[j] ) {
        parent[ i ] = preorder_parent[ j ];
        levels[ i ] = preorder_level[ j ];
      }
    }
  }

  // Get the number of edges for perfomance counting
  std::vector<std::size_t> nedge_bins;
  nedge_bins.resize(N);
  std::fill(nedge_bins.begin(),nedge_bins.end(),0);
  for (std::size_t i=0;i<nodelist.size();i++) {
    nedge_bins[nodelist[i] ] += 1;
    nedge_bins[neighborlist[i] ] += 1;
  }  

  num_edges = 0;
  for (std::size_t i=0;i<N;i++) {
    if ( parent[i] > 0 ) {
      num_edges += nedge_bins[i];
    }
  }
  // Volume/2
  num_edges = num_edges/2;

  if ( parent[searchkey] != searchkey ) {
    // the parent of the searchkey is always itself
    std::cout << " searchkey " << searchkey << " parent " << parent[searchkey] << std::endl;
    return 0;
  }  

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
    if ( k > N ) {
      // there is a cycle in the tree -- something wrong
      return -3;
    }
  } 

  // octave: lij = level (ij);
  std::vector<std::size_t> li,lj;
  li.resize(nodelist.size());
  lj.resize(nodelist.size());
  for (std::size_t i=0;i<nodelist.size();i++) {
    li[i] = level[ nodelist[i] ];
    lj[i] = level[ neighborlist[i] ];
  }

  // octave: neither_in = lij(1,:) == 0 & lij(2,:) == 0;
  // both_in = lij(1,:) > 0 & lij(2,:) > 0;
  std::vector<bool> neither_in,both_in;
  neither_in.resize(nodelist.size()); 
  both_in.resize(nodelist.size()); 
  for (std::size_t i=0;i<nodelist.size();i++) {
    if ( li[i] == 0 && lj[i] == 0 ) neither_in[i] = true;
    if ( li[i] > 0 && lj[i] > 0 ) both_in[i] = true;
  }

  // octave: 
  //  if any (not (neither_in | both_in)),
  //  out = -4;
  //  return
  //end
  for (std::size_t i=0;i<nodelist.size();i++) {
    if ( !(neither_in[i] || both_in[i] ) ) {
      return -4;
    }
  }

  // octave: respects_tree_level = abs (lij(1,:) - lij(2,:)) <= 1;
  std::vector<bool> respects_tree_level;
  respects_tree_level.resize( nodelist.size() );
  for (std::size_t i=0;i<nodelist.size();i++) {
    if ( abs( (int) (li[i] - lj[i]) ) <= 1 ) respects_tree_level[i] = true;
    else respects_tree_level[i] = false;
  }

  // octave:
  // if any (not (neither_in | respects_tree_level)),
  //  out = -5;
  //  return
  for (std::size_t i=0;i<nodelist.size();i++) {
    if ( !(neither_in[i] || respects_tree_level[i] ) ) {
      return -5;
    }
  }
  
  return 1;
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
