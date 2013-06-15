//  Copyright (c) 2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include<iostream>
#include<vector>
#include<math.h>

#include <hpx/hpx_fwd.hpp>
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
