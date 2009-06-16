//  Copyright (c) 2009 Maciej Brodowicz
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _rand_hpp
#define _rand_hpp

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <iostream>
#include <boost/random/linear_congruential.hpp>
#include <hpx/config/export_definitions.hpp>

#define RNDMAPSZ 10000

extern long *work;
extern int zone, nzones;
extern boost::rand48 random_numbers;

double normicdf(double);
HPX_COMPONENT_EXPORT void initrand(long, char, double, double, int, int, int);

#endif // _rand_hpp
