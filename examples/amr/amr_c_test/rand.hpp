#ifndef _rand_hpp
#define _rand_hpp

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <iostream>

#define RNDMAPSZ 10000

extern long *work;
extern int zone, nzones;

double normicdf(double);
void initrand(long, char, double, double, int, int, int);

#endif // _rand_hpp
