//  Copyright (c) 2009 Maciej Brodowicz
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "rand.hpp"

long *work = 0;
int zone = 0, nzones = 1;

boost::rand48 random_numbers;

double normicdf(double p)
{
#define  A1  (-3.969683028665376e+01)
#define  A2   2.209460984245205e+02
#define  A3  (-2.759285104469687e+02)
#define  A4   1.383577518672690e+02
#define  A5  (-3.066479806614716e+01)
#define  A6   2.506628277459239e+00

#define  B1  (-5.447609879822406e+01)
#define  B2   1.615858368580409e+02
#define  B3  (-1.556989798598866e+02)
#define  B4   6.680131188771972e+01
#define  B5  (-1.328068155288572e+01)

#define  C1  (-7.784894002430293e-03)
#define  C2  (-3.223964580411365e-01)
#define  C3  (-2.400758277161838e+00)
#define  C4  (-2.549732539343734e+00)
#define  C5   4.374664141464968e+00
#define  C6   2.938163982698783e+00

#define  D1   7.784695709041462e-03
#define  D2   3.224671290700398e-01
#define  D3   2.445134137142996e+00
#define  D4   3.754408661907416e+00

#define P_LOW   0.02425
/* P_HIGH = 1 - P_LOW */
#define P_HIGH  0.97575

    double x(0), q, r/*, u, e*/;

    if ((0 < p )  && (p < P_LOW))
    {
        q = sqrt(-2*log(p));
        x = (((((C1*q+C2)*q+C3)*q+C4)*q+C5)*q+C6)/((((D1*q+D2)*q+D3)*q+D4)*q+1);
    }
    else
    {
        if ((P_LOW <= p) && (p <= P_HIGH))
        {
            q = p - 0.5;
            r = q*q;
            x = (((((A1*r+A2)*r+A3)*r+A4)*r+A5)*r+A6)*q/(((((B1*r+B2)*r+B3)*r+B4)*r+B5)*r+1);
        }
        else
        {
            if ((P_HIGH < p) && (p < 1))
            {
                q = sqrt(-2*log(1-p));
                x = -(((((C1*q+C2)*q+C3)*q+C4)*q+C5)*q+C6)/((((D1*q+D2)*q+D3)*q+D4)*q+1);
            }
        }
    }

    return x;
}

void initrand(long seed, char dist, double mean, double stddev, int iters, int points, int nthr)
{
    nzones = nthr; 
    zone = (std::max)(1, points/nzones);
    random_numbers.seed(boost::int32_t(seed));

    int i, npos = 0, allz = iters*nzones;
    work = new long[allz];
    if (dist == 'u')
    { /* uniform distribution */
        double h = sqrt(3.0)*stddev;
    
        for (i = 0; i < allz; i++)
        {
            work[i] = long(mean-h+random_numbers()*(2*h));
            if (work[i] <= 0)
            {
                work[i] = 1; ++npos;
            }
        }
    }
    else if (dist == 'n')
    { /* normal distribution */
        for (i = 0; i < allz; i++)
        {
            work[i] = long(mean+stddev*normicdf(((double)i+0.5)/RNDMAPSZ));
            if (work[i] <= 0)
            {
                work[i] = 1; ++npos;
            }
        }
    }
    else
    {
        std::cerr << "Error: unsupported probability distribution: " << dist
                  << std::endl;
        exit(-4);
    }

    if (npos)
        std::cerr << "Warning: converted " << npos
                  <<" non-positive entries (probability distribution is skewed)!\n";
}
