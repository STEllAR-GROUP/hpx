################################################################################
#  Copyright (c) 2011 Bryce Lelbach
#
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
################################################################################

from sys import argv
from cmath import pi, sqrt, sin, exp
 
p = (
  0.99999999999980993
, 676.5203681218851
, -1259.1392167224028
, 771.32342877765313
, -176.61502916214059
, 12.507343278686905
, -0.13857109526572012
, 9.9843695780195716e-6
, 1.5056327351493116e-7
)
 
def lanczos_gamma(z):
  z = complex(z)

  if z.real < 0.5:
    return pi / (sin(pi * z) * lanczos_gamma(1 - z))

  else:
    x = p[0]

    z -= 1

    for i in range(1, len(p)):
      x += p[i] / (z+i)

    t = z + (len(p) - 2) + 0.5

    return sqrt(2 * pi) * (t ** (z + 0.5)) * exp(-t) * x

try:
  z = argv[1]
  r = lanczos_gamma(z)
  print "lanczos_gamma({0}) == {1}+{2}j".format(z, r.real, r.imag)
except:
  print "usage: {0} Z".format(argv[0])
 
