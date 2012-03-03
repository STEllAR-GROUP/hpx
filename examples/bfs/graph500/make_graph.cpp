/* Copyright (C) 2009-2010 The Trustees of Indiana University.             */
/*                                                                         */
/* Use, modification and distribution is subject to the Boost Software     */
/* License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at */
/* http://www.boost.org/LICENSE_1_0.txt)                                   */
/*                                                                         */
/*  Authors: Jeremiah Willcock                                             */
/*           Andrew Lumsdaine                                              */

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <assert.h>
#include <math.h>

#include "make_graph.hpp"
#include "server/graph_generator.hpp"
#include "server/splittable_mrg.hpp"

void make_mrg_seed(uint64_t userseed1, uint64_t userseed2, uint_fast32_t* seed);

/* PRNG interface for implementations; takes seed in same format as given by
 * users, and creates a vector of doubles in a reproducible (and
 * random-access) way. */
void make_random_numbers(
       /* in */ int64_t nvalues    /* Number of values to generate */,
       /* in */ uint64_t userseed1 /* Arbitrary 64-bit seed value */,
       /* in */ uint64_t userseed2 /* Arbitrary 64-bit seed value */,
       /* in */ int64_t position   /* Start index in random number stream */,
       /* out */ double* result    /* Returned array of values */
) {
  int64_t i;
  uint_fast32_t seed[5];
  make_mrg_seed(userseed1, userseed2, seed);

  mrg_state st;
  mrg_seed(&st, seed);

  mrg_skip(&st, 2, 0, 2 * position); /* Each double takes two PRNG outputs */

  for (i = 0; i < nvalues; ++i) {
    result[i] = mrg_get_double_orig(&st);
  }
}
