/* Copyright (C) 2010 The Trustees of Indiana University.                  */
/*                                                                         */
/* Use, modification and distribution is subject to the Boost Software     */
/* License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at */
/* http://www.boost.org/LICENSE_1_0.txt)                                   */
/*                                                                         */
/*  Authors: Jeremiah Willcock                                             */
/*           Andrew Lumsdaine                                              */

#ifndef SPLITTABLE_MRG_H
#define SPLITTABLE_MRG_H

#include <stdint.h>

/* Multiple recursive generator from L'Ecuyer, P., Blouin, F., and       */
/* Couture, R. 1993. A search for good multiple recursive random number  */
/* generators. ACM Trans. Model. Comput. Simul. 3, 2 (Apr. 1993), 87-98. */
/* DOI= http://doi.acm.org/10.1145/169702.169698 -- particular generator */
/* used is from table 3, entry for m = 2^31 - 1, k = 5 (same generator   */
/* is used in GNU Scientific Library).                                   */

/* See notes at top of splittable_mrg.c for information on this          */
/* implementation.                                                       */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mrg_state {
  uint_fast32_t z1, z2, z3, z4, z5;
} mrg_state;

/* Returns integer value in [0, 2^31-1) using original transition matrix */
uint_fast32_t mrg_get_uint_orig(mrg_state* state);

/* Returns real value in [0, 1) using original transition matrix */
double mrg_get_double_orig(mrg_state* state);

/* Seed PRNG with a given array of five values in the range [0, 0x7FFFFFFE] and
 * not all zero. */
void mrg_seed(mrg_state* st, const uint_fast32_t seed[5]);

/* Skip the PRNG ahead _exponent_ steps.  This code treats the exponent as a
 * 192-bit word, even though the PRNG period is less than that. */
void mrg_skip(mrg_state* state,
              uint_least64_t exponent_high,
              uint_least64_t exponent_middle,
              uint_least64_t exponent_low);

#ifdef __cplusplus
}
#endif

#endif /* SPLITTABLE_MRG_H */
