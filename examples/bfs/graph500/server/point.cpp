//  Copyright (c) 2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/iostreams.hpp>

#include "../stubs/point.hpp"
#include "point.hpp"

#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>

#include <iostream>
#include <fstream>

/* Spread the two 64-bit numbers into five nonzero values in the correct
 * range. */
void make_mrg_seed(uint64_t userseed1, uint64_t userseed2, uint_fast32_t* seed) {
  seed[0] = (userseed1 & 0x3FFFFFFF) + 1;
  seed[1] = ((userseed1 >> 30) & 0x3FFFFFFF) + 1;
  seed[2] = (userseed2 & 0x3FFFFFFF) + 1;
  seed[3] = ((userseed2 >> 30) & 0x3FFFFFFF) + 1;
  seed[4] = ((userseed2 >> 60) << 4) + (userseed1 >> 60) + 1;
}

static void compute_edge_range(int rank, int size, int64_t M, int64_t* start_idx, int64_t* end_idx) {
  int64_t rankc = (int64_t)(rank);
  int64_t sizec = (int64_t)(size);
  *start_idx = rankc * (M / sizec) + (rankc < (M % sizec) ? rankc : (M % sizec));
  *end_idx = (rankc + 1) * (M / sizec) + (rankc + 1 < (M % sizec) ? rankc + 1 : (M % sizec));
}

///////////////////////////////////////////////////////////////////////////////
namespace graph500 { namespace server
{
    void point::init(std::size_t objectid,std::size_t log_numverts,std::size_t number_partitions)
    {
      // Spread the two 64-bit numbers into five nonzero values in the correct
      //  range.
      uint_fast32_t seed[5];
      uint64_t userseed1 = 1;
      uint64_t userseed2 = 1;
      make_mrg_seed(userseed1, userseed2, seed);

      int64_t M = INT64_C(16) << log_numverts;

      int64_t start_idx, end_idx;
      compute_edge_range(objectid, number_partitions, M, &start_idx, &end_idx);
      int64_t nedges = end_idx - start_idx;

      std::vector<packed_edge> local_edges;
      local_edges.resize(nedges);

    //  generate_kronecker_range(seed, log_numverts, start_idx, end_idx, &*local_edges.begin());
    }

}}

