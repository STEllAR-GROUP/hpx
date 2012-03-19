//  Copyright (c) 2009-2011 Matt Anderson
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_4A2DF0C5_AD6E_489D_95C9_84ED1BACA41B)
#define HPX_4A2DF0C5_AD6E_489D_95C9_84ED1BACA41B

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/serialization.hpp>

#include <boost/serialization/detail/get_data.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

namespace boost { namespace serialization
{

template <typename Archive>
void serialize(Archive &ar, hpx::components::adaptive1d::detail::parameter& par,
               const unsigned int)
{
    ar & par.loglevel;
    ar & par.nt0;
    ar & par.nx0;
    ar & par.grain_size;
    ar & par.allowedl;
    ar & par.num_neighbors;
    ar & par.out_every;
    ar & par.outdir;
    ar & par.refine_every;

    ar & par.gr_sibling;
    ar & par.gr_minx;
    ar & par.gr_maxx;
    ar & par.gr_nx;
    ar & par.gr_h;
    ar & par.gr_lbox;
    ar & par.gr_rbox;
    ar & par.gr_left_neighbor;
    ar & par.gr_right_neighbor;
    ar & par.levelp;
    ar & par.item2gi;
    ar & par.gi2item;
    ar & par.prev_gi2item;
    ar & par.prev_gi;

    ar & par.minx0;
    ar & par.maxx0;
    ar & par.ethreshold;
    ar & par.ghostwidth;
    ar & par.num_rows;
    ar & par.rowsize;
    ar & par.level_row;

    ar & par.cfl;
    ar & par.disip;
    ar & par.Rmin;
    ar & par.Rout;
    ar & par.tau;
    ar & par.lambda;
    ar & par.v;
    ar & par.amp;
    ar & par.x0;
    ar & par.id_sigma;

    ar & par.h;
}

template <typename Archive>
void serialize(Archive &ar, hpx::components::adaptive1d::parameter& par,
               const unsigned int)
{ ar & par.p; }

}}

#endif

