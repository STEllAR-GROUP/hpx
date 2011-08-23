//  Copyright (c) 2009-2011 Matt Anderson
//  Copyright (c)      2011 Bryce Lelbach 
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_4A2DF0C5_AD6E_489D_95C9_84ED1BACA41B)
#define HPX_4A2DF0C5_AD6E_489D_95C9_84ED1BACA41B

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/detail/get_data.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

namespace boost { namespace serialization
{

template <typename Archive>
void serialize(Archive &ar, hpx::components::amr::detail::parameter& par,
               const unsigned int) 
{
    ar & par.lambda;
    ar & par.allowedl;
    ar & par.num_rows;

    ar & par.loglevel;
    ar & par.output;
    ar & par.output_stdout;
    ar & par.output_level;

    ar & par.nt0;
    ar & par.refine_every;
    ar & par.nx0;
    ar & par.ny0;
    ar & par.nz0;
    ar & par.shadow;
    ar & par.refine_level;
    ar & par.minx0;
    ar & par.maxx0;
    ar & par.miny0;
    ar & par.maxy0;
    ar & par.minz0;
    ar & par.maxz0;
    ar & par.h;
    ar & par.ethreshold;
    ar & par.minefficiency;
    ar & par.num_px_threads;
    ar & par.refine_every;
    ar & par.ghostwidth;
    ar & par.bound_width;
    ar & par.clusterstyle;
    ar & par.mindim;
    ar & par.refine_factor;

    ar & par.gr_sibling;
    ar & par.gr_t;
    ar & par.gr_minx;
    ar & par.gr_miny;
    ar & par.gr_minz;
    ar & par.gr_maxx;
    ar & par.gr_maxy;
    ar & par.gr_maxz;
    ar & par.gr_nx;
    ar & par.gr_ny;
    ar & par.gr_nz;
    ar & par.gr_h;
    ar & par.gr_alive;
    ar & par.levelp;
    ar & par.item2gi;
    ar & par.gi2item;
    ar & par.prev_gi2item;
    ar & par.prev_gi;

    ar & par.rowsize;
    ar & par.level_row;
}

template <typename Archive>
void serialize(Archive &ar, hpx::components::amr::parameter& par,
               const unsigned int) 
{ ar & par.p; }

}}

#endif

