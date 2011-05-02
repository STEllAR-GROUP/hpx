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

#include <boost/serialization/string.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/vector.hpp>

namespace boost { namespace serialization
{

template <typename Archive>
void serialize(Archive &ar, hpx::components::mesh::detail::parameter& par,
               const unsigned int) 
{
    ar & par.lambda;
    ar & par.allowedl;
    ar & par.loglevel;
    ar & par.output;
    ar & par.output_stdout;
    ar & par.nt0;
    ar & par.nx;
    ar & par.refine_level;
    ar & par.minx0;
    ar & par.maxx0;
    ar & par.dx0;
    ar & par.dt0;
    ar & par.ethreshold;
    ar & par.R0;
    ar & par.delta;
    ar & par.amp;
    ar & par.amp_dot;
    ar & par.eps;
    ar & par.output_level;
    ar & par.granularity;
    ar & par.time_granularity;
    ar & par.rowsize;
    ar & par.level_row;
    ar & par.level_begin;
    ar & par.level_end;
    ar & par.min;
    ar & par.max;
    ar & par.gw;
    ar & par.num_rows;
}

template <typename Archive>
void serialize(Archive &ar, hpx::components::mesh::parameter& par,
               const unsigned int) 
{ ar & par.p; }

}}

#endif

