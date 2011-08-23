//  Copyright (c) 2009-2011 Matt Anderson
//  Copyright (c)      2011 Bryce Lelbach
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_4A2DF0C5_AD6E_489D_95C9_84ED1BACA41B)
#define HPX_4A2DF0C5_AD6E_489D_95C9_84ED1BACA41B

#include <vector>

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

