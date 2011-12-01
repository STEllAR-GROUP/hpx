//  Copyright (c) 2009-2011 Matt Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_GTC_SERIALIZEPARAMETER_29NOV2011)
#define HPX_GTC_SERIALIZEPARAMETER_29NOV201

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
void serialize(Archive &ar, hpx::components::gtc::detail::parameter& par,
               const unsigned int)
{
    ar & par.irun;
    ar & par.mstep;
    ar & par.msnap;
    ar & par.ndiag;
    ar & par.nonlinear;
    ar & par.nhybrid;
    ar & par.paranl;
    ar & par.mode00;

    ar & par.tstep;
    ar & par.micell;
    ar & par.mecell;
    ar & par.mpsi;
    ar & par.mthetamax;
    ar & par.mzetamax;
    ar & par.npartdom;
    ar & par.ncycle;

    ar & par.a;
    ar & par.a0;
    ar & par.a1;
    ar & par.q0;
    ar & par.q1;
    ar & par.q2;
    ar & par.rc;
    ar & par.rw;

    ar & par.aion;
    ar & par.qion;
    ar & par.aelectron;
    ar & par.qelectron;

    ar & par.kappati;
    ar & par.kappate;
    ar & par.kappan;
    ar & par.fixed_Tprofile;
    ar & par.tite;
    ar & par.flow0;
    ar & par.flow1;
    ar & par.flow2;

    ar & par.r0;
    ar & par.b0;
    ar & par.temperature;
    ar & par.edensity0;

    ar & par.utime;
    ar & par.gyroradius;
    ar & par.tauii;

    ar & par.mflux;
    ar & par.num_mode;
    ar & par.m_poloidal;

    ar & par.output;
    ar & par.nbound;
    ar & par.umax;
    ar & par.iload;
    ar & par.track_particles;
    ar & par.nptrack;
    ar & par.rng_control;
    
    ar & par.isnap;
    ar & par.idiag1;
    ar & par.idiag2;

    ar & par.numberpe;
    ar & par.ntoroidal;

    ar & par.nmode;
    ar & par.mmode;
}

template <typename Archive>
void serialize(Archive &ar, hpx::components::gtc::parameter& par,
               const unsigned int)
{ ar & par.p; }

}}

#endif

