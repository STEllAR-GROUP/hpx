//  Copyright (c) 2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "particle.hpp"

#include <boost/lexical_cast.hpp>

#include <string>
#include <sstream>
#include <fstream>

///////////////////////////////////////////////////////////////////////////////
namespace gtc { namespace server
{
    void particle::init(std::size_t objectid,parameter const& par)
    {
        // 
        srand(objectid+5);

        // initial data

        mtheta_.resize(par->mpsi+1);
        // --- Define poloidal grid ---
        double tmp5 = (double) par->mpsi;
        deltar_ = (par->a1-par->a0)/tmp5;

        double pi = 4.0*atan(1.0);
        // grid shift associated with fieldline following coordinates
        double tmp6 = (double) par->mthetamax;
        double tdum = 2.0*pi*par->a1/tmp6;
        for (std::size_t i=0;i<par->mpsi+1;i++) {
          double r = par->a0 + deltar_*i;
          std::size_t two = 2;
          double tmp7 = pi*r/tdum + 0.5;
          std::size_t tmp8 = (std::size_t) tmp7;
          mtheta_[i] = std::max(two,std::min(par->mthetamax,two*tmp8)); // even # poloidal grid
        }

        // number of grids on a poloidal plane
        mgrid_ = 0;
        for (std::size_t i=0;i<mtheta_.size();i++) {
          mgrid_ += mtheta_[i] + 1;
        }
        mzeta_ = par->mzetamax/par->ntoroidal;
        std::size_t mi = par->micell*(mgrid_-par->mpsi)*mzeta_/par->npartdom; // # of ions per processor
    }
}}


