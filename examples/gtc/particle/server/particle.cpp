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
    void particle::init(std::size_t objectid,std::string const& particlefile)
    {
        idx_ = objectid;
        std::string line;
        std::string val1,val2,val3,val4;
        std::ifstream myfile;
        myfile.open(particlefile);
        if (myfile.is_open() ) {
            while (myfile.good()) { 
                while (std::getline(myfile,line)) {
                    std::istringstream isstream(line);
                    std::getline(isstream,val1,' ');
                    std::getline(isstream,val2,' ');
                    std::getline(isstream,val3,' ');
                    std::getline(isstream,val4,' ');
                    std::size_t node = boost::lexical_cast<std::size_t>(val1);   
                    double posx = boost::lexical_cast<double>(val2);   
                    double posy = boost::lexical_cast<double>(val3);   
                    double posz = boost::lexical_cast<double>(val4);   
                    if ( node == objectid ) {
                        posx_ = posx;
                        posy_ = posy;
                        posz_ = posz;
                    }
                }
            }
            myfile.close();
        } 
    }
}}


