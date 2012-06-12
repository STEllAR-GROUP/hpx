
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef OCLM_PROGRAM_HPP
#define OCLM_PROGRAM_HPP

#include <oclm/config.hpp>

#include <string>
#include <boost/filesystem/path.hpp>
#include <boost/range.hpp>

namespace oclm {

    struct program
    {
        /*
        program() : p(0) {}
        ~program()
        {
            //if(p != 0) ::clReleaseProgram(p);
        }

        program(program const & other)
            : p(other.p)
        {
            if(p != 0) ::clRetainProgram(p);
        }

        program & operator=(program const & other)
        {
            p = other.p;
            if(p != 0) ::clRetainProgram(p);

            return *this;
        }
        */

        // Creates a program from a strings ...
        template <typename Range>
        program(Range const & r)
            : content_(boost::begin(r), boost::end(r))
        {}
        //program(std::vector<std::string> const &);
        // Creates a program from a files ...
        //program(boost::filesystem::path const &);
        //program(std::vector<boost::filesystem::path> const &);

        // TODO: add constructors for binaries
 
        std::vector<char> content_;
    };
}

#endif
