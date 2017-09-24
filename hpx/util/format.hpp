//  Copyright (c) 2017 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// hpxinspect:nodeprecatedinclude:boost/format.hpp
// hpxinspect:nodeprecatedname:boost::format

#ifndef HPX_UTIL_FORMAT_HPP
#define HPX_UTIL_FORMAT_HPP

#include <hpx/config.hpp>

#include <boost/format.hpp>

#include <iosfwd>
#include <string>

namespace hpx { namespace util
{
    template <typename ...Args>
    std::string format(
        std::string const& format_str, Args const&... args)
    {
        boost::format fmt(format_str);
        int const _sequencer[] = { ((fmt % args), 0)... };
        (void)_sequencer;

        return boost::str(fmt);
    }

    template <typename ...Args>
    std::ostream& format_to(
        std::ostream& os,
        std::string const& format_str, Args const&... args)
    {
        boost::format fmt(format_str);
        int const _sequencer[] = { ((fmt % args), 0)... };
        (void)_sequencer;

        return os << fmt;
    }
}}

#endif /*HPX_UTIL_FORMAT_HPP*/
