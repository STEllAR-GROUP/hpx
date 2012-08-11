// Copyright (c) 2012 Vinay C Amatya
// Copyright (c) Bryce Adelstein-Lelbach
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>

#if defined(__linux) || defined(linux) || defined(linux__) || defined(__linux__)

#include <hpx/exception.hpp>
#include <boost/config/warning_disable.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/qi_uint.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>
#include <boost/spirit/include/phoenix_object.hpp>
#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/fusion/include/io.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/format.hpp>

#include <fstream>
#include <iostream>
#include <string>
#include <algorithm>
#include <iterator>
#include <vector>

namespace hpx { namespace performance_counters { namespace memory
{
    namespace qi = boost::spirit::qi;
    namespace ascii = boost::spirit::ascii;

    struct proc_statm
    {
        proc_statm()
          : size(0), resident(0), share(0), text(0), lib(0), data(0), dt(0)
        {}

        boost::uint64_t size;
        boost::uint64_t resident;
        boost::uint64_t share;
        boost::uint64_t text;
        boost::uint64_t lib;
        boost::uint64_t data;
        boost::uint64_t dt;
    };
}}}

BOOST_FUSION_ADAPT_STRUCT(
    hpx::performance_counters::memory::proc_statm,
    (boost::uint64_t, size)
    (boost::uint64_t, resident)
    (boost::uint64_t, share)
    (boost::uint64_t, text)
    (boost::uint64_t, lib)
    (boost::uint64_t, data)
    (boost::uint64_t, dt)
)

namespace hpx { namespace performance_counters { namespace memory
{
    template <typename Iterator>
    struct proc_statm_parser : qi::grammar<Iterator, proc_statm(), ascii::space_type>
    {
        proc_statm_parser() : proc_statm_parser::base_type(start)
        {
            using qi::ulong_;

            start = ulong_
                >> ulong_
                >> ulong_
                >> ulong_
                >> ulong_
                >> ulong_
                >> ulong_
                ;
        }

        qi::rule<Iterator, proc_statm(), ascii::space_type> start;
    };

    ///////////////////////////////////////////////////////////////////////////
    // returns virtual memory value
    boost::uint64_t read_psm_vm()
    {
        using boost::spirit::ascii::space;
        typedef std::string::const_iterator iterator_type;
        typedef proc_statm_parser<iterator_type> proc_statm_parser;

        proc_statm_parser psg;
        std::string in_string;
        boost::uint32_t pid = getpid();
        std::string filename = boost::str(boost::format("/proc/%1%/statm") % pid);

        {
            std::ifstream infile(filename.c_str());
            if (!infile.is_open())
            {
                HPX_THROW_EXCEPTION(
                    hpx::bad_request
                    "hpx::performance_counters::memory::read_psm_resident"
                    boost::str(boost::format("unable to open statm file '%s'") % filename));
                return boost::uint64_t(-1);
            }

            in_string << infile;
        }

        proc_statm psm;
        iterator_type itr = in_string.begin();
        iterator_type end = in_string.end();
        bool r = phrase_parse(itr, end, psg, space, psm);
        if (!r) {
            HPX_THROW_EXCEPTION(
                hpx::invalid_data
                "hpx::performance_counters::memory::read_psm_vm"
                boost::str(boost::format("failed to parse '%s'") % filename));
            return boost::uint64_t(-1);
        }

        // psm.size is in blocks, but we need to return the number of bytes
        return psm.size * 4096;
    }

    // returns resident memory value
    boost::uint64_t read_psm_resident()
    {
        using boost::spirit::ascii::space;
        typedef std::string::const_iterator iterator_type;
        typedef proc_statm_parser<iterator_type> proc_statm_parser;

        proc_statm_parser psg;
        std::string in_string;
        boost::uint32_t pid = getpid();
        std::string filename = boost::str(boost::format("/proc/%1%/statm") % pid);

        {
            std::ifstream infile(filename.c_str());
            if (!infile.is_open())
            {
                HPX_THROW_EXCEPTION(
                    hpx::bad_request
                    "hpx::performance_counters::memory::read_psm_resident"
                    boost::str(boost::format("unable to open statm file '%s'") % filename));
                return boost::uint64_t(-1);
            }

            in_string << infile;
        }

        proc_statm psm;
        iterator_type itr = in_string.begin();
        iterator_type end = in_string.end();
        bool r = phrase_parse(itr, end, psg, space, psm);
        if (!r) {
            HPX_THROW_EXCEPTION(
                hpx::invalid_data
                "hpx::performance_counters::memory::read_psm_resident"
                boost::str(boost::format("failed to parse '%s'") % filename));
            return boost::uint64_t(-1);
        }

        // psm.size is in blocks, but we need to return the number of bytes
        return psm.resident * 4096;
    }
}}}

#endif

#if defined(BOOST_WINDOWS)
#endif
