// Copyright (c) 2012 Vinay C Amatya
// Copyright (c) Bryce Adelstein-Lelbach
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(__linux) || defined(linux) || defined(linux__) || defined(__linux__)

#include <sys/mman.h>
#include <sys/param.h>

#include <hpx/exception.hpp>
#include <hpx/util/format.hpp>

#include <boost/config/warning_disable.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/qi_uint.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>
#include <boost/spirit/include/phoenix_object.hpp>
#include <boost/fusion/include/define_struct.hpp>
#include <boost/fusion/include/io.hpp>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

BOOST_FUSION_DEFINE_STRUCT(
    (hpx)(performance_counters)(memory),
    proc_statm,
    (std::uint64_t,   size)
    (std::uint64_t,   resident)
    (std::uint64_t,   share)
    (std::uint64_t,   text)
    (std::uint64_t,   lib)
    (std::uint64_t,   data)
    (std::uint64_t,   dt)
    )

namespace hpx { namespace performance_counters { namespace memory
{
    namespace qi = boost::spirit::qi;
    namespace ascii = boost::spirit::ascii;

    template <typename Iterator>
    struct proc_statm_grammar
      : qi::grammar<Iterator, proc_statm(), ascii::space_type>
    {
        proc_statm_grammar()
          : proc_statm_grammar::base_type(start)
        {
            start =  uint64_t_
                  >> uint64_t_
                  >> uint64_t_
                  >> uint64_t_
                  >> uint64_t_
                  >> uint64_t_
                  >> uint64_t_
                  ;
        }

        qi::rule<Iterator, proc_statm(), ascii::space_type> start;

        qi::uint_parser<std::uint64_t> uint64_t_;
    };

    struct ifstream_raii
    {
        ifstream_raii(char const* file, std::ios_base::openmode mode)
            : stm(file, mode)
        {}

        ~ifstream_raii()
        {
            if (stm.is_open())
                stm.close();
        }

        std::ifstream& get()
        {
            return stm;
        }

      private:
        std::ifstream stm;
    };

    bool read_proc_statm(proc_statm& ps, std::int32_t pid)
    {
        std::string filename
            = hpx::util::format("/proc/{1}/statm", pid);

        ifstream_raii in(filename.c_str(), std::ios_base::in);

        if (!in.get())
            return false;

        in.get().unsetf(std::ios::skipws); // No white space skipping!

        typedef boost::spirit::basic_istream_iterator<char> iterator;

        iterator it(in.get()), end;

        proc_statm_grammar<iterator> p;

        if (!qi::phrase_parse(it, end, p, ascii::space, ps))
            return false;
        else
            return true;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Returns virtual memory usage.
    std::uint64_t read_psm_virtual(bool)
    {
        proc_statm ps;

        if (!read_proc_statm(ps, getpid()))
        {
            HPX_THROW_EXCEPTION(
                hpx::invalid_data,
                "hpx::performance_counters::memory::read_psm_virtual",
                hpx::util::format("failed to parse '/proc/{1}/statm'",
                    getpid()));
            return std::uint64_t(-1);
        }

        // ps.size is in pages, but we need to return the number of bytes.
        return ps.size * EXEC_PAGESIZE;
    }

    // Returns resident memory usage.
    std::uint64_t read_psm_resident(bool)
    {
        proc_statm ps;

        if (!read_proc_statm(ps, getpid()))
        {
            HPX_THROW_EXCEPTION(
                hpx::invalid_data,
                "hpx::performance_counters::memory::read_psm_resident",
                hpx::util::format("failed to parse '/proc/{1}/statm'",
                    getpid()));
            return std::uint64_t(-1);
        }

        // ps.resident is in pages, but we need to return the number of bytes.
        return ps.resident * EXEC_PAGESIZE;
    }

    // Returns total available memory
    std::uint64_t read_total_mem_avail(bool)
    {
        std::string file = "/proc/meminfo";
        std::ifstream in;

        char buffer[1024];
        in.open(file.c_str());

        //Available Memory is on 3rd line
        for (int k = 0; k < 3; k++) {
          in.getline(buffer, 1024);
        }
        in.close();
        std::string tbuf = buffer;
        tbuf.copy(buffer, 11, 13);
        return atol(buffer);
    }

}}}

#endif

