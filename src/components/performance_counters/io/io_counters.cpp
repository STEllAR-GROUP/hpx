//  Copyright (c) 2015 Maciej Brodowicz
//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/performance_counters/manage_counter_type.hpp>
#include <hpx/runtime/components/component_factory_base.hpp>
#include <hpx/runtime/components/component_startup_shutdown.hpp>
#include <hpx/runtime/startup_function.hpp>
#include <hpx/util/function.hpp>

#include <hpx/components/performance_counters/io/io_counters.hpp>

#include <boost/format.hpp>
#include <hpx/exception.hpp>
#include <boost/config/warning_disable.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/qi_uint.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>
#include <boost/spirit/include/phoenix_object.hpp>
#include <boost/fusion/include/define_struct.hpp>
#include <boost/fusion/include/io.hpp>

#include <cstdint>
#include <fstream>
#include <iterator>
#include <string>

// type to store parser output
BOOST_FUSION_DEFINE_STRUCT(
    (hpx)(performance_counters)(io),
    proc_io,
    (std::uint64_t, riss)
    (std::uint64_t, wiss)
    (std::uint64_t, rsysc)
    (std::uint64_t, wsysc)
    (std::uint64_t, rstor)
    (std::uint64_t, wstor)
    (std::uint64_t, wcanc)
    )

///////////////////////////////////////////////////////////////////////////////
// Add factory registration functionality, We register the module dynamically
// as no executable links against it.
HPX_REGISTER_COMPONENT_MODULE_DYNAMIC()

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters { namespace io
{
#define PROC_IO_PATH "/proc/%1%/io"

    namespace qi = boost::spirit::qi;
    namespace ascii = boost::spirit::ascii;

    // grammar
    template<typename I>
    struct proc_io_parser: qi::grammar<I, proc_io(), ascii::space_type>
    {
        proc_io_parser(): proc_io_parser::base_type(start)
        {
            using qi::lit;

            start %=
                lit("rchar:")                 >> uint64_t_ >>
                lit("wchar:")                 >> uint64_t_ >>
                lit("syscr:")                 >> uint64_t_ >>
                lit("syscw:")                 >> uint64_t_ >>
                lit("read_bytes:")            >> uint64_t_ >>
                lit("write_bytes:")           >> uint64_t_ >>
                lit("cancelled_write_bytes:") >> uint64_t_;
          }

          qi::rule<I, proc_io(), ascii::space_type> start;
          qi::uint_parser<std::uint64_t> uint64_t_;
    };

    ///////////////////////////////////////////////////////////////////////////
    void parse_proc_io(proc_io& pio)
    {
        boost::format fmt(PROC_IO_PATH);
        pid_t pid = getpid();
        std::string fn = boost::str(fmt % pid);
        std::ifstream ins(fn);

        if (!ins.is_open())
            HPX_THROW_EXCEPTION(hpx::no_success,
                "hpx::performance_counters::io::parse_proc_io",
                boost::str(boost::format("failed to open " PROC_IO_PATH) % pid));

        typedef boost::spirit::basic_istream_iterator<char> iterator;
        iterator it(ins), end;
        proc_io_parser<iterator> p;

        if (!qi::phrase_parse(it, end, p, ascii::space, pio))
            HPX_THROW_EXCEPTION(hpx::no_success,
                "hpx::performance_counters::io::parse_proc_io",
                boost::str(boost::format("failed to parse " PROC_IO_PATH) % pid));
    }

    std::uint64_t get_pio_riss(bool)
    {
        proc_io pio;
        parse_proc_io(pio);
        return pio.riss;
    }

    std::uint64_t get_pio_wiss(bool)
    {
        proc_io pio;
        parse_proc_io(pio);
        return pio.wiss;
    }

    std::uint64_t get_pio_rsysc(bool)
    {
        proc_io pio;
        parse_proc_io(pio);
        return pio.rsysc;
    }

    std::uint64_t get_pio_wsysc(bool)
    {
        proc_io pio;
        parse_proc_io(pio);
        return pio.wsysc;
    }

    std::uint64_t get_pio_rstor(bool)
    {
        proc_io pio;
        parse_proc_io(pio);
        return pio.rstor;
    }

    std::uint64_t get_pio_wstor(bool)
    {
        proc_io pio;
        parse_proc_io(pio);
        return pio.wstor;
    }

    std::uint64_t get_pio_wcanc(bool)
    {
        proc_io pio;
        parse_proc_io(pio);
        return pio.wcanc;
    }

    ///////////////////////////////////////////////////////////////////////////
    void register_counter_types()
    {
        namespace pc = hpx::performance_counters;
        pc::install_counter_type(
            "/runtime/io/read_bytes_issued",
            &get_pio_riss,
            "number of bytes read by process (aggregate of count "
            "arguments passed to read() call or its analogues)",
            "bytes");
        pc::install_counter_type(
            "/runtime/io/write_bytes_issued",
            &get_pio_wiss,
            "number of bytes the process has caused or shall cause to be "
            "written (aggregate of count arguments passed to write() call or "
            "its analogues)",
            "bytes");
        pc::install_counter_type(
            "/runtime/io/read_syscalls",
            &get_pio_rsysc,
            "number of system calls that perform I/O reads",
            "");
        pc::install_counter_type(
            "/runtime/io/write_syscalls",
            &get_pio_wsysc,
            "number of system calls that perform I/O writes",
            "");
        pc::install_counter_type(
            "/runtime/io/read_bytes_transferred",
            &get_pio_rstor,
            "number of bytes retrieved from storage by I/O operations",
            "bytes");
        pc::install_counter_type(
            "/runtime/io/write_bytes_transferred",
            &get_pio_wstor,
            "number of bytes transferred to storage by I/O operations",
            "bytes");
        pc::install_counter_type(
            "/runtime/io/write_bytes_cancelled", &get_pio_wcanc,
            "number of bytes accounted by write_bytes_transferred that has not "
            "been ultimately stored due to truncation or deletion",
            "bytes");
    }

    bool get_startup(hpx::startup_function_type& startup_func,
        bool& pre_startup)
    {
#if defined(__linux) || defined(linux) || defined(__linux__) || defined(__gnu_linux__)
        startup_func = register_counter_types;
        pre_startup = true;
        return true;
#else
        return false;
#endif
    }
}}}

// register component's startup function
HPX_REGISTER_STARTUP_MODULE_DYNAMIC(
    hpx::performance_counters::io::get_startup);
