//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_TIME_LOGGER_NOV_26_0548PM)
#define HPX_UTIL_TIME_LOGGER_NOV_26_0548PM

#include <fstream>
#include <boost/version.hpp>
#include <boost/cstdint.hpp>

#include <hpx/config.hpp>
#include <hpx/util/logging.hpp>

#if defined(HPX_WINDOWS)
#  include <process.h>
#elif defined(HPX_HAVE_UNISTD_H)
#  include <unistd.h>
#endif

///////////////////////////////////////////////////////////////////////////////
#define HPX_INITIAL_TIMES_SIZE 64000

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{
    inline boost::uint64_t hrtimer_ticks()
    {
        return 0;   // FIXME!
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        struct time_logger_ref_time
        {
            time_logger_ref_time()
            {
                start_ = hrtimer_ticks();
            }
            boost::uint64_t start_;
        };

        time_logger_ref_time const time_logger_ref_time_;
    }

    ///////////////////////////////////////////////////////////////////////////
    /// The \a time_logger class can be used to collect timings for a block
    /// of code. It measures the execution time for each of the executions and
    /// collects the number of invocations, the average, and the variance of
    /// the measured execution times.
    class time_logger
    {
    private:
        enum { hpx_initial_times_size = 64000 };

    public:
        time_logger(char const* const description, int thread_num,
                bool enabled = false)
          : description_(description), thread_num_(thread_num),
            enabled_(enabled && LTIM_ENABLED(warning))
        {
            if (enabled_)
                times_.reserve(hpx_initial_times_size);
        }

        ~time_logger()
        {
            if (!enabled_)
                return;     // generate output only if logging is enabled

            std::string name(description_);
            name += "." + boost::lexical_cast<std::string>(getpid());
            name += "." + boost::lexical_cast<std::string>(thread_num_);
            name += ".csv";

            std::ofstream out(name.c_str());

            int i = 0;
            std::vector<boost::uint64_t>::iterator eit = times_.end();
            std::vector<boost::uint64_t>::iterator bit = times_.begin();
            for (std::vector<boost::uint64_t>::iterator it = bit; it != eit; ++it, ++i)
            {
                boost::uint64_t t = *it - detail::time_logger_ref_time_.start_;
                out << t << "," << 2*thread_num_ + ((i % 2) ? 1 : 0) << std::endl;
                out << t+1 << "," << 2*thread_num_ + ((i % 2) ? 0 : 1) << std::endl;
            }
        }

        void tick()
        {
            if (enabled_)
                times_.push_back(hrtimer_ticks());
        }

        void tock()
        {
            if (enabled_)
                times_.push_back(hrtimer_ticks());
        }

    private:
        char const* const description_;
        int thread_num_;
        std::vector<boost::uint64_t> times_;
        bool enabled_;
    };

}}

#endif
