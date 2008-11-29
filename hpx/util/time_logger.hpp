//  Copyright (c) 2007-2008 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_TIME_LOGGER_NOV_26_0548PM)
#define HPX_UTIL_TIME_LOGGER_NOV_26_0548PM

#include <fstream>
#include <boost/version.hpp>
#include <boost/cstdint.hpp>

#include <hpx/util/logging.hpp>

///////////////////////////////////////////////////////////////////////////////
#if defined(__GNUC__) 

inline boost::uint64_t hrtimer_ticks()
{
    boost::uint32_t _lo, _hi;
    __asm__ __volatile__ (
          "movl %%ebx,%%esi\n"
          "cpuid\n"
          "rdtsc\n"
          "movl %%esi,%%ebx\n"
        : "=a" (_lo), "=d" (_hi)
        :
        : "%esi", "%ecx"
    );
    return ((boost::uint64_t)_hi << 32) | _lo;
}

#elif defined(BOOST_WINDOWS)

inline boost::uint64_t hrtimer_ticks()
{
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    return (boost::uint64_t)now.QuadPart;
}

#else

// no timings are recorded
inline boost::uint64_t hrtimer_ticks()
{
    return 0;
}

#endif

#define HPX_INITIAL_TIMES_SIZE 64000

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    /// The \a block_profiler class can be used to collect timings for a block
    /// of code. It measures the execution time for each of the executions and
    /// collects the number of invocations, the average, and the variance of 
    /// the measured execution times.
    class time_logger
    {
    public:
        time_logger(char const* const description, int thread_num,
              boost::uint64_t start)
          : description_(description), thread_num_(thread_num), start_(start)
        {
            if (LTIM_ENABLED(fatal)) 
                times_.reserve(HPX_INITIAL_TIMES_SIZE);
        }

        ~time_logger()
        {
            if (!LTIM_ENABLED(fatal)) 
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
                boost::uint64_t t = *it - start_;
                out << t << "," << 2*thread_num_ + ((i % 2) ? 1 : 0) << std::endl;
                out << t+1 << "," << 2*thread_num_ + ((i % 2) ? 0 : 1) << std::endl;
            }
        }

        void tick()
        {
            if (LTIM_ENABLED(fatal)) 
                times_.push_back(hrtimer_ticks());
        }

        void tock()
        {
            if (LTIM_ENABLED(fatal)) 
                times_.push_back(hrtimer_ticks());
        }

        struct ref_time
        {
            ref_time()
            {
                start_ = hrtimer_ticks();
            }
            boost::uint64_t start_;
        };

    private:
        char const* const description_;
        int thread_num_;
        boost::uint64_t start_;
        std::vector<boost::uint64_t> times_;
    };

    time_logger::ref_time const ref_time_;

}}

#endif
