//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_VALUE_LOGGER_DEC_08_0548PM)
#define HPX_UTIL_VALUE_LOGGER_DEC_08_0548PM

#include <fstream>
#include <boost/lockfree/primitives.hpp>
#include <boost/lexical_cast.hpp>

#include <hpx/config.hpp>

#if defined(HPX_WINDOWS)
#  include <process.h>
#elif defined(HPX_HAVE_UNISTD_H)
#  include <unistd.h>
#endif

#include <hpx/util/logging.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        struct value_logger_ref_time
        {
            value_logger_ref_time()
            {
                start_ = boost::lockfree::hrtimer_ticks();
            }
            boost::uint64_t start_;
        };

        value_logger_ref_time const value_logger_ref_time_;
    }

    ///////////////////////////////////////////////////////////////////////////
    /// The \a value_logger class can be used to collect timings for a block
    /// of code. It measures the execution time for each of the executions and
    /// collects the number of invocations, the average, and the variance of
    /// the measured execution times.
    template <typename T>
    class value_logger
    {
    private:
        enum { hpx_initial_times_size = 64000 };
        typedef std::vector<std::pair<boost::uint64_t, T> > values_type;

    public:
        value_logger(char const* const description, bool enabled = true)
          : description_(description), enabled_(enabled && LTIM_ENABLED(fatal))
        {
            if (enabled_)
                values_.reserve(hpx_initial_times_size);
        }

        ~value_logger()
        {
            if (!enabled_)
                return;     // generate output only if logging is enabled

            std::string name(description_);
            name += "." + boost::lexical_cast<std::string>(getpid());
            name += ".csv";

            std::ofstream out(name.c_str());

            typename values_type::iterator eit = values_.end();
            typename values_type::iterator bit = values_.begin();
            for (typename values_type::iterator it = bit; it != eit; ++it)
            {
                boost::uint64_t t = (*it).first - detail::value_logger_ref_time_.start_;
                out << t << "," << (*it).second << std::endl;
            }
        }

        void snapshot(T const& t)
        {
            if (enabled_)
                values_.push_back(std::make_pair(boost::lockfree::hrtimer_ticks(), t));
        }

    private:
        char const* const description_;
        values_type values_;
        bool enabled_;
    };

}}

#endif
