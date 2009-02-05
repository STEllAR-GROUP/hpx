//  Copyright (c) 2007-2009 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_BLOCK_PROFILER_NOV_16_0811PM)
#define HPX_UTIL_BLOCK_PROFILER_NOV_16_0811PM

#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/util/static.hpp>
#include <hpx/util/logging.hpp>

#include <boost/version.hpp>
#if BOOST_VERSION >= 103600
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/sum.hpp>
#include <boost/accumulators/statistics/count.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/moment.hpp>
#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
#if BOOST_VERSION >= 103600
        class accumulator_stats
        {
        private:
            typedef boost::accumulators::stats<
                    boost::accumulators::tag::sum, 
                    boost::accumulators::tag::count, 
                    boost::accumulators::tag::mean, 
                    boost::accumulators::tag::moment<2> 
            > accumulator_stats_type;
            typedef boost::accumulators::accumulator_set<
                double, accumulator_stats_type
            > totals_type;

        public:
            accumulator_stats(char const* const description)
              : description_(description)
            {}

            void print_stats()
            {
                LTIM_(fatal) << "profiler: " << description_ << ": "
                            << boost::accumulators::sum(totals_) << " (" 
                            << boost::accumulators::count(totals_) << ", " 
                            << boost::accumulators::mean(totals_) << ", " 
                            << boost::accumulators::extract::moment<2>(totals_)
                            << ")";
            }

            void add(double val)
            {
                totals_(val);
            }

        private:
            totals_type totals_;
            char const* const description_;
        };
#else
        class accumulator_stats
        {
        private:
            typedef std::pair<double, std::size_t> totals_type;

        public:
            accumulator_stats(char const* const description)
              : description_(description)
            {}

            static inline double extract_count(totals_type const& p)
            {
                return double(p.second);
            }
            static inline double extract_mean(totals_type const& p)
            {
                return (0 != p.second) ? p.first / p.second : 0.0;
            }

            void print_stats()
            {
                LTIM_(fatal) << "profiler: " << description_ << ": "
                            << extract_count(totals_) << ", " 
                            << extract_mean(totals_);
            }

            void add(double val)
            {
                totals_.first += val;
                ++totals_.second;
            }

        private:
            totals_type totals_;
            char const* const description_;
        };
#endif
    }

    ///////////////////////////////////////////////////////////////////////////
    /// The \a block_profiler class can be used to collect timings for a block
    /// of code. It measures the execution time for each of the executions and
    /// collects the number of invocations, the average, and the variance of 
    /// the measured execution times.
    template <typename Tag>
    class block_profiler
    {
    public:
        block_profiler(char const* const description)
          : description_(description), measuring_(false)
        {
            restart();
        }

        ~block_profiler()
        {
            measure();
        }

        void restart()
        {
            measuring_ = true;
            timer_.restart();
        }
        void measure()
        {
            if (measuring_) {
                get_stats(description_).add(timer_.elapsed());
                measuring_ = false;
            }
        }

        static void print_stats()
        {
            get_stats().print_stats();
        }

    private:
        static detail::accumulator_stats& get_stats(char const* const description = "")
        {
            static_<detail::accumulator_stats, Tag> stats(description);
            return stats.get();
        };

    private:
        high_resolution_timer timer_;
        char const* const description_;
        bool measuring_;
    };

}}

#endif
