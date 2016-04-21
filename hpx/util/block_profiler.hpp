//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_BLOCK_PROFILER_NOV_16_0811PM)
#define HPX_UTIL_BLOCK_PROFILER_NOV_16_0811PM

#include <hpx/config.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/reinitializable_static.hpp>

#include <boost/version.hpp>
#if defined(HPX_USE_ACCUMULATOR_LIBRARY)
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/sum.hpp>
#include <boost/accumulators/statistics/count.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/moment.hpp>
#endif

#include <string>

#define HPX_DONT_USE_BLOCK_PROFILER

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{
#if !defined(HPX_DONT_USE_BLOCK_PROFILER)
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
#if defined(HPX_USE_ACCUMULATOR_LIBRARY)
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

            accumulator_stats* This() { return this; }

        public:
            accumulator_stats(char const* const description)
              : description_(description), registered_on_exit_(false)
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

            void add()
            {
                totals_(timer_.elapsed());
            }

            void add(double started)
            {
                totals_(timer_.elapsed() - started);
            }

            void restart()
            {
                if (!registered_on_exit_)
                {
                    registered_on_exit_ = hpx::register_on_exit(
                        util::bind(&accumulator_stats::print_stats, This()));
                }
                timer_.restart();
            }

            double elapsed()
            {
                if (!registered_on_exit_)
                {
                    registered_on_exit_ = hpx::register_on_exit(
                        util::bind(&accumulator_stats::print_stats, This()));
                }
                return timer_.elapsed();
            }

        private:
            high_resolution_timer timer_;
            totals_type totals_;
            std::string description_;
            bool registered_on_exit_;
        };
#else
        class accumulator_stats
        {
        private:
            typedef std::pair<double, std::size_t> totals_type;

            accumulator_stats* This() { return this; }

        public:
            accumulator_stats(char const* const description)
              : description_(description), registered_on_exit_(false)
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

            void add()
            {
                totals_.first += timer_.elapsed();
                ++totals_.second;
            }

            void add(double started)
            {
                totals_.first += timer_.elapsed() - started;
                ++totals_.second;
            }

            void restart()
            {
                if (!registered_on_exit_)
                {
                    registered_on_exit_ = hpx::register_on_exit(
                        util::bind(&accumulator_stats::print_stats, This()));
                }
                timer_.restart();
            }

            double elapsed()
            {
                if (!registered_on_exit_)
                {
                    registered_on_exit_ = hpx::register_on_exit(
                        util::bind(&accumulator_stats::print_stats, This()));
                }
                return timer_.elapsed();
            }

        private:
            high_resolution_timer timer_;
            totals_type totals_;
            std::string description_;
            bool registered_on_exit_;
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
        block_profiler(char const* description, bool enable_logging = true)
          : stats_(get_stats(description)), measuring_(false),
            enable_logging_(enable_logging)
        {
            restart();
        }

        ~block_profiler()
        {
            measure();
        }

        // direct interface if used on the stack
        void restart()
        {
            if (enable_logging_)
            {
                measuring_ = true;
                stats_.restart();
            }
        }
        void measure()
        {
            if (measuring_ && enable_logging_)
            {
                stats_.add();
                measuring_ = false;
            }
        }

        // special interface for block_profiler_wrapper below
        void measure(double started)
        {
            if (enable_logging_ && LTIM_ENABLED(fatal))
                stats_.add(started);
        }

        double elapsed()
        {
            return enable_logging_ && LTIM_ENABLED(fatal) ? stats_.elapsed() : 0;
        }

    private:
        static detail::accumulator_stats&
        get_stats(char const* description)
        {
            reinitializable_static<detail::accumulator_stats, Tag> stats(description);
            return stats.get();
        };

    private:
        detail::accumulator_stats& stats_;
        bool measuring_;
        bool enable_logging_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Tag>
    class block_profiler_wrapper
    {
    public:
        block_profiler_wrapper(block_profiler<Tag>& profile)
          : profile_(profile), started_(profile_.elapsed())
        {
        }

        ~block_profiler_wrapper()
        {
            profile_.measure(started_);
        }

    private:
        block_profiler<Tag>& profile_;
        double started_;
    };
#else
    ///////////////////////////////////////////////////////////////////////////
    // define dummy classes
    template <typename Tag>
    class block_profiler
    {
    public:
        block_profiler(char const*, bool = true) {}
        ~block_profiler() {}

        void restart() {}
        void measure() {}
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Tag>
    class block_profiler_wrapper
    {
    public:
        block_profiler_wrapper(block_profiler<Tag>&)
        {
        }

        ~block_profiler_wrapper()
        {
        }
    };
#endif

}}

#endif
