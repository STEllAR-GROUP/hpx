//  Copyright (c) 2011-2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2013-2014 Thomas Heller
//  Copyright (c) 2013-2014 Patricia Grubel
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#define HPX_NO_VERSION_CHECK

#include "htts2.hpp"

#if !defined(HPX_MOVABLE_BUT_NOT_COPYABLE)
#if defined(HPX_HAVE_CXX11_DELETED_FUNCTIONS)
#define HPX_MOVABLE_BUT_NOT_COPYABLE(TYPE)                                    \
    public:                                                                   \
        TYPE(TYPE const &) = delete;                                          \
        TYPE& operator=(TYPE const &) = delete;                               \
    private:                                                                  \
/**/
#else
#define HPX_MOVABLE_BUT_NOT_COPYABLE(TYPE)                                    \
    private:                                                                  \
        TYPE(TYPE const &);                                                   \
        TYPE& operator=(TYPE const &);                                        \
/**/
#endif
#endif

template <typename BaseClock = boost::chrono::steady_clock>
struct payload_precision_tracker : htts2::clocksource<BaseClock>
{
    typedef typename htts2::clocksource<BaseClock>::rep rep;

    // 'expected' is the expected payload in nanosecond
    payload_precision_tracker(rep expected)
      : expected_(expected)
      , mean_lost_(0.0)
      , stdev_lost_(0.0)
      , samples_(0)
    {}

    // Returns: stdev of the mean lost payload.
    double precision_stat_uncertainty() const
    {
        return stdev_lost_;
    }

    // Returns: the uncertainty of the mean lost payload.
    double precision_uncertainty() const
    {
        return (std::max)(this->clock_uncertainty()
                      , precision_stat_uncertainty());
    }

    double average_precision() const
    {
        return mean_lost_;
    }

    // Returns: the number of samples.
    rep samples() const
    {
        return samples_;
    }

    // Performs approximately 'expected_' nanoseconds of artificial work.
    void operator()()
    {
        double const measured =
            static_cast<double>(htts2::payload<BaseClock>(expected_));

        double const lost = double(measured - expected_);

        update_lost(lost);
    }

  private:
    void update_lost(double lost)
    {
        // Based on Boost.Accumulators.
        //   mean_n = (mean_{n-1}*(n-1) + x_n) / n
        //   sigma_n ~= sqrt(((n-1)/n)*(sigma_{n-1}^2) + (1/n)*(x_n-mean_n)^2)

        ++samples_;

        mean_lost_ =
            (mean_lost_*(samples_-1.0) + lost)/samples_;

        stdev_lost_ =
            std::sqrt( ((samples_-1.0)/samples_)*(stdev_lost_*stdev_lost_)
                     + (1.0/samples_)*(lost-mean_lost_)*(lost-mean_lost_));
    }

    rep expected_;
    double mean_lost_;
    double stdev_lost_;
    rep samples_;
};

template <typename BaseClock = boost::chrono::steady_clock>
struct payload_precision_driver : htts2::driver
{
    payload_precision_driver(int argc, char** argv)
      : htts2::driver(argc, argv)
    {}

    void run() const
    {
        // Cold run
        kernel();

        // Hot run
        results_type results = kernel();
        print_results(results);
    }

  private:
    struct results_type
    {
        results_type()
          : average_precision_(0.0)
          , precision_uncertainty_(0.0)
          , amortized_overhead_(0.0)
          , overhead_uncertainty_(0.0)
        {}

        results_type(
            double average_precision,       // nanoseconds
            double precision_uncertainty,   // nanoseconds
            double amortized_overhead,      // nanoseconds
            double overhead_uncertainty     // nanoseconds
            )
          : average_precision_(average_precision)
          , precision_uncertainty_(precision_uncertainty)
          , amortized_overhead_(amortized_overhead)
          , overhead_uncertainty_(overhead_uncertainty)
        {}

        double average_precision_;
        double precision_uncertainty_;
        double amortized_overhead_;
        double overhead_uncertainty_;
    };

    results_type kernel() const
    {
        ///////////////////////////////////////////////////////////////////////

        payload_precision_tracker<BaseClock>
            p(this->payload_duration_ /* = p */);

        htts2::timer<BaseClock> t;

        for (boost::uint64_t i = 0; i < this->tasks_ /* = m */; ++i)
            p();

        // w_M [nanoseconds]
        double measured_walltime = static_cast<double>(t.elapsed());

        // w_T [nanoseconds]
        double theoretical_walltime =
            static_cast<double>(this->tasks_ * this->payload_duration_);

        // Overhead = Measured Walltime - Theoretical Walltime
        double overhead = measured_walltime - theoretical_walltime;

        results_type results(
            p.average_precision()
          , p.precision_uncertainty()
            // Average Overhead per Task
          , overhead / this->tasks_
            // f = A/a with a : arbitrary constant,
            // then sigma_f = sqrt((1/a)^2*sigma_A^2)
          , std::sqrt( (1.0/this->tasks_)*(1.0/this->tasks_)
                     * p.clock_uncertainty()*p.clock_uncertainty())
            );

        return results;

        ///////////////////////////////////////////////////////////////////////
    }

    void print_results(results_type results) const
    {
        if (this->io_ == htts2::csv_with_headers)
            std::cout
                << "OS-threads (Independent Variable),"
                << "Tasks per OS-thread (Control Variable) [tasks/OS-threads],"
                << "Payload Duration (Control Variable) [nanoseconds],"
                << "Average Precision per Task [nanoseconds/task],"
                << "Precision Uncertainty per Task [nanoseconds/task],"
                << "Amortized Overhead per Task [nanoseconds/task],"
                << "Amortized Overhead per Task Uncertainty [nanoseconds/task]"
                << "\n";

        std::cout
            << ( boost::format("%lu,%lu,%lu,%.14g,%.14g,%.14g,%.14g\n")
               % this->osthreads_
               % this->tasks_
               % this->payload_duration_
               % results.average_precision_
               % results.precision_uncertainty_
               % results.amortized_overhead_
               % results.overhead_uncertainty_
               )
            ;
    }
};

int main(int argc, char** argv)
{
    payload_precision_driver<> d(argc, argv);

    d.run();

    return 0;
}
