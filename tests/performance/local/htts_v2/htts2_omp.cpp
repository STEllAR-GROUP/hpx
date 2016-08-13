//  Copyright (c) 2011-2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2013-2014 Thomas Heller
//  Copyright (c) 2013-2014 Patricia Grubel
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define HPX_NO_VERSION_CHECK
#include "htts2.hpp"

#include <chrono>
#include <cstdint>

#include <omp.h>

template <typename BaseClock = std::chrono::steady_clock>
struct omp_driver : htts2::driver
{
    omp_driver(int argc, char** argv)
      : htts2::driver(argc, argv)
    {}

    void run()
    {
        omp_set_num_threads(this->osthreads_);

        // Cold run
        //kernel();

        // Hot run
        results_type results = kernel();
        print_results(results);
    }

  private:
    typedef double results_type;

    results_type kernel()
    {
        ///////////////////////////////////////////////////////////////////////

        results_type results;

        htts2::timer<BaseClock> t;

        #pragma omp parallel
        #pragma omp single
        {
            // One stager per OS-thread.
            for (std::uint64_t n = 0; n < this->osthreads_; ++n)
                #if _OPENMP>=200805
                #pragma omp task untied
                #endif
                for (std::uint64_t m = 0; m < this->tasks_; ++m)
                    #if _OPENMP>=200805
                    #pragma omp task untied
                    #endif
                    htts2::payload<BaseClock>(this->payload_duration_ /* = p */);

            #if _OPENMP>=200805
            #pragma omp taskwait
            #endif

            // w_M [nanoseconds]
            results = t.elapsed();
        }

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
                << "Total Walltime [nanoseconds]"
                << "\n";

        std::cout
            << ( boost::format("%lu,%lu,%lu,%.14g\n")
               % this->osthreads_
               % this->tasks_
               % this->payload_duration_
               % results
               )
            ;
    }
};

int main(int argc, char** argv)
{
    omp_driver<> d(argc, argv);

    d.run();

    return 0;
}

