//  Copyright (c) 2011-2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2013-2014 Thomas Heller
//  Copyright (c) 2013-2014 Patricia Grubel
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "htts2.hpp"

#include <boost/atomic.hpp>
#include <boost/lexical_cast.hpp>

#include <qthread/qthread.h>
#include <qthread/qloop.h>

typedef boost::chrono::steady_clock BaseClock;

extern "C" void stage_tasks(
    size_t start,
    size_t stop,
    void* payload_duration_
    )
{
    htts2::payload<BaseClock>(reinterpret_cast<boost::uint64_t>
        (payload_duration_ /* = p */));
}

struct qthreads_driver : htts2::driver
{
    qthreads_driver(int argc, char** argv)
      : htts2::driver(argc, argv)
    {}

    void run()
    {
        setenv("QT_NUM_SHEPHERDS",
            boost::lexical_cast<std::string>(this->osthreads_).c_str(), 1);
        setenv("QT_NUM_WORKERS_PER_SHEPHERD", "1", 1);

        qthread_initialize();

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

        qt_loop(0, this->tasks_ * this->osthreads_, stage_tasks,
            reinterpret_cast<void*>(this->payload_duration_));

        // w_M [nanoseconds]
        results = t.elapsed();

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
    qthreads_driver d(argc, argv);

    d.run();

    return 0;
}

