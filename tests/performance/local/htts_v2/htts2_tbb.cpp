//  Copyright (c) 2011-2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2013-2014 Thomas Heller
//  Copyright (c) 2013-2014 Patricia Grubel
//
//  Distributed under the Boost Software License, Version 1.0. (See acctbbanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "htts2.hpp"

#include <tbb/task.h>
#include <tbb/task_scheduler_init.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include <chrono>
#include <cstdint>

template <typename BaseClock = std::chrono::steady_clock>
struct payload_functor : tbb::task
{
    std::uint64_t const payload_duration_;

    payload_functor(std::uint64_t payload_duration)
      : payload_duration_(payload_duration)
    {}

    tbb::task* execute()
    {
        htts2::payload<BaseClock>(payload_duration_ /* = p */);
        return 0;
    }
};

template <typename BaseClock = std::chrono::steady_clock>
struct stage_tasks_functor : public tbb::task
{
  private:
    struct range_spawner
    {
      private:
        stage_tasks_functor &outer;

      public:
        void operator() (const tbb::blocked_range<std::uint64_t>& r) const
        {
            for (std::uint64_t i = r.begin(); i != r.end(); ++i)
            {
                payload_functor<BaseClock> &a
                    = *new (outer.allocate_child())
                        payload_functor<BaseClock>(outer.payload_duration_);
                outer.spawn(a);
            }
        }

        range_spawner(stage_tasks_functor &ref) : outer(ref) {}
    };

  public:
    std::uint64_t osthreads_;
    std::uint64_t tasks_;
    std::uint64_t payload_duration_;

    stage_tasks_functor(
        std::uint64_t osthreads
      , std::uint64_t tasks
      , std::uint64_t payload_duration
        )
      : osthreads_(osthreads)
      , tasks_(tasks)
      , payload_duration_(payload_duration)
    {}

    tbb::task *execute()
    {
        set_ref_count(osthreads_ * tasks_); // Note the lack of a +1

        // Note the -2; this task counts as one of the ones we're spawning
        parallel_for(tbb::blocked_range<std::uint64_t>
                        (0, (osthreads_ * tasks_) - 2),
                     range_spawner(*this),
                     tbb::auto_partitioner());

        {
            payload_functor<BaseClock> &a = *new (tbb::task::allocate_child())
                payload_functor<BaseClock>(this->payload_duration_);

            spawn_and_wait_for_all(a);
        }

        return nullptr;
    }
};

template <typename BaseClock = std::chrono::steady_clock>
struct tbb_driver : htts2::driver
{
    tbb_driver(int argc, char** argv)
      : htts2::driver(argc, argv)
    {}

    void run()
    {
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

        {
            tbb::task_scheduler_init init(this->osthreads_);

            {
                stage_tasks_functor<BaseClock>& a =
                    *new (tbb::task::allocate_root())
                        stage_tasks_functor<BaseClock>
                            ( this->osthreads_
                            , this->tasks_
                            , this->payload_duration_);

                tbb::task::spawn_root_and_wait(a);
            }
        }

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
    tbb_driver<> d(argc, argv);

    d.run();

    return 0;
}

