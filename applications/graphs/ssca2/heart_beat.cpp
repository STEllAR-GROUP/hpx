//  Copyright (c) 2009-2010 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>

#include <hpx/hpx.hpp>
#include <hpx/runtime/actions/plain_action.hpp>

#include <boost/program_options.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/export.hpp>

#include <hpx/performance_counters/stubs/performance_counter.hpp>

using namespace hpx;
namespace po = boost::program_options;

///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
int monitor(double frequency, double duration, double rate);

typedef
    actions::plain_result_action3<int, double, double, double, monitor>
monitor_action;

///////////////////////////////////////////////////////////////////////////////
// This is one long-running thread.
//
// Currently only monitors "this" locality, and never stops :-).

HPX_REGISTER_ACTION(monitor_action);

int monitor(double frequency, double duration, double rate)
{
  typedef hpx::naming::id_type gid_type;
  typedef std::vector<gid_type> gids_type;
  typedef util::high_resolution_timer timer_type;

  int num_times_empty = 0;
  int empty_threshold = 10000;
  bool keep_running = true;

  hpx::naming::resolver_client const& agas =
      hpx::applier::get_applier().get_agas_client();

  gid_type here = applier::get_applier().get_runtime_support_gid();

  // Build full performance counter name
  std::string queue("/queue(");
  queue += boost::lexical_cast<std::string>(here);
  queue += "/threadmanager)/length";

  // Get GID of performance counter
  naming::gid_type gid;
  agas.queryid(queue, gid);

  std::cout << "Begin timing block" << std::endl;

  // Start segment
  timer_type t;
  double current_time;

  while(keep_running)
  {
      std::cout << "\tBegin segment" << std::endl;

      double segment_start = t.elapsed();
      do {
          std::cout << "\t\tBegin monitoring block" << std::endl;

          // Start monitoring phase
          double monitor_start = t.elapsed();
          do{
              current_time = t.elapsed();

              // Access value of performance counter
              hpx::performance_counters::counter_value value;
              value =
                  hpx::performance_counters::stubs::
                    performance_counter::get_value(gid);

              if (hpx::performance_counters::status_valid_data == value.status_)
              {
                  std::cout << "run:" << current_time << " "
                            << value.value_ << std::endl;
              }

              value.value_ == 0 ? ++num_times_empty : num_times_empty=0;
              if (num_times_empty > empty_threshold)
                  keep_running = false;

              // Adjust rate of pinging values
              double delay_start = t.elapsed();
              do {
                  current_time = t.elapsed();
              } while(current_time - delay_start < rate);
          } while (current_time - monitor_start < duration && keep_running);
      } while (current_time - segment_start < frequency && keep_running);

      if (keep_running)
      {
          // Adjust rate of monitoring phases
          double pause_start = t.elapsed();
          do {
              current_time = t.elapsed();
          } while(current_time - pause_start < (frequency-duration));
      }
  }

  return 0;
}
