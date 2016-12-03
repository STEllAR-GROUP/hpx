//  Copyright (c) 2014-2015 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/components/iostreams/standard_streams.hpp>

#include <boost/format.hpp>

#include <algorithm>
#include <iostream>
#include <vector>
#include <memory>
#include <tuple>
#include <cstdio>
#include <string>
#include <map>

namespace hpx { namespace util {

//----------------------------------------------------------------------------
// an experimental profiling class which times sections of code and collects
// the timing from a tree of profiler objects to break down the time spend
// in nested sections
//
class simple_profiler {
  public:
    // time, level, count
    typedef std::tuple<double,int,int> valtype;

    simple_profiler(const char *title) {
        this->_parent = nullptr;
        this->_title  = title;
        this->_done   = false;
    }

    simple_profiler(simple_profiler &parent, const char *title) {
        this->_parent = &parent;
        this->_title  = title;
        this->_done   = false;
    }

    ~simple_profiler() {
      if (!this->_done) done();
    };

    void done() {
        double elapsed = this->_timer.elapsed();
        if (this->_parent) {
          this->_parent->addProfile(this->_title, std::make_tuple(elapsed,0,1));
          std::for_each(this->_profiles.begin(), this->_profiles.end(),
            [=](std::map<const char *, valtype>::value_type &p) {
              this->_parent->addProfile(p.first, p.second);
            }
          );
        }
        else {
          // get the max depth of the profile tree so we can prepare string lengths
          int maxlevel = 0;
          std::for_each(this->_profiles.begin(), this->_profiles.end(),
              [&](std::map<const char *, valtype>::value_type &p) {
              maxlevel = (std::max)(maxlevel, std::get<1>(p.second));
            }
          );
          // prepare format string for output
          char const* fmt1 = "Profile %20s : %2i %5i %9.3f %s %7.3f";
          std::string fmt2 = "Total   " + std::string(41,' ') + " %s %7.3f";
          // add this level to top of list
          this->_profiles[this->_title] = std::make_tuple(elapsed,0,1);
          // print each of the sub nodes
          std::vector<double> level_totals(5,0);
          int last_level = 0;
          hpx::cout << std::string(58+maxlevel*9,'-') << "\n";
          for (auto p=this->_profiles.begin(); p!=this->_profiles.end(); ) {
              int &level = std::get<1>(p->second);
              level_totals[level] += std::get<0>(p->second);
              if (level<last_level) {
                hpx::cout << std::string(52,' ') << std::string (last_level*9, ' ')
                    << "------\n";
                hpx::cout << (boost::format(fmt2)
                  % std::string (last_level*9, ' ')
                  % (100.0*level_totals[last_level]/elapsed)) << "\n";
                last_level = level;
              }
              else if (level>last_level) {
                last_level = level;
              }
              hpx::cout << (boost::format(fmt1)
                  % p->first
                  % level
                  % std::get<2>(p->second)
                  % std::get<0>(p->second)
                  % std::string (level*9, ' ')
                  % (100.0*std::get<0>(p->second)/elapsed)) << "\n";
              if ((++p)==this->_profiles.end()) {
                hpx::cout << std::string(52,' ') << std::string (last_level*9, ' ')
                    << "------\n";
                hpx::cout << (boost::format(fmt2)
                  % std::string (last_level*9, ' ')
                  % (100.0*level_totals[last_level]/elapsed)) << "\n";
                last_level = level;
              }
          }
          hpx::cout << std::string(58+maxlevel*9,'-') << "\n";
        }
        this->_done = true;
    }

    void addProfile(const char *title, valtype value)
    {
        if (this->_profiles.find(title) == this->_profiles.end()) {
            std::get<1>(value) += 1;                 // level
            this->_profiles[title] = value;
        }
        else {
            valtype &val = this->_profiles[title];
            std::get<0>(val)  += std::get<0>(value); // time
            std::get<2>(val)  += 1;                  // count
        }
    }
    //
    simple_profiler                              *_parent;
    hpx::util::high_resolution_timer              _timer;
    const char *                                  _title;
    std::map<const char *, valtype>               _profiles;
    bool                                          _done;
};

} }
