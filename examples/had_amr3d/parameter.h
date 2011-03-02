//  Copyright (c) 2009 Matt Anderson
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_PARAMETER_OCT_23_2009_1249PM)
#define HPX_COMPONENTS_PARAMETER_OCT_23_2009_1249PM

#include "had_config.hpp"
#include <boost/serialization/vector.hpp>

class Array3D {
    size_t m_width, m_height;
    std::vector<int> m_data;
  public:
    Array3D(size_t x, size_t y, size_t z, int init = 0):
         m_width(x), m_height(y), m_data(x*y*z, init)
      {}
    int& operator()(size_t x, size_t y, size_t z) {
      return m_data.at(x + y * m_width + z * m_width * m_height);
    }
    int operator()(size_t x, size_t y, size_t z) const {
      return m_data.at(x + y * m_width + z * m_width * m_height);
    }
  private:
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) 
    {
        ar & m_width & m_height & m_data;
    }
};

#if defined(__cplusplus)
extern "C" {
#endif

struct Par {
      had_double_type lambda;
      int allowedl;
      int loglevel;
      had_double_type output;
      int output_stdout;
      int nt0;
      int nx[maxlevels];
      double refine_level[maxlevels];
      had_double_type minx0;
      had_double_type maxx0;
      had_double_type dx0;
      had_double_type dt0;
      had_double_type ethreshold;
      had_double_type R0;
      had_double_type delta;
      had_double_type amp;
      had_double_type amp_dot;
      had_double_type eps;
      int output_level;
      int granularity;
      int gw;
      std::vector<std::size_t> rowsize,level_row;
      std::vector<std::size_t> level_begin, level_end;
      hpx::memory::default_vector<had_double_type>::type min;
      hpx::memory::default_vector<had_double_type>::type max;
};

#if defined(__cplusplus)
}
#endif

#endif

