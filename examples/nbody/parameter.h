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

struct body {
    int node_type;
    double mass;
    double px, py, pz;
    double vx, vy, vz;
    double ax, ay, az;
private:
    // serialization support
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & node_type;
        ar & mass;
        ar & px;
        ar & py;
        ar & pz;
        ar & vx;
        ar & vy;
        ar & vz;
        ar & ax;
        ar & ay;
        ar & az;
    }
};


#if defined(__cplusplus)
extern "C" {
#endif




struct Par {
      int loglevel;
      had_double_type output;
      int output_stdout;
      int rowsize;
      std::string input_file;
      std::vector <std::vector<int> > iList;
      std::vector <std::vector<int> > bilist;
      std::vector<body> bodies; 
      double dtime;
      double eps;
      double tolerance;
      double half_dt;
      double softening_2;
      double inv_tolerance_2;
      int iter;
      int num_bodies;
      int num_iterations;
      double part_mass;
      int granularity;
      int num_pxpar;
      int extra_pxpar;
};

#if defined(__cplusplus)
}
#endif

#endif

