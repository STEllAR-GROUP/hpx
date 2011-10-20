//  Copyright (c) 2009-2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#if !defined(HPX_REFINE_SEP_07_2011_0834AM)
#define HPX_REFINE_SEP_07_2011_0834AM

const int maxgids = 1000;

#include <hpx/hpx.hpp>
#include <examples/adaptive1d/dataflow/dynamic_stencil_value.hpp>
#include <examples/adaptive1d/dataflow/functional_component.hpp>
#include <examples/adaptive1d/dataflow/dataflow_stencil.hpp>
#include <examples/adaptive1d/stencil/stencil.hpp>
#include <examples/adaptive1d/stencil/stencil_data.hpp>
#include <examples/adaptive1d/stencil/stencil_functions.hpp>
#include <examples/adaptive1d/stencil/logging.hpp>

using hpx::components::adaptive1d::parameter;
using hpx::naming::id_type;

int level_refine(int level,parameter &par,
                 boost::shared_ptr<std::vector<id_type> > &result_data,
                 double time);

int compute_error(std::vector<double> &error,int nx0,
                                double minx0,
                                double maxx0,
                                double h,double t,
                                int gi,
                boost::shared_ptr<std::vector<id_type> > &result_data,
                                parameter &par);
int level_combine(std::vector<double> &error, std::vector<double> &localerror,
                  int mini,
                  int nxl, int nx);
int grid_return_existence(int gridnum,parameter &par);
int grid_find_bounds(int gi,double &minx,double &maxx,
                            parameter &par);
int level_return_start(int level,parameter &par);
int level_return_start(int level,parameter &par);
int level_find_bounds(int level, double &minx, double &maxx,
                                 parameter &par);
int compute_numrows(parameter &par);
int compute_rowsize(parameter &par);
int increment_gi(int level,int nx,
                 double lminx, double lmaxx,
                 double hl,int refine_factor,parameter &par);
bool intersection(double xmin,double xmax,double xmin2,double xmax2);
bool floatcmp_le(double const& x1, double const& x2);
int floatcmp(double const& x1, double const& x2);
int level_makeflag_simple(std::vector<int> &flag,std::vector<double> &error,int nxl,double ethreshold);
int level_bbox(int level,parameter &par);
int ballpark(double const& x1, double const& x2,double const& epsilon);

#endif
