//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c)      2011 Matt Anderson
//  Copyright (c)      2011 Bryce Lelbach
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include <examples/marduk/mesh/dynamic_stencil_value.hpp>
#include <examples/marduk/mesh/functional_component.hpp>
#include <examples/marduk/mesh/unigrid_mesh.hpp>
#include <examples/marduk/amr_c/stencil.hpp>
#include <examples/marduk/amr_c/stencil_data.hpp>
#include <examples/marduk/amr_c/stencil_functions.hpp>
#include <examples/marduk/amr_c/logging.hpp>
#include "fname.h"
#if defined(RNPL_FOUND)
#include <sdf.h>
#endif

#include <boost/format.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;
using boost::lexical_cast;

using hpx::bad_parameter;

using hpx::applier::get_applier;

using hpx::components::component_type;
using hpx::components::component_invalid;
using hpx::components::get_component_type;

using hpx::naming::id_type;

using hpx::util::section;
using hpx::util::high_resolution_timer;

using hpx::init;
using hpx::finalize;

using hpx::components::amr::parameter;
using hpx::components::amr::stencil;
using hpx::components::amr::unigrid_mesh;
using hpx::components::amr::server::logging;

void appconfig_option(std::string const& name, section const& pars,
                      std::string& data)
{
    if (pars.has_entry(name))
        data = pars.get_entry(name);
}

template <typename T>
void appconfig_option(std::string const& name, section const& pars, T& data)
{
    try {
        if (pars.has_entry(name))
            data = lexical_cast<T>(pars.get_entry(name));
    } catch (...) {
        std::string msg = boost::str(boost::format(
            "\"%1%\" is not a valid value for %2%")
            % pars.get_entry(name) % name);
        HPX_THROW_IN_CURRENT_FUNC(bad_parameter, msg);
    }
}

const int maxgids = 1000;
const int ALIVE = 1;
const int DEAD = 0;
const int PENDING = -1;

extern "C" {void FNAME(level_cluster)(double *flag,
                          double *sigi,double *sigj,double *sigk,
                          double *lapi, double *lapj, double *lapk,
                          double *asigi, double *asigj, double *asigk,
                          double *alapi, double *alapj, double *alapk,
                          double *time,
                          int *b_minx, int *b_maxx,int *b_miny,int *b_maxy,
                          int *b_minz, int *b_maxz,
                          double *minx, double *maxx,
                          double *miny, double *maxy,
                          double *minz, double *maxz,
                          int *numbox, int *nx, int *ny, int *nz,
                          int *clusterstyle, double *minefficiency, int *mindim,
                          int *ghostwidth, int *refine_factor,
                          double *minx0,double *miny0, double *minz0,
                          double *maxx0,double *maxy0, double *maxz0); }

extern "C" {void FNAME(load_scal_mult3d)(double *,double *,
                                         double *,int *,int *,int *); }
extern "C" {void FNAME(level_makeflag_simple)(double *flag,double *error,int *level,
                                       double *minx,double *miny,double *minz,
                                       double *h,int *nx,int *ny,int *nz,
                                       double *ethreshold,int *ghostwidth); }

extern "C" {void FNAME(level_clusterdd)(double *tmp_mini,double *tmp_maxi,
                         double *tmp_minj,double *tmp_maxj,
                         double *tmp_mink,double *tmp_maxk,
                         int *b_mini,int *b_maxi,
                         int *b_minj,int *b_maxj,
                         int *b_mink,int *b_maxk,
                         int *numbox,int *numprocs,int *maxbboxsize,
                         int *ghostwidth,int *refine_factor,int *mindim,
                         int *bound_width); }

int dataflow(std::vector<int> comm_list[maxgids],std::vector<int> prolong_list[maxgids],
             std::vector<int> restrict_list[maxgids],parameter &par);
int compute_error(std::vector<double> &error,int nx0, int ny0, int nz0,
                                double minx0,double miny0,double minz0,
                                double maxx0,double maxy0,double maxz0,
                                double h,double time,
                                int gi,
                                boost::shared_ptr<std::vector<id_type> > &result_data,
                                parameter &par);
int level_find_bounds(int level, double &minx, double &maxx,
                                 double &miny, double &maxy,
                                 double &minz, double &maxz, parameter &par);

int level_refine(int level,parameter &par,boost::shared_ptr<std::vector<id_type> > &result_data, double);
int level_mkall_dead(int level,parameter &par);
int level_return_start(int level,parameter &par);
int grid_return_existence(int gridnum,parameter &par);
int level_combine(std::vector<double> &error, std::vector<double> &localerror,
                  int mini,int minj,int mink,
                  int nxl,int nyl,int nzl,
                  int nx,int ny,int nz);
int grid_find_bounds(int gi,double &minx,double &maxx,
                     double &miny,double &maxy,
                     double &minz,double &maxz,parameter &par);
int compute_rowsize(parameter &par);
int compute_numrows(parameter &par);
int increment_gi(int level,int nx,int ny,int nz,
                 double lminx,double lminy,double lminz,
                 double lmaxx,double lmaxy,double lmaxz,
                 double hl,int refine_factor,parameter &par);
int write_sdf(double time,
              boost::shared_ptr<std::vector<id_type> > &result_data,
              parameter &par);

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{

    parameter par;

    // default pars
    par->loglevel       = 2;
    par->output         = 1.0;
    par->output_stdout  = 1;
    par->output_level   = 0;

    par->nx0            = 39;
    par->ny0            = 39;
    par->nz0            = 39;
    par->allowedl       = 2;
    par->lambda         = 0.025;
    par->nt0            = 100;
    par->refine_every   = 10;
    par->minx0          = -15.0;
    par->maxx0          =  15.0;
    par->miny0          = -15.0;
    par->maxy0          =  15.0;
    par->minz0          = -15.0;
    par->maxz0          =  15.0;
    par->ethreshold     = 1.e-4;
    par->ghostwidth     = 9;
    par->bound_width    = 5;
    par->shadow         = 0;
    par->refine_factor  = 2;
    par->minefficiency  = 0.9;
    par->clusterstyle   = 0;
    par->mindim         = 6;

    par->num_px_threads = 40;

    for (std::size_t i = 0; i < HPX_SMP_AMR3D_MAX_LEVELS; i++)
        par->refine_level[i] = 1.0;

    id_type rt_id = get_applier().get_runtime_support_gid();

    section root;
    hpx::components::stubs::runtime_support::get_config(rt_id, root);

    if (root.has_section("marduk"))
    {
        section pars = *(root.get_section("marduk"));

        appconfig_option<double>("lambda", pars, par->lambda);
        appconfig_option<std::size_t>("allowedl", pars, par->allowedl);
        appconfig_option<std::size_t>("loglevel", pars, par->loglevel);
        appconfig_option<double>("output", pars, par->output);
        appconfig_option<std::size_t>("output_stdout", pars, par->output_stdout);
        appconfig_option<std::size_t>("output_level", pars, par->output_level);
        appconfig_option<std::size_t>("nx0", pars, par->nx0);
        appconfig_option<std::size_t>("ny0", pars, par->ny0);
        appconfig_option<std::size_t>("nz0", pars, par->nz0);
        appconfig_option<std::size_t>("timesteps", pars, par->nt0);
        appconfig_option<std::size_t>("refine_every", pars, par->refine_every);
        appconfig_option<double>("maxx0", pars, par->maxx0);
        appconfig_option<double>("minx0", pars, par->minx0);
        appconfig_option<double>("maxy0", pars, par->maxy0);
        appconfig_option<double>("miny0", pars, par->miny0);
        appconfig_option<double>("maxz0", pars, par->maxz0);
        appconfig_option<double>("minz0", pars, par->minz0);
        appconfig_option<double>("ethreshold", pars, par->ethreshold);
        appconfig_option<std::size_t>("ghostwidth", pars, par->ghostwidth);
        appconfig_option<std::size_t>("bound_width", pars, par->bound_width);
        appconfig_option<std::size_t>("num_px_threads", pars, par->num_px_threads);
        appconfig_option<double>("minefficiency", pars, par->minefficiency);
        appconfig_option<std::size_t>("mindim", pars, par->mindim);
        appconfig_option<std::size_t>("shadow", pars, par->shadow);

        for (std::size_t i = 0; i < par->allowedl; i++)
        {
            std::string ref_level = "refine_level_"
                                  + lexical_cast<std::string>(i);
            appconfig_option<double>(ref_level, pars, par->refine_level[i]);
        }
    }

    // derived parameters
    if ( (par->nx0 % 2) == 0 || (par->ny0 % 2) == 0 || (par->nz0 % 2) == 0 ) 
    {
        std::string msg = boost::str(boost::format(
            "coarse mesh dimensions (%1%) (%2%) (%3%) must each be odd "
            ) % par->nx0 % par->ny0 % par->nz0);
        HPX_THROW_IN_CURRENT_FUNC(bad_parameter, msg);
    }
    if ( (par->nt0 % 2) != 0 )
    {
        std::string msg = boost::str(boost::format(
            "nt0 needs to be even: (%1%) "
            ) % par->nt0);
        HPX_THROW_IN_CURRENT_FUNC(bad_parameter, msg);
    } 

    if ( par->refine_every < 1 ) {
      // fixed mesh -- no AMR
      par->refine_every = par->nt0;
    }

    if ( (par->refine_every % 2) != 0 )
    {
        std::string msg = boost::str(boost::format(
            "refine_every needs to be even: (%1%) "
            ) % par->refine_every);
        HPX_THROW_IN_CURRENT_FUNC(bad_parameter, msg);
    } 

    if ( (par->nt0 % par->refine_every ) != 0 )
    {
        std::string msg = boost::str(boost::format(
            "nt0 (%1%) needs to be evenly divisible by refine_every (%2%) "
            ) % par->nt0 % par->refine_every);
        HPX_THROW_IN_CURRENT_FUNC(bad_parameter, msg);
    } 

    std::size_t time_grain_size = par->nt0/par->refine_every;

    std::cout << " Parameters    : " << std::endl;
    std::cout << " lambda        : " << par->lambda << std::endl;
    std::cout << " allowedl      : " << par->allowedl << std::endl;
    std::cout << " nx0           : " << par->nx0 << std::endl;
    std::cout << " ny0           : " << par->ny0 << std::endl;
    std::cout << " nz0           : " << par->nz0 << std::endl;
    std::cout << " timesteps     : " << par->nt0 << std::endl;
    std::cout << " refine_every  : " << par->refine_every << std::endl;
    std::cout << " minx0         : " << par->minx0 << std::endl;
    std::cout << " maxx0         : " << par->maxx0 << std::endl;
    std::cout << " miny0         : " << par->miny0 << std::endl;
    std::cout << " maxy0         : " << par->maxy0 << std::endl;
    std::cout << " miny0         : " << par->miny0 << std::endl;
    std::cout << " maxy0         : " << par->maxy0 << std::endl;
    std::cout << " minz0         : " << par->minz0 << std::endl;
    std::cout << " maxz0         : " << par->maxz0 << std::endl;
    std::cout << " ethreshold    : " << par->ethreshold << std::endl;
    std::cout << " ghostwidth    : " << par->ghostwidth << std::endl;
    std::cout << " bound_width   : " << par->bound_width << std::endl;
    std::cout << " num_px_threads: " << par->num_px_threads << std::endl;
    std::cout << " minefficiency : " << par->minefficiency << std::endl;
    std::cout << " mindim        : " << par->mindim << std::endl;
    std::cout << " shadow        : " << par->shadow << std::endl;
    std::cout << " --------------: " << std::endl;
    std::cout << " loglevel      : " << par->loglevel << std::endl;
    std::cout << " output        : " << par->output   << std::endl;
    std::cout << " output_stdout : " << par->output_stdout  << std::endl;
    std::cout << " output_level  : " << par->output_level  << std::endl;
    std::cout << " --------------: " << std::endl;
    for (std::size_t i = 0; i < par->allowedl; i++) {
      std::string ref_level = "refine_level_" + lexical_cast<std::string>(i);
      std::cout << " " << ref_level << ": " << par->refine_level[i] << std::endl;
    }

    // Derived parameters
    double hx = (par->maxx0 - par->minx0)/(par->nx0-1);
    double hy = (par->maxy0 - par->miny0)/(par->ny0-1);
    double hz = (par->maxz0 - par->minz0)/(par->nz0-1);
    double h = hx;
  
    // verify hx == hy == hz
    if ( fabs(hx - hy) < 1.e-10 && fabs(hy - hz) < 1.e-10 ) {
      h = hx;
    } else {
      std::string msg = boost::str(boost::format(
          "marduk does not handle unequal spacings (%1%) (%2%) (%3%) "
          ) % hx % hy % hz);
      HPX_THROW_IN_CURRENT_FUNC(bad_parameter, msg);
    }   
    par->h = h;

    //std::vector<double> localerror[maxgids];

    // memory allocation
    par->levelp.resize(par->allowedl+1);
    par->levelp[0] = 0; 

    // Compute the grid sizes
    boost::shared_ptr<std::vector<id_type> > placeholder;
    double initial_time = 0.0;
    int rc = level_refine(-1,par,placeholder,initial_time);
    for (std::size_t i=0;i<par->allowedl;i++) {
      rc = level_refine(i,par,placeholder,initial_time);
    }  

    // Output error
    //write_sdf(initial_time,placeholder,par);

    //std::vector<int> comm_list[maxgids];
    //std::vector<int> prolong_list[maxgids];
    //std::vector<int> restrict_list[maxgids];
    //rc = dataflow(comm_list,prolong_list,restrict_list,par);

    // compute the number of rows needed for two timesteps and the rowsize of each row
    rc = compute_numrows(par);
    rc = compute_rowsize(par);

    // get component types needed below
    component_type function_type = get_component_type<stencil>();
    component_type logging_type = get_component_type<logging>();

    {
        id_type here = get_applier().get_runtime_support_gid();

        high_resolution_timer t;

        unigrid_mesh um;
        um.create(here);
        int numsteps = par->refine_every/2;
        boost::shared_ptr<std::vector<id_type> > result_data(new std::vector<id_type>);

        for (std::size_t j=0;j<time_grain_size;j++) {
          double time = j*par->refine_every*par->lambda*par->h;

          result_data = um.init_execute(*result_data,time,function_type, 
                                        par->rowsize[0], numsteps,
                                par->loglevel ? logging_type : component_invalid, par);

          // Regrid
          time = (j+1)*par->refine_every*par->lambda*par->h;
          std::cout << " Completed time: " << time << std::endl;

          // grab the gi's associated with the steps just completed
          par->prev_gi.resize(0);
          for (std::size_t i=0;i<=par->allowedl;i++) {
            int gi = level_return_start(i,par);
            par->prev_gi.push_back(gi);
            while ( grid_return_existence(gi,par) ) {
              gi = par->gr_sibling[gi];
              par->prev_gi.push_back(gi);
            }
          }
          par->prev_gi2item = par->gi2item;
  
          for (std::size_t i=0;i<par->allowedl;i++) {
            rc = level_refine(i,par,result_data,time);
          }  
  
          // Output error
          //write_sdf(time,result_data,par);
          //std::cout << " PRE TEST rowsize " << par->rowsize[0] << std::endl;
  
          // re-compute rowsize information
          rc = compute_rowsize(par);
  
          //std::cout << " TEST rowsize " << par->rowsize[0] << std::endl;
        }

        std::cout << "Elapsed time: " << t.elapsed() << " [s]" << std::endl;

        for (std::size_t i = 0; i < result_data->size(); ++i)
            hpx::components::stubs::memory_block::free((*result_data)[i]);
    } // mesh needs to go out of scope before shutdown
    // initiate shutdown of the runtime systems on all localities

    finalize();
    return 0;
}

// dataflow {{{
int dataflow(std::vector<int> comm_list[maxgids],std::vector<int> prolong_list[maxgids],
             std::vector<int> restrict_list[maxgids],parameter &par) 
{
    // determine the communication pattern
  for (std::size_t i=0;i<=par->allowedl;i++) {
    std::vector<int> level_gids;
    int gi = level_return_start(i,par);
    level_gids.push_back(gi);
    gi = par->gr_sibling[gi];
    while ( grid_return_existence(gi,par) ) {
      level_gids.push_back(gi);
      gi = par->gr_sibling[gi];
    }

    // figure out the interaction list
    for (std::size_t j=0;j<level_gids.size();j++) {
      gi = level_gids[j];
      for (std::size_t k=j;k<level_gids.size();k++) {
        int gi2 = level_gids[k];

        if ( intersection(par->gr_minx[gi],par->gr_maxx[gi],
                          par->gr_miny[gi],par->gr_maxy[gi],
                          par->gr_minz[gi],par->gr_maxz[gi],
                          par->gr_minx[gi2],par->gr_maxx[gi2],
                          par->gr_miny[gi2],par->gr_maxy[gi2],
                          par->gr_minz[gi2],par->gr_maxz[gi2]) ) {
          comm_list[gi].push_back(gi2);
          if ( gi != gi2 ) comm_list[gi2].push_back(gi);
        }
      }
    }
  }

  // prolongation pattern
  // determine the communication pattern
  for (std::size_t i=0;i<par->allowedl;i++) {
    int gi = level_return_start(i,par);
    while ( grid_return_existence(gi,par) ) {
      int gi2 = level_return_start(i+1,par);
      while ( grid_return_existence(gi2,par) ) {
        double h = par->gr_h[gi2];
        if ( intersection(par->gr_minx[gi],par->gr_maxx[gi],
                          par->gr_miny[gi],par->gr_maxy[gi],
                          par->gr_minz[gi],par->gr_maxz[gi],
                          par->gr_minx[gi2],par->gr_minx[gi2]+h*par->ghostwidth/2,
                          par->gr_miny[gi2],par->gr_maxy[gi2],
                          par->gr_minz[gi2],par->gr_maxz[gi2]) ||
             intersection(par->gr_minx[gi],par->gr_maxx[gi],
                          par->gr_miny[gi],par->gr_maxy[gi],
                          par->gr_minz[gi],par->gr_maxz[gi],
                          par->gr_maxx[gi2]-h*par->ghostwidth/2,par->gr_maxx[gi2],
                          par->gr_miny[gi2],par->gr_maxy[gi2],
                          par->gr_minz[gi2],par->gr_maxz[gi2]) ||
             intersection(par->gr_minx[gi],par->gr_maxx[gi],
                          par->gr_miny[gi],par->gr_maxy[gi],
                          par->gr_minz[gi],par->gr_maxz[gi],
                          par->gr_minx[gi2],par->gr_maxx[gi2],
                          par->gr_miny[gi2],par->gr_miny[gi2]+h*par->ghostwidth/2,
                          par->gr_minz[gi2],par->gr_maxz[gi2]) ||
             intersection(par->gr_minx[gi],par->gr_maxx[gi],
                          par->gr_miny[gi],par->gr_maxy[gi],
                          par->gr_minz[gi],par->gr_maxz[gi],
                          par->gr_minx[gi2],par->gr_maxx[gi2],
                          par->gr_maxy[gi2]-h*par->ghostwidth/2,par->gr_maxy[gi2],
                          par->gr_minz[gi2],par->gr_maxz[gi2]) ||
             intersection(par->gr_minx[gi],par->gr_maxx[gi],
                          par->gr_miny[gi],par->gr_maxy[gi],
                          par->gr_minz[gi],par->gr_maxz[gi],
                          par->gr_minx[gi2],par->gr_maxx[gi2],
                          par->gr_miny[gi2],par->gr_maxy[gi2],
                          par->gr_minz[gi2],par->gr_minz[gi2]+h*par->ghostwidth/2) ||
             intersection(par->gr_minx[gi],par->gr_maxx[gi],
                          par->gr_miny[gi],par->gr_maxy[gi],
                          par->gr_minz[gi],par->gr_maxz[gi],
                          par->gr_minx[gi2],par->gr_maxx[gi2],
                          par->gr_miny[gi2],par->gr_maxy[gi2],
                          par->gr_maxz[gi2]-h*par->ghostwidth/2,par->gr_maxz[gi2]) ) {
          prolong_list[gi].push_back(gi2);
        }
        gi2 = par->gr_sibling[gi2];
      }
      gi = par->gr_sibling[gi];
    }
  }

  // restriction pattern
  // determine the communication pattern
  for (int i=par->allowedl;i>0;i--) {
    int gi = level_return_start(i,par);
    while ( grid_return_existence(gi,par) ) {
      int gi2 = level_return_start(i-1,par);
      while ( grid_return_existence(gi2,par) ) {
        gi2 = par->gr_sibling[gi2];
        if ( intersection(par->gr_minx[gi],par->gr_maxx[gi],
                          par->gr_miny[gi],par->gr_maxy[gi],
                          par->gr_minz[gi],par->gr_maxz[gi],
                          par->gr_minx[gi2],par->gr_maxx[gi2],
                          par->gr_miny[gi2],par->gr_maxy[gi2],
                          par->gr_minz[gi2],par->gr_maxz[gi2]) ) {
          restrict_list[gi].push_back(gi2);
        }
      }
      gi = par->gr_sibling[gi];
    }
  }

  // TEST
  for (std::size_t i=0;i<par->gr_minx.size();i++) {
    std::cout << " gi " << i << " has " << comm_list[i].size() << " interactions " << std::endl;
    std::cout << "                    " << restrict_list[i].size() << " restrict " << std::endl;
    std::cout << "                    " << prolong_list[i].size() << " prolong " << std::endl;
  }

  return 0;

}
// }}}

// level_refine {{{
int level_refine(int level,parameter &par,boost::shared_ptr<std::vector<id_type> > &result_data, double time)
{
  int rc;
  int numprocs = par->num_px_threads;
  int ghostwidth = par->ghostwidth;
  int bound_width = par->bound_width;
  double ethreshold = par->ethreshold;
  double minx0 = par->minx0;
  double miny0 = par->miny0;
  double minz0 = par->minz0;
  double maxx0 = par->maxx0;
  double maxy0 = par->maxy0;
  double maxz0 = par->maxz0;
  double h = par->h;
  int clusterstyle = par->clusterstyle;
  double minefficiency = par->minefficiency;
  int mindim = par->mindim;
  int refine_factor = par->refine_factor;

  // local vars
  std::vector<int> b_minx,b_maxx,b_miny,b_maxy,b_minz,b_maxz;
  std::vector<double> tmp_mini,tmp_maxi,tmp_minj,tmp_maxj,tmp_mink,tmp_maxk;
  int numbox;

  // hard coded parameters -- eventually to be made into full parameters
  int maxbboxsize = 1000000;

  int maxlevel = par->allowedl;
  if ( level == maxlevel ) {
    return 0;
  }
 
  double minx,miny,minz,maxx,maxy,maxz;
  double hl;
  int gi;
  if ( level == -1 ) {
    // we're creating the coarse grid
    minx = minx0;
    miny = miny0;
    minz = minz0;
    maxx = maxx0;
    maxy = maxy0;
    maxz = maxz0;
    gi = -1;
    hl = h * refine_factor;
  } else {
    // find the bounds of the level
    rc = level_find_bounds(level,minx,maxx,miny,maxy,minz,maxz,par);

    // find the grid index of the beginning of the level
    gi = level_return_start(level,par);

    // grid spacing for the level
    hl = par->gr_h[gi];
  }

  int nxl   =  (int) ((maxx-minx)/hl+0.5);
  int nyl   =  (int) ((maxy-miny)/hl+0.5);
  int nzl   =  (int) ((maxz-minz)/hl+0.5);
  nxl++;
  nyl++;
  nzl++;

  std::vector<double> error,localerror,flag;
  error.resize(nxl*nyl*nzl);
  flag.resize(nxl*nyl*nzl);

  if ( level == -1 ) {
    b_minx.resize(maxbboxsize);
    b_maxx.resize(maxbboxsize);
    b_miny.resize(maxbboxsize);
    b_maxy.resize(maxbboxsize);
    b_minz.resize(maxbboxsize);
    b_maxz.resize(maxbboxsize);

    tmp_mini.resize(maxbboxsize);
    tmp_maxi.resize(maxbboxsize);
    tmp_minj.resize(maxbboxsize);
    tmp_maxj.resize(maxbboxsize);
    tmp_mink.resize(maxbboxsize);
    tmp_maxk.resize(maxbboxsize);

    int numbox = 1;
    b_minx[0] = 1;
    b_maxx[0] = nxl;
    b_miny[0] = 1;
    b_maxy[0] = nyl;
    b_minz[0] = 1;
    b_maxz[0] = nzl;
    FNAME(level_clusterdd)(&*tmp_mini.begin(),&*tmp_maxi.begin(),
                           &*tmp_minj.begin(),&*tmp_maxj.begin(),
                           &*tmp_mink.begin(),&*tmp_maxk.begin(),
                           &*b_minx.begin(),&*b_maxx.begin(),
                           &*b_miny.begin(),&*b_maxy.begin(),
                           &*b_minz.begin(),&*b_maxz.begin(),
                           &numbox,&numprocs,&maxbboxsize,
                           &ghostwidth,&refine_factor,&mindim,
                           &bound_width);

    //std::cout << " numbox post DD " << numbox << std::endl;
    for (int i=0;i<numbox;i++) {
      if (i == numbox-1 ) {
        par->gr_sibling.push_back(-1);
      } else {
        par->gr_sibling.push_back(i+1);
      }

      int nx = (b_maxx[i] - b_minx[i])*refine_factor+1;
      int ny = (b_maxy[i] - b_miny[i])*refine_factor+1;
      int nz = (b_maxz[i] - b_minz[i])*refine_factor+1;

      double lminx = minx + (b_minx[i]-1)*hl;
      double lminy = miny + (b_miny[i]-1)*hl;
      double lminz = minz + (b_minz[i]-1)*hl;
      double lmaxx = minx + (b_maxx[i]-1)*hl;
      double lmaxy = miny + (b_maxy[i]-1)*hl;
      double lmaxz = minz + (b_maxz[i]-1)*hl;

      par->gr_minx.push_back(lminx);
      par->gr_miny.push_back(lminy);
      par->gr_minz.push_back(lminz);
      par->gr_maxx.push_back(lmaxx);
      par->gr_maxy.push_back(lmaxy);
      par->gr_maxz.push_back(lmaxz);
      par->gr_h.push_back(hl/refine_factor);
      par->gr_alive.push_back(PENDING);
      par->gr_nx.push_back(nx);
      par->gr_ny.push_back(ny);
      par->gr_nz.push_back(nz);
    }

    return 0;
  } else {
    // loop over all grids in level and get its error data
    gi = level_return_start(level,par);

    while ( grid_return_existence(gi,par) ) {
      int nx = par->gr_nx[gi];
      int ny = par->gr_ny[gi];
      int nz = par->gr_nz[gi];

      localerror.resize(nx*ny*nz);

      double lminx = par->gr_minx[gi];
      double lminy = par->gr_miny[gi];
      double lminz = par->gr_minz[gi];
      double lmaxx = par->gr_maxx[gi];
      double lmaxy = par->gr_maxy[gi];
      double lmaxz = par->gr_maxz[gi];

      rc = compute_error(localerror,nx,ny,nz,
                         lminx,lminy,lminz,
                         lmaxx,lmaxy,lmaxz,
                         par->gr_h[gi],time,gi,result_data,par);

      int mini = (int) ((lminx - minx)/hl+0.5);
      int minj = (int) ((lminy - miny)/hl+0.5);
      int mink = (int) ((lminz - minz)/hl+0.5);

      // sanity check
      if ( floatcmp(lminx-(minx + mini*hl),0.0) == 0 ||
           floatcmp(lminy-(miny + minj*hl),0.0) == 0 ||
           floatcmp(lminz-(minz + mink*hl),0.0) == 0 ||
           floatcmp(lmaxx-(minx + (mini+nx-1)*hl),0.0) == 0 ||
           floatcmp(lmaxy-(miny + (minj+ny-1)*hl),0.0) == 0 ||
           floatcmp(lmaxz-(minz + (mink+nz-1)*hl),0.0) == 0 ) {
        std::cerr << " Index problem " << std::endl;
        std::cerr << " lminx " << lminx << " " << minx + mini*hl << std::endl;
        std::cerr << " lminy " << lminy << " " << miny + minj*hl << std::endl;
        std::cerr << " lminz " << lminz << " " << minz + mink*hl << std::endl;
        std::cerr << " lmaxx " << lminx << " " << minx + (mini+nx-1)*hl << std::endl;
        std::cerr << " lmaxy " << lminy << " " << miny + (minj+ny-1)*hl << std::endl;
        std::cerr << " lmaxz " << lminz << " " << minz + (mink+nz-1)*hl << std::endl;
        exit(0);
      }

      rc = level_combine(error,localerror,
                         mini,minj,mink,
                         nxl,nyl,nzl,
                         nx,ny,nz);

      gi = par->gr_sibling[gi];
    }
  }

  rc = level_mkall_dead(level+1,par);

  double scalar = par->refine_level[level];
  FNAME(load_scal_mult3d)(&*error.begin(),&*error.begin(),&scalar,&nxl,&nyl,&nzl);
  int gw = par->ghostwidth;
  FNAME(level_makeflag_simple)(&*flag.begin(),&*error.begin(),&level,
                                                 &minx,&miny,&minz,&h,
                                                 &nxl,&nyl,&nzl,&ethreshold,&gw);

  // level_cluster
  std::vector<double> sigi,sigj,sigk;
  std::vector<double> asigi,asigj,asigk;
  std::vector<double> lapi,lapj,lapk;
  std::vector<double> alapi,alapj,alapk;

   sigi.resize(nxl); sigj.resize(nyl); sigk.resize(nzl);
  asigi.resize(nxl);asigj.resize(nyl);asigk.resize(nzl);
   lapi.resize(nxl); lapj.resize(nyl); lapk.resize(nzl);
  alapi.resize(nxl);alapj.resize(nyl);alapk.resize(nzl);

  b_minx.resize(maxbboxsize);
  b_maxx.resize(maxbboxsize);
  b_miny.resize(maxbboxsize);
  b_maxy.resize(maxbboxsize);
  b_minz.resize(maxbboxsize);
  b_maxz.resize(maxbboxsize);

  FNAME(level_cluster)(&*flag.begin(),&*sigi.begin(),&*sigj.begin(),&*sigk.begin(),
                       &*lapi.begin(),&*lapj.begin(),&*lapk.begin(),
                       &*asigi.begin(),&*asigj.begin(),&*asigk.begin(),
                       &*alapi.begin(),&*alapj.begin(),&*alapk.begin(),
                       &time,
                       &*b_minx.begin(),&*b_maxx.begin(),
                       &*b_miny.begin(),&*b_maxy.begin(),
                       &*b_minz.begin(),&*b_maxz.begin(),
                       &minx,&maxx,
                       &miny,&maxy,
                       &minz,&maxz,
                       &numbox,&nxl,&nyl,&nzl,
                       &clusterstyle,&minefficiency,&mindim,
                       &ghostwidth,&refine_factor,
                       &minx0,&miny0,&minz0,
                       &maxx0,&maxy0,&maxz0);
  //std::cout << " pre DD numbox TEST " << numbox << std::endl;

  //for (int i=0;i<numbox;i++) {
  //  std::cout << " bbox : " << b_minx[i] << " " << b_maxx[i] << std::endl; 
  //  std::cout << "      : " << b_miny[i] << " " << b_maxy[i] << std::endl; 
  //  std::cout << "      : " << b_minz[i] << " " << b_maxz[i] << std::endl; 
  //}

  // Take the set of subgrids to create and then output the same set but domain decomposed
  tmp_mini.resize(maxbboxsize);
  tmp_maxi.resize(maxbboxsize);
  tmp_minj.resize(maxbboxsize);
  tmp_maxj.resize(maxbboxsize);
  tmp_mink.resize(maxbboxsize);
  tmp_maxk.resize(maxbboxsize);

  FNAME(level_clusterdd)(&*tmp_mini.begin(),&*tmp_maxi.begin(),
                         &*tmp_minj.begin(),&*tmp_maxj.begin(),
                         &*tmp_mink.begin(),&*tmp_maxk.begin(),
                         &*b_minx.begin(),&*b_maxx.begin(),
                         &*b_miny.begin(),&*b_maxy.begin(),
                         &*b_minz.begin(),&*b_maxz.begin(),
                         &numbox,&numprocs,&maxbboxsize,
                         &ghostwidth,&refine_factor,&mindim,
                         &bound_width);

  //std::cout << " numbox post DD " << numbox << std::endl;
  int prev_tgi = 0;
  int tgi = 0;
  for (int i=0;i<numbox;i++) {
    //std::cout << " bbox: " << b_minx[i] << " " << b_maxx[i] << std::endl;
    //std::cout << "       " << b_miny[i] << " " << b_maxy[i] << std::endl;
    //std::cout << "       " << b_minz[i] << " " << b_maxz[i] << std::endl;
    int nx = (b_maxx[i] - b_minx[i])*refine_factor+1;
    int ny = (b_maxy[i] - b_miny[i])*refine_factor+1;
    int nz = (b_maxz[i] - b_minz[i])*refine_factor+1;

    double lminx = minx + (b_minx[i]-1)*hl;
    double lminy = miny + (b_miny[i]-1)*hl;
    double lminz = minz + (b_minz[i]-1)*hl;
    double lmaxx = minx + (b_maxx[i]-1)*hl;
    double lmaxy = miny + (b_maxy[i]-1)*hl;
    double lmaxz = minz + (b_maxz[i]-1)*hl;

    if ( i != 0 ) {
      prev_tgi = tgi;
    }

    // look for matching grid on level
    tgi = increment_gi(level,nx,ny,nz,lminx,lminy,lminz,
                     lmaxx,lmaxy,lmaxz,hl,refine_factor,par);

    if ( i == 0 ) {
      par->levelp[level+1] = tgi;
    }

    if (i == numbox-1) {
      par->gr_sibling[prev_tgi] = tgi;
      par->gr_sibling[tgi] = -1;
    } else if (i != 0 ) {
      par->gr_sibling[prev_tgi] = tgi;
    }

  }

  return 0;
}
// }}}

// compute_error {{{
int compute_error(std::vector<double> &error,int nx0, int ny0, int nz0,
                                double minx0,double miny0,double minz0,
                                double maxx0,double maxy0,double maxz0,
                                double h,double t,
                                int gi,
                boost::shared_ptr<std::vector<id_type> > &result_data,
                                parameter &par)
{
    // initialize some positive error
    if ( t < 1.e-8 ) {
      for (int k=0;k<nz0;k++) {
        double z = minz0 + k*h;
      for (int j=0;j<ny0;j++) {
        double y = miny0 + j*h;
      for (int i=0;i<nx0;i++) {
        double x = minx0 + i*h;

        // Provide initial error
        double d = 11.0;
        double A = -(x-0.5*d*cos(t))*(x-0.5*d*cos(t)) - (y+0.5*d*sin(t))*(y+0.5*d*sin(t)) - z*z;
        double B = -(x+0.5*d*cos(t))*(x+0.5*d*cos(t)) - (y-0.5*d*sin(t))*(y-0.5*d*sin(t)) - z*z;
        double Phi = exp(A) + exp(B);
        error[i+nx0*(j+ny0*k)] = Phi;
      } } }
    } else {
#if 0
      // TEST
      for (int k=0;k<nz0;k++) {
        double z = minz0 + k*h;
      for (int j=0;j<ny0;j++) {
        double y = miny0 + j*h;
      for (int i=0;i<nx0;i++) {
        double x = minx0 + i*h;

        // Provide initial error
        double d = 11.0;
        double A = -(x-0.5*d*cos(t))*(x-0.5*d*cos(t)) - (y+0.5*d*sin(t))*(y+0.5*d*sin(t)) - z*z;
        double B = -(x+0.5*d*cos(t))*(x+0.5*d*cos(t)) - (y-0.5*d*sin(t))*(y-0.5*d*sin(t)) - z*z;
        double Phi = exp(A) + exp(B);
        error[i+nx0*(j+ny0*k)] = Phi;
      } } }
#endif
//#if 0
      // This is the old mesh structure; we need the error for the new mesh structure
      // Some re-assembly is necessary
      // go through all of the old mesh gi's; see if they overlap this new mesh
      for (int step=0;step<par->prev_gi.size();step++) {
        // see if the new gi is the same as the old
        int gi = par->prev_gi[step];
        if ( gi != -1 ) {
          if ( floatcmp(minx0,par->gr_minx[gi]) && 
               floatcmp(miny0,par->gr_miny[gi]) && 
               floatcmp(minz0,par->gr_minz[gi]) && 
               floatcmp(h,par->gr_h[gi]) && 
               nx0 == par->gr_nx[gi] && 
               ny0 == par->gr_ny[gi] && 
               nz0 == par->gr_nz[gi] 
             ) {
            hpx::components::access_memory_block<hpx::components::amr::stencil_data>
                result( hpx::components::stubs::memory_block::get((*result_data)[par->gi2item[gi]]) );
            for (int k=0;k<nz0;k++) {
            for (int j=0;j<ny0;j++) {
            for (int i=0;i<nx0;i++) {
              error[i+nx0*(j+ny0*k)] = result->value_[i+nx0*(j+ny0*k)].error;
            } } }
          } else if ( intersection(minx0,maxx0,
                                   miny0,maxy0,
                                   minz0,maxz0,
                                   par->gr_minx[gi],par->gr_maxx[gi],
                                   par->gr_miny[gi],par->gr_maxy[gi],
                                   par->gr_minz[gi],par->gr_maxz[gi]) &&
                      floatcmp(h,par->gr_h[gi])
                    ) { 
            hpx::components::access_memory_block<hpx::components::amr::stencil_data>
                result( hpx::components::stubs::memory_block::get((*result_data)[par->gi2item[gi]]) );

            // find the intersection index
            double x1 = (std::max)(minx0,par->gr_minx[gi]); 
            double x2 = (std::min)(maxx0,par->gr_maxx[gi]); 
            double y1 = (std::max)(miny0,par->gr_miny[gi]); 
            double y2 = (std::min)(maxy0,par->gr_maxy[gi]); 
            double z1 = (std::max)(minz0,par->gr_minz[gi]); 
            double z2 = (std::min)(maxz0,par->gr_maxz[gi]); 
  
            //std::cout << " x TEST " << minx0 << " " << maxx0 << std::endl;
            //std::cout << " y TEST " << miny0 << " " << maxy0 << std::endl;
            //std::cout << " z TEST " << minz0 << " " << maxz0 << std::endl;
            //std::cout << " x gr TEST " << par->gr_minx[gi] << " " << par->gr_maxx[gi] << std::endl;
            //std::cout << " y gr TEST " << par->gr_miny[gi] << " " << par->gr_maxy[gi] << std::endl;
            //std::cout << " z gr TEST " << par->gr_minz[gi] << " " << par->gr_maxz[gi] << std::endl;
            //std::cout << " TEST " << x1 << " " << x2 << " " << y1 << " " << y2 << " " << z1 << " " << z2 << std::endl;
            //std::cout << " HELLO WORLD " << result->value_.size() << std::endl;

            int isize = (int) ( (x2-x1)/h );
            int jsize = (int) ( (y2-y1)/h );
            int ksize = (int) ( (z2-z1)/h );
  
            int lnx = par->gr_nx[gi]; 
            int lny = par->gr_ny[gi]; 
  
            int istart_dst = (int) ( (x1 - minx0)/h );
            int jstart_dst = (int) ( (y1 - miny0)/h );
            int kstart_dst = (int) ( (z1 - minz0)/h );
  
            int istart_src = (int) ( (x1 - par->gr_minx[gi])/h );
            int jstart_src = (int) ( (y1 - par->gr_miny[gi])/h );
            int kstart_src = (int) ( (z1 - par->gr_minz[gi])/h );
  
         //   std::cout << " TEST " << x1 << " " << x2 << " " << y1 << " " << y2 << " " << z1 << " " << z2 << std::endl;
         //   std::cout << " result val size " << result->value_.size() << std::endl;
         //   std::cout << " istart_src " << istart_src << std::endl;
         //   std::cout << " jstart_src " << jstart_src << std::endl;
         //   std::cout << " kstart_src " << kstart_src << std::endl;
         //   std::cout << " isize " << isize << std::endl;
         //   std::cout << " jsize " << jsize << std::endl;
         //   std::cout << " ksize " << ksize << std::endl;
         //   std::cout << " lnx " << lnx << std::endl;
         //   std::cout << " lny " << lny << std::endl;
         //   std::cout << " lnz " << par->gr_nz[i] << std::endl;
  
            for (int kk=0;kk<=ksize;kk++) {
            for (int jj=0;jj<=jsize;jj++) {
            for (int ii=0;ii<=isize;ii++) {
              int i = ii + istart_dst;
              int j = jj + jstart_dst;
              int k = kk + kstart_dst;
  
              int si = ii + istart_src;
              int sj = jj + jstart_src;
              int sk = kk + kstart_src;
              BOOST_ASSERT(i+nx0*(j+ny0*k) < error.size());
              BOOST_ASSERT(si+lnx*(sj+lny*sk) < result->value_.size());
              error[i+nx0*(j+ny0*k)] = result->value_[si+lnx*(sj+lny*sk)].error;
            } } }

          }
        }
      }
//#endif
    }

    return 0;
}
// }}}

// level_combine {{{
int level_combine(std::vector<double> &error, std::vector<double> &localerror,
                  int mini,int minj,int mink,
                  int nxl,int nyl,int nzl,
                  int nx,int ny,int nz)
{
  int il,jl,kl;
  // combine local grid error into global grid error
  for (int k=0;k<nz;k++) {
    kl = k + mink;
  for (int j=0;j<ny;j++) {
    jl = j + minj;
  for (int i=0;i<nx;i++) {
    il = i + mini;
    if ( error[il + nxl*(jl + nyl*kl)] < localerror[i+nx*(j+ny*k)] ) {
      error[il + nxl*(jl + nyl*kl)] = localerror[i+nx*(j+ny*k)];
    }
  } } }

  return 0;
}
// }}}

// level_mkall_dead {{{
int level_mkall_dead(int level,parameter &par)
{
  // Find the beginning of level
  int gi = level_return_start(level,par);

  while ( grid_return_existence(gi,par) ) {
    par->gr_alive[gi] = DEAD;
    gi = par->gr_sibling[gi];
  }

  return 0;
}
// }}}

// grid_return_existence {{{
int grid_return_existence(int gridnum,parameter &par)
{
  int maxsize = par->gr_minx.size();
  if ( (gridnum >= 0 ) && (gridnum < maxsize) ) {
    return 1;
  } else {
    return 0;
  }
}
// }}}

// grid_find_bounds {{{
int grid_find_bounds(int gi,double &minx,double &maxx,
                            double &miny,double &maxy,
                            double &minz,double &maxz,parameter &par)
{
  minx = par->gr_minx[gi];
  miny = par->gr_miny[gi];
  minz = par->gr_minz[gi];

  maxx = par->gr_maxx[gi];
  maxy = par->gr_maxy[gi];
  maxz = par->gr_maxz[gi];

  return 0;
}
// }}}

// level_return_start {{{
int level_return_start(int level,parameter &par)
{
  int maxsize = par->levelp.size();
  if ( level >= maxsize || level < 0 ) {
    return -1;
  } else {
    return par->levelp[level];
  }
}
// }}}

// level_find_bounds {{{
int level_find_bounds(int level, double &minx, double &maxx,
                                 double &miny, double &maxy,
                                 double &minz, double &maxz, parameter &par)
{
  int rc;
  minx = 0.0;
  miny = 0.0;
  minz = 0.0;
  maxx = 0.0;
  maxy = 0.0;
  maxz = 0.0;

  double tminx,tmaxx,tminy,tmaxy,tminz,tmaxz;

  int gi = level_return_start(level,par);

  if ( !grid_return_existence(gi,par) ) {
    std::cerr << " level_find_bounds PROBLEM: level doesn't exist " << level << std::endl;
    exit(0);
  }

  rc = grid_find_bounds(gi,minx,maxx,miny,maxy,minz,maxz,par);

  gi = par->gr_sibling[gi];
  while ( grid_return_existence(gi,par) ) {
    rc = grid_find_bounds(gi,tminx,tmaxx,tminy,tmaxy,tminz,tmaxz,par);
    if ( tminx < minx ) minx = tminx;
    if ( tminy < miny ) miny = tminy;
    if ( tminz < minz ) minz = tminz;
    if ( tmaxx > maxx ) maxx = tmaxx;
    if ( tmaxy > maxy ) maxy = tmaxy;
    if ( tmaxz > maxz ) maxz = tmaxz;
    gi = par->gr_sibling[gi];
  }

  return 0;
}
// }}}

// compute_numrows {{{
int compute_numrows(parameter &par)
{
    // for each row, record what the lowest level on the row is
    std::size_t num_rows = 1 << par->allowedl;

    // account for prolongation and restriction (which is done every other step)
    if (par->allowedl > 0)
        num_rows += (1 << par->allowedl) / 2;

    num_rows *= 2; // we take two timesteps in the mesh
    par->num_rows = num_rows;

    int ii = -1; 
    for (std::size_t i = 0; i < num_rows; ++i)
    {
        if (((i + 5) % 3) != 0 || (par->allowedl == 0))
            ii++;

        std::size_t level = 0;
        for (std::size_t j = par->allowedl; j>=0; --j)
        {
            if ((ii % (1 << j)) == 0)
            {
                level = par->allowedl - j;
                par->level_row.push_back(level);
                break;
            }
        }
    }
    return 0;
}
// }}}

// compute_rowsize {{{
int compute_rowsize(parameter &par)
{
    // discover each rowsize
    std::size_t count;
    par->rowsize.resize(par->allowedl+1);
    for (std::size_t i=0;i<=par->allowedl;i++) {
      count = 0; 
      int gi = level_return_start(i,par);
      count++;
      gi = par->gr_sibling[gi];
      while ( grid_return_existence(gi,par) ) {
        count++;
        gi = par->gr_sibling[gi];
      }
      par->rowsize[i] = count;
    } 
    for (std::size_t i=0;i<=par->allowedl;i++) {
      for (std::size_t j=i+1;j<=par->allowedl;j++) {
        par->rowsize[i] += par->rowsize[j];
      }
    }

    // here we create a correspondence getween the gi number used in 'had' 
    // and the index number used in hpx
    par->item2gi.resize(par->rowsize[0]);
    par->gi2item.resize(maxgids);
    count = 0;
    for (int i=par->allowedl;i>=0;i--) {
      int gi = level_return_start(i,par);
      par->item2gi[count] = gi; 
      par->gi2item[gi] = count; 
      count++;
      gi = par->gr_sibling[gi];
      while ( grid_return_existence(gi,par) ) {
        par->item2gi[count] = gi; 
        par->gi2item[gi] = count; 
        count++;
        gi = par->gr_sibling[gi];
      }
    } 

    return 0;
}
// }}}

// increment_gi {{{
int increment_gi(int level,int nx,int ny,int nz,
                 double lminx,double lminy,double lminz,
                 double lmaxx,double lmaxy,double lmaxz,
                 double hl,int refine_factor,parameter &par)
{
    int gi_tmp = level_return_start(level+1,par);
    bool found = false;
    while ( grid_return_existence(gi_tmp,par) ) {
      int lnx = par->gr_nx[gi_tmp];
      int lny = par->gr_ny[gi_tmp];
      int lnz = par->gr_nz[gi_tmp];
      if ( floatcmp(lminx,par->gr_minx[gi_tmp]) &&
           floatcmp(lminy,par->gr_miny[gi_tmp]) &&
           floatcmp(lminz,par->gr_minz[gi_tmp]) &&
           nx == lnx && ny == lny && nz == lnz ) {
        par->gr_alive[gi_tmp] = PENDING;
        found = true;
        break;
      }
      gi_tmp = par->gr_sibling[gi_tmp];
    }
    if ( found ) return gi_tmp;

    // none found; this is a new gi
    int gi = par->gr_h.size();
    par->gr_alive.resize(gi+1);
    par->gr_minx.resize(gi+1);
    par->gr_miny.resize(gi+1);
    par->gr_minz.resize(gi+1);
    par->gr_maxx.resize(gi+1);
    par->gr_maxy.resize(gi+1);
    par->gr_maxz.resize(gi+1);
    par->gr_h.resize(gi+1);
    par->gr_nx.resize(gi+1);
    par->gr_ny.resize(gi+1);
    par->gr_nz.resize(gi+1);
    par->gr_sibling.resize(gi+1);

    par->gr_alive[gi] = PENDING;
    par->gr_minx[gi]  = lminx;
    par->gr_miny[gi]  = lminy;
    par->gr_minz[gi]  = lminz;
    par->gr_maxx[gi]  = lmaxx;
    par->gr_maxy[gi]  = lmaxy;
    par->gr_maxz[gi]  = lmaxz;
    par->gr_h[gi]     = hl/refine_factor;
    par->gr_nx[gi]    = nx;
    par->gr_ny[gi]    = ny;
    par->gr_nz[gi]    = nz;
    
    return gi;

};
// }}}

// write_sdf {{{
int write_sdf(double time,
              boost::shared_ptr<std::vector<id_type> > &result_data,
              parameter &par)
{

  for (std::size_t i=0;i<=par->allowedl;i++) {
    int gi = level_return_start(i,par);
    while ( grid_return_existence(gi,par) ) {
      int nx = par->gr_nx[gi];
      int ny = par->gr_ny[gi];
      int nz = par->gr_nz[gi];
      double h = par->gr_h[gi];
      double minx = par->gr_minx[gi];
      double miny = par->gr_miny[gi];
      double minz = par->gr_minz[gi];
      double maxx = par->gr_maxx[gi];
      double maxy = par->gr_maxy[gi];
      double maxz = par->gr_maxz[gi];

      std::vector<double> localerror;
      localerror.resize(nx*ny*nz);
      int rc = compute_error(localerror,nx,ny,nz,
                   minx,miny,minz,maxx,maxy,maxz,h,time,gi,result_data,par);
      int shape[3];
      std::vector<double> coord;
      coord.resize(nx+ny+nz);
      for (int i=0;i<nx;i++) {
        coord[i] = minx + i*h;
      }
      for (int i=0;i<ny;i++) {
        coord[i+nx] = miny + i*h;
      }
      for (int i=0;i<nz;i++) {
        coord[i+nx+ny] = minz + i*h;
      }

      shape[0] = nx;
      shape[1] = ny;
      shape[2] = nz;
      char nme[80];
      char crdnme[80];
      sprintf(nme,"error");
      sprintf(crdnme,"x|y|z");
#if defined(RNPL_FOUND)
      gft_out_full(nme,time,shape,crdnme, 3,&*coord.begin(),&*localerror.begin());
#endif
      gi = par->gr_sibling[gi];
    }
  }

  return 0;
}
// }}}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description
       desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    // Initialize and run HPX
    return init(desc_commandline, argc, argv);
}

