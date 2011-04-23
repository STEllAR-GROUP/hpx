//  Copyright (c) 2007-2011 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <cmath>

//#include "../nbody_c/stencil.hpp"
#include "../nbody_c/stencil_data.hpp"
#include "../nbody_c/stencil_functions.hpp"
#include "../had_config.hpp"
#include <stdio.h>

#include <boost/scoped_array.hpp>

#define UGLIFY 1

///////////////////////////////////////////////////////////////////////////////
// windows needs to initialize MPFR in each shared library
#if defined(BOOST_WINDOWS) 

#include "../init_mpfr.hpp"

namespace hpx { namespace components { namespace nbody 
{
    // initialize mpreal default precision
    init_mpfr init_;
}}}
#endif

///////////////////////////////////////////////////////////////////////////////
// local functions
inline int floatcmp(had_double_type const& x1, had_double_type const& x2) 
{
  // compare to floating point numbers
  static had_double_type const epsilon = 1.e-8;
  if ( x1 + epsilon >= x2 && x1 - epsilon <= x2 ) {
    // the numbers are close enough for coordinate comparison
    return 1;
  } else {
    return 0;
  }
}

///////////////////////////////////////////////////////////////////////////
int generate_initial_data(stencil_data* val, int item, int maxitems, int row,
    Par const& par)
{
//     // provide initial data for the given data value 
//     val->x = item;
//     val->y = item + 100; 
//     val->z = item + 1000;
// 
//     val->vx = 0.0;
//     val->vy = 0.0;
//     val->vz = 0.0;
// 
//     val->ax = 0.0;
//     val->ay = 0.0;
//     val->az = 0.0;
//     
//     val->row = row;
//     val->column = item;

  // std::cout << "gen_init_data: item : " << item << " item.size : " << par.bilist[item].size() << std::endl;
  

  
//  if (par.bilist[item].size() != 0)
//  {  
        unsigned long num_real_par = 0;
        if (par.extra_pxpar != 0)
        {
            if (item < par.num_pxpar-1)
                num_real_par = par.granularity;
            else if (item == par.num_pxpar-1)
                num_real_par = par.extra_pxpar;
            else if (item >= par.num_pxpar)
                BOOST_ASSERT("ERROR: Compute_index is more than number of PX particles");
        }
        else if(par.extra_pxpar == 0)
        {
            if (item < par.num_pxpar)
                num_real_par = par.granularity;
            else if (item >= par.num_pxpar)
                BOOST_ASSERT("ERROR: Compute_index is more than number of PX particles");
        }

//         if (item < par.num_pxpar-1)
//             num_real_par = par.granularity;
//         else if (item == par.num_pxpar-1 && par.extra_pxpar == 0)
//             num_real_par = par.granularity;
//         else if (item == par.num_pxpar-1 && par.extra_pxpar != 0)
//             num_real_par = par.extra_pxpar;
//         else if (item >= par.num_pxpar)
//             BOOST_ASSERT("ERROR: Compute_index is more than number of PX particles");
     
        for (unsigned long i = 0; i < num_real_par; ++i)
        {
            unsigned long iidx = (item * par.granularity) + i;
        //         int iidx = 0;
        //         for (int k =0; k < item; ++k)
        //             iidx += par.bilist[k].size();
        //         iidx = iidx+i;
        //         std::cout << "gen_init_data::   old index " << (item * par.granularity) + i << " new index " << iidx << std::endl;
                val->node_type.push_back(par.bodies[iidx].node_type); 
                val->mass.push_back(par.bodies[iidx].mass);
                val->x.push_back(par.bodies[iidx].px); 
                val->y.push_back(par.bodies[iidx].py);
                val->z.push_back(par.bodies[iidx].pz);  
                if (par.bodies[iidx].node_type == 1 && par.iter > 0)
                {
                    val->vx.push_back((par.bodies[iidx].vx+(0 - par.bodies[iidx].ax)*par.half_dt)); 
                    val->vy.push_back((par.bodies[iidx].vy+(0 - par.bodies[iidx].ay)*par.half_dt)); 
                    val->vz.push_back((par.bodies[iidx].vz+(0 - par.bodies[iidx].az)*par.half_dt));
                }
                else
                {
                    val->vx.push_back(par.bodies[iidx].vx);
                    val->vy.push_back(par.bodies[iidx].vy); 
                    val->vz.push_back(par.bodies[iidx].vz);
                }
                
                
                
//                 if (par.bodies[iidx].node_type == 1 && par.iter > 0)
//                 {
//                     val->ax.push_back((0 -par.bodies[iidx].ax)*par.half_dt); 
//                     val->ay.push_back((0 -par.bodies[iidx].ay)*par.half_dt); 
//                     val->az.push_back((0 -par.bodies[iidx].az)*par.half_dt);
//                 }
//                 else
//                 {
                    val->ax.push_back(0.0); 
                    val->ay.push_back(0.0); 
                    val->az.push_back(0.0); 
//                 }
//                 std::cout << "gen_init_data: PX Par-item " << item << " par.bodies.size " << par.bodies.size() << " global index " << iidx << " num_real_par " << num_real_par << " NodeType " <<  par.bodies[iidx].node_type << " AX val " << par.bodies[iidx].ax <<std::endl;
        //       std::cout << "gen_init_data: Row: " << row << " item: " << item << " x: " << val->x[i] << " y: " << val->y[i] << " z: " << val->z[i] << " iidx : " << iidx << std::endl;
            //     std::cout << " Maxitems " << maxitems << std::endl;
        //        std::cout << "I get till here"<< std::endl;
                val->row = row;
                val->column = item;
        }
//  }
//  
//  if(par.extra_pxpar == 0)
//  {
//         num_real_par = (par.num_pxpar * par.granularity) + par.extra_pxpar;
//         for (int i = 0; i < num_real_par; ++i)
//         {
//             int iidx = i;
//             val->node_type.push_back(par.bodies[i].node_type); 
//             val->x.push_back(par.bodies[i].px); 
//             val->y.push_back(par.bodies[i].py);
//             val->z.push_back(par.bodies[i].pz);  
// 
//             val->vx.push_back(par.bodies[i].vx);
//             val->vy.push_back(par.bodies[i].vy); 
//             val->vz.push_back(par.bodies[i].vz);
//             val->ax.push_back(0.0); 
//             val->ay.push_back(0.0); 
//             val->az.push_back(0.0); 
//             std::cout << "gen_init_data: PX Par-item " << item << " par.bodies.size " << par.bodies.size() << " global index " << iidx << std::endl;
// 
//         }
//         val->row = row;
//         val->column = item;
//  } else if(par.extra_pxpar != 0)
//  {
//       for (int i = 0; i < par.extra_pxpar; ++i)
//         {
//             int iidx = (item * par.granularity) + i;
//             val->node_type.push_back(par.bodies[iidx].node_type); 
//             val->x.push_back(par.bodies[iidx].px); 
//             val->y.push_back(par.bodies[iidx].py);
//             val->z.push_back(par.bodies[iidx].pz);  
// 
//             val->vx.push_back(par.bodies[iidx].vx);
//             val->vy.push_back(par.bodies[iidx].vy); 
//             val->vz.push_back(par.bodies[iidx].vz);
//             val->ax.push_back(0.0); 
//             val->ay.push_back(0.0); 
//             val->az.push_back(0.0); 
// 
//                 std::cout << "gen_init_data: PX Par-item " << item << " par.bodies.size " << par.bodies.size() << " global index " << iidx << " par.extra_pxpar " << par.extra_pxpar << std::endl;
//         //       std::cout << "gen_init_data: Row: " << row << " item: " << item << " x: " << val->x[i] << " y: " << val->y[i] << " z: " << val->z[i] << " iidx : " << iidx << std::endl;
//                 val->row = row;
//                 val->column = item;
//         }
//  }
// 

    return 1;
}

int rkupdate(std::vector< nodedata* > const& vecval, stencil_data* result, 
             int compute_index, Par const& par)
{
  return 1;
}
