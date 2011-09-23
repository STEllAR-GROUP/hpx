//  Copyright (c) 2007-2011 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_DISTPX_MAR_24_2008_1118AM)
#define HPX_DISTPX_MAR_24_2008_1118AM

#include <hpx/config.hpp>
#include <hpx/version.hpp>
#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/include/runtime.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/performance_counters.hpp>

/// \mainpage
///
/// \section intro_sec Introduction
/// Current advances in high performance computing continue to suffer from the
/// issues plaguing parallel computation. These issues include, but are not 
/// limited to, ease of programming, inability to handle dynamically changing 
/// workloads, scalability and efficient utilization of system resources. 
/// Emerging technological trends such as multi-core processors further 
/// highlight limitations of existing parallel computation models. To mitigate 
/// the aforementioned problems, it is necessary to rethink the approach to 
/// parallelization models. ParalleX contains mechanisms such as 
/// multi-threading, parcels, global name space support, percolation and local 
/// control objects (LCO). By design, ParalleX overcomes limitations of current 
/// models of parallelism by alleviating contention, latency, overhead and 
/// starvation. With ParalleX, it is further possible to increase performance 
/// by at least an order of magnitude on challenging parallel algorithms, e.g., 
/// dynamic directed graph algorithms. Finally, an additional benefit of 
/// ParalleX may manifest itself through a reduction in power consumption.
///
/// ParalleX (PX) is a parallel execution model that offers an alternative to 
/// the conventional computation models, such as message passing. ParalleX 
/// distinguishes itself by:
///   - Split-phase transaction model
///   - Message-driven
///   - Distributed shared memory (not cache coherent)
///   - Multi-threaded
///   - Futures Synchronization
///   - Local Control Objects (LCO)
///   - Synchronization for anonymous producer-consumer scenarios
///   - Percolation (pre-staging of task data)
///
/// The ParalleX model is intrinsically latency hiding, delivering an abundance 
/// of parallelism in their diversity of form and size within a hierarchical 
/// distributed shared named space environment. The goal of this innovative 
/// strategy is to enable future systems delivering very high efficiency, 
/// increased scalability and ease of programming (through custom developed 
/// programming language, Agincourt). ParalleX can contribute to the 
/// significant improvements in the design of all levels of computing systems 
/// and their usage from application algorithms and their programming languages 
/// to system architecture and hardware design together with their supporting 
/// compilers and operating system software.
///
/// HPX is a second-generation implementation of the ParalleX execution model 
/// which is currently under development. Building upon the experience gained 
/// from DistPX, HPX is designed to be a modular, feature-complete, and 
/// performance-oriented representation of the ParalleX model. Specifically, 
/// HPX will provide complete critical path handling mechanisms. HPX will 
/// provide the notion of processes that transcend multiple localities. 
/// Percolation will also be implemented in HPX, to further facilitate 
/// heterogeneous computing. The modular design of HPX will allow for software 
/// optimizations at a component level, with the future goal of moving some 
/// functionality into hardware.
/// 

#endif

