// Copyright (c) 2010-2011 Dylan Stark
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying 
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(PXGL_PXGL_20100827T1126)
#define PXGL_PXGL_20100827T1126

#define YAP_T_begin(name) { hpx::util::high_resolution_timer name;
#define YAP_T_end(name) fprintf(stderr, #name ": %f\n", name.elapsed()); }

#define YAP_T_nb_begin(name) hpx::util::high_resolution_timer name;
#define YAP_T_nb_end(name) fprintf(stderr, #name ": %f\n", name.elapsed());

#define YAPs(str,...) \
do { \
  fprintf(stderr, str, __VA_ARGS__); \
  fflush(stderr); \
} while (0)

////////////////////////////////////////////////////////////////////////////////
// Timing support
#define YAP_now(major,minor) \
do { \
  boost::uint32_t prefix = 0; \
  hpx::applier::applier* appl = hpx::applier::get_applier_ptr(); \
  if (appl) \
    prefix = hpx::naming::get_prefix_from_gid(appl->get_prefix()); \
  hpx::threads::thread_self* self = hpx::threads::get_self_ptr(); \
  if (0 != self) { \
    std::stringstream out; \
    out << std::hex << std::setw(8) << std::setfill('0') \
        << self->get_thread_id() << "." << self->get_thread_phase(); \
    fprintf(stderr, "%u---%f---T%s---%s---%s---\n", \
        prefix, \
        hpx::util::high_resolution_timer::now(), \
        out.str().c_str(), major, minor); \
    fflush(stderr); \
  } else { \
    fprintf(stderr, "%u---%f---T---%s---%s---\n", \
        prefix,\
        hpx::util::high_resolution_timer::now(), major, minor); \
    fflush(stderr); \
  } \
} while (0)

#endif
