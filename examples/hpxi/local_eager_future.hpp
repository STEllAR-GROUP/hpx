////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file BOOST_LICENSE_1_0.rst or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(PHXPR_16B9AB37_5CB9_45BE_BABF_599E0424D99D)
#define PHXPR_16B9AB37_5CB9_45BE_BABF_599E0424D99D

#include <phxpr/primitives/binary.hpp>

namespace phxpr {
namespace hpx { 

utree local_eager_future_function (utree const&);
 
typedef ::hpx::actions::plain_result_action1<
  utree, utree const&, local_eager_future_function
> local_eager_future_action;

struct local_future_wait: actor<local_future_wait> {
  ::hpx::naming::id_type gid;
  utree arg;
  ::hpx::lcos::eager_future<local_eager_future_action> future;

  local_future_wait (::hpx::naming::id_type const& gid_, utree const& arg_):
    gid(gid_), arg(arg_), future(gid_, arg_) { }
  
  utree eval (utree& ut) const
  { return future.get(); }


  // REVIEW: Will passing shallow references via smart pointers screw with
  // assignment once it's implemented?
  function_base* copy (void) const
  { return new local_future_wait(gid, arg); }
};


struct local_eager_future: phxpr::binary<local_eager_future> {
  // TODO: This should call through to apply, which means I need to write apply.
  utree eval (utree const& f, utree const& arg) const {
    ::hpx::naming::id_type prefix =
      ::hpx::applier::get_applier().get_runtime_support_gid();

    // REVIEW: This probably needs to be in the GPT, which means we need GPT
    // access from this intrinsic.
    return utree(new local_future_wait(prefix, arg)); 
  }
  
  function_base* duplicate (void) const
  { return new local_eager_future; } 
};

} // hpx
} // phxpr

#endif // PHXPR_16B9AB37_5CB9_45BE_BABF_599E0424D99D

