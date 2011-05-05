//  Copyright (c) 2010-2011 Dylan Stark
//  Copyright (c)      2011 Bryce Lelbach
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/components/plain_component_factory.hpp>

#include <boost/program_options.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/export.hpp>

using namespace hpx;
namespace po = boost::program_options;

typedef hpx::lcos::detail::dataflow_variable<int, int> detail_dataflow_type;
HPX_DEFINE_GET_COMPONENT_TYPE(detail_dataflow_type);

///////////////////////////////////////////////////////////////////////////////
// More helpers
template<typename Action, typename Arg0>
inline void apply(Arg0 arg0, naming::id_type k, naming::id_type h)
{
  hpx::applier::apply<Action>(get_runtime().get_process().next(), arg0, k, h);
}

template<typename Action, typename Arg0, typename Arg1>
inline void apply(Arg0 arg0, Arg1 arg1, naming::id_type k, naming::id_type h)
{
  hpx::applier::apply<Action>(get_runtime().get_process().next(), arg0, arg1, k, h);
}

template<typename Arg0>
inline void trigger(naming::id_type k, Arg0 arg0)
{
  typedef typename
      lcos::template base_lco_with_value<Arg0>::set_result_action set_action;
  hpx::applier::apply<set_action>(k,arg0);
}

template<typename Value>
inline Value get(naming::id_type k)
{
  typedef typename
      lcos::template base_lco_with_value<Value>::get_value_action get_action;
  return lcos::eager_future<get_action>(k).get();
}

///////////////////////////////////////////////////////////////////////////////
// The following CPS example was adapted from 
// http://en.wikipedia.org/wiki/Continuation-passing_style (Jun 30, 2010).
//
// def factorial(n, k):
//     factorial_aux(n, 1, k)
//
// def factorial_aux(n, a, k):
//     if n == 0:
//         k(a)
//     else:
//         factorial_aux(n-1, n*a, k)
//
// >>> k = dataflow_variable()
// >>> factorial(3, k)
//
// >>> print k
// 6

///////////////////////////////////////////////////////////////////////////////
void factorial_get_aux(int, int, naming::id_type, naming::id_type);
typedef actions::plain_action4<int, int, naming::id_type, naming::id_type, factorial_get_aux> 
    factorial_get_aux_action;
HPX_REGISTER_PLAIN_ACTION(factorial_get_aux_action);

void factorial_get(int, naming::id_type, naming::id_type);
typedef actions::plain_action3<int, naming::id_type, naming::id_type, factorial_get> 
    factorial_get_action;
HPX_REGISTER_PLAIN_ACTION(factorial_get_action);

void factorial_get_aux(int n, int a, naming::id_type k, naming::id_type h)
{
  if (n == 0)
  {
    std::cout << "Triggering k(" << a << ")" << std::endl;
    trigger<int>(k, a);

    std::cout << "Getting value of k" << std::endl;
    get<int>(k);

    std::cout << "Triggering h(0)" << std::endl;
    trigger<int>(h, 0);
  }
  else
  {
    std::cout << "Applying factorial_aux(" << n-1 
              << ", " << n*a << ", k, h)" << std::endl;
    apply<factorial_get_aux_action>(n-1, n*a, k, h);
  }
}

void factorial_get(int n, naming::id_type k, naming::id_type h)
{
  std::cout << "Applying factorial_aux(" << n << ", 1, k, h)" << std::endl;
  apply<factorial_get_aux_action>(n, 1, k, h);
}

///////////////////////////////////////////////////////////////////////////////
typedef hpx::lcos::dataflow_variable<int,int> dataflow_int_type;

///////////////////////////////////////////////////////////////////////////////
int hpx_main(po::variables_map &vm)
{
    int n = 0;

    get_option(vm, "value", n);

    // Create DFV to guard against premature termination of main thread
    dataflow_int_type halt;

    // Create DFV for storing final value
    std::cout << ">>> kn = dataflow()" << std::endl;
    dataflow_int_type kn;

    if (n >= 0)
    {
      std::cout << ">>> factorial(" << n << ", kn, halt)" << std::endl;
      std::cout << "Applying factorial(" << n << ", kn, halt)" << std::endl;
      apply<factorial_get_action>(n, kn.get_gid(), halt.get_gid());

      std::cout << ">>> print kn" << std::endl;
      std::cout << kn.get() << std::endl;
    }

    std::cout << "Getting value of halt" << std::endl;
    halt.get();

    // initiate shutdown of the runtime systems on all localities
    hpx::finalize();

    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
  po::options_description 
      desc_commandline ("Usage: factorial_get [hpx_options] [options]");
  desc_commandline.add_options()
      ("value,v", po::value<int>(), 
       "the number to be used as the argument to factorial (default is 0)")
      ;

  return hpx::init(desc_commandline, argc, argv); 
}

