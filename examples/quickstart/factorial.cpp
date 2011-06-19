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

typedef hpx::lcos::detail::local_dataflow_variable<int, int>
    detail_dataflow_type;
HPX_DEFINE_GET_COMPONENT_TYPE(detail_dataflow_type);

///////////////////////////////////////////////////////////////////////////////
// More helpers
template<typename Action, typename Arg0>
inline void apply(Arg0 arg0, naming::id_type k)
{
  hpx::applier::apply<Action>(find_here(), arg0, k);
}

template<typename Action, typename Arg0, typename Arg1>
inline void apply(Arg0 arg0, Arg1 arg1, naming::id_type k)
{
  hpx::applier::apply<Action>(find_here(), arg0, arg1, k);
}

template<typename Arg0>
inline void trigger(naming::id_type k, Arg0 arg0)
{
  actions::continuation(k).trigger<Arg0>(arg0);
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
void factorial_aux(int, int, naming::id_type);
typedef actions::plain_action3<int, int, naming::id_type, factorial_aux> 
    factorial_aux_action;
HPX_REGISTER_PLAIN_ACTION(factorial_aux_action);

void factorial(int, naming::id_type);
typedef actions::plain_action2<int, naming::id_type, factorial> 
    factorial_action;
HPX_REGISTER_PLAIN_ACTION(factorial_action);

void factorial_aux(int n, int a, naming::id_type k)
{
  if (n == 0)
    trigger<int>(k, a);
  else
    apply<factorial_aux_action>(n-1, n*a, k);
}

void factorial(int n, naming::id_type k)
{
  apply<factorial_aux_action>(n, 1, k);
}

///////////////////////////////////////////////////////////////////////////////
typedef hpx::lcos::local_dataflow_variable<int,int> dataflow_int_type;

///////////////////////////////////////////////////////////////////////////////
int hpx_main(po::variables_map &vm)
{
    int n = -1;
    
    get_option(vm, "value", n);

    {
        std::cout << ">>> k1 = dataflow_variable()" << std::endl;
        dataflow_int_type k1;

        std::cout << ">>> factorial(0, k1)" << std::endl;
        apply<factorial_action>(0, k1.get_gid());

        std::cout << ">>> print k1" << std::endl;
        std::cout << k1.get() << std::endl;

        std::cout << ">>> k2 = dataflow()" << std::endl;
        dataflow_int_type k2;

        std::cout << ">>> factorial(1, k2)" << std::endl;
        apply<factorial_action>(1, k2.get_gid());

        std::cout << ">>> print k2" << std::endl;
        std::cout << k2.get() << std::endl;

        std::cout << ">>> k3 = dataflow()" << std::endl;
        dataflow_int_type k3;

        std::cout << ">>> factorial(3, k3)" << std::endl;
        apply<factorial_action>(3, k3.get_gid());

        std::cout << ">>> print k3" << std::endl;
        std::cout << k3.get() << std::endl;

        if (n >= 0)
        {
          std::cout << ">>> kn = dataflow()" << std::endl;
          dataflow_int_type kn;

          std::cout << ">>> factorial(" << n << ", kn)" << std::endl;
          apply<factorial_action>(n, kn.get_gid());

          std::cout << ">>> print kn" << std::endl;
          std::cout << kn.get() << std::endl;
        }
    }

    // initiate shutdown of the runtime systems on all localities
    hpx::finalize();

    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
  // Configure application-specific options
  po::options_description 
      desc_commandline("Usage: factorial [hpx_options] [options]");
  desc_commandline.add_options()
      ("value,v", po::value<int>(), 
       "the number to be used as the argument to fib (default is 10)")
      ;

  // Initialize and run HPX
  return hpx::init(desc_commandline, argc, argv); 
}

