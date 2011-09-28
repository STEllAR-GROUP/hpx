// Copyright (c) 2010-2011 Dylan Stark
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying 
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_PP_IS_ITERATING

#if !defined(PXGL_AUX_CONTROL_IMPLEMENTATIONS_20101102T1048)
#define PXGL_AUX_CONTROL_IMPLEMENTATIONS_20101102T1048

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repeat.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>

#define BOOST_PP_ITERATION_PARAMS_1                                            \
    (3, (1, 6, "examples/pxgl/pxgl/xua/control_implementations.hpp"))          \
    /**/
    
#include BOOST_PP_ITERATE()

#endif

///////////////////////////////////////////////////////////////////////////////
//  Preprocessor vertical repetition code
///////////////////////////////////////////////////////////////////////////////
#else // defined(BOOST_PP_IS_ITERATING)

#define N BOOST_PP_ITERATION()

  template <
      typename Container, typename Action, 
      BOOST_PP_ENUM_PARAMS(N, typename Arg)>
  inline void for_each(
      id_type const & container_id, 
      BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const & arg))
  {
    typedef unsigned long size_type;

    Container container(container_id);
    size_type const extent = container.get_distribution().size();

    for (size_type i=0; i < extent; i++)
    {
      hpx::applier::apply<Action>(
          container.local_to(i), container.local_to(i),
          BOOST_PP_ENUM_PARAMS(N, arg));
    }
  }

  template <
      typename C0, typename C1, typename Action,
      BOOST_PP_ENUM_PARAMS(N, typename Arg)>
  inline void for_each_aligned(
      id_type const & c0_id, 
      id_type const & c1_id,
      BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const & arg))
  {
    typedef hpx::naming::gid_type gid_type;
    typedef std::vector<gid_type> gids_type;

    typedef unsigned long size_type;

    C0 c0(c0_id);
    C1 c1(c1_id);
 
    size_type const extent = c0.get_distribution().size();

    for (size_type i=0; i < extent; i++)
    {
      hpx::applier::apply<Action>(
          c0.local_to(i), c0.local_to(i), c1.local_to(i),
          BOOST_PP_ENUM_PARAMS(N, arg));
    }
  }

  template <
      typename C0, typename C1, typename C2, typename Action,
      BOOST_PP_ENUM_PARAMS(N, typename Arg)>
  inline void for_each_aligned(
      id_type const & c0_id, 
      id_type const & c1_id, 
      id_type const & c2_id,
      BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const & arg))
  {
    typedef hpx::naming::gid_type gid_type;
    typedef std::vector<gid_type> gids_type;

    typedef unsigned long size_type;

    C0 c0(c0_id);
    C1 c1(c1_id);
    C2 c2(c2_id);
 
    size_type const extent = c0.get_distribution().size();

    for (size_type i=0; i < extent; i++)
    {
      hpx::applier::apply<Action>(
          c0.local_to(i), c0.local_to(i), c1.local_to(i), c2.local_to(i),
          BOOST_PP_ENUM_PARAMS(N, arg));
    }
  }

  template <typename Container, typename Action,
      BOOST_PP_ENUM_PARAMS(N, typename Arg)>
  inline void blocking_for_each(id_type const & container_id,
      BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const & arg))
  {
    typedef unsigned long size_type;

    typedef hpx::lcos::promise<typename Action::result_type> future_type;
    typedef std::vector<future_type> futures_type;
  
    Container container(container_id);
  
    futures_type outstanding_actions;
    size_type const extent = container.get_distribution().size();
    for (size_type i=0; i < extent; i++)
    {
      id_type const local_container_id(container.local_to(i));
      outstanding_actions.push_back(
          hpx::lcos::eager_future<Action>(
              local_container_id, local_container_id,
              BOOST_PP_ENUM_PARAMS(N, arg)));
    }
  
    while (outstanding_actions.size() > 0)
    {
      outstanding_actions.back().get();
      outstanding_actions.pop_back();
    }
  }

  template <typename Container, typename Action,
      BOOST_PP_ENUM_PARAMS(N, typename Arg)>
  inline void blocking_for_each_comp(id_type const & container_id,
      BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const & arg))
  {
    typedef unsigned long size_type;

    typedef hpx::lcos::promise<typename Action::result_type> future_type;
    typedef std::vector<future_type> futures_type;
  
    Container container(container_id);
  
    futures_type outstanding_actions;
    size_type const extent = container.get_distribution().size();
    for (size_type i=0; i < extent; i++)
    {
      outstanding_actions.push_back(
          hpx::lcos::eager_future<Action>(
              container.local_to(i),
              BOOST_PP_ENUM_PARAMS(N, arg)));
    }
  
    while (outstanding_actions.size() > 0)
    {
      outstanding_actions.back().get();
      outstanding_actions.pop_back();
    }
  }

  template <typename Container, typename Action, typename Result,
      BOOST_PP_ENUM_PARAMS(N, typename Arg)>
  Result blocking_reduce(
      id_type const & container_id, 
      Result result,
      BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const & arg))
  {
    typedef unsigned long size_type;

    typedef hpx::lcos::promise<Result> future_value_type;
    typedef std::vector<future_value_type> future_values_type;
   
    Container container(container_id);
 
    future_values_type outstanding_results;
    size_type const extent = container.get_distribution().size();
    for (size_type i=0; i < extent; i++)
    {
      id_type const local_container_id(container.local_to(i));
      outstanding_results.push_back(
          hpx::lcos::eager_future<Action>(
              local_container_id, 
              local_container_id,
              BOOST_PP_ENUM_PARAMS(N, arg)));
    }
  
    while (outstanding_results.size() > 0)
    {
      result += outstanding_results.back().get();
      outstanding_results.pop_back();
    }

    return result;
  }

#undef N

#endif
