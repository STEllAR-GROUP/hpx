// Copyright (c) 2010-2011 Dylan Stark
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying 
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(PXGL_AUX_CONTROL_20101102T1048)
#define PXGL_AUX_CONTROL_20101102T1048

////////////////////////////////////////////////////////////////////////////////
// Define logging helper
#define LCTRL_LOG_fatal 0
#define LCTRL_LOG_info  0
#define LCTRL_LOG_debug 0
#define LCTRL_LOG__ping 0

#if LCTRL_LOG_ping == 1
#  define LCTRL_ping(major,minor) YAP_now(major,minor)
#else
#  define LCTRL_ping(major,minor) do {} while(0)
#endif

#if LCTRL_LOG_debug == 1
#define LCTRL_debug(str,...) YAPs(str,__VA_ARGS__)
#else
#define LCTRL_debug(str,...) do {} while(0)
#endif

#if LCTRL_LOG_info == 1
#define LCTRL_info(str,...) YAPs(str,__VA_ARGS__)
#else
#define LCTRL_info(str,...) do {} while(0)
#endif

#if LCTRL_LOG_fatal == 1
#define LCTRL_fatal(str,...) YAPs(str,__VA_ARGS__)
#else
#define LCTRL_fatal(str,...) do {} while(0)
#endif

namespace pxgl { namespace xua {
  typedef hpx::naming::id_type id_type;
  //////////////////////////////////////////////////////////////////////////////
  // For-each

  ///
  /// \brief Invoke the action once on each sub-container
  ///
  /// Applies the action to each of the sub-containers, does not wait for all
  /// results to finish.
  ///
  template <typename Action>
  inline void for_all(void)
  {
    typedef hpx::naming::id_type id_type;
    typedef std::vector<id_type> ids_type;

    id_type const here = hpx::get_runtime().get_process().here();
    ids_type const & localities = hpx::get_runtime().get_process().localities();

    BOOST_FOREACH(id_type const & there, localities)
    {
      if (there != here)
      {
        hpx::applier::apply<Action>(
            id_type(there, id_type::unmanaged));
      }
    }
  }

  template <typename Container, typename Action>
  inline void for_each(id_type const & container_id)
  {
    typedef unsigned long size_type;

    Container container(container_id);
    size_type const extent = container.get_distribution().size();

    for (size_type i=0; i < extent; i++)
      hpx::applier::apply<Action>(
          container.local_to(i), container.local_to(i));
  }

  template <typename C0, typename C1, typename Action>
  inline void for_each_aligned(
      id_type const & c0_id, 
      id_type const & c1_id)
  {
    typedef unsigned long size_type;

    C0 c0(c0_id);
    C1 c1(c1_id);
 
    size_type const extent = c0.get_distribution().size();

    for (size_type i=0; i < extent; i++)
    {
      hpx::applier::apply<Action>(
          c0.local_to(i), c0.local_to(i), c1.local_to(i));
    }
  }

  template <typename Action, typename C0, typename C1>
  inline void for_each_aligned_client(
      C0 const & c0, 
      C1 const & c1)
  {
    typedef unsigned long size_type;

    size_type const extent = c0.get_distribution().size();

    for (size_type i=0; i < extent; i++)
    {
      hpx::applier::apply<Action>(
          c0.local_to(i), c0.local_to(i), c1.local_to(i));
    }
  }

  template <typename Action, typename C0, typename C1, typename Arg0>
  inline void for_each_aligned_client1(
      C0 const & c0, 
      C1 const & c1,
      Arg0 const & arg0)
  {
    typedef unsigned long size_type;

    size_type const extent = c0.get_distribution().size();

    for (size_type i=0; i < extent; i++)
    {
      hpx::applier::apply<Action>(
          c0.local_to(i), c0.local_to(i), c1.local_to(i), arg0);
    }
  }

  template <typename Action, typename C0, typename C1, typename C2>
  inline void for_each_aligned_client(
      C0 const & c0, 
      C1 const & c1,
      C2 const & c2)
  {
    typedef unsigned long size_type;

    size_type const extent = c0.get_distribution().size();

    for (size_type i=0; i < extent; i++)
    {
      hpx::applier::apply<Action>(
          c0.local_to(i), c0.local_to(i), c1.local_to(i), c2.local_to(i));
    }
  }

  template <typename Action, typename C0, typename C1, typename C2>
  inline void blocking_for_each_aligned_client(
      C0 const & c0, 
      C1 const & c1,
      C2 const & c2)
  {
    typedef unsigned long size_type;

    typedef hpx::lcos::promise<typename Action::result_type> future_type;
    typedef std::vector<future_type> futures_type;

    size_type const extent = c0.get_distribution().size();

    futures_type outstanding_actions;
    for (size_type i=0; i < extent; i++)
    {
      outstanding_actions.push_back(
          hpx::lcos::eager_future<Action>(
              c0.local_to(i), c0.local_to(i), c1.local_to(i), c2.local_to(i)));
    }

    while (outstanding_actions.size() > 0)
    {
      outstanding_actions.back().get();
      outstanding_actions.pop_back();
    }
  }

  template <typename C0, typename C1, typename C2, typename Action>
  inline void for_each_aligned(
      id_type const & c0_id, 
      id_type const & c1_id, 
      id_type const & c2_id)
  {
    typedef unsigned long size_type;

    C0 c0(c0_id);
    C1 c1(c1_id);
    C2 c2(c2_id);
 
    size_type const extent = c0.get_distribution().size();

    for (size_type i=0; i < extent; i++)
    {
      hpx::applier::apply<Action>(
          c0.local_to(i), c0.local_to(i), c1.local_to(i), c2.local_to(i));
    }
  }

  template <typename C0, typename C1, typename C2, typename Action>
  inline void serial_for_each_aligned(
      id_type const & c0_id, 
      id_type const & c1_id, 
      id_type const & c2_id)
  {
    typedef unsigned long size_type;

    C0 c0(c0_id);
    C1 c1(c1_id);
    C2 c2(c2_id);
 
    size_type const extent = c0.get_distribution().size();

    for (size_type i=0; i < extent; i++)
    {
      hpx::lcos::eager_future<Action>(
          c0.local_to(i), c0.local_to(i), c1.local_to(i), c2.local_to(i)).get();
    }
  }

  // For use with plain actions
  template <typename Container, typename Action>
  inline void blocking_for_each(id_type const & container_id)
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
          hpx::lcos::eager_future<Action>(local_container_id, 
              local_container_id));
    }
  
    while (outstanding_actions.size() > 0)
    {
      outstanding_actions.back().get();
      outstanding_actions.pop_back();
    }
  }

  template <typename Container, typename Action>
  inline void blocking_for_each_comp(id_type const & container_id)
  {
    typedef unsigned long size_type;

    typedef hpx::lcos::promise<typename Action::result_type> future_type;
    typedef std::vector<future_type> futures_type;
  
    Container container(container_id);
  
    futures_type outstanding_actions;
    size_type const extent = container.get_distribution().size();
    LCTRL_debug("Blocking for-each comp. over %u localities.\n", extent);

    for (size_type i=0; i < extent; i++)
    {
      outstanding_actions.push_back(
          hpx::lcos::eager_future<Action>(container.local_to(i)));
    }
  
    while (outstanding_actions.size() > 0)
    {
      outstanding_actions.back().get();
      outstanding_actions.pop_back();
    }
  }

  template <typename Container, typename Action, typename Result>
  Result blocking_reduce(id_type const & container_id, Result result)
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
              local_container_id));
    }
  
    while (outstanding_results.size() > 0)
    {
      result += outstanding_results.back().get();
      outstanding_results.pop_back();
    }

    return result;
  }

// Pull in remaining implementations
#include "../../pxgl/xua/control_implementations.hpp"

  ///
  /// \brief Invoke the component action once on each sub-container
  ///
  /// Applies the action to each of the sub-containers, does not wait for all
  /// results to finish.
  ///
  template <typename Container, typename Action, typename Arg0>
  inline void for_each_comp(id_type container_id, Arg0 arg0)
  {
    typedef hpx::naming::id_type id_type;
    typedef std::vector<id_type> ids_type;

    typedef unsigned long size_type;
  
    Container container(container_id);
  
    typename Container::distribution_type 
        distribution(container.get_distribution());
    ids_type locales = distribution.coverage();
  
    for (size_type i=0; i < locales.size(); i++)
      hpx::applier::apply<Action>(
          container.local_to(i), arg0);
  }

  ///
  /// \brief Invoke the component action once on each sub-container
  ///
  /// Applies the action to each of the sub-containers, does not wait for all
  /// results to finish.
  ///
  template <typename Container, typename Action, 
            typename Arg0, typename Arg1>
  inline void for_each_comp(id_type container_id, 
                            Arg0 arg0, Arg1 arg1)
  {
    typedef hpx::naming::id_type id_type;
    typedef std::vector<id_type> ids_type;

    typedef unsigned long size_type;

    Container container(container_id);
  
    typename Container::distribution_type 
        distribution(container.get_distribution());
    ids_type locales = distribution.coverage();
  
    for (size_type i=0; i < locales.size(); i++)
      hpx::applier::apply<Action>(container.local_to(i), arg0, arg1);
  }

  ///
  /// \brief Invoke the component action once on each sub-container
  ///
  /// Applies the action to each of the sub-containers, does not wait for all
  /// results to finish.
  ///
  /// Assumes C0 and C1 have same distribution
  template <typename C0, typename C1, typename Action, 
            typename Arg0>
  inline void for_each_aligned_comp(id_type c0_id, id_type c1_id,
                                    Arg0 arg0)
  {
    typedef hpx::naming::id_type id_type;
    typedef std::vector<id_type> ids_type;

    typedef unsigned long size_type;

    C0 c0(c0_id);
    C1 c1(c1_id);
  
    typename C0::distribution_type 
        distribution(c0.get_distribution());
    ids_type locales = distribution.coverage();
  
    for (size_type i=0; i < locales.size(); i++)
      hpx::applier::apply<Action>(c0.local_to(i), c1.local_to(i), arg0);
  }

  template <typename C0, typename C1, typename C2, typename Action>
  inline void for_each_aligned_comp(id_type c0_id, id_type c1_id, id_type c2_id)
  {
    typedef hpx::naming::id_type id_type;
    typedef std::vector<id_type> ids_type;

    typedef unsigned long size_type;

    C0 c0(c0_id);
    C1 c1(c1_id);
    C2 c2(c2_id);
  
    typename C0::distribution_type 
        distribution(c0.get_distribution());
    ids_type locales = distribution.coverage();
  
    for (size_type i=0; i < locales.size(); i++)
      hpx::applier::apply<Action>(
          c0.local_to(i), c1.local_to(i), c2.local_to(i));
  }

  ///
  /// \brief Invoke the component action once on each sub-container
  ///
  /// Applies the action to each of the sub-containers, does not wait for all
  /// results to finish.
  ///
  /// Assumes C0 and C1 have same distribution
  template <typename C0, typename C1, typename Action, 
            typename Arg0, typename Arg1>
  inline void for_each_aligned_comp(id_type c0_id, id_type c1_id,
                                    Arg0 arg0, Arg1 arg1)
  {
    typedef hpx::naming::id_type id_type;
    typedef std::vector<id_type> ids_type;

    typedef unsigned long size_type;

    C0 c0(c0_id);
    C1 c1(c1_id);
  
    typename C0::distribution_type 
        distribution(c0.get_distribution());
    ids_type locales = distribution.coverage();
  
    for (size_type i=0; i < locales.size(); i++)
      hpx::applier::apply<Action>(c0.local_to(i), c1.local_to(i), arg0, arg1);
  }

  ///
  /// \brief Invoke the component action once on each sub-container
  ///
  /// Applies the action to each of the sub-containers, does not wait for all
  /// results to finish.
  ///
  template <typename Container, typename Action, 
            typename Arg0, typename Arg1, typename Arg2>
  inline void for_each_comp(id_type container_id, 
                            Arg0 arg0, Arg1 arg1, Arg2 arg2)
  {
    typedef hpx::naming::id_type id_type;
    typedef std::vector<id_type> ids_type;

    typedef unsigned long size_type;

    Container container(container_id);
  
    typename Container::distribution_type 
        distribution(container.get_distribution());
    ids_type locales = distribution.coverage();
  
    for (size_type i=0; i < locales.size(); i++)
      hpx::applier::apply<Action>(container.local_to(i), arg0, arg1, arg2);
  }

}}

#endif

