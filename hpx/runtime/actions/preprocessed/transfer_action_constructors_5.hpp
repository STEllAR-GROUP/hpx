// Copyright (c) 2007-2012 Hartmut Kaiser
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


    template <typename Arg0 , typename Arg1>
    transfer_action(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1)
        : arguments_(boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 )),
          parent_locality_(transfer_action::get_locality_id()),
          parent_id_(reinterpret_cast<std::size_t>(threads::get_parent_id())),
          parent_phase_(threads::get_parent_phase()),
          priority_(
                detail::thread_priority<
                    static_cast<threads::thread_priority>(priority_value)
                >::call(priority_value))
    {}
    template <typename Arg0 , typename Arg1>
    transfer_action(threads::thread_priority priority,
              BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1)
        : arguments_(boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 )),
          parent_locality_(transfer_action::get_locality_id()),
          parent_id_(reinterpret_cast<std::size_t>(threads::get_parent_id())),
          parent_phase_(threads::get_parent_phase()),
          priority_(
                detail::thread_priority<
                    static_cast<threads::thread_priority>(priority_value)
                >::call(priority))
    {}
    template <typename Arg0 , typename Arg1 , typename Arg2>
    transfer_action(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2)
        : arguments_(boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 )),
          parent_locality_(transfer_action::get_locality_id()),
          parent_id_(reinterpret_cast<std::size_t>(threads::get_parent_id())),
          parent_phase_(threads::get_parent_phase()),
          priority_(
                detail::thread_priority<
                    static_cast<threads::thread_priority>(priority_value)
                >::call(priority_value))
    {}
    template <typename Arg0 , typename Arg1 , typename Arg2>
    transfer_action(threads::thread_priority priority,
              BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2)
        : arguments_(boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 )),
          parent_locality_(transfer_action::get_locality_id()),
          parent_id_(reinterpret_cast<std::size_t>(threads::get_parent_id())),
          parent_phase_(threads::get_parent_phase()),
          priority_(
                detail::thread_priority<
                    static_cast<threads::thread_priority>(priority_value)
                >::call(priority))
    {}
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    transfer_action(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3)
        : arguments_(boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 )),
          parent_locality_(transfer_action::get_locality_id()),
          parent_id_(reinterpret_cast<std::size_t>(threads::get_parent_id())),
          parent_phase_(threads::get_parent_phase()),
          priority_(
                detail::thread_priority<
                    static_cast<threads::thread_priority>(priority_value)
                >::call(priority_value))
    {}
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    transfer_action(threads::thread_priority priority,
              BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3)
        : arguments_(boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 )),
          parent_locality_(transfer_action::get_locality_id()),
          parent_id_(reinterpret_cast<std::size_t>(threads::get_parent_id())),
          parent_phase_(threads::get_parent_phase()),
          priority_(
                detail::thread_priority<
                    static_cast<threads::thread_priority>(priority_value)
                >::call(priority))
    {}
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    transfer_action(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4)
        : arguments_(boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 )),
          parent_locality_(transfer_action::get_locality_id()),
          parent_id_(reinterpret_cast<std::size_t>(threads::get_parent_id())),
          parent_phase_(threads::get_parent_phase()),
          priority_(
                detail::thread_priority<
                    static_cast<threads::thread_priority>(priority_value)
                >::call(priority_value))
    {}
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    transfer_action(threads::thread_priority priority,
              BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4)
        : arguments_(boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 )),
          parent_locality_(transfer_action::get_locality_id()),
          parent_id_(reinterpret_cast<std::size_t>(threads::get_parent_id())),
          parent_phase_(threads::get_parent_phase()),
          priority_(
                detail::thread_priority<
                    static_cast<threads::thread_priority>(priority_value)
                >::call(priority))
    {}
