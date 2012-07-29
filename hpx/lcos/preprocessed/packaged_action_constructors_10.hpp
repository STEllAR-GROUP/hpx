// Copyright (c) 2007-2012 Hartmut Kaiser
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


    template <typename Arg0 , typename Arg1>
    void apply(naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1)
    {
        util::block_profiler_wrapper<profiler_tag> bp(apply_logger_);
        hpx::apply_c<action_type>(this->get_gid(), gid,
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ));
    }
    template <typename Arg0 , typename Arg1>
    void apply_p(naming::id_type const& gid,
        threads::thread_priority priority, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1)
    {
        util::block_profiler_wrapper<profiler_tag> bp(apply_logger_);
        hpx::apply_c_p<action_type>(this->get_gid(), gid, priority,
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ));
    }
    
    template <typename Arg0 , typename Arg1>
    packaged_action(naming::id_type const& gid,
            BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1)
      : apply_logger_("packaged_action::apply")
    {
        LLCO_(info) << "packaged_action::packaged_action("
                    << hpx::actions::detail::get_action_name<action_type>()
                    << ", "
                    << gid
                    << ") args(" << (2 + 1) << ")";
        apply(gid, boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ));
    }
    template <typename Arg0 , typename Arg1>
    packaged_action(naming::id_type const& gid,
            completed_callback_type const& data_sink,
            BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1)
      : base_type(data_sink),
        apply_logger_("packaged_action::apply")
    {
        LLCO_(info) << "packaged_action::packaged_action("
                    << hpx::actions::detail::get_action_name<action_type>()
                    << ", "
                    << gid
                    << ") args(" << (2 + 1) << ")";
        apply(gid, boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ));
    }
    template <typename Arg0 , typename Arg1>
    packaged_action(naming::gid_type const& gid,
            threads::thread_priority priority,
            BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1)
      : apply_logger_("packaged_action::apply")
    {
        LLCO_(info) << "packaged_action::packaged_action("
                    << hpx::actions::detail::get_action_name<action_type>()
                    << ", "
                    << gid
                    << ") args(" << (2 + 1) << ")";
        apply_p(naming::id_type(gid, naming::id_type::unmanaged),
            priority, boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ));
    }
    template <typename Arg0 , typename Arg1>
    packaged_action(naming::id_type const& gid,
            threads::thread_priority priority,
            completed_callback_type const& data_sink,
            BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1)
      : base_type(data_sink),
        apply_logger_("packaged_action::apply")
    {
        LLCO_(info) << "packaged_action::packaged_action("
                    << hpx::actions::detail::get_action_name<action_type>()
                    << ", "
                    << gid
                    << ") args(" << (2 + 1) << ")";
        apply_p(gid, priority, boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ));
    }
    template <typename Arg0 , typename Arg1 , typename Arg2>
    void apply(naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2)
    {
        util::block_profiler_wrapper<profiler_tag> bp(apply_logger_);
        hpx::apply_c<action_type>(this->get_gid(), gid,
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ));
    }
    template <typename Arg0 , typename Arg1 , typename Arg2>
    void apply_p(naming::id_type const& gid,
        threads::thread_priority priority, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2)
    {
        util::block_profiler_wrapper<profiler_tag> bp(apply_logger_);
        hpx::apply_c_p<action_type>(this->get_gid(), gid, priority,
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ));
    }
    
    template <typename Arg0 , typename Arg1 , typename Arg2>
    packaged_action(naming::id_type const& gid,
            BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2)
      : apply_logger_("packaged_action::apply")
    {
        LLCO_(info) << "packaged_action::packaged_action("
                    << hpx::actions::detail::get_action_name<action_type>()
                    << ", "
                    << gid
                    << ") args(" << (3 + 1) << ")";
        apply(gid, boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ));
    }
    template <typename Arg0 , typename Arg1 , typename Arg2>
    packaged_action(naming::id_type const& gid,
            completed_callback_type const& data_sink,
            BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2)
      : base_type(data_sink),
        apply_logger_("packaged_action::apply")
    {
        LLCO_(info) << "packaged_action::packaged_action("
                    << hpx::actions::detail::get_action_name<action_type>()
                    << ", "
                    << gid
                    << ") args(" << (3 + 1) << ")";
        apply(gid, boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ));
    }
    template <typename Arg0 , typename Arg1 , typename Arg2>
    packaged_action(naming::gid_type const& gid,
            threads::thread_priority priority,
            BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2)
      : apply_logger_("packaged_action::apply")
    {
        LLCO_(info) << "packaged_action::packaged_action("
                    << hpx::actions::detail::get_action_name<action_type>()
                    << ", "
                    << gid
                    << ") args(" << (3 + 1) << ")";
        apply_p(naming::id_type(gid, naming::id_type::unmanaged),
            priority, boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ));
    }
    template <typename Arg0 , typename Arg1 , typename Arg2>
    packaged_action(naming::id_type const& gid,
            threads::thread_priority priority,
            completed_callback_type const& data_sink,
            BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2)
      : base_type(data_sink),
        apply_logger_("packaged_action::apply")
    {
        LLCO_(info) << "packaged_action::packaged_action("
                    << hpx::actions::detail::get_action_name<action_type>()
                    << ", "
                    << gid
                    << ") args(" << (3 + 1) << ")";
        apply_p(gid, priority, boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ));
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    void apply(naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3)
    {
        util::block_profiler_wrapper<profiler_tag> bp(apply_logger_);
        hpx::apply_c<action_type>(this->get_gid(), gid,
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ));
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    void apply_p(naming::id_type const& gid,
        threads::thread_priority priority, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3)
    {
        util::block_profiler_wrapper<profiler_tag> bp(apply_logger_);
        hpx::apply_c_p<action_type>(this->get_gid(), gid, priority,
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ));
    }
    
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    packaged_action(naming::id_type const& gid,
            BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3)
      : apply_logger_("packaged_action::apply")
    {
        LLCO_(info) << "packaged_action::packaged_action("
                    << hpx::actions::detail::get_action_name<action_type>()
                    << ", "
                    << gid
                    << ") args(" << (4 + 1) << ")";
        apply(gid, boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ));
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    packaged_action(naming::id_type const& gid,
            completed_callback_type const& data_sink,
            BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3)
      : base_type(data_sink),
        apply_logger_("packaged_action::apply")
    {
        LLCO_(info) << "packaged_action::packaged_action("
                    << hpx::actions::detail::get_action_name<action_type>()
                    << ", "
                    << gid
                    << ") args(" << (4 + 1) << ")";
        apply(gid, boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ));
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    packaged_action(naming::gid_type const& gid,
            threads::thread_priority priority,
            BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3)
      : apply_logger_("packaged_action::apply")
    {
        LLCO_(info) << "packaged_action::packaged_action("
                    << hpx::actions::detail::get_action_name<action_type>()
                    << ", "
                    << gid
                    << ") args(" << (4 + 1) << ")";
        apply_p(naming::id_type(gid, naming::id_type::unmanaged),
            priority, boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ));
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    packaged_action(naming::id_type const& gid,
            threads::thread_priority priority,
            completed_callback_type const& data_sink,
            BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3)
      : base_type(data_sink),
        apply_logger_("packaged_action::apply")
    {
        LLCO_(info) << "packaged_action::packaged_action("
                    << hpx::actions::detail::get_action_name<action_type>()
                    << ", "
                    << gid
                    << ") args(" << (4 + 1) << ")";
        apply_p(gid, priority, boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ));
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    void apply(naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4)
    {
        util::block_profiler_wrapper<profiler_tag> bp(apply_logger_);
        hpx::apply_c<action_type>(this->get_gid(), gid,
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ));
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    void apply_p(naming::id_type const& gid,
        threads::thread_priority priority, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4)
    {
        util::block_profiler_wrapper<profiler_tag> bp(apply_logger_);
        hpx::apply_c_p<action_type>(this->get_gid(), gid, priority,
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ));
    }
    
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    packaged_action(naming::id_type const& gid,
            BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4)
      : apply_logger_("packaged_action::apply")
    {
        LLCO_(info) << "packaged_action::packaged_action("
                    << hpx::actions::detail::get_action_name<action_type>()
                    << ", "
                    << gid
                    << ") args(" << (5 + 1) << ")";
        apply(gid, boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ));
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    packaged_action(naming::id_type const& gid,
            completed_callback_type const& data_sink,
            BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4)
      : base_type(data_sink),
        apply_logger_("packaged_action::apply")
    {
        LLCO_(info) << "packaged_action::packaged_action("
                    << hpx::actions::detail::get_action_name<action_type>()
                    << ", "
                    << gid
                    << ") args(" << (5 + 1) << ")";
        apply(gid, boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ));
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    packaged_action(naming::gid_type const& gid,
            threads::thread_priority priority,
            BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4)
      : apply_logger_("packaged_action::apply")
    {
        LLCO_(info) << "packaged_action::packaged_action("
                    << hpx::actions::detail::get_action_name<action_type>()
                    << ", "
                    << gid
                    << ") args(" << (5 + 1) << ")";
        apply_p(naming::id_type(gid, naming::id_type::unmanaged),
            priority, boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ));
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    packaged_action(naming::id_type const& gid,
            threads::thread_priority priority,
            completed_callback_type const& data_sink,
            BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4)
      : base_type(data_sink),
        apply_logger_("packaged_action::apply")
    {
        LLCO_(info) << "packaged_action::packaged_action("
                    << hpx::actions::detail::get_action_name<action_type>()
                    << ", "
                    << gid
                    << ") args(" << (5 + 1) << ")";
        apply_p(gid, priority, boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ));
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    void apply(naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5)
    {
        util::block_profiler_wrapper<profiler_tag> bp(apply_logger_);
        hpx::apply_c<action_type>(this->get_gid(), gid,
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ));
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    void apply_p(naming::id_type const& gid,
        threads::thread_priority priority, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5)
    {
        util::block_profiler_wrapper<profiler_tag> bp(apply_logger_);
        hpx::apply_c_p<action_type>(this->get_gid(), gid, priority,
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ));
    }
    
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    packaged_action(naming::id_type const& gid,
            BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5)
      : apply_logger_("packaged_action::apply")
    {
        LLCO_(info) << "packaged_action::packaged_action("
                    << hpx::actions::detail::get_action_name<action_type>()
                    << ", "
                    << gid
                    << ") args(" << (6 + 1) << ")";
        apply(gid, boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ));
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    packaged_action(naming::id_type const& gid,
            completed_callback_type const& data_sink,
            BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5)
      : base_type(data_sink),
        apply_logger_("packaged_action::apply")
    {
        LLCO_(info) << "packaged_action::packaged_action("
                    << hpx::actions::detail::get_action_name<action_type>()
                    << ", "
                    << gid
                    << ") args(" << (6 + 1) << ")";
        apply(gid, boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ));
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    packaged_action(naming::gid_type const& gid,
            threads::thread_priority priority,
            BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5)
      : apply_logger_("packaged_action::apply")
    {
        LLCO_(info) << "packaged_action::packaged_action("
                    << hpx::actions::detail::get_action_name<action_type>()
                    << ", "
                    << gid
                    << ") args(" << (6 + 1) << ")";
        apply_p(naming::id_type(gid, naming::id_type::unmanaged),
            priority, boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ));
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    packaged_action(naming::id_type const& gid,
            threads::thread_priority priority,
            completed_callback_type const& data_sink,
            BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5)
      : base_type(data_sink),
        apply_logger_("packaged_action::apply")
    {
        LLCO_(info) << "packaged_action::packaged_action("
                    << hpx::actions::detail::get_action_name<action_type>()
                    << ", "
                    << gid
                    << ") args(" << (6 + 1) << ")";
        apply_p(gid, priority, boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ));
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    void apply(naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6)
    {
        util::block_profiler_wrapper<profiler_tag> bp(apply_logger_);
        hpx::apply_c<action_type>(this->get_gid(), gid,
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ));
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    void apply_p(naming::id_type const& gid,
        threads::thread_priority priority, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6)
    {
        util::block_profiler_wrapper<profiler_tag> bp(apply_logger_);
        hpx::apply_c_p<action_type>(this->get_gid(), gid, priority,
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ));
    }
    
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    packaged_action(naming::id_type const& gid,
            BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6)
      : apply_logger_("packaged_action::apply")
    {
        LLCO_(info) << "packaged_action::packaged_action("
                    << hpx::actions::detail::get_action_name<action_type>()
                    << ", "
                    << gid
                    << ") args(" << (7 + 1) << ")";
        apply(gid, boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ));
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    packaged_action(naming::id_type const& gid,
            completed_callback_type const& data_sink,
            BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6)
      : base_type(data_sink),
        apply_logger_("packaged_action::apply")
    {
        LLCO_(info) << "packaged_action::packaged_action("
                    << hpx::actions::detail::get_action_name<action_type>()
                    << ", "
                    << gid
                    << ") args(" << (7 + 1) << ")";
        apply(gid, boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ));
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    packaged_action(naming::gid_type const& gid,
            threads::thread_priority priority,
            BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6)
      : apply_logger_("packaged_action::apply")
    {
        LLCO_(info) << "packaged_action::packaged_action("
                    << hpx::actions::detail::get_action_name<action_type>()
                    << ", "
                    << gid
                    << ") args(" << (7 + 1) << ")";
        apply_p(naming::id_type(gid, naming::id_type::unmanaged),
            priority, boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ));
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    packaged_action(naming::id_type const& gid,
            threads::thread_priority priority,
            completed_callback_type const& data_sink,
            BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6)
      : base_type(data_sink),
        apply_logger_("packaged_action::apply")
    {
        LLCO_(info) << "packaged_action::packaged_action("
                    << hpx::actions::detail::get_action_name<action_type>()
                    << ", "
                    << gid
                    << ") args(" << (7 + 1) << ")";
        apply_p(gid, priority, boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ));
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    void apply(naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7)
    {
        util::block_profiler_wrapper<profiler_tag> bp(apply_logger_);
        hpx::apply_c<action_type>(this->get_gid(), gid,
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ));
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    void apply_p(naming::id_type const& gid,
        threads::thread_priority priority, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7)
    {
        util::block_profiler_wrapper<profiler_tag> bp(apply_logger_);
        hpx::apply_c_p<action_type>(this->get_gid(), gid, priority,
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ));
    }
    
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    packaged_action(naming::id_type const& gid,
            BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7)
      : apply_logger_("packaged_action::apply")
    {
        LLCO_(info) << "packaged_action::packaged_action("
                    << hpx::actions::detail::get_action_name<action_type>()
                    << ", "
                    << gid
                    << ") args(" << (8 + 1) << ")";
        apply(gid, boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ));
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    packaged_action(naming::id_type const& gid,
            completed_callback_type const& data_sink,
            BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7)
      : base_type(data_sink),
        apply_logger_("packaged_action::apply")
    {
        LLCO_(info) << "packaged_action::packaged_action("
                    << hpx::actions::detail::get_action_name<action_type>()
                    << ", "
                    << gid
                    << ") args(" << (8 + 1) << ")";
        apply(gid, boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ));
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    packaged_action(naming::gid_type const& gid,
            threads::thread_priority priority,
            BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7)
      : apply_logger_("packaged_action::apply")
    {
        LLCO_(info) << "packaged_action::packaged_action("
                    << hpx::actions::detail::get_action_name<action_type>()
                    << ", "
                    << gid
                    << ") args(" << (8 + 1) << ")";
        apply_p(naming::id_type(gid, naming::id_type::unmanaged),
            priority, boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ));
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    packaged_action(naming::id_type const& gid,
            threads::thread_priority priority,
            completed_callback_type const& data_sink,
            BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7)
      : base_type(data_sink),
        apply_logger_("packaged_action::apply")
    {
        LLCO_(info) << "packaged_action::packaged_action("
                    << hpx::actions::detail::get_action_name<action_type>()
                    << ", "
                    << gid
                    << ") args(" << (8 + 1) << ")";
        apply_p(gid, priority, boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ));
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
    void apply(naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8)
    {
        util::block_profiler_wrapper<profiler_tag> bp(apply_logger_);
        hpx::apply_c<action_type>(this->get_gid(), gid,
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ));
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
    void apply_p(naming::id_type const& gid,
        threads::thread_priority priority, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8)
    {
        util::block_profiler_wrapper<profiler_tag> bp(apply_logger_);
        hpx::apply_c_p<action_type>(this->get_gid(), gid, priority,
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ));
    }
    
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
    packaged_action(naming::id_type const& gid,
            BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8)
      : apply_logger_("packaged_action::apply")
    {
        LLCO_(info) << "packaged_action::packaged_action("
                    << hpx::actions::detail::get_action_name<action_type>()
                    << ", "
                    << gid
                    << ") args(" << (9 + 1) << ")";
        apply(gid, boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ));
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
    packaged_action(naming::id_type const& gid,
            completed_callback_type const& data_sink,
            BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8)
      : base_type(data_sink),
        apply_logger_("packaged_action::apply")
    {
        LLCO_(info) << "packaged_action::packaged_action("
                    << hpx::actions::detail::get_action_name<action_type>()
                    << ", "
                    << gid
                    << ") args(" << (9 + 1) << ")";
        apply(gid, boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ));
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
    packaged_action(naming::gid_type const& gid,
            threads::thread_priority priority,
            BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8)
      : apply_logger_("packaged_action::apply")
    {
        LLCO_(info) << "packaged_action::packaged_action("
                    << hpx::actions::detail::get_action_name<action_type>()
                    << ", "
                    << gid
                    << ") args(" << (9 + 1) << ")";
        apply_p(naming::id_type(gid, naming::id_type::unmanaged),
            priority, boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ));
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
    packaged_action(naming::id_type const& gid,
            threads::thread_priority priority,
            completed_callback_type const& data_sink,
            BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8)
      : base_type(data_sink),
        apply_logger_("packaged_action::apply")
    {
        LLCO_(info) << "packaged_action::packaged_action("
                    << hpx::actions::detail::get_action_name<action_type>()
                    << ", "
                    << gid
                    << ") args(" << (9 + 1) << ")";
        apply_p(gid, priority, boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ));
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
    void apply(naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9)
    {
        util::block_profiler_wrapper<profiler_tag> bp(apply_logger_);
        hpx::apply_c<action_type>(this->get_gid(), gid,
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ));
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
    void apply_p(naming::id_type const& gid,
        threads::thread_priority priority, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9)
    {
        util::block_profiler_wrapper<profiler_tag> bp(apply_logger_);
        hpx::apply_c_p<action_type>(this->get_gid(), gid, priority,
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ));
    }
    
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
    packaged_action(naming::id_type const& gid,
            BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9)
      : apply_logger_("packaged_action::apply")
    {
        LLCO_(info) << "packaged_action::packaged_action("
                    << hpx::actions::detail::get_action_name<action_type>()
                    << ", "
                    << gid
                    << ") args(" << (10 + 1) << ")";
        apply(gid, boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ));
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
    packaged_action(naming::id_type const& gid,
            completed_callback_type const& data_sink,
            BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9)
      : base_type(data_sink),
        apply_logger_("packaged_action::apply")
    {
        LLCO_(info) << "packaged_action::packaged_action("
                    << hpx::actions::detail::get_action_name<action_type>()
                    << ", "
                    << gid
                    << ") args(" << (10 + 1) << ")";
        apply(gid, boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ));
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
    packaged_action(naming::gid_type const& gid,
            threads::thread_priority priority,
            BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9)
      : apply_logger_("packaged_action::apply")
    {
        LLCO_(info) << "packaged_action::packaged_action("
                    << hpx::actions::detail::get_action_name<action_type>()
                    << ", "
                    << gid
                    << ") args(" << (10 + 1) << ")";
        apply_p(naming::id_type(gid, naming::id_type::unmanaged),
            priority, boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ));
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
    packaged_action(naming::id_type const& gid,
            threads::thread_priority priority,
            completed_callback_type const& data_sink,
            BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9)
      : base_type(data_sink),
        apply_logger_("packaged_action::apply")
    {
        LLCO_(info) << "packaged_action::packaged_action("
                    << hpx::actions::detail::get_action_name<action_type>()
                    << ", "
                    << gid
                    << ") args(" << (10 + 1) << ")";
        apply_p(gid, priority, boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ));
    }
