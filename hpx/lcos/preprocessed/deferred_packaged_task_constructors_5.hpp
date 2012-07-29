// Copyright (c) 2007-2012 Hartmut Kaiser
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


    template <typename Arg0 , typename Arg1>
    void apply(naming::id_type const& gid,
        HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)))
    {
        util::block_profiler_wrapper<deferred_packaged_task_tag> bp(apply_logger_);
        hpx::apply_c<Action>(
            this->get_gid(), gid, HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)));
    }
private:
    template <typename Arg0 , typename Arg1>
    static void invoke2(
        hpx::lcos::deferred_packaged_task<Action,Result> *th,
        naming::id_type const& gid,
        HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)))
    {
        if (!((*th->impl_)->is_ready()))
            th->apply(gid, HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)));
    }
public:
    template <typename Arg0 , typename Arg1>
    deferred_packaged_task(naming::gid_type const& gid,
            HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)))
      : apply_logger_("deferred_packaged_task::apply"),
        closure_(boost::bind(
          &deferred_packaged_task::template invoke2<Arg0 , Arg1>,
            this_(), naming::id_type(gid, naming::id_type::unmanaged),
            HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg))))
    {
        LLCO_(info) << "deferred_packaged_task::deferred_packaged_task("
                    << hpx::actions::detail::get_action_name<Action>()
                    << ", "
                    << gid
                    << ") args(" << (2 + 1) << ")";
    }
    template <typename Arg0 , typename Arg1>
    deferred_packaged_task(naming::id_type const& gid,
            HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)))
      : apply_logger_("deferred_packaged_task::apply"),
        closure_(boost::bind(
          &deferred_packaged_task::template invoke2<Arg0 , Arg1>,
            this_(), gid,
            HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg))))
    {
        LLCO_(info) << "deferred_packaged_task::deferred_packaged_task("
                    << hpx::actions::detail::get_action_name<Action>()
                    << ", "
                    << gid
                    << ") args(" << (2 + 1) << ")";
    }
    template <typename Arg0 , typename Arg1 , typename Arg2>
    void apply(naming::id_type const& gid,
        HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)))
    {
        util::block_profiler_wrapper<deferred_packaged_task_tag> bp(apply_logger_);
        hpx::apply_c<Action>(
            this->get_gid(), gid, HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)));
    }
private:
    template <typename Arg0 , typename Arg1 , typename Arg2>
    static void invoke3(
        hpx::lcos::deferred_packaged_task<Action,Result> *th,
        naming::id_type const& gid,
        HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)))
    {
        if (!((*th->impl_)->is_ready()))
            th->apply(gid, HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)));
    }
public:
    template <typename Arg0 , typename Arg1 , typename Arg2>
    deferred_packaged_task(naming::gid_type const& gid,
            HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)))
      : apply_logger_("deferred_packaged_task::apply"),
        closure_(boost::bind(
          &deferred_packaged_task::template invoke3<Arg0 , Arg1 , Arg2>,
            this_(), naming::id_type(gid, naming::id_type::unmanaged),
            HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg))))
    {
        LLCO_(info) << "deferred_packaged_task::deferred_packaged_task("
                    << hpx::actions::detail::get_action_name<Action>()
                    << ", "
                    << gid
                    << ") args(" << (3 + 1) << ")";
    }
    template <typename Arg0 , typename Arg1 , typename Arg2>
    deferred_packaged_task(naming::id_type const& gid,
            HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)))
      : apply_logger_("deferred_packaged_task::apply"),
        closure_(boost::bind(
          &deferred_packaged_task::template invoke3<Arg0 , Arg1 , Arg2>,
            this_(), gid,
            HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg))))
    {
        LLCO_(info) << "deferred_packaged_task::deferred_packaged_task("
                    << hpx::actions::detail::get_action_name<Action>()
                    << ", "
                    << gid
                    << ") args(" << (3 + 1) << ")";
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    void apply(naming::id_type const& gid,
        HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)))
    {
        util::block_profiler_wrapper<deferred_packaged_task_tag> bp(apply_logger_);
        hpx::apply_c<Action>(
            this->get_gid(), gid, HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)));
    }
private:
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    static void invoke4(
        hpx::lcos::deferred_packaged_task<Action,Result> *th,
        naming::id_type const& gid,
        HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)))
    {
        if (!((*th->impl_)->is_ready()))
            th->apply(gid, HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)));
    }
public:
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    deferred_packaged_task(naming::gid_type const& gid,
            HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)))
      : apply_logger_("deferred_packaged_task::apply"),
        closure_(boost::bind(
          &deferred_packaged_task::template invoke4<Arg0 , Arg1 , Arg2 , Arg3>,
            this_(), naming::id_type(gid, naming::id_type::unmanaged),
            HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg))))
    {
        LLCO_(info) << "deferred_packaged_task::deferred_packaged_task("
                    << hpx::actions::detail::get_action_name<Action>()
                    << ", "
                    << gid
                    << ") args(" << (4 + 1) << ")";
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    deferred_packaged_task(naming::id_type const& gid,
            HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)))
      : apply_logger_("deferred_packaged_task::apply"),
        closure_(boost::bind(
          &deferred_packaged_task::template invoke4<Arg0 , Arg1 , Arg2 , Arg3>,
            this_(), gid,
            HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg))))
    {
        LLCO_(info) << "deferred_packaged_task::deferred_packaged_task("
                    << hpx::actions::detail::get_action_name<Action>()
                    << ", "
                    << gid
                    << ") args(" << (4 + 1) << ")";
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    void apply(naming::id_type const& gid,
        HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)))
    {
        util::block_profiler_wrapper<deferred_packaged_task_tag> bp(apply_logger_);
        hpx::apply_c<Action>(
            this->get_gid(), gid, HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)));
    }
private:
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    static void invoke5(
        hpx::lcos::deferred_packaged_task<Action,Result> *th,
        naming::id_type const& gid,
        HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)))
    {
        if (!((*th->impl_)->is_ready()))
            th->apply(gid, HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)));
    }
public:
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    deferred_packaged_task(naming::gid_type const& gid,
            HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)))
      : apply_logger_("deferred_packaged_task::apply"),
        closure_(boost::bind(
          &deferred_packaged_task::template invoke5<Arg0 , Arg1 , Arg2 , Arg3 , Arg4>,
            this_(), naming::id_type(gid, naming::id_type::unmanaged),
            HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg))))
    {
        LLCO_(info) << "deferred_packaged_task::deferred_packaged_task("
                    << hpx::actions::detail::get_action_name<Action>()
                    << ", "
                    << gid
                    << ") args(" << (5 + 1) << ")";
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    deferred_packaged_task(naming::id_type const& gid,
            HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)))
      : apply_logger_("deferred_packaged_task::apply"),
        closure_(boost::bind(
          &deferred_packaged_task::template invoke5<Arg0 , Arg1 , Arg2 , Arg3 , Arg4>,
            this_(), gid,
            HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg))))
    {
        LLCO_(info) << "deferred_packaged_task::deferred_packaged_task("
                    << hpx::actions::detail::get_action_name<Action>()
                    << ", "
                    << gid
                    << ") args(" << (5 + 1) << ")";
    }
