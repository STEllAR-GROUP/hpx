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
        util::block_profiler_wrapper<deferred_packaged_task_direct_tag> bp(apply_logger_);
        naming::address addr;
        if (agas::is_local_address(gid, addr)) {
            
            (void)( (!!(components::types_are_compatible(addr.type_, components::get_component_type<typename Action::component_type>()))) || (_wassert(L"components::types_are_compatible(addr.type_, components::get_component_type<typename Action::component_type>())", L"D:/Devel\\hpx\\hpx\\lcos\\deferred_packaged_task_constructors_direct.hpp", 59), 0) );
            (*this->impl_)->set_data(Action::execute_function(addr.address_,
                util::forward_as_tuple(HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)))));
        }
        else {
            
            hpx::applier::detail::apply_c<Action>(addr, this->get_gid(), gid,
                HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)));
        }
    }
private:
    template <typename Arg0 , typename Arg1>
    static void invoke2(
        hpx::lcos::deferred_packaged_task<Action,Result,boost::mpl::true_> *th,
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
      : apply_logger_("deferred_packaged_task_direct::apply"),
        closure_(boost::bind(
            &deferred_packaged_task::template invoke2<Arg0 , Arg1>,
            this, naming::id_type(gid, naming::id_type::unmanaged),
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
      : apply_logger_("deferred_packaged_task_direct::apply"),
        closure_(boost::bind(
            &deferred_packaged_task::template invoke2<Arg0 , Arg1>,
            this, gid,
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
        util::block_profiler_wrapper<deferred_packaged_task_direct_tag> bp(apply_logger_);
        naming::address addr;
        if (agas::is_local_address(gid, addr)) {
            
            (void)( (!!(components::types_are_compatible(addr.type_, components::get_component_type<typename Action::component_type>()))) || (_wassert(L"components::types_are_compatible(addr.type_, components::get_component_type<typename Action::component_type>())", L"D:/Devel\\hpx\\hpx\\lcos\\deferred_packaged_task_constructors_direct.hpp", 59), 0) );
            (*this->impl_)->set_data(Action::execute_function(addr.address_,
                util::forward_as_tuple(HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)))));
        }
        else {
            
            hpx::applier::detail::apply_c<Action>(addr, this->get_gid(), gid,
                HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)));
        }
    }
private:
    template <typename Arg0 , typename Arg1 , typename Arg2>
    static void invoke3(
        hpx::lcos::deferred_packaged_task<Action,Result,boost::mpl::true_> *th,
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
      : apply_logger_("deferred_packaged_task_direct::apply"),
        closure_(boost::bind(
            &deferred_packaged_task::template invoke3<Arg0 , Arg1 , Arg2>,
            this, naming::id_type(gid, naming::id_type::unmanaged),
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
      : apply_logger_("deferred_packaged_task_direct::apply"),
        closure_(boost::bind(
            &deferred_packaged_task::template invoke3<Arg0 , Arg1 , Arg2>,
            this, gid,
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
        util::block_profiler_wrapper<deferred_packaged_task_direct_tag> bp(apply_logger_);
        naming::address addr;
        if (agas::is_local_address(gid, addr)) {
            
            (void)( (!!(components::types_are_compatible(addr.type_, components::get_component_type<typename Action::component_type>()))) || (_wassert(L"components::types_are_compatible(addr.type_, components::get_component_type<typename Action::component_type>())", L"D:/Devel\\hpx\\hpx\\lcos\\deferred_packaged_task_constructors_direct.hpp", 59), 0) );
            (*this->impl_)->set_data(Action::execute_function(addr.address_,
                util::forward_as_tuple(HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)))));
        }
        else {
            
            hpx::applier::detail::apply_c<Action>(addr, this->get_gid(), gid,
                HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)));
        }
    }
private:
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    static void invoke4(
        hpx::lcos::deferred_packaged_task<Action,Result,boost::mpl::true_> *th,
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
      : apply_logger_("deferred_packaged_task_direct::apply"),
        closure_(boost::bind(
            &deferred_packaged_task::template invoke4<Arg0 , Arg1 , Arg2 , Arg3>,
            this, naming::id_type(gid, naming::id_type::unmanaged),
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
      : apply_logger_("deferred_packaged_task_direct::apply"),
        closure_(boost::bind(
            &deferred_packaged_task::template invoke4<Arg0 , Arg1 , Arg2 , Arg3>,
            this, gid,
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
        util::block_profiler_wrapper<deferred_packaged_task_direct_tag> bp(apply_logger_);
        naming::address addr;
        if (agas::is_local_address(gid, addr)) {
            
            (void)( (!!(components::types_are_compatible(addr.type_, components::get_component_type<typename Action::component_type>()))) || (_wassert(L"components::types_are_compatible(addr.type_, components::get_component_type<typename Action::component_type>())", L"D:/Devel\\hpx\\hpx\\lcos\\deferred_packaged_task_constructors_direct.hpp", 59), 0) );
            (*this->impl_)->set_data(Action::execute_function(addr.address_,
                util::forward_as_tuple(HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)))));
        }
        else {
            
            hpx::applier::detail::apply_c<Action>(addr, this->get_gid(), gid,
                HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)));
        }
    }
private:
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    static void invoke5(
        hpx::lcos::deferred_packaged_task<Action,Result,boost::mpl::true_> *th,
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
      : apply_logger_("deferred_packaged_task_direct::apply"),
        closure_(boost::bind(
            &deferred_packaged_task::template invoke5<Arg0 , Arg1 , Arg2 , Arg3 , Arg4>,
            this, naming::id_type(gid, naming::id_type::unmanaged),
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
      : apply_logger_("deferred_packaged_task_direct::apply"),
        closure_(boost::bind(
            &deferred_packaged_task::template invoke5<Arg0 , Arg1 , Arg2 , Arg3 , Arg4>,
            this, gid,
            HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg))))
    {
        LLCO_(info) << "deferred_packaged_task::deferred_packaged_task("
                    << hpx::actions::detail::get_action_name<Action>()
                    << ", "
                    << gid
                    << ") args(" << (5 + 1) << ")";
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    void apply(naming::id_type const& gid,
        HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)))
    {
        util::block_profiler_wrapper<deferred_packaged_task_direct_tag> bp(apply_logger_);
        naming::address addr;
        if (agas::is_local_address(gid, addr)) {
            
            (void)( (!!(components::types_are_compatible(addr.type_, components::get_component_type<typename Action::component_type>()))) || (_wassert(L"components::types_are_compatible(addr.type_, components::get_component_type<typename Action::component_type>())", L"D:/Devel\\hpx\\hpx\\lcos\\deferred_packaged_task_constructors_direct.hpp", 59), 0) );
            (*this->impl_)->set_data(Action::execute_function(addr.address_,
                util::forward_as_tuple(HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)))));
        }
        else {
            
            hpx::applier::detail::apply_c<Action>(addr, this->get_gid(), gid,
                HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)));
        }
    }
private:
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    static void invoke6(
        hpx::lcos::deferred_packaged_task<Action,Result,boost::mpl::true_> *th,
        naming::id_type const& gid,
        HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)))
    {
        if (!((*th->impl_)->is_ready()))
            th->apply(gid, HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)));
    }
public:
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    deferred_packaged_task(naming::gid_type const& gid,
            HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)))
      : apply_logger_("deferred_packaged_task_direct::apply"),
        closure_(boost::bind(
            &deferred_packaged_task::template invoke6<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5>,
            this, naming::id_type(gid, naming::id_type::unmanaged),
            HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg))))
    {
        LLCO_(info) << "deferred_packaged_task::deferred_packaged_task("
                    << hpx::actions::detail::get_action_name<Action>()
                    << ", "
                    << gid
                    << ") args(" << (6 + 1) << ")";
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    deferred_packaged_task(naming::id_type const& gid,
            HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)))
      : apply_logger_("deferred_packaged_task_direct::apply"),
        closure_(boost::bind(
            &deferred_packaged_task::template invoke6<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5>,
            this, gid,
            HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg))))
    {
        LLCO_(info) << "deferred_packaged_task::deferred_packaged_task("
                    << hpx::actions::detail::get_action_name<Action>()
                    << ", "
                    << gid
                    << ") args(" << (6 + 1) << ")";
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    void apply(naming::id_type const& gid,
        HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)))
    {
        util::block_profiler_wrapper<deferred_packaged_task_direct_tag> bp(apply_logger_);
        naming::address addr;
        if (agas::is_local_address(gid, addr)) {
            
            (void)( (!!(components::types_are_compatible(addr.type_, components::get_component_type<typename Action::component_type>()))) || (_wassert(L"components::types_are_compatible(addr.type_, components::get_component_type<typename Action::component_type>())", L"D:/Devel\\hpx\\hpx\\lcos\\deferred_packaged_task_constructors_direct.hpp", 59), 0) );
            (*this->impl_)->set_data(Action::execute_function(addr.address_,
                util::forward_as_tuple(HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)))));
        }
        else {
            
            hpx::applier::detail::apply_c<Action>(addr, this->get_gid(), gid,
                HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)));
        }
    }
private:
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    static void invoke7(
        hpx::lcos::deferred_packaged_task<Action,Result,boost::mpl::true_> *th,
        naming::id_type const& gid,
        HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)))
    {
        if (!((*th->impl_)->is_ready()))
            th->apply(gid, HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)));
    }
public:
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    deferred_packaged_task(naming::gid_type const& gid,
            HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)))
      : apply_logger_("deferred_packaged_task_direct::apply"),
        closure_(boost::bind(
            &deferred_packaged_task::template invoke7<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6>,
            this, naming::id_type(gid, naming::id_type::unmanaged),
            HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg))))
    {
        LLCO_(info) << "deferred_packaged_task::deferred_packaged_task("
                    << hpx::actions::detail::get_action_name<Action>()
                    << ", "
                    << gid
                    << ") args(" << (7 + 1) << ")";
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    deferred_packaged_task(naming::id_type const& gid,
            HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)))
      : apply_logger_("deferred_packaged_task_direct::apply"),
        closure_(boost::bind(
            &deferred_packaged_task::template invoke7<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6>,
            this, gid,
            HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg))))
    {
        LLCO_(info) << "deferred_packaged_task::deferred_packaged_task("
                    << hpx::actions::detail::get_action_name<Action>()
                    << ", "
                    << gid
                    << ") args(" << (7 + 1) << ")";
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    void apply(naming::id_type const& gid,
        HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)))
    {
        util::block_profiler_wrapper<deferred_packaged_task_direct_tag> bp(apply_logger_);
        naming::address addr;
        if (agas::is_local_address(gid, addr)) {
            
            (void)( (!!(components::types_are_compatible(addr.type_, components::get_component_type<typename Action::component_type>()))) || (_wassert(L"components::types_are_compatible(addr.type_, components::get_component_type<typename Action::component_type>())", L"D:/Devel\\hpx\\hpx\\lcos\\deferred_packaged_task_constructors_direct.hpp", 59), 0) );
            (*this->impl_)->set_data(Action::execute_function(addr.address_,
                util::forward_as_tuple(HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)))));
        }
        else {
            
            hpx::applier::detail::apply_c<Action>(addr, this->get_gid(), gid,
                HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)));
        }
    }
private:
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    static void invoke8(
        hpx::lcos::deferred_packaged_task<Action,Result,boost::mpl::true_> *th,
        naming::id_type const& gid,
        HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)))
    {
        if (!((*th->impl_)->is_ready()))
            th->apply(gid, HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)));
    }
public:
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    deferred_packaged_task(naming::gid_type const& gid,
            HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)))
      : apply_logger_("deferred_packaged_task_direct::apply"),
        closure_(boost::bind(
            &deferred_packaged_task::template invoke8<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7>,
            this, naming::id_type(gid, naming::id_type::unmanaged),
            HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg))))
    {
        LLCO_(info) << "deferred_packaged_task::deferred_packaged_task("
                    << hpx::actions::detail::get_action_name<Action>()
                    << ", "
                    << gid
                    << ") args(" << (8 + 1) << ")";
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    deferred_packaged_task(naming::id_type const& gid,
            HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)))
      : apply_logger_("deferred_packaged_task_direct::apply"),
        closure_(boost::bind(
            &deferred_packaged_task::template invoke8<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7>,
            this, gid,
            HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg))))
    {
        LLCO_(info) << "deferred_packaged_task::deferred_packaged_task("
                    << hpx::actions::detail::get_action_name<Action>()
                    << ", "
                    << gid
                    << ") args(" << (8 + 1) << ")";
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
    void apply(naming::id_type const& gid,
        HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)))
    {
        util::block_profiler_wrapper<deferred_packaged_task_direct_tag> bp(apply_logger_);
        naming::address addr;
        if (agas::is_local_address(gid, addr)) {
            
            (void)( (!!(components::types_are_compatible(addr.type_, components::get_component_type<typename Action::component_type>()))) || (_wassert(L"components::types_are_compatible(addr.type_, components::get_component_type<typename Action::component_type>())", L"D:/Devel\\hpx\\hpx\\lcos\\deferred_packaged_task_constructors_direct.hpp", 59), 0) );
            (*this->impl_)->set_data(Action::execute_function(addr.address_,
                util::forward_as_tuple(HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 8, ( Arg, arg)))));
        }
        else {
            
            hpx::applier::detail::apply_c<Action>(addr, this->get_gid(), gid,
                HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 8, ( Arg, arg)));
        }
    }
private:
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
    static void invoke9(
        hpx::lcos::deferred_packaged_task<Action,Result,boost::mpl::true_> *th,
        naming::id_type const& gid,
        HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)))
    {
        if (!((*th->impl_)->is_ready()))
            th->apply(gid, HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 8, ( Arg, arg)));
    }
public:
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
    deferred_packaged_task(naming::gid_type const& gid,
            HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)))
      : apply_logger_("deferred_packaged_task_direct::apply"),
        closure_(boost::bind(
            &deferred_packaged_task::template invoke9<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8>,
            this, naming::id_type(gid, naming::id_type::unmanaged),
            HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 8, ( Arg, arg))))
    {
        LLCO_(info) << "deferred_packaged_task::deferred_packaged_task("
                    << hpx::actions::detail::get_action_name<Action>()
                    << ", "
                    << gid
                    << ") args(" << (9 + 1) << ")";
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
    deferred_packaged_task(naming::id_type const& gid,
            HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)))
      : apply_logger_("deferred_packaged_task_direct::apply"),
        closure_(boost::bind(
            &deferred_packaged_task::template invoke9<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8>,
            this, gid,
            HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 8, ( Arg, arg))))
    {
        LLCO_(info) << "deferred_packaged_task::deferred_packaged_task("
                    << hpx::actions::detail::get_action_name<Action>()
                    << ", "
                    << gid
                    << ") args(" << (9 + 1) << ")";
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
    void apply(naming::id_type const& gid,
        HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)))
    {
        util::block_profiler_wrapper<deferred_packaged_task_direct_tag> bp(apply_logger_);
        naming::address addr;
        if (agas::is_local_address(gid, addr)) {
            
            (void)( (!!(components::types_are_compatible(addr.type_, components::get_component_type<typename Action::component_type>()))) || (_wassert(L"components::types_are_compatible(addr.type_, components::get_component_type<typename Action::component_type>())", L"D:/Devel\\hpx\\hpx\\lcos\\deferred_packaged_task_constructors_direct.hpp", 59), 0) );
            (*this->impl_)->set_data(Action::execute_function(addr.address_,
                util::forward_as_tuple(HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 8, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 9, ( Arg, arg)))));
        }
        else {
            
            hpx::applier::detail::apply_c<Action>(addr, this->get_gid(), gid,
                HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 8, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 9, ( Arg, arg)));
        }
    }
private:
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
    static void invoke10(
        hpx::lcos::deferred_packaged_task<Action,Result,boost::mpl::true_> *th,
        naming::id_type const& gid,
        HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)))
    {
        if (!((*th->impl_)->is_ready()))
            th->apply(gid, HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 8, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 9, ( Arg, arg)));
    }
public:
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
    deferred_packaged_task(naming::gid_type const& gid,
            HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)))
      : apply_logger_("deferred_packaged_task_direct::apply"),
        closure_(boost::bind(
            &deferred_packaged_task::template invoke10<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9>,
            this, naming::id_type(gid, naming::id_type::unmanaged),
            HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 8, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 9, ( Arg, arg))))
    {
        LLCO_(info) << "deferred_packaged_task::deferred_packaged_task("
                    << hpx::actions::detail::get_action_name<Action>()
                    << ", "
                    << gid
                    << ") args(" << (10 + 1) << ")";
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
    deferred_packaged_task(naming::id_type const& gid,
            HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)))
      : apply_logger_("deferred_packaged_task_direct::apply"),
        closure_(boost::bind(
            &deferred_packaged_task::template invoke10<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9>,
            this, gid,
            HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 8, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 9, ( Arg, arg))))
    {
        LLCO_(info) << "deferred_packaged_task::deferred_packaged_task("
                    << hpx::actions::detail::get_action_name<Action>()
                    << ", "
                    << gid
                    << ") args(" << (10 + 1) << ")";
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10>
    void apply(naming::id_type const& gid,
        HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)) , HPX_FWD_ARGS(2, 10, ( Arg, arg)))
    {
        util::block_profiler_wrapper<deferred_packaged_task_direct_tag> bp(apply_logger_);
        naming::address addr;
        if (agas::is_local_address(gid, addr)) {
            
            (void)( (!!(components::types_are_compatible(addr.type_, components::get_component_type<typename Action::component_type>()))) || (_wassert(L"components::types_are_compatible(addr.type_, components::get_component_type<typename Action::component_type>())", L"D:/Devel\\hpx\\hpx\\lcos\\deferred_packaged_task_constructors_direct.hpp", 59), 0) );
            (*this->impl_)->set_data(Action::execute_function(addr.address_,
                util::forward_as_tuple(HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 8, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 9, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 10, ( Arg, arg)))));
        }
        else {
            
            hpx::applier::detail::apply_c<Action>(addr, this->get_gid(), gid,
                HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 8, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 9, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 10, ( Arg, arg)));
        }
    }
private:
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10>
    static void invoke11(
        hpx::lcos::deferred_packaged_task<Action,Result,boost::mpl::true_> *th,
        naming::id_type const& gid,
        HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)) , HPX_FWD_ARGS(2, 10, ( Arg, arg)))
    {
        if (!((*th->impl_)->is_ready()))
            th->apply(gid, HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 8, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 9, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 10, ( Arg, arg)));
    }
public:
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10>
    deferred_packaged_task(naming::gid_type const& gid,
            HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)) , HPX_FWD_ARGS(2, 10, ( Arg, arg)))
      : apply_logger_("deferred_packaged_task_direct::apply"),
        closure_(boost::bind(
            &deferred_packaged_task::template invoke11<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10>,
            this, naming::id_type(gid, naming::id_type::unmanaged),
            HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 8, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 9, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 10, ( Arg, arg))))
    {
        LLCO_(info) << "deferred_packaged_task::deferred_packaged_task("
                    << hpx::actions::detail::get_action_name<Action>()
                    << ", "
                    << gid
                    << ") args(" << (11 + 1) << ")";
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10>
    deferred_packaged_task(naming::id_type const& gid,
            HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)) , HPX_FWD_ARGS(2, 10, ( Arg, arg)))
      : apply_logger_("deferred_packaged_task_direct::apply"),
        closure_(boost::bind(
            &deferred_packaged_task::template invoke11<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10>,
            this, gid,
            HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 8, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 9, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 10, ( Arg, arg))))
    {
        LLCO_(info) << "deferred_packaged_task::deferred_packaged_task("
                    << hpx::actions::detail::get_action_name<Action>()
                    << ", "
                    << gid
                    << ") args(" << (11 + 1) << ")";
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11>
    void apply(naming::id_type const& gid,
        HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)) , HPX_FWD_ARGS(2, 10, ( Arg, arg)) , HPX_FWD_ARGS(2, 11, ( Arg, arg)))
    {
        util::block_profiler_wrapper<deferred_packaged_task_direct_tag> bp(apply_logger_);
        naming::address addr;
        if (agas::is_local_address(gid, addr)) {
            
            (void)( (!!(components::types_are_compatible(addr.type_, components::get_component_type<typename Action::component_type>()))) || (_wassert(L"components::types_are_compatible(addr.type_, components::get_component_type<typename Action::component_type>())", L"D:/Devel\\hpx\\hpx\\lcos\\deferred_packaged_task_constructors_direct.hpp", 59), 0) );
            (*this->impl_)->set_data(Action::execute_function(addr.address_,
                util::forward_as_tuple(HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 8, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 9, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 10, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 11, ( Arg, arg)))));
        }
        else {
            
            hpx::applier::detail::apply_c<Action>(addr, this->get_gid(), gid,
                HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 8, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 9, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 10, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 11, ( Arg, arg)));
        }
    }
private:
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11>
    static void invoke12(
        hpx::lcos::deferred_packaged_task<Action,Result,boost::mpl::true_> *th,
        naming::id_type const& gid,
        HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)) , HPX_FWD_ARGS(2, 10, ( Arg, arg)) , HPX_FWD_ARGS(2, 11, ( Arg, arg)))
    {
        if (!((*th->impl_)->is_ready()))
            th->apply(gid, HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 8, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 9, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 10, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 11, ( Arg, arg)));
    }
public:
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11>
    deferred_packaged_task(naming::gid_type const& gid,
            HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)) , HPX_FWD_ARGS(2, 10, ( Arg, arg)) , HPX_FWD_ARGS(2, 11, ( Arg, arg)))
      : apply_logger_("deferred_packaged_task_direct::apply"),
        closure_(boost::bind(
            &deferred_packaged_task::template invoke12<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11>,
            this, naming::id_type(gid, naming::id_type::unmanaged),
            HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 8, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 9, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 10, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 11, ( Arg, arg))))
    {
        LLCO_(info) << "deferred_packaged_task::deferred_packaged_task("
                    << hpx::actions::detail::get_action_name<Action>()
                    << ", "
                    << gid
                    << ") args(" << (12 + 1) << ")";
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11>
    deferred_packaged_task(naming::id_type const& gid,
            HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)) , HPX_FWD_ARGS(2, 10, ( Arg, arg)) , HPX_FWD_ARGS(2, 11, ( Arg, arg)))
      : apply_logger_("deferred_packaged_task_direct::apply"),
        closure_(boost::bind(
            &deferred_packaged_task::template invoke12<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11>,
            this, gid,
            HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 8, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 9, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 10, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 11, ( Arg, arg))))
    {
        LLCO_(info) << "deferred_packaged_task::deferred_packaged_task("
                    << hpx::actions::detail::get_action_name<Action>()
                    << ", "
                    << gid
                    << ") args(" << (12 + 1) << ")";
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12>
    void apply(naming::id_type const& gid,
        HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)) , HPX_FWD_ARGS(2, 10, ( Arg, arg)) , HPX_FWD_ARGS(2, 11, ( Arg, arg)) , HPX_FWD_ARGS(2, 12, ( Arg, arg)))
    {
        util::block_profiler_wrapper<deferred_packaged_task_direct_tag> bp(apply_logger_);
        naming::address addr;
        if (agas::is_local_address(gid, addr)) {
            
            (void)( (!!(components::types_are_compatible(addr.type_, components::get_component_type<typename Action::component_type>()))) || (_wassert(L"components::types_are_compatible(addr.type_, components::get_component_type<typename Action::component_type>())", L"D:/Devel\\hpx\\hpx\\lcos\\deferred_packaged_task_constructors_direct.hpp", 59), 0) );
            (*this->impl_)->set_data(Action::execute_function(addr.address_,
                util::forward_as_tuple(HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 8, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 9, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 10, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 11, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 12, ( Arg, arg)))));
        }
        else {
            
            hpx::applier::detail::apply_c<Action>(addr, this->get_gid(), gid,
                HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 8, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 9, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 10, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 11, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 12, ( Arg, arg)));
        }
    }
private:
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12>
    static void invoke13(
        hpx::lcos::deferred_packaged_task<Action,Result,boost::mpl::true_> *th,
        naming::id_type const& gid,
        HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)) , HPX_FWD_ARGS(2, 10, ( Arg, arg)) , HPX_FWD_ARGS(2, 11, ( Arg, arg)) , HPX_FWD_ARGS(2, 12, ( Arg, arg)))
    {
        if (!((*th->impl_)->is_ready()))
            th->apply(gid, HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 8, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 9, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 10, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 11, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 12, ( Arg, arg)));
    }
public:
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12>
    deferred_packaged_task(naming::gid_type const& gid,
            HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)) , HPX_FWD_ARGS(2, 10, ( Arg, arg)) , HPX_FWD_ARGS(2, 11, ( Arg, arg)) , HPX_FWD_ARGS(2, 12, ( Arg, arg)))
      : apply_logger_("deferred_packaged_task_direct::apply"),
        closure_(boost::bind(
            &deferred_packaged_task::template invoke13<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12>,
            this, naming::id_type(gid, naming::id_type::unmanaged),
            HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 8, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 9, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 10, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 11, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 12, ( Arg, arg))))
    {
        LLCO_(info) << "deferred_packaged_task::deferred_packaged_task("
                    << hpx::actions::detail::get_action_name<Action>()
                    << ", "
                    << gid
                    << ") args(" << (13 + 1) << ")";
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12>
    deferred_packaged_task(naming::id_type const& gid,
            HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)) , HPX_FWD_ARGS(2, 10, ( Arg, arg)) , HPX_FWD_ARGS(2, 11, ( Arg, arg)) , HPX_FWD_ARGS(2, 12, ( Arg, arg)))
      : apply_logger_("deferred_packaged_task_direct::apply"),
        closure_(boost::bind(
            &deferred_packaged_task::template invoke13<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12>,
            this, gid,
            HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 8, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 9, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 10, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 11, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 12, ( Arg, arg))))
    {
        LLCO_(info) << "deferred_packaged_task::deferred_packaged_task("
                    << hpx::actions::detail::get_action_name<Action>()
                    << ", "
                    << gid
                    << ") args(" << (13 + 1) << ")";
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13>
    void apply(naming::id_type const& gid,
        HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)) , HPX_FWD_ARGS(2, 10, ( Arg, arg)) , HPX_FWD_ARGS(2, 11, ( Arg, arg)) , HPX_FWD_ARGS(2, 12, ( Arg, arg)) , HPX_FWD_ARGS(2, 13, ( Arg, arg)))
    {
        util::block_profiler_wrapper<deferred_packaged_task_direct_tag> bp(apply_logger_);
        naming::address addr;
        if (agas::is_local_address(gid, addr)) {
            
            (void)( (!!(components::types_are_compatible(addr.type_, components::get_component_type<typename Action::component_type>()))) || (_wassert(L"components::types_are_compatible(addr.type_, components::get_component_type<typename Action::component_type>())", L"D:/Devel\\hpx\\hpx\\lcos\\deferred_packaged_task_constructors_direct.hpp", 59), 0) );
            (*this->impl_)->set_data(Action::execute_function(addr.address_,
                util::forward_as_tuple(HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 8, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 9, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 10, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 11, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 12, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 13, ( Arg, arg)))));
        }
        else {
            
            hpx::applier::detail::apply_c<Action>(addr, this->get_gid(), gid,
                HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 8, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 9, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 10, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 11, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 12, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 13, ( Arg, arg)));
        }
    }
private:
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13>
    static void invoke14(
        hpx::lcos::deferred_packaged_task<Action,Result,boost::mpl::true_> *th,
        naming::id_type const& gid,
        HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)) , HPX_FWD_ARGS(2, 10, ( Arg, arg)) , HPX_FWD_ARGS(2, 11, ( Arg, arg)) , HPX_FWD_ARGS(2, 12, ( Arg, arg)) , HPX_FWD_ARGS(2, 13, ( Arg, arg)))
    {
        if (!((*th->impl_)->is_ready()))
            th->apply(gid, HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 8, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 9, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 10, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 11, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 12, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 13, ( Arg, arg)));
    }
public:
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13>
    deferred_packaged_task(naming::gid_type const& gid,
            HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)) , HPX_FWD_ARGS(2, 10, ( Arg, arg)) , HPX_FWD_ARGS(2, 11, ( Arg, arg)) , HPX_FWD_ARGS(2, 12, ( Arg, arg)) , HPX_FWD_ARGS(2, 13, ( Arg, arg)))
      : apply_logger_("deferred_packaged_task_direct::apply"),
        closure_(boost::bind(
            &deferred_packaged_task::template invoke14<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13>,
            this, naming::id_type(gid, naming::id_type::unmanaged),
            HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 8, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 9, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 10, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 11, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 12, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 13, ( Arg, arg))))
    {
        LLCO_(info) << "deferred_packaged_task::deferred_packaged_task("
                    << hpx::actions::detail::get_action_name<Action>()
                    << ", "
                    << gid
                    << ") args(" << (14 + 1) << ")";
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13>
    deferred_packaged_task(naming::id_type const& gid,
            HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)) , HPX_FWD_ARGS(2, 10, ( Arg, arg)) , HPX_FWD_ARGS(2, 11, ( Arg, arg)) , HPX_FWD_ARGS(2, 12, ( Arg, arg)) , HPX_FWD_ARGS(2, 13, ( Arg, arg)))
      : apply_logger_("deferred_packaged_task_direct::apply"),
        closure_(boost::bind(
            &deferred_packaged_task::template invoke14<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13>,
            this, gid,
            HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 8, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 9, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 10, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 11, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 12, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 13, ( Arg, arg))))
    {
        LLCO_(info) << "deferred_packaged_task::deferred_packaged_task("
                    << hpx::actions::detail::get_action_name<Action>()
                    << ", "
                    << gid
                    << ") args(" << (14 + 1) << ")";
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14>
    void apply(naming::id_type const& gid,
        HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)) , HPX_FWD_ARGS(2, 10, ( Arg, arg)) , HPX_FWD_ARGS(2, 11, ( Arg, arg)) , HPX_FWD_ARGS(2, 12, ( Arg, arg)) , HPX_FWD_ARGS(2, 13, ( Arg, arg)) , HPX_FWD_ARGS(2, 14, ( Arg, arg)))
    {
        util::block_profiler_wrapper<deferred_packaged_task_direct_tag> bp(apply_logger_);
        naming::address addr;
        if (agas::is_local_address(gid, addr)) {
            
            (void)( (!!(components::types_are_compatible(addr.type_, components::get_component_type<typename Action::component_type>()))) || (_wassert(L"components::types_are_compatible(addr.type_, components::get_component_type<typename Action::component_type>())", L"D:/Devel\\hpx\\hpx\\lcos\\deferred_packaged_task_constructors_direct.hpp", 59), 0) );
            (*this->impl_)->set_data(Action::execute_function(addr.address_,
                util::forward_as_tuple(HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 8, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 9, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 10, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 11, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 12, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 13, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 14, ( Arg, arg)))));
        }
        else {
            
            hpx::applier::detail::apply_c<Action>(addr, this->get_gid(), gid,
                HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 8, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 9, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 10, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 11, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 12, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 13, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 14, ( Arg, arg)));
        }
    }
private:
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14>
    static void invoke15(
        hpx::lcos::deferred_packaged_task<Action,Result,boost::mpl::true_> *th,
        naming::id_type const& gid,
        HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)) , HPX_FWD_ARGS(2, 10, ( Arg, arg)) , HPX_FWD_ARGS(2, 11, ( Arg, arg)) , HPX_FWD_ARGS(2, 12, ( Arg, arg)) , HPX_FWD_ARGS(2, 13, ( Arg, arg)) , HPX_FWD_ARGS(2, 14, ( Arg, arg)))
    {
        if (!((*th->impl_)->is_ready()))
            th->apply(gid, HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 8, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 9, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 10, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 11, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 12, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 13, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 14, ( Arg, arg)));
    }
public:
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14>
    deferred_packaged_task(naming::gid_type const& gid,
            HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)) , HPX_FWD_ARGS(2, 10, ( Arg, arg)) , HPX_FWD_ARGS(2, 11, ( Arg, arg)) , HPX_FWD_ARGS(2, 12, ( Arg, arg)) , HPX_FWD_ARGS(2, 13, ( Arg, arg)) , HPX_FWD_ARGS(2, 14, ( Arg, arg)))
      : apply_logger_("deferred_packaged_task_direct::apply"),
        closure_(boost::bind(
            &deferred_packaged_task::template invoke15<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14>,
            this, naming::id_type(gid, naming::id_type::unmanaged),
            HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 8, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 9, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 10, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 11, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 12, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 13, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 14, ( Arg, arg))))
    {
        LLCO_(info) << "deferred_packaged_task::deferred_packaged_task("
                    << hpx::actions::detail::get_action_name<Action>()
                    << ", "
                    << gid
                    << ") args(" << (15 + 1) << ")";
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14>
    deferred_packaged_task(naming::id_type const& gid,
            HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)) , HPX_FWD_ARGS(2, 10, ( Arg, arg)) , HPX_FWD_ARGS(2, 11, ( Arg, arg)) , HPX_FWD_ARGS(2, 12, ( Arg, arg)) , HPX_FWD_ARGS(2, 13, ( Arg, arg)) , HPX_FWD_ARGS(2, 14, ( Arg, arg)))
      : apply_logger_("deferred_packaged_task_direct::apply"),
        closure_(boost::bind(
            &deferred_packaged_task::template invoke15<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14>,
            this, gid,
            HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 8, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 9, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 10, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 11, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 12, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 13, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 14, ( Arg, arg))))
    {
        LLCO_(info) << "deferred_packaged_task::deferred_packaged_task("
                    << hpx::actions::detail::get_action_name<Action>()
                    << ", "
                    << gid
                    << ") args(" << (15 + 1) << ")";
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15>
    void apply(naming::id_type const& gid,
        HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)) , HPX_FWD_ARGS(2, 10, ( Arg, arg)) , HPX_FWD_ARGS(2, 11, ( Arg, arg)) , HPX_FWD_ARGS(2, 12, ( Arg, arg)) , HPX_FWD_ARGS(2, 13, ( Arg, arg)) , HPX_FWD_ARGS(2, 14, ( Arg, arg)) , HPX_FWD_ARGS(2, 15, ( Arg, arg)))
    {
        util::block_profiler_wrapper<deferred_packaged_task_direct_tag> bp(apply_logger_);
        naming::address addr;
        if (agas::is_local_address(gid, addr)) {
            
            (void)( (!!(components::types_are_compatible(addr.type_, components::get_component_type<typename Action::component_type>()))) || (_wassert(L"components::types_are_compatible(addr.type_, components::get_component_type<typename Action::component_type>())", L"D:/Devel\\hpx\\hpx\\lcos\\deferred_packaged_task_constructors_direct.hpp", 59), 0) );
            (*this->impl_)->set_data(Action::execute_function(addr.address_,
                util::forward_as_tuple(HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 8, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 9, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 10, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 11, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 12, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 13, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 14, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 15, ( Arg, arg)))));
        }
        else {
            
            hpx::applier::detail::apply_c<Action>(addr, this->get_gid(), gid,
                HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 8, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 9, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 10, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 11, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 12, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 13, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 14, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 15, ( Arg, arg)));
        }
    }
private:
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15>
    static void invoke16(
        hpx::lcos::deferred_packaged_task<Action,Result,boost::mpl::true_> *th,
        naming::id_type const& gid,
        HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)) , HPX_FWD_ARGS(2, 10, ( Arg, arg)) , HPX_FWD_ARGS(2, 11, ( Arg, arg)) , HPX_FWD_ARGS(2, 12, ( Arg, arg)) , HPX_FWD_ARGS(2, 13, ( Arg, arg)) , HPX_FWD_ARGS(2, 14, ( Arg, arg)) , HPX_FWD_ARGS(2, 15, ( Arg, arg)))
    {
        if (!((*th->impl_)->is_ready()))
            th->apply(gid, HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 8, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 9, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 10, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 11, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 12, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 13, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 14, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 15, ( Arg, arg)));
    }
public:
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15>
    deferred_packaged_task(naming::gid_type const& gid,
            HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)) , HPX_FWD_ARGS(2, 10, ( Arg, arg)) , HPX_FWD_ARGS(2, 11, ( Arg, arg)) , HPX_FWD_ARGS(2, 12, ( Arg, arg)) , HPX_FWD_ARGS(2, 13, ( Arg, arg)) , HPX_FWD_ARGS(2, 14, ( Arg, arg)) , HPX_FWD_ARGS(2, 15, ( Arg, arg)))
      : apply_logger_("deferred_packaged_task_direct::apply"),
        closure_(boost::bind(
            &deferred_packaged_task::template invoke16<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15>,
            this, naming::id_type(gid, naming::id_type::unmanaged),
            HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 8, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 9, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 10, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 11, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 12, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 13, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 14, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 15, ( Arg, arg))))
    {
        LLCO_(info) << "deferred_packaged_task::deferred_packaged_task("
                    << hpx::actions::detail::get_action_name<Action>()
                    << ", "
                    << gid
                    << ") args(" << (16 + 1) << ")";
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15>
    deferred_packaged_task(naming::id_type const& gid,
            HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)) , HPX_FWD_ARGS(2, 10, ( Arg, arg)) , HPX_FWD_ARGS(2, 11, ( Arg, arg)) , HPX_FWD_ARGS(2, 12, ( Arg, arg)) , HPX_FWD_ARGS(2, 13, ( Arg, arg)) , HPX_FWD_ARGS(2, 14, ( Arg, arg)) , HPX_FWD_ARGS(2, 15, ( Arg, arg)))
      : apply_logger_("deferred_packaged_task_direct::apply"),
        closure_(boost::bind(
            &deferred_packaged_task::template invoke16<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15>,
            this, gid,
            HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 8, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 9, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 10, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 11, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 12, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 13, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 14, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 15, ( Arg, arg))))
    {
        LLCO_(info) << "deferred_packaged_task::deferred_packaged_task("
                    << hpx::actions::detail::get_action_name<Action>()
                    << ", "
                    << gid
                    << ") args(" << (16 + 1) << ")";
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16>
    void apply(naming::id_type const& gid,
        HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)) , HPX_FWD_ARGS(2, 10, ( Arg, arg)) , HPX_FWD_ARGS(2, 11, ( Arg, arg)) , HPX_FWD_ARGS(2, 12, ( Arg, arg)) , HPX_FWD_ARGS(2, 13, ( Arg, arg)) , HPX_FWD_ARGS(2, 14, ( Arg, arg)) , HPX_FWD_ARGS(2, 15, ( Arg, arg)) , HPX_FWD_ARGS(2, 16, ( Arg, arg)))
    {
        util::block_profiler_wrapper<deferred_packaged_task_direct_tag> bp(apply_logger_);
        naming::address addr;
        if (agas::is_local_address(gid, addr)) {
            
            (void)( (!!(components::types_are_compatible(addr.type_, components::get_component_type<typename Action::component_type>()))) || (_wassert(L"components::types_are_compatible(addr.type_, components::get_component_type<typename Action::component_type>())", L"D:/Devel\\hpx\\hpx\\lcos\\deferred_packaged_task_constructors_direct.hpp", 59), 0) );
            (*this->impl_)->set_data(Action::execute_function(addr.address_,
                util::forward_as_tuple(HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 8, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 9, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 10, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 11, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 12, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 13, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 14, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 15, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 16, ( Arg, arg)))));
        }
        else {
            
            hpx::applier::detail::apply_c<Action>(addr, this->get_gid(), gid,
                HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 8, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 9, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 10, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 11, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 12, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 13, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 14, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 15, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 16, ( Arg, arg)));
        }
    }
private:
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16>
    static void invoke17(
        hpx::lcos::deferred_packaged_task<Action,Result,boost::mpl::true_> *th,
        naming::id_type const& gid,
        HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)) , HPX_FWD_ARGS(2, 10, ( Arg, arg)) , HPX_FWD_ARGS(2, 11, ( Arg, arg)) , HPX_FWD_ARGS(2, 12, ( Arg, arg)) , HPX_FWD_ARGS(2, 13, ( Arg, arg)) , HPX_FWD_ARGS(2, 14, ( Arg, arg)) , HPX_FWD_ARGS(2, 15, ( Arg, arg)) , HPX_FWD_ARGS(2, 16, ( Arg, arg)))
    {
        if (!((*th->impl_)->is_ready()))
            th->apply(gid, HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 8, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 9, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 10, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 11, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 12, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 13, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 14, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 15, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 16, ( Arg, arg)));
    }
public:
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16>
    deferred_packaged_task(naming::gid_type const& gid,
            HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)) , HPX_FWD_ARGS(2, 10, ( Arg, arg)) , HPX_FWD_ARGS(2, 11, ( Arg, arg)) , HPX_FWD_ARGS(2, 12, ( Arg, arg)) , HPX_FWD_ARGS(2, 13, ( Arg, arg)) , HPX_FWD_ARGS(2, 14, ( Arg, arg)) , HPX_FWD_ARGS(2, 15, ( Arg, arg)) , HPX_FWD_ARGS(2, 16, ( Arg, arg)))
      : apply_logger_("deferred_packaged_task_direct::apply"),
        closure_(boost::bind(
            &deferred_packaged_task::template invoke17<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16>,
            this, naming::id_type(gid, naming::id_type::unmanaged),
            HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 8, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 9, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 10, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 11, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 12, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 13, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 14, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 15, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 16, ( Arg, arg))))
    {
        LLCO_(info) << "deferred_packaged_task::deferred_packaged_task("
                    << hpx::actions::detail::get_action_name<Action>()
                    << ", "
                    << gid
                    << ") args(" << (17 + 1) << ")";
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16>
    deferred_packaged_task(naming::id_type const& gid,
            HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)) , HPX_FWD_ARGS(2, 10, ( Arg, arg)) , HPX_FWD_ARGS(2, 11, ( Arg, arg)) , HPX_FWD_ARGS(2, 12, ( Arg, arg)) , HPX_FWD_ARGS(2, 13, ( Arg, arg)) , HPX_FWD_ARGS(2, 14, ( Arg, arg)) , HPX_FWD_ARGS(2, 15, ( Arg, arg)) , HPX_FWD_ARGS(2, 16, ( Arg, arg)))
      : apply_logger_("deferred_packaged_task_direct::apply"),
        closure_(boost::bind(
            &deferred_packaged_task::template invoke17<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16>,
            this, gid,
            HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 8, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 9, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 10, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 11, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 12, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 13, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 14, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 15, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 16, ( Arg, arg))))
    {
        LLCO_(info) << "deferred_packaged_task::deferred_packaged_task("
                    << hpx::actions::detail::get_action_name<Action>()
                    << ", "
                    << gid
                    << ") args(" << (17 + 1) << ")";
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17>
    void apply(naming::id_type const& gid,
        HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)) , HPX_FWD_ARGS(2, 10, ( Arg, arg)) , HPX_FWD_ARGS(2, 11, ( Arg, arg)) , HPX_FWD_ARGS(2, 12, ( Arg, arg)) , HPX_FWD_ARGS(2, 13, ( Arg, arg)) , HPX_FWD_ARGS(2, 14, ( Arg, arg)) , HPX_FWD_ARGS(2, 15, ( Arg, arg)) , HPX_FWD_ARGS(2, 16, ( Arg, arg)) , HPX_FWD_ARGS(2, 17, ( Arg, arg)))
    {
        util::block_profiler_wrapper<deferred_packaged_task_direct_tag> bp(apply_logger_);
        naming::address addr;
        if (agas::is_local_address(gid, addr)) {
            
            (void)( (!!(components::types_are_compatible(addr.type_, components::get_component_type<typename Action::component_type>()))) || (_wassert(L"components::types_are_compatible(addr.type_, components::get_component_type<typename Action::component_type>())", L"D:/Devel\\hpx\\hpx\\lcos\\deferred_packaged_task_constructors_direct.hpp", 59), 0) );
            (*this->impl_)->set_data(Action::execute_function(addr.address_,
                util::forward_as_tuple(HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 8, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 9, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 10, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 11, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 12, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 13, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 14, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 15, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 16, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 17, ( Arg, arg)))));
        }
        else {
            
            hpx::applier::detail::apply_c<Action>(addr, this->get_gid(), gid,
                HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 8, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 9, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 10, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 11, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 12, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 13, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 14, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 15, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 16, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 17, ( Arg, arg)));
        }
    }
private:
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17>
    static void invoke18(
        hpx::lcos::deferred_packaged_task<Action,Result,boost::mpl::true_> *th,
        naming::id_type const& gid,
        HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)) , HPX_FWD_ARGS(2, 10, ( Arg, arg)) , HPX_FWD_ARGS(2, 11, ( Arg, arg)) , HPX_FWD_ARGS(2, 12, ( Arg, arg)) , HPX_FWD_ARGS(2, 13, ( Arg, arg)) , HPX_FWD_ARGS(2, 14, ( Arg, arg)) , HPX_FWD_ARGS(2, 15, ( Arg, arg)) , HPX_FWD_ARGS(2, 16, ( Arg, arg)) , HPX_FWD_ARGS(2, 17, ( Arg, arg)))
    {
        if (!((*th->impl_)->is_ready()))
            th->apply(gid, HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 8, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 9, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 10, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 11, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 12, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 13, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 14, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 15, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 16, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 17, ( Arg, arg)));
    }
public:
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17>
    deferred_packaged_task(naming::gid_type const& gid,
            HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)) , HPX_FWD_ARGS(2, 10, ( Arg, arg)) , HPX_FWD_ARGS(2, 11, ( Arg, arg)) , HPX_FWD_ARGS(2, 12, ( Arg, arg)) , HPX_FWD_ARGS(2, 13, ( Arg, arg)) , HPX_FWD_ARGS(2, 14, ( Arg, arg)) , HPX_FWD_ARGS(2, 15, ( Arg, arg)) , HPX_FWD_ARGS(2, 16, ( Arg, arg)) , HPX_FWD_ARGS(2, 17, ( Arg, arg)))
      : apply_logger_("deferred_packaged_task_direct::apply"),
        closure_(boost::bind(
            &deferred_packaged_task::template invoke18<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17>,
            this, naming::id_type(gid, naming::id_type::unmanaged),
            HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 8, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 9, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 10, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 11, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 12, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 13, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 14, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 15, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 16, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 17, ( Arg, arg))))
    {
        LLCO_(info) << "deferred_packaged_task::deferred_packaged_task("
                    << hpx::actions::detail::get_action_name<Action>()
                    << ", "
                    << gid
                    << ") args(" << (18 + 1) << ")";
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17>
    deferred_packaged_task(naming::id_type const& gid,
            HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)) , HPX_FWD_ARGS(2, 10, ( Arg, arg)) , HPX_FWD_ARGS(2, 11, ( Arg, arg)) , HPX_FWD_ARGS(2, 12, ( Arg, arg)) , HPX_FWD_ARGS(2, 13, ( Arg, arg)) , HPX_FWD_ARGS(2, 14, ( Arg, arg)) , HPX_FWD_ARGS(2, 15, ( Arg, arg)) , HPX_FWD_ARGS(2, 16, ( Arg, arg)) , HPX_FWD_ARGS(2, 17, ( Arg, arg)))
      : apply_logger_("deferred_packaged_task_direct::apply"),
        closure_(boost::bind(
            &deferred_packaged_task::template invoke18<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17>,
            this, gid,
            HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 8, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 9, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 10, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 11, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 12, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 13, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 14, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 15, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 16, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 17, ( Arg, arg))))
    {
        LLCO_(info) << "deferred_packaged_task::deferred_packaged_task("
                    << hpx::actions::detail::get_action_name<Action>()
                    << ", "
                    << gid
                    << ") args(" << (18 + 1) << ")";
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17 , typename Arg18>
    void apply(naming::id_type const& gid,
        HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)) , HPX_FWD_ARGS(2, 10, ( Arg, arg)) , HPX_FWD_ARGS(2, 11, ( Arg, arg)) , HPX_FWD_ARGS(2, 12, ( Arg, arg)) , HPX_FWD_ARGS(2, 13, ( Arg, arg)) , HPX_FWD_ARGS(2, 14, ( Arg, arg)) , HPX_FWD_ARGS(2, 15, ( Arg, arg)) , HPX_FWD_ARGS(2, 16, ( Arg, arg)) , HPX_FWD_ARGS(2, 17, ( Arg, arg)) , HPX_FWD_ARGS(2, 18, ( Arg, arg)))
    {
        util::block_profiler_wrapper<deferred_packaged_task_direct_tag> bp(apply_logger_);
        naming::address addr;
        if (agas::is_local_address(gid, addr)) {
            
            (void)( (!!(components::types_are_compatible(addr.type_, components::get_component_type<typename Action::component_type>()))) || (_wassert(L"components::types_are_compatible(addr.type_, components::get_component_type<typename Action::component_type>())", L"D:/Devel\\hpx\\hpx\\lcos\\deferred_packaged_task_constructors_direct.hpp", 59), 0) );
            (*this->impl_)->set_data(Action::execute_function(addr.address_,
                util::forward_as_tuple(HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 8, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 9, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 10, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 11, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 12, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 13, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 14, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 15, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 16, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 17, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 18, ( Arg, arg)))));
        }
        else {
            
            hpx::applier::detail::apply_c<Action>(addr, this->get_gid(), gid,
                HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 8, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 9, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 10, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 11, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 12, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 13, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 14, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 15, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 16, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 17, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 18, ( Arg, arg)));
        }
    }
private:
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17 , typename Arg18>
    static void invoke19(
        hpx::lcos::deferred_packaged_task<Action,Result,boost::mpl::true_> *th,
        naming::id_type const& gid,
        HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)) , HPX_FWD_ARGS(2, 10, ( Arg, arg)) , HPX_FWD_ARGS(2, 11, ( Arg, arg)) , HPX_FWD_ARGS(2, 12, ( Arg, arg)) , HPX_FWD_ARGS(2, 13, ( Arg, arg)) , HPX_FWD_ARGS(2, 14, ( Arg, arg)) , HPX_FWD_ARGS(2, 15, ( Arg, arg)) , HPX_FWD_ARGS(2, 16, ( Arg, arg)) , HPX_FWD_ARGS(2, 17, ( Arg, arg)) , HPX_FWD_ARGS(2, 18, ( Arg, arg)))
    {
        if (!((*th->impl_)->is_ready()))
            th->apply(gid, HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 8, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 9, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 10, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 11, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 12, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 13, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 14, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 15, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 16, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 17, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 18, ( Arg, arg)));
    }
public:
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17 , typename Arg18>
    deferred_packaged_task(naming::gid_type const& gid,
            HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)) , HPX_FWD_ARGS(2, 10, ( Arg, arg)) , HPX_FWD_ARGS(2, 11, ( Arg, arg)) , HPX_FWD_ARGS(2, 12, ( Arg, arg)) , HPX_FWD_ARGS(2, 13, ( Arg, arg)) , HPX_FWD_ARGS(2, 14, ( Arg, arg)) , HPX_FWD_ARGS(2, 15, ( Arg, arg)) , HPX_FWD_ARGS(2, 16, ( Arg, arg)) , HPX_FWD_ARGS(2, 17, ( Arg, arg)) , HPX_FWD_ARGS(2, 18, ( Arg, arg)))
      : apply_logger_("deferred_packaged_task_direct::apply"),
        closure_(boost::bind(
            &deferred_packaged_task::template invoke19<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17 , Arg18>,
            this, naming::id_type(gid, naming::id_type::unmanaged),
            HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 8, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 9, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 10, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 11, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 12, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 13, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 14, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 15, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 16, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 17, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 18, ( Arg, arg))))
    {
        LLCO_(info) << "deferred_packaged_task::deferred_packaged_task("
                    << hpx::actions::detail::get_action_name<Action>()
                    << ", "
                    << gid
                    << ") args(" << (19 + 1) << ")";
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17 , typename Arg18>
    deferred_packaged_task(naming::id_type const& gid,
            HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)) , HPX_FWD_ARGS(2, 10, ( Arg, arg)) , HPX_FWD_ARGS(2, 11, ( Arg, arg)) , HPX_FWD_ARGS(2, 12, ( Arg, arg)) , HPX_FWD_ARGS(2, 13, ( Arg, arg)) , HPX_FWD_ARGS(2, 14, ( Arg, arg)) , HPX_FWD_ARGS(2, 15, ( Arg, arg)) , HPX_FWD_ARGS(2, 16, ( Arg, arg)) , HPX_FWD_ARGS(2, 17, ( Arg, arg)) , HPX_FWD_ARGS(2, 18, ( Arg, arg)))
      : apply_logger_("deferred_packaged_task_direct::apply"),
        closure_(boost::bind(
            &deferred_packaged_task::template invoke19<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17 , Arg18>,
            this, gid,
            HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 8, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 9, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 10, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 11, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 12, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 13, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 14, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 15, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 16, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 17, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 18, ( Arg, arg))))
    {
        LLCO_(info) << "deferred_packaged_task::deferred_packaged_task("
                    << hpx::actions::detail::get_action_name<Action>()
                    << ", "
                    << gid
                    << ") args(" << (19 + 1) << ")";
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17 , typename Arg18 , typename Arg19>
    void apply(naming::id_type const& gid,
        HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)) , HPX_FWD_ARGS(2, 10, ( Arg, arg)) , HPX_FWD_ARGS(2, 11, ( Arg, arg)) , HPX_FWD_ARGS(2, 12, ( Arg, arg)) , HPX_FWD_ARGS(2, 13, ( Arg, arg)) , HPX_FWD_ARGS(2, 14, ( Arg, arg)) , HPX_FWD_ARGS(2, 15, ( Arg, arg)) , HPX_FWD_ARGS(2, 16, ( Arg, arg)) , HPX_FWD_ARGS(2, 17, ( Arg, arg)) , HPX_FWD_ARGS(2, 18, ( Arg, arg)) , HPX_FWD_ARGS(2, 19, ( Arg, arg)))
    {
        util::block_profiler_wrapper<deferred_packaged_task_direct_tag> bp(apply_logger_);
        naming::address addr;
        if (agas::is_local_address(gid, addr)) {
            
            (void)( (!!(components::types_are_compatible(addr.type_, components::get_component_type<typename Action::component_type>()))) || (_wassert(L"components::types_are_compatible(addr.type_, components::get_component_type<typename Action::component_type>())", L"D:/Devel\\hpx\\hpx\\lcos\\deferred_packaged_task_constructors_direct.hpp", 59), 0) );
            (*this->impl_)->set_data(Action::execute_function(addr.address_,
                util::forward_as_tuple(HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 8, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 9, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 10, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 11, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 12, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 13, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 14, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 15, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 16, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 17, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 18, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 19, ( Arg, arg)))));
        }
        else {
            
            hpx::applier::detail::apply_c<Action>(addr, this->get_gid(), gid,
                HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 8, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 9, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 10, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 11, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 12, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 13, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 14, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 15, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 16, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 17, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 18, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 19, ( Arg, arg)));
        }
    }
private:
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17 , typename Arg18 , typename Arg19>
    static void invoke20(
        hpx::lcos::deferred_packaged_task<Action,Result,boost::mpl::true_> *th,
        naming::id_type const& gid,
        HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)) , HPX_FWD_ARGS(2, 10, ( Arg, arg)) , HPX_FWD_ARGS(2, 11, ( Arg, arg)) , HPX_FWD_ARGS(2, 12, ( Arg, arg)) , HPX_FWD_ARGS(2, 13, ( Arg, arg)) , HPX_FWD_ARGS(2, 14, ( Arg, arg)) , HPX_FWD_ARGS(2, 15, ( Arg, arg)) , HPX_FWD_ARGS(2, 16, ( Arg, arg)) , HPX_FWD_ARGS(2, 17, ( Arg, arg)) , HPX_FWD_ARGS(2, 18, ( Arg, arg)) , HPX_FWD_ARGS(2, 19, ( Arg, arg)))
    {
        if (!((*th->impl_)->is_ready()))
            th->apply(gid, HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 8, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 9, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 10, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 11, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 12, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 13, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 14, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 15, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 16, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 17, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 18, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 19, ( Arg, arg)));
    }
public:
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17 , typename Arg18 , typename Arg19>
    deferred_packaged_task(naming::gid_type const& gid,
            HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)) , HPX_FWD_ARGS(2, 10, ( Arg, arg)) , HPX_FWD_ARGS(2, 11, ( Arg, arg)) , HPX_FWD_ARGS(2, 12, ( Arg, arg)) , HPX_FWD_ARGS(2, 13, ( Arg, arg)) , HPX_FWD_ARGS(2, 14, ( Arg, arg)) , HPX_FWD_ARGS(2, 15, ( Arg, arg)) , HPX_FWD_ARGS(2, 16, ( Arg, arg)) , HPX_FWD_ARGS(2, 17, ( Arg, arg)) , HPX_FWD_ARGS(2, 18, ( Arg, arg)) , HPX_FWD_ARGS(2, 19, ( Arg, arg)))
      : apply_logger_("deferred_packaged_task_direct::apply"),
        closure_(boost::bind(
            &deferred_packaged_task::template invoke20<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17 , Arg18 , Arg19>,
            this, naming::id_type(gid, naming::id_type::unmanaged),
            HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 8, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 9, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 10, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 11, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 12, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 13, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 14, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 15, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 16, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 17, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 18, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 19, ( Arg, arg))))
    {
        LLCO_(info) << "deferred_packaged_task::deferred_packaged_task("
                    << hpx::actions::detail::get_action_name<Action>()
                    << ", "
                    << gid
                    << ") args(" << (20 + 1) << ")";
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17 , typename Arg18 , typename Arg19>
    deferred_packaged_task(naming::id_type const& gid,
            HPX_FWD_ARGS(2, 0, ( Arg, arg)) , HPX_FWD_ARGS(2, 1, ( Arg, arg)) , HPX_FWD_ARGS(2, 2, ( Arg, arg)) , HPX_FWD_ARGS(2, 3, ( Arg, arg)) , HPX_FWD_ARGS(2, 4, ( Arg, arg)) , HPX_FWD_ARGS(2, 5, ( Arg, arg)) , HPX_FWD_ARGS(2, 6, ( Arg, arg)) , HPX_FWD_ARGS(2, 7, ( Arg, arg)) , HPX_FWD_ARGS(2, 8, ( Arg, arg)) , HPX_FWD_ARGS(2, 9, ( Arg, arg)) , HPX_FWD_ARGS(2, 10, ( Arg, arg)) , HPX_FWD_ARGS(2, 11, ( Arg, arg)) , HPX_FWD_ARGS(2, 12, ( Arg, arg)) , HPX_FWD_ARGS(2, 13, ( Arg, arg)) , HPX_FWD_ARGS(2, 14, ( Arg, arg)) , HPX_FWD_ARGS(2, 15, ( Arg, arg)) , HPX_FWD_ARGS(2, 16, ( Arg, arg)) , HPX_FWD_ARGS(2, 17, ( Arg, arg)) , HPX_FWD_ARGS(2, 18, ( Arg, arg)) , HPX_FWD_ARGS(2, 19, ( Arg, arg)))
      : apply_logger_("deferred_packaged_task_direct::apply"),
        closure_(boost::bind(
            &deferred_packaged_task::template invoke20<Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17 , Arg18 , Arg19>,
            this, gid,
            HPX_FORWARD_ARGS(2, 0, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 1, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 2, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 3, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 4, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 5, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 6, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 7, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 8, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 9, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 10, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 11, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 12, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 13, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 14, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 15, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 16, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 17, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 18, ( Arg, arg)) , HPX_FORWARD_ARGS(2, 19, ( Arg, arg))))
    {
        LLCO_(info) << "deferred_packaged_task::deferred_packaged_task("
                    << hpx::actions::detail::get_action_name<Action>()
                    << ", "
                    << gid
                    << ") args(" << (20 + 1) << ")";
    }
