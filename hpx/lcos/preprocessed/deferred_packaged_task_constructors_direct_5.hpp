// Copyright (c) 2007-2012 Hartmut Kaiser
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


    template <typename Arg0 , typename Arg1>
    void apply(naming::id_type const& gid,
        BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1)
    {
        util::block_profiler_wrapper<deferred_packaged_task_direct_tag> bp(apply_logger_);
        naming::address addr;
        if (agas::is_local_address(gid, addr)) {
            
            (void)( (!!(components::types_are_compatible(addr.type_, components::get_component_type<typename Action::component_type>()))) || (_wassert(L"components::types_are_compatible(addr.type_, components::get_component_type<typename Action::component_type>())", L"D:/Devel\\hpx\\hpx\\lcos\\deferred_packaged_task_constructors_direct.hpp", 59), 0) );
            (*this->impl_)->set_data(Action::execute_function(addr.address_,
                util::forward_as_tuple(boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ))));
        }
        else {
            
            hpx::applier::detail::apply_c<Action>(addr, this->get_gid(), gid,
                boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ));
        }
    }
private:
    template <typename Arg0 , typename Arg1>
    static void invoke2(
        hpx::lcos::deferred_packaged_task<Action,Result,boost::mpl::true_> *th,
        naming::id_type const& gid,
        BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1)
    {
        if (!((*th->impl_)->is_ready()))
            th->apply(gid, boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ));
    }
public:
    template <typename Arg0 , typename Arg1>
    deferred_packaged_task(naming::gid_type const& gid,
            BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1)
      : apply_logger_("deferred_packaged_task_direct::apply"),
        closure_(boost::bind(
            &deferred_packaged_task::template invoke2<Arg0 , Arg1>,
            this, naming::id_type(gid, naming::id_type::unmanaged),
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 )))
    {
        LLCO_(info) << "deferred_packaged_task::deferred_packaged_task("
                    << hpx::actions::detail::get_action_name<Action>()
                    << ", "
                    << gid
                    << ") args(" << (2 + 1) << ")";
    }
    template <typename Arg0 , typename Arg1>
    deferred_packaged_task(naming::id_type const& gid,
            BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1)
      : apply_logger_("deferred_packaged_task_direct::apply"),
        closure_(boost::bind(
            &deferred_packaged_task::template invoke2<Arg0 , Arg1>,
            this, gid,
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 )))
    {
        LLCO_(info) << "deferred_packaged_task::deferred_packaged_task("
                    << hpx::actions::detail::get_action_name<Action>()
                    << ", "
                    << gid
                    << ") args(" << (2 + 1) << ")";
    }
    template <typename Arg0 , typename Arg1 , typename Arg2>
    void apply(naming::id_type const& gid,
        BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2)
    {
        util::block_profiler_wrapper<deferred_packaged_task_direct_tag> bp(apply_logger_);
        naming::address addr;
        if (agas::is_local_address(gid, addr)) {
            
            (void)( (!!(components::types_are_compatible(addr.type_, components::get_component_type<typename Action::component_type>()))) || (_wassert(L"components::types_are_compatible(addr.type_, components::get_component_type<typename Action::component_type>())", L"D:/Devel\\hpx\\hpx\\lcos\\deferred_packaged_task_constructors_direct.hpp", 59), 0) );
            (*this->impl_)->set_data(Action::execute_function(addr.address_,
                util::forward_as_tuple(boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ))));
        }
        else {
            
            hpx::applier::detail::apply_c<Action>(addr, this->get_gid(), gid,
                boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ));
        }
    }
private:
    template <typename Arg0 , typename Arg1 , typename Arg2>
    static void invoke3(
        hpx::lcos::deferred_packaged_task<Action,Result,boost::mpl::true_> *th,
        naming::id_type const& gid,
        BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2)
    {
        if (!((*th->impl_)->is_ready()))
            th->apply(gid, boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ));
    }
public:
    template <typename Arg0 , typename Arg1 , typename Arg2>
    deferred_packaged_task(naming::gid_type const& gid,
            BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2)
      : apply_logger_("deferred_packaged_task_direct::apply"),
        closure_(boost::bind(
            &deferred_packaged_task::template invoke3<Arg0 , Arg1 , Arg2>,
            this, naming::id_type(gid, naming::id_type::unmanaged),
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 )))
    {
        LLCO_(info) << "deferred_packaged_task::deferred_packaged_task("
                    << hpx::actions::detail::get_action_name<Action>()
                    << ", "
                    << gid
                    << ") args(" << (3 + 1) << ")";
    }
    template <typename Arg0 , typename Arg1 , typename Arg2>
    deferred_packaged_task(naming::id_type const& gid,
            BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2)
      : apply_logger_("deferred_packaged_task_direct::apply"),
        closure_(boost::bind(
            &deferred_packaged_task::template invoke3<Arg0 , Arg1 , Arg2>,
            this, gid,
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 )))
    {
        LLCO_(info) << "deferred_packaged_task::deferred_packaged_task("
                    << hpx::actions::detail::get_action_name<Action>()
                    << ", "
                    << gid
                    << ") args(" << (3 + 1) << ")";
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    void apply(naming::id_type const& gid,
        BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3)
    {
        util::block_profiler_wrapper<deferred_packaged_task_direct_tag> bp(apply_logger_);
        naming::address addr;
        if (agas::is_local_address(gid, addr)) {
            
            (void)( (!!(components::types_are_compatible(addr.type_, components::get_component_type<typename Action::component_type>()))) || (_wassert(L"components::types_are_compatible(addr.type_, components::get_component_type<typename Action::component_type>())", L"D:/Devel\\hpx\\hpx\\lcos\\deferred_packaged_task_constructors_direct.hpp", 59), 0) );
            (*this->impl_)->set_data(Action::execute_function(addr.address_,
                util::forward_as_tuple(boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ))));
        }
        else {
            
            hpx::applier::detail::apply_c<Action>(addr, this->get_gid(), gid,
                boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ));
        }
    }
private:
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    static void invoke4(
        hpx::lcos::deferred_packaged_task<Action,Result,boost::mpl::true_> *th,
        naming::id_type const& gid,
        BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3)
    {
        if (!((*th->impl_)->is_ready()))
            th->apply(gid, boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ));
    }
public:
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    deferred_packaged_task(naming::gid_type const& gid,
            BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3)
      : apply_logger_("deferred_packaged_task_direct::apply"),
        closure_(boost::bind(
            &deferred_packaged_task::template invoke4<Arg0 , Arg1 , Arg2 , Arg3>,
            this, naming::id_type(gid, naming::id_type::unmanaged),
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 )))
    {
        LLCO_(info) << "deferred_packaged_task::deferred_packaged_task("
                    << hpx::actions::detail::get_action_name<Action>()
                    << ", "
                    << gid
                    << ") args(" << (4 + 1) << ")";
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    deferred_packaged_task(naming::id_type const& gid,
            BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3)
      : apply_logger_("deferred_packaged_task_direct::apply"),
        closure_(boost::bind(
            &deferred_packaged_task::template invoke4<Arg0 , Arg1 , Arg2 , Arg3>,
            this, gid,
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 )))
    {
        LLCO_(info) << "deferred_packaged_task::deferred_packaged_task("
                    << hpx::actions::detail::get_action_name<Action>()
                    << ", "
                    << gid
                    << ") args(" << (4 + 1) << ")";
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    void apply(naming::id_type const& gid,
        BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4)
    {
        util::block_profiler_wrapper<deferred_packaged_task_direct_tag> bp(apply_logger_);
        naming::address addr;
        if (agas::is_local_address(gid, addr)) {
            
            (void)( (!!(components::types_are_compatible(addr.type_, components::get_component_type<typename Action::component_type>()))) || (_wassert(L"components::types_are_compatible(addr.type_, components::get_component_type<typename Action::component_type>())", L"D:/Devel\\hpx\\hpx\\lcos\\deferred_packaged_task_constructors_direct.hpp", 59), 0) );
            (*this->impl_)->set_data(Action::execute_function(addr.address_,
                util::forward_as_tuple(boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ))));
        }
        else {
            
            hpx::applier::detail::apply_c<Action>(addr, this->get_gid(), gid,
                boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ));
        }
    }
private:
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    static void invoke5(
        hpx::lcos::deferred_packaged_task<Action,Result,boost::mpl::true_> *th,
        naming::id_type const& gid,
        BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4)
    {
        if (!((*th->impl_)->is_ready()))
            th->apply(gid, boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ));
    }
public:
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    deferred_packaged_task(naming::gid_type const& gid,
            BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4)
      : apply_logger_("deferred_packaged_task_direct::apply"),
        closure_(boost::bind(
            &deferred_packaged_task::template invoke5<Arg0 , Arg1 , Arg2 , Arg3 , Arg4>,
            this, naming::id_type(gid, naming::id_type::unmanaged),
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 )))
    {
        LLCO_(info) << "deferred_packaged_task::deferred_packaged_task("
                    << hpx::actions::detail::get_action_name<Action>()
                    << ", "
                    << gid
                    << ") args(" << (5 + 1) << ")";
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    deferred_packaged_task(naming::id_type const& gid,
            BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4)
      : apply_logger_("deferred_packaged_task_direct::apply"),
        closure_(boost::bind(
            &deferred_packaged_task::template invoke5<Arg0 , Arg1 , Arg2 , Arg3 , Arg4>,
            this, gid,
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 )))
    {
        LLCO_(info) << "deferred_packaged_task::deferred_packaged_task("
                    << hpx::actions::detail::get_action_name<Action>()
                    << ", "
                    << gid
                    << ") args(" << (5 + 1) << ")";
    }
