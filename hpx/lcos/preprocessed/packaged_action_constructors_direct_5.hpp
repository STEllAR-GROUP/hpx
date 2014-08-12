// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


    template <typename Arg0>
    void apply(BOOST_SCOPED_ENUM(launch) , naming::id_type const& gid,
        Arg0 && arg0)
    {
        util::block_profiler_wrapper<profiler_tag> bp(apply_logger_);
        naming::address addr;
        if (agas::is_local_address_cached(gid, addr)) {
            
            HPX_ASSERT(traits::component_type_is_compatible<
                typename Action::component_type>::call(addr));
            (*this->impl_)->set_data(
                std::move(action_type::execute_function(addr.address_,
                    util::forward_as_tuple(std::forward<Arg0>( arg0 ))))
            );
        }
        else {
            
            hpx::applier::detail::apply_c_cb<action_type>(
                std::move(addr), this->get_gid(), gid,
                util::bind(&packaged_action::parcel_write_handler,
                    this->impl_, util::placeholders::_1),
                std::forward<Arg0>( arg0 ));
        }
    }
    template <typename Arg0>
    void apply(BOOST_SCOPED_ENUM(launch) , naming::address&& addr,
        naming::id_type const& gid, Arg0 && arg0)
    {
        util::block_profiler_wrapper<profiler_tag> bp(apply_logger_);
        if (addr.locality_ == hpx::get_locality()) {
            
            HPX_ASSERT(traits::component_type_is_compatible<
                typename Action::component_type>::call(addr));
            (*this->impl_)->set_data(
                std::move(action_type::execute_function(addr.address_,
                    util::forward_as_tuple(std::forward<Arg0>( arg0 ))))
            );
        }
        else {
            
            hpx::applier::detail::apply_c_cb<action_type>(
                std::move(addr), this->get_gid(), gid,
                util::bind(&packaged_action::parcel_write_handler,
                    this->impl_, util::placeholders::_1),
                std::forward<Arg0>( arg0 ));
        }
    }
    template <typename Arg0>
    packaged_action(naming::id_type const& gid,
            Arg0 && arg0)
      : apply_logger_("packaged_action_direct::apply")
    {
        LLCO_(info) << "packaged_action::packaged_action("
                    << hpx::actions::detail::get_action_name<action_type>()
                    << ", "
                    << gid
                    << ") args(" << (1 + 1) << ")";
        apply(launch::all, gid, std::forward<Arg0>( arg0 ));
    }
    template <typename Arg0 , typename Arg1>
    void apply(BOOST_SCOPED_ENUM(launch) , naming::id_type const& gid,
        Arg0 && arg0 , Arg1 && arg1)
    {
        util::block_profiler_wrapper<profiler_tag> bp(apply_logger_);
        naming::address addr;
        if (agas::is_local_address_cached(gid, addr)) {
            
            HPX_ASSERT(traits::component_type_is_compatible<
                typename Action::component_type>::call(addr));
            (*this->impl_)->set_data(
                std::move(action_type::execute_function(addr.address_,
                    util::forward_as_tuple(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ))))
            );
        }
        else {
            
            hpx::applier::detail::apply_c_cb<action_type>(
                std::move(addr), this->get_gid(), gid,
                util::bind(&packaged_action::parcel_write_handler,
                    this->impl_, util::placeholders::_1),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
        }
    }
    template <typename Arg0 , typename Arg1>
    void apply(BOOST_SCOPED_ENUM(launch) , naming::address&& addr,
        naming::id_type const& gid, Arg0 && arg0 , Arg1 && arg1)
    {
        util::block_profiler_wrapper<profiler_tag> bp(apply_logger_);
        if (addr.locality_ == hpx::get_locality()) {
            
            HPX_ASSERT(traits::component_type_is_compatible<
                typename Action::component_type>::call(addr));
            (*this->impl_)->set_data(
                std::move(action_type::execute_function(addr.address_,
                    util::forward_as_tuple(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ))))
            );
        }
        else {
            
            hpx::applier::detail::apply_c_cb<action_type>(
                std::move(addr), this->get_gid(), gid,
                util::bind(&packaged_action::parcel_write_handler,
                    this->impl_, util::placeholders::_1),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
        }
    }
    template <typename Arg0 , typename Arg1>
    packaged_action(naming::id_type const& gid,
            Arg0 && arg0 , Arg1 && arg1)
      : apply_logger_("packaged_action_direct::apply")
    {
        LLCO_(info) << "packaged_action::packaged_action("
                    << hpx::actions::detail::get_action_name<action_type>()
                    << ", "
                    << gid
                    << ") args(" << (2 + 1) << ")";
        apply(launch::all, gid, std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
    }
    template <typename Arg0 , typename Arg1 , typename Arg2>
    void apply(BOOST_SCOPED_ENUM(launch) , naming::id_type const& gid,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
    {
        util::block_profiler_wrapper<profiler_tag> bp(apply_logger_);
        naming::address addr;
        if (agas::is_local_address_cached(gid, addr)) {
            
            HPX_ASSERT(traits::component_type_is_compatible<
                typename Action::component_type>::call(addr));
            (*this->impl_)->set_data(
                std::move(action_type::execute_function(addr.address_,
                    util::forward_as_tuple(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ))))
            );
        }
        else {
            
            hpx::applier::detail::apply_c_cb<action_type>(
                std::move(addr), this->get_gid(), gid,
                util::bind(&packaged_action::parcel_write_handler,
                    this->impl_, util::placeholders::_1),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
        }
    }
    template <typename Arg0 , typename Arg1 , typename Arg2>
    void apply(BOOST_SCOPED_ENUM(launch) , naming::address&& addr,
        naming::id_type const& gid, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
    {
        util::block_profiler_wrapper<profiler_tag> bp(apply_logger_);
        if (addr.locality_ == hpx::get_locality()) {
            
            HPX_ASSERT(traits::component_type_is_compatible<
                typename Action::component_type>::call(addr));
            (*this->impl_)->set_data(
                std::move(action_type::execute_function(addr.address_,
                    util::forward_as_tuple(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ))))
            );
        }
        else {
            
            hpx::applier::detail::apply_c_cb<action_type>(
                std::move(addr), this->get_gid(), gid,
                util::bind(&packaged_action::parcel_write_handler,
                    this->impl_, util::placeholders::_1),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
        }
    }
    template <typename Arg0 , typename Arg1 , typename Arg2>
    packaged_action(naming::id_type const& gid,
            Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
      : apply_logger_("packaged_action_direct::apply")
    {
        LLCO_(info) << "packaged_action::packaged_action("
                    << hpx::actions::detail::get_action_name<action_type>()
                    << ", "
                    << gid
                    << ") args(" << (3 + 1) << ")";
        apply(launch::all, gid, std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    void apply(BOOST_SCOPED_ENUM(launch) , naming::id_type const& gid,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
    {
        util::block_profiler_wrapper<profiler_tag> bp(apply_logger_);
        naming::address addr;
        if (agas::is_local_address_cached(gid, addr)) {
            
            HPX_ASSERT(traits::component_type_is_compatible<
                typename Action::component_type>::call(addr));
            (*this->impl_)->set_data(
                std::move(action_type::execute_function(addr.address_,
                    util::forward_as_tuple(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ))))
            );
        }
        else {
            
            hpx::applier::detail::apply_c_cb<action_type>(
                std::move(addr), this->get_gid(), gid,
                util::bind(&packaged_action::parcel_write_handler,
                    this->impl_, util::placeholders::_1),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
        }
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    void apply(BOOST_SCOPED_ENUM(launch) , naming::address&& addr,
        naming::id_type const& gid, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
    {
        util::block_profiler_wrapper<profiler_tag> bp(apply_logger_);
        if (addr.locality_ == hpx::get_locality()) {
            
            HPX_ASSERT(traits::component_type_is_compatible<
                typename Action::component_type>::call(addr));
            (*this->impl_)->set_data(
                std::move(action_type::execute_function(addr.address_,
                    util::forward_as_tuple(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ))))
            );
        }
        else {
            
            hpx::applier::detail::apply_c_cb<action_type>(
                std::move(addr), this->get_gid(), gid,
                util::bind(&packaged_action::parcel_write_handler,
                    this->impl_, util::placeholders::_1),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
        }
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    packaged_action(naming::id_type const& gid,
            Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
      : apply_logger_("packaged_action_direct::apply")
    {
        LLCO_(info) << "packaged_action::packaged_action("
                    << hpx::actions::detail::get_action_name<action_type>()
                    << ", "
                    << gid
                    << ") args(" << (4 + 1) << ")";
        apply(launch::all, gid, std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    void apply(BOOST_SCOPED_ENUM(launch) , naming::id_type const& gid,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
    {
        util::block_profiler_wrapper<profiler_tag> bp(apply_logger_);
        naming::address addr;
        if (agas::is_local_address_cached(gid, addr)) {
            
            HPX_ASSERT(traits::component_type_is_compatible<
                typename Action::component_type>::call(addr));
            (*this->impl_)->set_data(
                std::move(action_type::execute_function(addr.address_,
                    util::forward_as_tuple(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ))))
            );
        }
        else {
            
            hpx::applier::detail::apply_c_cb<action_type>(
                std::move(addr), this->get_gid(), gid,
                util::bind(&packaged_action::parcel_write_handler,
                    this->impl_, util::placeholders::_1),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
        }
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    void apply(BOOST_SCOPED_ENUM(launch) , naming::address&& addr,
        naming::id_type const& gid, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
    {
        util::block_profiler_wrapper<profiler_tag> bp(apply_logger_);
        if (addr.locality_ == hpx::get_locality()) {
            
            HPX_ASSERT(traits::component_type_is_compatible<
                typename Action::component_type>::call(addr));
            (*this->impl_)->set_data(
                std::move(action_type::execute_function(addr.address_,
                    util::forward_as_tuple(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ))))
            );
        }
        else {
            
            hpx::applier::detail::apply_c_cb<action_type>(
                std::move(addr), this->get_gid(), gid,
                util::bind(&packaged_action::parcel_write_handler,
                    this->impl_, util::placeholders::_1),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
        }
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    packaged_action(naming::id_type const& gid,
            Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
      : apply_logger_("packaged_action_direct::apply")
    {
        LLCO_(info) << "packaged_action::packaged_action("
                    << hpx::actions::detail::get_action_name<action_type>()
                    << ", "
                    << gid
                    << ") args(" << (5 + 1) << ")";
        apply(launch::all, gid, std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
    }
