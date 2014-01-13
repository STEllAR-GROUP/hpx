// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


    
    template <typename LocalResult>
    struct sync_invoke_1
    {
        template <typename Arg0>
        BOOST_FORCEINLINE static LocalResult call(
            boost::mpl::false_, BOOST_SCOPED_ENUM(launch) policy,
            naming::id_type const& id, Arg0 && arg0,
            error_code& ec)
        {
            return hpx::async<action>(policy, id,
                std::forward<Arg0>( arg0 )).get(ec);
        }
        template <typename Arg0>
        BOOST_FORCEINLINE static LocalResult call(
            boost::mpl::true_, BOOST_SCOPED_ENUM(launch) policy,
            naming::id_type const& id, Arg0 && arg0,
            error_code& ec)
        {
            return hpx::async<action>(policy, id,
                std::forward<Arg0>( arg0 ));
        }
    };
    template <typename IdType, typename Arg0>
    BOOST_FORCEINLINE typename boost::enable_if<
        boost::mpl::and_<
            boost::mpl::bool_<
                util::tuple_size<arguments_type>::value == 1>,
            boost::is_same<IdType, naming::id_type> >,
        local_result_type
    >::type
    operator()(BOOST_SCOPED_ENUM(launch) policy, IdType const& id,
        Arg0 && arg0, error_code& ec = throws) const
    {
        return util::void_guard<local_result_type>(),
            sync_invoke_1<local_result_type>::call(
                is_future_pred(), policy, id,
                std::forward<Arg0>( arg0 ), ec);
    }
    template <typename IdType, typename Arg0>
    BOOST_FORCEINLINE typename boost::enable_if<
        boost::mpl::and_<
            boost::mpl::bool_<
                util::tuple_size<arguments_type>::value == 1>,
            boost::is_same<IdType, naming::id_type> >,
        local_result_type
    >::type
    operator()(IdType const& id, Arg0 && arg0,
        error_code& ec = throws) const
    {
        return util::void_guard<local_result_type>(),
            sync_invoke_1<local_result_type>::call(
                is_future_pred(), launch::sync, id,
                std::forward<Arg0>( arg0 ), ec);
    }
    
    template <typename LocalResult>
    struct sync_invoke_2
    {
        template <typename Arg0 , typename Arg1>
        BOOST_FORCEINLINE static LocalResult call(
            boost::mpl::false_, BOOST_SCOPED_ENUM(launch) policy,
            naming::id_type const& id, Arg0 && arg0 , Arg1 && arg1,
            error_code& ec)
        {
            return hpx::async<action>(policy, id,
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 )).get(ec);
        }
        template <typename Arg0 , typename Arg1>
        BOOST_FORCEINLINE static LocalResult call(
            boost::mpl::true_, BOOST_SCOPED_ENUM(launch) policy,
            naming::id_type const& id, Arg0 && arg0 , Arg1 && arg1,
            error_code& ec)
        {
            return hpx::async<action>(policy, id,
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
        }
    };
    template <typename IdType, typename Arg0 , typename Arg1>
    BOOST_FORCEINLINE typename boost::enable_if<
        boost::mpl::and_<
            boost::mpl::bool_<
                util::tuple_size<arguments_type>::value == 2>,
            boost::is_same<IdType, naming::id_type> >,
        local_result_type
    >::type
    operator()(BOOST_SCOPED_ENUM(launch) policy, IdType const& id,
        Arg0 && arg0 , Arg1 && arg1, error_code& ec = throws) const
    {
        return util::void_guard<local_result_type>(),
            sync_invoke_2<local_result_type>::call(
                is_future_pred(), policy, id,
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ), ec);
    }
    template <typename IdType, typename Arg0 , typename Arg1>
    BOOST_FORCEINLINE typename boost::enable_if<
        boost::mpl::and_<
            boost::mpl::bool_<
                util::tuple_size<arguments_type>::value == 2>,
            boost::is_same<IdType, naming::id_type> >,
        local_result_type
    >::type
    operator()(IdType const& id, Arg0 && arg0 , Arg1 && arg1,
        error_code& ec = throws) const
    {
        return util::void_guard<local_result_type>(),
            sync_invoke_2<local_result_type>::call(
                is_future_pred(), launch::sync, id,
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ), ec);
    }
    
    template <typename LocalResult>
    struct sync_invoke_3
    {
        template <typename Arg0 , typename Arg1 , typename Arg2>
        BOOST_FORCEINLINE static LocalResult call(
            boost::mpl::false_, BOOST_SCOPED_ENUM(launch) policy,
            naming::id_type const& id, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2,
            error_code& ec)
        {
            return hpx::async<action>(policy, id,
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 )).get(ec);
        }
        template <typename Arg0 , typename Arg1 , typename Arg2>
        BOOST_FORCEINLINE static LocalResult call(
            boost::mpl::true_, BOOST_SCOPED_ENUM(launch) policy,
            naming::id_type const& id, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2,
            error_code& ec)
        {
            return hpx::async<action>(policy, id,
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
        }
    };
    template <typename IdType, typename Arg0 , typename Arg1 , typename Arg2>
    BOOST_FORCEINLINE typename boost::enable_if<
        boost::mpl::and_<
            boost::mpl::bool_<
                util::tuple_size<arguments_type>::value == 3>,
            boost::is_same<IdType, naming::id_type> >,
        local_result_type
    >::type
    operator()(BOOST_SCOPED_ENUM(launch) policy, IdType const& id,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2, error_code& ec = throws) const
    {
        return util::void_guard<local_result_type>(),
            sync_invoke_3<local_result_type>::call(
                is_future_pred(), policy, id,
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ), ec);
    }
    template <typename IdType, typename Arg0 , typename Arg1 , typename Arg2>
    BOOST_FORCEINLINE typename boost::enable_if<
        boost::mpl::and_<
            boost::mpl::bool_<
                util::tuple_size<arguments_type>::value == 3>,
            boost::is_same<IdType, naming::id_type> >,
        local_result_type
    >::type
    operator()(IdType const& id, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2,
        error_code& ec = throws) const
    {
        return util::void_guard<local_result_type>(),
            sync_invoke_3<local_result_type>::call(
                is_future_pred(), launch::sync, id,
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ), ec);
    }
    
    template <typename LocalResult>
    struct sync_invoke_4
    {
        template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
        BOOST_FORCEINLINE static LocalResult call(
            boost::mpl::false_, BOOST_SCOPED_ENUM(launch) policy,
            naming::id_type const& id, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3,
            error_code& ec)
        {
            return hpx::async<action>(policy, id,
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 )).get(ec);
        }
        template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
        BOOST_FORCEINLINE static LocalResult call(
            boost::mpl::true_, BOOST_SCOPED_ENUM(launch) policy,
            naming::id_type const& id, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3,
            error_code& ec)
        {
            return hpx::async<action>(policy, id,
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
        }
    };
    template <typename IdType, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    BOOST_FORCEINLINE typename boost::enable_if<
        boost::mpl::and_<
            boost::mpl::bool_<
                util::tuple_size<arguments_type>::value == 4>,
            boost::is_same<IdType, naming::id_type> >,
        local_result_type
    >::type
    operator()(BOOST_SCOPED_ENUM(launch) policy, IdType const& id,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3, error_code& ec = throws) const
    {
        return util::void_guard<local_result_type>(),
            sync_invoke_4<local_result_type>::call(
                is_future_pred(), policy, id,
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ), ec);
    }
    template <typename IdType, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    BOOST_FORCEINLINE typename boost::enable_if<
        boost::mpl::and_<
            boost::mpl::bool_<
                util::tuple_size<arguments_type>::value == 4>,
            boost::is_same<IdType, naming::id_type> >,
        local_result_type
    >::type
    operator()(IdType const& id, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3,
        error_code& ec = throws) const
    {
        return util::void_guard<local_result_type>(),
            sync_invoke_4<local_result_type>::call(
                is_future_pred(), launch::sync, id,
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ), ec);
    }
    
    template <typename LocalResult>
    struct sync_invoke_5
    {
        template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
        BOOST_FORCEINLINE static LocalResult call(
            boost::mpl::false_, BOOST_SCOPED_ENUM(launch) policy,
            naming::id_type const& id, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4,
            error_code& ec)
        {
            return hpx::async<action>(policy, id,
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 )).get(ec);
        }
        template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
        BOOST_FORCEINLINE static LocalResult call(
            boost::mpl::true_, BOOST_SCOPED_ENUM(launch) policy,
            naming::id_type const& id, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4,
            error_code& ec)
        {
            return hpx::async<action>(policy, id,
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
        }
    };
    template <typename IdType, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    BOOST_FORCEINLINE typename boost::enable_if<
        boost::mpl::and_<
            boost::mpl::bool_<
                util::tuple_size<arguments_type>::value == 5>,
            boost::is_same<IdType, naming::id_type> >,
        local_result_type
    >::type
    operator()(BOOST_SCOPED_ENUM(launch) policy, IdType const& id,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4, error_code& ec = throws) const
    {
        return util::void_guard<local_result_type>(),
            sync_invoke_5<local_result_type>::call(
                is_future_pred(), policy, id,
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ), ec);
    }
    template <typename IdType, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    BOOST_FORCEINLINE typename boost::enable_if<
        boost::mpl::and_<
            boost::mpl::bool_<
                util::tuple_size<arguments_type>::value == 5>,
            boost::is_same<IdType, naming::id_type> >,
        local_result_type
    >::type
    operator()(IdType const& id, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4,
        error_code& ec = throws) const
    {
        return util::void_guard<local_result_type>(),
            sync_invoke_5<local_result_type>::call(
                is_future_pred(), launch::sync, id,
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ), ec);
    }
