// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


namespace hpx { namespace lcos { namespace local
{
    
    template <typename Func, typename F0>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_future_or_future_range<
            typename util::decay<Func>::type
        >
      , detail::dataflow_frame<
            BOOST_SCOPED_ENUM(launch)
          , typename hpx::util::decay<Func>::type
          , hpx::util::tuple<
                typename util::decay<F0>::type
            >
        >
    >::type
    dataflow(
        BOOST_SCOPED_ENUM(launch) policy
      , Func && func
      , F0 && f0
    )
    {
        typedef
            detail::dataflow_frame<
                BOOST_SCOPED_ENUM(launch)
              , typename hpx::util::decay<Func>::type
              , hpx::util::tuple<
                    typename util::decay<F0>::type
                >
            >
            frame_type;
        boost::intrusive_ptr<frame_type> p(new frame_type(
                policy
              , std::forward<Func>(func)
              , hpx::util::forward_as_tuple(std::forward<F0>( f0 ))
            ));
        p->await();
        using traits::future_access;
        return future_access<typename frame_type::type>::create(std::move(p));
    }
    template <typename Func, typename F0>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_future_or_future_range<
            typename util::decay<Func>::type
        >
      , detail::dataflow_frame<
            threads::executor
          , typename hpx::util::decay<Func>::type
          , hpx::util::tuple<
                typename util::decay<F0>::type
            >
        >
    >::type
    dataflow(
        threads::executor& sched
      , Func && func
      , F0 && f0
    )
    {
        typedef
            detail::dataflow_frame<
                threads::executor
              , typename hpx::util::decay<Func>::type
              , hpx::util::tuple<
                    typename util::decay<F0>::type
                >
            >
            frame_type;
        boost::intrusive_ptr<frame_type> p(new frame_type(
                sched
              , std::forward<Func>(func)
              , hpx::util::forward_as_tuple(std::forward<F0>( f0 ))
            ));
        p->await();
        using traits::future_access;
        return future_access<typename frame_type::type>::create(std::move(p));
    }
    template <typename Func, typename F0>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        traits::is_launch_policy<
            typename util::decay<Func>::type
        >
      , detail::dataflow_frame<
            BOOST_SCOPED_ENUM(launch)
          , typename hpx::util::decay<Func>::type
          , hpx::util::tuple<
                typename util::decay<F0>::type
            >
        >
    >::type
    dataflow(Func && func, F0 && f0)
    {
        typedef
            detail::dataflow_frame<
                BOOST_SCOPED_ENUM(launch)
              , typename hpx::util::decay<Func>::type
              , hpx::util::tuple<
                    typename util::decay<F0>::type
                >
            >
            frame_type;
        boost::intrusive_ptr<frame_type> p(new frame_type(
                launch::all
              , std::forward<Func>(func)
              , hpx::util::forward_as_tuple(std::forward<F0>( f0 ))
            ));
        p->await();
        using traits::future_access;
        return future_access<typename frame_type::type>::create(std::move(p));
    }
}}}
namespace hpx { namespace lcos { namespace local
{
    
    template <typename Func, typename F0 , typename F1>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_future_or_future_range<
            typename util::decay<Func>::type
        >
      , detail::dataflow_frame<
            BOOST_SCOPED_ENUM(launch)
          , typename hpx::util::decay<Func>::type
          , hpx::util::tuple<
                typename util::decay<F0>::type , typename util::decay<F1>::type
            >
        >
    >::type
    dataflow(
        BOOST_SCOPED_ENUM(launch) policy
      , Func && func
      , F0 && f0 , F1 && f1
    )
    {
        typedef
            detail::dataflow_frame<
                BOOST_SCOPED_ENUM(launch)
              , typename hpx::util::decay<Func>::type
              , hpx::util::tuple<
                    typename util::decay<F0>::type , typename util::decay<F1>::type
                >
            >
            frame_type;
        boost::intrusive_ptr<frame_type> p(new frame_type(
                policy
              , std::forward<Func>(func)
              , hpx::util::forward_as_tuple(std::forward<F0>( f0 ) , std::forward<F1>( f1 ))
            ));
        p->await();
        using traits::future_access;
        return future_access<typename frame_type::type>::create(std::move(p));
    }
    template <typename Func, typename F0 , typename F1>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_future_or_future_range<
            typename util::decay<Func>::type
        >
      , detail::dataflow_frame<
            threads::executor
          , typename hpx::util::decay<Func>::type
          , hpx::util::tuple<
                typename util::decay<F0>::type , typename util::decay<F1>::type
            >
        >
    >::type
    dataflow(
        threads::executor& sched
      , Func && func
      , F0 && f0 , F1 && f1
    )
    {
        typedef
            detail::dataflow_frame<
                threads::executor
              , typename hpx::util::decay<Func>::type
              , hpx::util::tuple<
                    typename util::decay<F0>::type , typename util::decay<F1>::type
                >
            >
            frame_type;
        boost::intrusive_ptr<frame_type> p(new frame_type(
                sched
              , std::forward<Func>(func)
              , hpx::util::forward_as_tuple(std::forward<F0>( f0 ) , std::forward<F1>( f1 ))
            ));
        p->await();
        using traits::future_access;
        return future_access<typename frame_type::type>::create(std::move(p));
    }
    template <typename Func, typename F0 , typename F1>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        traits::is_launch_policy<
            typename util::decay<Func>::type
        >
      , detail::dataflow_frame<
            BOOST_SCOPED_ENUM(launch)
          , typename hpx::util::decay<Func>::type
          , hpx::util::tuple<
                typename util::decay<F0>::type , typename util::decay<F1>::type
            >
        >
    >::type
    dataflow(Func && func, F0 && f0 , F1 && f1)
    {
        typedef
            detail::dataflow_frame<
                BOOST_SCOPED_ENUM(launch)
              , typename hpx::util::decay<Func>::type
              , hpx::util::tuple<
                    typename util::decay<F0>::type , typename util::decay<F1>::type
                >
            >
            frame_type;
        boost::intrusive_ptr<frame_type> p(new frame_type(
                launch::all
              , std::forward<Func>(func)
              , hpx::util::forward_as_tuple(std::forward<F0>( f0 ) , std::forward<F1>( f1 ))
            ));
        p->await();
        using traits::future_access;
        return future_access<typename frame_type::type>::create(std::move(p));
    }
}}}
namespace hpx { namespace lcos { namespace local
{
    
    template <typename Func, typename F0 , typename F1 , typename F2>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_future_or_future_range<
            typename util::decay<Func>::type
        >
      , detail::dataflow_frame<
            BOOST_SCOPED_ENUM(launch)
          , typename hpx::util::decay<Func>::type
          , hpx::util::tuple<
                typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type
            >
        >
    >::type
    dataflow(
        BOOST_SCOPED_ENUM(launch) policy
      , Func && func
      , F0 && f0 , F1 && f1 , F2 && f2
    )
    {
        typedef
            detail::dataflow_frame<
                BOOST_SCOPED_ENUM(launch)
              , typename hpx::util::decay<Func>::type
              , hpx::util::tuple<
                    typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type
                >
            >
            frame_type;
        boost::intrusive_ptr<frame_type> p(new frame_type(
                policy
              , std::forward<Func>(func)
              , hpx::util::forward_as_tuple(std::forward<F0>( f0 ) , std::forward<F1>( f1 ) , std::forward<F2>( f2 ))
            ));
        p->await();
        using traits::future_access;
        return future_access<typename frame_type::type>::create(std::move(p));
    }
    template <typename Func, typename F0 , typename F1 , typename F2>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_future_or_future_range<
            typename util::decay<Func>::type
        >
      , detail::dataflow_frame<
            threads::executor
          , typename hpx::util::decay<Func>::type
          , hpx::util::tuple<
                typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type
            >
        >
    >::type
    dataflow(
        threads::executor& sched
      , Func && func
      , F0 && f0 , F1 && f1 , F2 && f2
    )
    {
        typedef
            detail::dataflow_frame<
                threads::executor
              , typename hpx::util::decay<Func>::type
              , hpx::util::tuple<
                    typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type
                >
            >
            frame_type;
        boost::intrusive_ptr<frame_type> p(new frame_type(
                sched
              , std::forward<Func>(func)
              , hpx::util::forward_as_tuple(std::forward<F0>( f0 ) , std::forward<F1>( f1 ) , std::forward<F2>( f2 ))
            ));
        p->await();
        using traits::future_access;
        return future_access<typename frame_type::type>::create(std::move(p));
    }
    template <typename Func, typename F0 , typename F1 , typename F2>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        traits::is_launch_policy<
            typename util::decay<Func>::type
        >
      , detail::dataflow_frame<
            BOOST_SCOPED_ENUM(launch)
          , typename hpx::util::decay<Func>::type
          , hpx::util::tuple<
                typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type
            >
        >
    >::type
    dataflow(Func && func, F0 && f0 , F1 && f1 , F2 && f2)
    {
        typedef
            detail::dataflow_frame<
                BOOST_SCOPED_ENUM(launch)
              , typename hpx::util::decay<Func>::type
              , hpx::util::tuple<
                    typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type
                >
            >
            frame_type;
        boost::intrusive_ptr<frame_type> p(new frame_type(
                launch::all
              , std::forward<Func>(func)
              , hpx::util::forward_as_tuple(std::forward<F0>( f0 ) , std::forward<F1>( f1 ) , std::forward<F2>( f2 ))
            ));
        p->await();
        using traits::future_access;
        return future_access<typename frame_type::type>::create(std::move(p));
    }
}}}
namespace hpx { namespace lcos { namespace local
{
    
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_future_or_future_range<
            typename util::decay<Func>::type
        >
      , detail::dataflow_frame<
            BOOST_SCOPED_ENUM(launch)
          , typename hpx::util::decay<Func>::type
          , hpx::util::tuple<
                typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type
            >
        >
    >::type
    dataflow(
        BOOST_SCOPED_ENUM(launch) policy
      , Func && func
      , F0 && f0 , F1 && f1 , F2 && f2 , F3 && f3
    )
    {
        typedef
            detail::dataflow_frame<
                BOOST_SCOPED_ENUM(launch)
              , typename hpx::util::decay<Func>::type
              , hpx::util::tuple<
                    typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type
                >
            >
            frame_type;
        boost::intrusive_ptr<frame_type> p(new frame_type(
                policy
              , std::forward<Func>(func)
              , hpx::util::forward_as_tuple(std::forward<F0>( f0 ) , std::forward<F1>( f1 ) , std::forward<F2>( f2 ) , std::forward<F3>( f3 ))
            ));
        p->await();
        using traits::future_access;
        return future_access<typename frame_type::type>::create(std::move(p));
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_future_or_future_range<
            typename util::decay<Func>::type
        >
      , detail::dataflow_frame<
            threads::executor
          , typename hpx::util::decay<Func>::type
          , hpx::util::tuple<
                typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type
            >
        >
    >::type
    dataflow(
        threads::executor& sched
      , Func && func
      , F0 && f0 , F1 && f1 , F2 && f2 , F3 && f3
    )
    {
        typedef
            detail::dataflow_frame<
                threads::executor
              , typename hpx::util::decay<Func>::type
              , hpx::util::tuple<
                    typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type
                >
            >
            frame_type;
        boost::intrusive_ptr<frame_type> p(new frame_type(
                sched
              , std::forward<Func>(func)
              , hpx::util::forward_as_tuple(std::forward<F0>( f0 ) , std::forward<F1>( f1 ) , std::forward<F2>( f2 ) , std::forward<F3>( f3 ))
            ));
        p->await();
        using traits::future_access;
        return future_access<typename frame_type::type>::create(std::move(p));
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        traits::is_launch_policy<
            typename util::decay<Func>::type
        >
      , detail::dataflow_frame<
            BOOST_SCOPED_ENUM(launch)
          , typename hpx::util::decay<Func>::type
          , hpx::util::tuple<
                typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type
            >
        >
    >::type
    dataflow(Func && func, F0 && f0 , F1 && f1 , F2 && f2 , F3 && f3)
    {
        typedef
            detail::dataflow_frame<
                BOOST_SCOPED_ENUM(launch)
              , typename hpx::util::decay<Func>::type
              , hpx::util::tuple<
                    typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type
                >
            >
            frame_type;
        boost::intrusive_ptr<frame_type> p(new frame_type(
                launch::all
              , std::forward<Func>(func)
              , hpx::util::forward_as_tuple(std::forward<F0>( f0 ) , std::forward<F1>( f1 ) , std::forward<F2>( f2 ) , std::forward<F3>( f3 ))
            ));
        p->await();
        using traits::future_access;
        return future_access<typename frame_type::type>::create(std::move(p));
    }
}}}
namespace hpx { namespace lcos { namespace local
{
    
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_future_or_future_range<
            typename util::decay<Func>::type
        >
      , detail::dataflow_frame<
            BOOST_SCOPED_ENUM(launch)
          , typename hpx::util::decay<Func>::type
          , hpx::util::tuple<
                typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type
            >
        >
    >::type
    dataflow(
        BOOST_SCOPED_ENUM(launch) policy
      , Func && func
      , F0 && f0 , F1 && f1 , F2 && f2 , F3 && f3 , F4 && f4
    )
    {
        typedef
            detail::dataflow_frame<
                BOOST_SCOPED_ENUM(launch)
              , typename hpx::util::decay<Func>::type
              , hpx::util::tuple<
                    typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type
                >
            >
            frame_type;
        boost::intrusive_ptr<frame_type> p(new frame_type(
                policy
              , std::forward<Func>(func)
              , hpx::util::forward_as_tuple(std::forward<F0>( f0 ) , std::forward<F1>( f1 ) , std::forward<F2>( f2 ) , std::forward<F3>( f3 ) , std::forward<F4>( f4 ))
            ));
        p->await();
        using traits::future_access;
        return future_access<typename frame_type::type>::create(std::move(p));
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_future_or_future_range<
            typename util::decay<Func>::type
        >
      , detail::dataflow_frame<
            threads::executor
          , typename hpx::util::decay<Func>::type
          , hpx::util::tuple<
                typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type
            >
        >
    >::type
    dataflow(
        threads::executor& sched
      , Func && func
      , F0 && f0 , F1 && f1 , F2 && f2 , F3 && f3 , F4 && f4
    )
    {
        typedef
            detail::dataflow_frame<
                threads::executor
              , typename hpx::util::decay<Func>::type
              , hpx::util::tuple<
                    typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type
                >
            >
            frame_type;
        boost::intrusive_ptr<frame_type> p(new frame_type(
                sched
              , std::forward<Func>(func)
              , hpx::util::forward_as_tuple(std::forward<F0>( f0 ) , std::forward<F1>( f1 ) , std::forward<F2>( f2 ) , std::forward<F3>( f3 ) , std::forward<F4>( f4 ))
            ));
        p->await();
        using traits::future_access;
        return future_access<typename frame_type::type>::create(std::move(p));
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        traits::is_launch_policy<
            typename util::decay<Func>::type
        >
      , detail::dataflow_frame<
            BOOST_SCOPED_ENUM(launch)
          , typename hpx::util::decay<Func>::type
          , hpx::util::tuple<
                typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type
            >
        >
    >::type
    dataflow(Func && func, F0 && f0 , F1 && f1 , F2 && f2 , F3 && f3 , F4 && f4)
    {
        typedef
            detail::dataflow_frame<
                BOOST_SCOPED_ENUM(launch)
              , typename hpx::util::decay<Func>::type
              , hpx::util::tuple<
                    typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type
                >
            >
            frame_type;
        boost::intrusive_ptr<frame_type> p(new frame_type(
                launch::all
              , std::forward<Func>(func)
              , hpx::util::forward_as_tuple(std::forward<F0>( f0 ) , std::forward<F1>( f1 ) , std::forward<F2>( f2 ) , std::forward<F3>( f3 ) , std::forward<F4>( f4 ))
            ));
        p->await();
        using traits::future_access;
        return future_access<typename frame_type::type>::create(std::move(p));
    }
}}}
namespace hpx { namespace lcos { namespace local
{
    
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_future_or_future_range<
            typename util::decay<Func>::type
        >
      , detail::dataflow_frame<
            BOOST_SCOPED_ENUM(launch)
          , typename hpx::util::decay<Func>::type
          , hpx::util::tuple<
                typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type , typename util::decay<F5>::type
            >
        >
    >::type
    dataflow(
        BOOST_SCOPED_ENUM(launch) policy
      , Func && func
      , F0 && f0 , F1 && f1 , F2 && f2 , F3 && f3 , F4 && f4 , F5 && f5
    )
    {
        typedef
            detail::dataflow_frame<
                BOOST_SCOPED_ENUM(launch)
              , typename hpx::util::decay<Func>::type
              , hpx::util::tuple<
                    typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type , typename util::decay<F5>::type
                >
            >
            frame_type;
        boost::intrusive_ptr<frame_type> p(new frame_type(
                policy
              , std::forward<Func>(func)
              , hpx::util::forward_as_tuple(std::forward<F0>( f0 ) , std::forward<F1>( f1 ) , std::forward<F2>( f2 ) , std::forward<F3>( f3 ) , std::forward<F4>( f4 ) , std::forward<F5>( f5 ))
            ));
        p->await();
        using traits::future_access;
        return future_access<typename frame_type::type>::create(std::move(p));
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_future_or_future_range<
            typename util::decay<Func>::type
        >
      , detail::dataflow_frame<
            threads::executor
          , typename hpx::util::decay<Func>::type
          , hpx::util::tuple<
                typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type , typename util::decay<F5>::type
            >
        >
    >::type
    dataflow(
        threads::executor& sched
      , Func && func
      , F0 && f0 , F1 && f1 , F2 && f2 , F3 && f3 , F4 && f4 , F5 && f5
    )
    {
        typedef
            detail::dataflow_frame<
                threads::executor
              , typename hpx::util::decay<Func>::type
              , hpx::util::tuple<
                    typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type , typename util::decay<F5>::type
                >
            >
            frame_type;
        boost::intrusive_ptr<frame_type> p(new frame_type(
                sched
              , std::forward<Func>(func)
              , hpx::util::forward_as_tuple(std::forward<F0>( f0 ) , std::forward<F1>( f1 ) , std::forward<F2>( f2 ) , std::forward<F3>( f3 ) , std::forward<F4>( f4 ) , std::forward<F5>( f5 ))
            ));
        p->await();
        using traits::future_access;
        return future_access<typename frame_type::type>::create(std::move(p));
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        traits::is_launch_policy<
            typename util::decay<Func>::type
        >
      , detail::dataflow_frame<
            BOOST_SCOPED_ENUM(launch)
          , typename hpx::util::decay<Func>::type
          , hpx::util::tuple<
                typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type , typename util::decay<F5>::type
            >
        >
    >::type
    dataflow(Func && func, F0 && f0 , F1 && f1 , F2 && f2 , F3 && f3 , F4 && f4 , F5 && f5)
    {
        typedef
            detail::dataflow_frame<
                BOOST_SCOPED_ENUM(launch)
              , typename hpx::util::decay<Func>::type
              , hpx::util::tuple<
                    typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type , typename util::decay<F5>::type
                >
            >
            frame_type;
        boost::intrusive_ptr<frame_type> p(new frame_type(
                launch::all
              , std::forward<Func>(func)
              , hpx::util::forward_as_tuple(std::forward<F0>( f0 ) , std::forward<F1>( f1 ) , std::forward<F2>( f2 ) , std::forward<F3>( f3 ) , std::forward<F4>( f4 ) , std::forward<F5>( f5 ))
            ));
        p->await();
        using traits::future_access;
        return future_access<typename frame_type::type>::create(std::move(p));
    }
}}}
namespace hpx { namespace lcos { namespace local
{
    
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_future_or_future_range<
            typename util::decay<Func>::type
        >
      , detail::dataflow_frame<
            BOOST_SCOPED_ENUM(launch)
          , typename hpx::util::decay<Func>::type
          , hpx::util::tuple<
                typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type , typename util::decay<F5>::type , typename util::decay<F6>::type
            >
        >
    >::type
    dataflow(
        BOOST_SCOPED_ENUM(launch) policy
      , Func && func
      , F0 && f0 , F1 && f1 , F2 && f2 , F3 && f3 , F4 && f4 , F5 && f5 , F6 && f6
    )
    {
        typedef
            detail::dataflow_frame<
                BOOST_SCOPED_ENUM(launch)
              , typename hpx::util::decay<Func>::type
              , hpx::util::tuple<
                    typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type , typename util::decay<F5>::type , typename util::decay<F6>::type
                >
            >
            frame_type;
        boost::intrusive_ptr<frame_type> p(new frame_type(
                policy
              , std::forward<Func>(func)
              , hpx::util::forward_as_tuple(std::forward<F0>( f0 ) , std::forward<F1>( f1 ) , std::forward<F2>( f2 ) , std::forward<F3>( f3 ) , std::forward<F4>( f4 ) , std::forward<F5>( f5 ) , std::forward<F6>( f6 ))
            ));
        p->await();
        using traits::future_access;
        return future_access<typename frame_type::type>::create(std::move(p));
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_future_or_future_range<
            typename util::decay<Func>::type
        >
      , detail::dataflow_frame<
            threads::executor
          , typename hpx::util::decay<Func>::type
          , hpx::util::tuple<
                typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type , typename util::decay<F5>::type , typename util::decay<F6>::type
            >
        >
    >::type
    dataflow(
        threads::executor& sched
      , Func && func
      , F0 && f0 , F1 && f1 , F2 && f2 , F3 && f3 , F4 && f4 , F5 && f5 , F6 && f6
    )
    {
        typedef
            detail::dataflow_frame<
                threads::executor
              , typename hpx::util::decay<Func>::type
              , hpx::util::tuple<
                    typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type , typename util::decay<F5>::type , typename util::decay<F6>::type
                >
            >
            frame_type;
        boost::intrusive_ptr<frame_type> p(new frame_type(
                sched
              , std::forward<Func>(func)
              , hpx::util::forward_as_tuple(std::forward<F0>( f0 ) , std::forward<F1>( f1 ) , std::forward<F2>( f2 ) , std::forward<F3>( f3 ) , std::forward<F4>( f4 ) , std::forward<F5>( f5 ) , std::forward<F6>( f6 ))
            ));
        p->await();
        using traits::future_access;
        return future_access<typename frame_type::type>::create(std::move(p));
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        traits::is_launch_policy<
            typename util::decay<Func>::type
        >
      , detail::dataflow_frame<
            BOOST_SCOPED_ENUM(launch)
          , typename hpx::util::decay<Func>::type
          , hpx::util::tuple<
                typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type , typename util::decay<F5>::type , typename util::decay<F6>::type
            >
        >
    >::type
    dataflow(Func && func, F0 && f0 , F1 && f1 , F2 && f2 , F3 && f3 , F4 && f4 , F5 && f5 , F6 && f6)
    {
        typedef
            detail::dataflow_frame<
                BOOST_SCOPED_ENUM(launch)
              , typename hpx::util::decay<Func>::type
              , hpx::util::tuple<
                    typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type , typename util::decay<F5>::type , typename util::decay<F6>::type
                >
            >
            frame_type;
        boost::intrusive_ptr<frame_type> p(new frame_type(
                launch::all
              , std::forward<Func>(func)
              , hpx::util::forward_as_tuple(std::forward<F0>( f0 ) , std::forward<F1>( f1 ) , std::forward<F2>( f2 ) , std::forward<F3>( f3 ) , std::forward<F4>( f4 ) , std::forward<F5>( f5 ) , std::forward<F6>( f6 ))
            ));
        p->await();
        using traits::future_access;
        return future_access<typename frame_type::type>::create(std::move(p));
    }
}}}
namespace hpx { namespace lcos { namespace local
{
    
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_future_or_future_range<
            typename util::decay<Func>::type
        >
      , detail::dataflow_frame<
            BOOST_SCOPED_ENUM(launch)
          , typename hpx::util::decay<Func>::type
          , hpx::util::tuple<
                typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type , typename util::decay<F5>::type , typename util::decay<F6>::type , typename util::decay<F7>::type
            >
        >
    >::type
    dataflow(
        BOOST_SCOPED_ENUM(launch) policy
      , Func && func
      , F0 && f0 , F1 && f1 , F2 && f2 , F3 && f3 , F4 && f4 , F5 && f5 , F6 && f6 , F7 && f7
    )
    {
        typedef
            detail::dataflow_frame<
                BOOST_SCOPED_ENUM(launch)
              , typename hpx::util::decay<Func>::type
              , hpx::util::tuple<
                    typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type , typename util::decay<F5>::type , typename util::decay<F6>::type , typename util::decay<F7>::type
                >
            >
            frame_type;
        boost::intrusive_ptr<frame_type> p(new frame_type(
                policy
              , std::forward<Func>(func)
              , hpx::util::forward_as_tuple(std::forward<F0>( f0 ) , std::forward<F1>( f1 ) , std::forward<F2>( f2 ) , std::forward<F3>( f3 ) , std::forward<F4>( f4 ) , std::forward<F5>( f5 ) , std::forward<F6>( f6 ) , std::forward<F7>( f7 ))
            ));
        p->await();
        using traits::future_access;
        return future_access<typename frame_type::type>::create(std::move(p));
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_future_or_future_range<
            typename util::decay<Func>::type
        >
      , detail::dataflow_frame<
            threads::executor
          , typename hpx::util::decay<Func>::type
          , hpx::util::tuple<
                typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type , typename util::decay<F5>::type , typename util::decay<F6>::type , typename util::decay<F7>::type
            >
        >
    >::type
    dataflow(
        threads::executor& sched
      , Func && func
      , F0 && f0 , F1 && f1 , F2 && f2 , F3 && f3 , F4 && f4 , F5 && f5 , F6 && f6 , F7 && f7
    )
    {
        typedef
            detail::dataflow_frame<
                threads::executor
              , typename hpx::util::decay<Func>::type
              , hpx::util::tuple<
                    typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type , typename util::decay<F5>::type , typename util::decay<F6>::type , typename util::decay<F7>::type
                >
            >
            frame_type;
        boost::intrusive_ptr<frame_type> p(new frame_type(
                sched
              , std::forward<Func>(func)
              , hpx::util::forward_as_tuple(std::forward<F0>( f0 ) , std::forward<F1>( f1 ) , std::forward<F2>( f2 ) , std::forward<F3>( f3 ) , std::forward<F4>( f4 ) , std::forward<F5>( f5 ) , std::forward<F6>( f6 ) , std::forward<F7>( f7 ))
            ));
        p->await();
        using traits::future_access;
        return future_access<typename frame_type::type>::create(std::move(p));
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        traits::is_launch_policy<
            typename util::decay<Func>::type
        >
      , detail::dataflow_frame<
            BOOST_SCOPED_ENUM(launch)
          , typename hpx::util::decay<Func>::type
          , hpx::util::tuple<
                typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type , typename util::decay<F5>::type , typename util::decay<F6>::type , typename util::decay<F7>::type
            >
        >
    >::type
    dataflow(Func && func, F0 && f0 , F1 && f1 , F2 && f2 , F3 && f3 , F4 && f4 , F5 && f5 , F6 && f6 , F7 && f7)
    {
        typedef
            detail::dataflow_frame<
                BOOST_SCOPED_ENUM(launch)
              , typename hpx::util::decay<Func>::type
              , hpx::util::tuple<
                    typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type , typename util::decay<F5>::type , typename util::decay<F6>::type , typename util::decay<F7>::type
                >
            >
            frame_type;
        boost::intrusive_ptr<frame_type> p(new frame_type(
                launch::all
              , std::forward<Func>(func)
              , hpx::util::forward_as_tuple(std::forward<F0>( f0 ) , std::forward<F1>( f1 ) , std::forward<F2>( f2 ) , std::forward<F3>( f3 ) , std::forward<F4>( f4 ) , std::forward<F5>( f5 ) , std::forward<F6>( f6 ) , std::forward<F7>( f7 ))
            ));
        p->await();
        using traits::future_access;
        return future_access<typename frame_type::type>::create(std::move(p));
    }
}}}
namespace hpx { namespace lcos { namespace local
{
    
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_future_or_future_range<
            typename util::decay<Func>::type
        >
      , detail::dataflow_frame<
            BOOST_SCOPED_ENUM(launch)
          , typename hpx::util::decay<Func>::type
          , hpx::util::tuple<
                typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type , typename util::decay<F5>::type , typename util::decay<F6>::type , typename util::decay<F7>::type , typename util::decay<F8>::type
            >
        >
    >::type
    dataflow(
        BOOST_SCOPED_ENUM(launch) policy
      , Func && func
      , F0 && f0 , F1 && f1 , F2 && f2 , F3 && f3 , F4 && f4 , F5 && f5 , F6 && f6 , F7 && f7 , F8 && f8
    )
    {
        typedef
            detail::dataflow_frame<
                BOOST_SCOPED_ENUM(launch)
              , typename hpx::util::decay<Func>::type
              , hpx::util::tuple<
                    typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type , typename util::decay<F5>::type , typename util::decay<F6>::type , typename util::decay<F7>::type , typename util::decay<F8>::type
                >
            >
            frame_type;
        boost::intrusive_ptr<frame_type> p(new frame_type(
                policy
              , std::forward<Func>(func)
              , hpx::util::forward_as_tuple(std::forward<F0>( f0 ) , std::forward<F1>( f1 ) , std::forward<F2>( f2 ) , std::forward<F3>( f3 ) , std::forward<F4>( f4 ) , std::forward<F5>( f5 ) , std::forward<F6>( f6 ) , std::forward<F7>( f7 ) , std::forward<F8>( f8 ))
            ));
        p->await();
        using traits::future_access;
        return future_access<typename frame_type::type>::create(std::move(p));
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_future_or_future_range<
            typename util::decay<Func>::type
        >
      , detail::dataflow_frame<
            threads::executor
          , typename hpx::util::decay<Func>::type
          , hpx::util::tuple<
                typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type , typename util::decay<F5>::type , typename util::decay<F6>::type , typename util::decay<F7>::type , typename util::decay<F8>::type
            >
        >
    >::type
    dataflow(
        threads::executor& sched
      , Func && func
      , F0 && f0 , F1 && f1 , F2 && f2 , F3 && f3 , F4 && f4 , F5 && f5 , F6 && f6 , F7 && f7 , F8 && f8
    )
    {
        typedef
            detail::dataflow_frame<
                threads::executor
              , typename hpx::util::decay<Func>::type
              , hpx::util::tuple<
                    typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type , typename util::decay<F5>::type , typename util::decay<F6>::type , typename util::decay<F7>::type , typename util::decay<F8>::type
                >
            >
            frame_type;
        boost::intrusive_ptr<frame_type> p(new frame_type(
                sched
              , std::forward<Func>(func)
              , hpx::util::forward_as_tuple(std::forward<F0>( f0 ) , std::forward<F1>( f1 ) , std::forward<F2>( f2 ) , std::forward<F3>( f3 ) , std::forward<F4>( f4 ) , std::forward<F5>( f5 ) , std::forward<F6>( f6 ) , std::forward<F7>( f7 ) , std::forward<F8>( f8 ))
            ));
        p->await();
        using traits::future_access;
        return future_access<typename frame_type::type>::create(std::move(p));
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        traits::is_launch_policy<
            typename util::decay<Func>::type
        >
      , detail::dataflow_frame<
            BOOST_SCOPED_ENUM(launch)
          , typename hpx::util::decay<Func>::type
          , hpx::util::tuple<
                typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type , typename util::decay<F5>::type , typename util::decay<F6>::type , typename util::decay<F7>::type , typename util::decay<F8>::type
            >
        >
    >::type
    dataflow(Func && func, F0 && f0 , F1 && f1 , F2 && f2 , F3 && f3 , F4 && f4 , F5 && f5 , F6 && f6 , F7 && f7 , F8 && f8)
    {
        typedef
            detail::dataflow_frame<
                BOOST_SCOPED_ENUM(launch)
              , typename hpx::util::decay<Func>::type
              , hpx::util::tuple<
                    typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type , typename util::decay<F5>::type , typename util::decay<F6>::type , typename util::decay<F7>::type , typename util::decay<F8>::type
                >
            >
            frame_type;
        boost::intrusive_ptr<frame_type> p(new frame_type(
                launch::all
              , std::forward<Func>(func)
              , hpx::util::forward_as_tuple(std::forward<F0>( f0 ) , std::forward<F1>( f1 ) , std::forward<F2>( f2 ) , std::forward<F3>( f3 ) , std::forward<F4>( f4 ) , std::forward<F5>( f5 ) , std::forward<F6>( f6 ) , std::forward<F7>( f7 ) , std::forward<F8>( f8 ))
            ));
        p->await();
        using traits::future_access;
        return future_access<typename frame_type::type>::create(std::move(p));
    }
}}}
namespace hpx { namespace lcos { namespace local
{
    
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8 , typename F9>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_future_or_future_range<
            typename util::decay<Func>::type
        >
      , detail::dataflow_frame<
            BOOST_SCOPED_ENUM(launch)
          , typename hpx::util::decay<Func>::type
          , hpx::util::tuple<
                typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type , typename util::decay<F5>::type , typename util::decay<F6>::type , typename util::decay<F7>::type , typename util::decay<F8>::type , typename util::decay<F9>::type
            >
        >
    >::type
    dataflow(
        BOOST_SCOPED_ENUM(launch) policy
      , Func && func
      , F0 && f0 , F1 && f1 , F2 && f2 , F3 && f3 , F4 && f4 , F5 && f5 , F6 && f6 , F7 && f7 , F8 && f8 , F9 && f9
    )
    {
        typedef
            detail::dataflow_frame<
                BOOST_SCOPED_ENUM(launch)
              , typename hpx::util::decay<Func>::type
              , hpx::util::tuple<
                    typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type , typename util::decay<F5>::type , typename util::decay<F6>::type , typename util::decay<F7>::type , typename util::decay<F8>::type , typename util::decay<F9>::type
                >
            >
            frame_type;
        boost::intrusive_ptr<frame_type> p(new frame_type(
                policy
              , std::forward<Func>(func)
              , hpx::util::forward_as_tuple(std::forward<F0>( f0 ) , std::forward<F1>( f1 ) , std::forward<F2>( f2 ) , std::forward<F3>( f3 ) , std::forward<F4>( f4 ) , std::forward<F5>( f5 ) , std::forward<F6>( f6 ) , std::forward<F7>( f7 ) , std::forward<F8>( f8 ) , std::forward<F9>( f9 ))
            ));
        p->await();
        using traits::future_access;
        return future_access<typename frame_type::type>::create(std::move(p));
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8 , typename F9>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_future_or_future_range<
            typename util::decay<Func>::type
        >
      , detail::dataflow_frame<
            threads::executor
          , typename hpx::util::decay<Func>::type
          , hpx::util::tuple<
                typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type , typename util::decay<F5>::type , typename util::decay<F6>::type , typename util::decay<F7>::type , typename util::decay<F8>::type , typename util::decay<F9>::type
            >
        >
    >::type
    dataflow(
        threads::executor& sched
      , Func && func
      , F0 && f0 , F1 && f1 , F2 && f2 , F3 && f3 , F4 && f4 , F5 && f5 , F6 && f6 , F7 && f7 , F8 && f8 , F9 && f9
    )
    {
        typedef
            detail::dataflow_frame<
                threads::executor
              , typename hpx::util::decay<Func>::type
              , hpx::util::tuple<
                    typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type , typename util::decay<F5>::type , typename util::decay<F6>::type , typename util::decay<F7>::type , typename util::decay<F8>::type , typename util::decay<F9>::type
                >
            >
            frame_type;
        boost::intrusive_ptr<frame_type> p(new frame_type(
                sched
              , std::forward<Func>(func)
              , hpx::util::forward_as_tuple(std::forward<F0>( f0 ) , std::forward<F1>( f1 ) , std::forward<F2>( f2 ) , std::forward<F3>( f3 ) , std::forward<F4>( f4 ) , std::forward<F5>( f5 ) , std::forward<F6>( f6 ) , std::forward<F7>( f7 ) , std::forward<F8>( f8 ) , std::forward<F9>( f9 ))
            ));
        p->await();
        using traits::future_access;
        return future_access<typename frame_type::type>::create(std::move(p));
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8 , typename F9>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        traits::is_launch_policy<
            typename util::decay<Func>::type
        >
      , detail::dataflow_frame<
            BOOST_SCOPED_ENUM(launch)
          , typename hpx::util::decay<Func>::type
          , hpx::util::tuple<
                typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type , typename util::decay<F5>::type , typename util::decay<F6>::type , typename util::decay<F7>::type , typename util::decay<F8>::type , typename util::decay<F9>::type
            >
        >
    >::type
    dataflow(Func && func, F0 && f0 , F1 && f1 , F2 && f2 , F3 && f3 , F4 && f4 , F5 && f5 , F6 && f6 , F7 && f7 , F8 && f8 , F9 && f9)
    {
        typedef
            detail::dataflow_frame<
                BOOST_SCOPED_ENUM(launch)
              , typename hpx::util::decay<Func>::type
              , hpx::util::tuple<
                    typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type , typename util::decay<F5>::type , typename util::decay<F6>::type , typename util::decay<F7>::type , typename util::decay<F8>::type , typename util::decay<F9>::type
                >
            >
            frame_type;
        boost::intrusive_ptr<frame_type> p(new frame_type(
                launch::all
              , std::forward<Func>(func)
              , hpx::util::forward_as_tuple(std::forward<F0>( f0 ) , std::forward<F1>( f1 ) , std::forward<F2>( f2 ) , std::forward<F3>( f3 ) , std::forward<F4>( f4 ) , std::forward<F5>( f5 ) , std::forward<F6>( f6 ) , std::forward<F7>( f7 ) , std::forward<F8>( f8 ) , std::forward<F9>( f9 ))
            ));
        p->await();
        using traits::future_access;
        return future_access<typename frame_type::type>::create(std::move(p));
    }
}}}
