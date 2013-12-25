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
      , BOOST_FWD_REF(Func) func
      , BOOST_FWD_REF(F0) f0
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
              , boost::forward<Func>(func)
              , hpx::util::forward_as_tuple(boost::forward<F0>( f0 ))
            ));
        p->await();
        using lcos::detail::future_access;
        return future_access::create<typename frame_type::type>(
            boost::move(p));
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
      , BOOST_FWD_REF(Func) func
      , BOOST_FWD_REF(F0) f0
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
              , boost::forward<Func>(func)
              , hpx::util::forward_as_tuple(boost::forward<F0>( f0 ))
            ));
        p->await();
        using lcos::detail::future_access;
        return future_access::create<typename frame_type::type>(
            boost::move(p));
    }
    template <typename Func, typename F0>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_launch_policy<
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
    dataflow(BOOST_FWD_REF(Func) func, BOOST_FWD_REF(F0) f0)
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
              , boost::forward<Func>(func)
              , hpx::util::forward_as_tuple(boost::forward<F0>( f0 ))
            ));
        p->await();
        using lcos::detail::future_access;
        return future_access::create<typename frame_type::type>(
            boost::move(p));
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
      , BOOST_FWD_REF(Func) func
      , BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1
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
              , boost::forward<Func>(func)
              , hpx::util::forward_as_tuple(boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ))
            ));
        p->await();
        using lcos::detail::future_access;
        return future_access::create<typename frame_type::type>(
            boost::move(p));
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
      , BOOST_FWD_REF(Func) func
      , BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1
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
              , boost::forward<Func>(func)
              , hpx::util::forward_as_tuple(boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ))
            ));
        p->await();
        using lcos::detail::future_access;
        return future_access::create<typename frame_type::type>(
            boost::move(p));
    }
    template <typename Func, typename F0 , typename F1>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_launch_policy<
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
    dataflow(BOOST_FWD_REF(Func) func, BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1)
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
              , boost::forward<Func>(func)
              , hpx::util::forward_as_tuple(boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ))
            ));
        p->await();
        using lcos::detail::future_access;
        return future_access::create<typename frame_type::type>(
            boost::move(p));
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
      , BOOST_FWD_REF(Func) func
      , BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2
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
              , boost::forward<Func>(func)
              , hpx::util::forward_as_tuple(boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ))
            ));
        p->await();
        using lcos::detail::future_access;
        return future_access::create<typename frame_type::type>(
            boost::move(p));
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
      , BOOST_FWD_REF(Func) func
      , BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2
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
              , boost::forward<Func>(func)
              , hpx::util::forward_as_tuple(boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ))
            ));
        p->await();
        using lcos::detail::future_access;
        return future_access::create<typename frame_type::type>(
            boost::move(p));
    }
    template <typename Func, typename F0 , typename F1 , typename F2>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_launch_policy<
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
    dataflow(BOOST_FWD_REF(Func) func, BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2)
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
              , boost::forward<Func>(func)
              , hpx::util::forward_as_tuple(boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ))
            ));
        p->await();
        using lcos::detail::future_access;
        return future_access::create<typename frame_type::type>(
            boost::move(p));
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
      , BOOST_FWD_REF(Func) func
      , BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3
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
              , boost::forward<Func>(func)
              , hpx::util::forward_as_tuple(boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ))
            ));
        p->await();
        using lcos::detail::future_access;
        return future_access::create<typename frame_type::type>(
            boost::move(p));
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
      , BOOST_FWD_REF(Func) func
      , BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3
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
              , boost::forward<Func>(func)
              , hpx::util::forward_as_tuple(boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ))
            ));
        p->await();
        using lcos::detail::future_access;
        return future_access::create<typename frame_type::type>(
            boost::move(p));
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_launch_policy<
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
    dataflow(BOOST_FWD_REF(Func) func, BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3)
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
              , boost::forward<Func>(func)
              , hpx::util::forward_as_tuple(boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ))
            ));
        p->await();
        using lcos::detail::future_access;
        return future_access::create<typename frame_type::type>(
            boost::move(p));
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
      , BOOST_FWD_REF(Func) func
      , BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4
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
              , boost::forward<Func>(func)
              , hpx::util::forward_as_tuple(boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ))
            ));
        p->await();
        using lcos::detail::future_access;
        return future_access::create<typename frame_type::type>(
            boost::move(p));
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
      , BOOST_FWD_REF(Func) func
      , BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4
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
              , boost::forward<Func>(func)
              , hpx::util::forward_as_tuple(boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ))
            ));
        p->await();
        using lcos::detail::future_access;
        return future_access::create<typename frame_type::type>(
            boost::move(p));
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_launch_policy<
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
    dataflow(BOOST_FWD_REF(Func) func, BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4)
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
              , boost::forward<Func>(func)
              , hpx::util::forward_as_tuple(boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ))
            ));
        p->await();
        using lcos::detail::future_access;
        return future_access::create<typename frame_type::type>(
            boost::move(p));
    }
}}}
