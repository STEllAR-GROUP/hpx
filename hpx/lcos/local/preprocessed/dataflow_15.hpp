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
      , BOOST_FWD_REF(Func) func
      , BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5
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
              , boost::forward<Func>(func)
              , hpx::util::forward_as_tuple(boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ))
            ));
        p->await();
        using lcos::detail::future_access;
        return future_access::create<typename frame_type::type>(
            boost::move(p));
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
      , BOOST_FWD_REF(Func) func
      , BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5
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
              , boost::forward<Func>(func)
              , hpx::util::forward_as_tuple(boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ))
            ));
        p->await();
        using lcos::detail::future_access;
        return future_access::create<typename frame_type::type>(
            boost::move(p));
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_launch_policy<
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
    dataflow(BOOST_FWD_REF(Func) func, BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5)
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
              , boost::forward<Func>(func)
              , hpx::util::forward_as_tuple(boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ))
            ));
        p->await();
        using lcos::detail::future_access;
        return future_access::create<typename frame_type::type>(
            boost::move(p));
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
      , BOOST_FWD_REF(Func) func
      , BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5 , BOOST_FWD_REF(F6) f6
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
              , boost::forward<Func>(func)
              , hpx::util::forward_as_tuple(boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ))
            ));
        p->await();
        using lcos::detail::future_access;
        return future_access::create<typename frame_type::type>(
            boost::move(p));
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
      , BOOST_FWD_REF(Func) func
      , BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5 , BOOST_FWD_REF(F6) f6
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
              , boost::forward<Func>(func)
              , hpx::util::forward_as_tuple(boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ))
            ));
        p->await();
        using lcos::detail::future_access;
        return future_access::create<typename frame_type::type>(
            boost::move(p));
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_launch_policy<
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
    dataflow(BOOST_FWD_REF(Func) func, BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5 , BOOST_FWD_REF(F6) f6)
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
              , boost::forward<Func>(func)
              , hpx::util::forward_as_tuple(boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ))
            ));
        p->await();
        using lcos::detail::future_access;
        return future_access::create<typename frame_type::type>(
            boost::move(p));
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
      , BOOST_FWD_REF(Func) func
      , BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5 , BOOST_FWD_REF(F6) f6 , BOOST_FWD_REF(F7) f7
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
              , boost::forward<Func>(func)
              , hpx::util::forward_as_tuple(boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ) , boost::forward<F7>( f7 ))
            ));
        p->await();
        using lcos::detail::future_access;
        return future_access::create<typename frame_type::type>(
            boost::move(p));
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
      , BOOST_FWD_REF(Func) func
      , BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5 , BOOST_FWD_REF(F6) f6 , BOOST_FWD_REF(F7) f7
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
              , boost::forward<Func>(func)
              , hpx::util::forward_as_tuple(boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ) , boost::forward<F7>( f7 ))
            ));
        p->await();
        using lcos::detail::future_access;
        return future_access::create<typename frame_type::type>(
            boost::move(p));
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_launch_policy<
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
    dataflow(BOOST_FWD_REF(Func) func, BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5 , BOOST_FWD_REF(F6) f6 , BOOST_FWD_REF(F7) f7)
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
              , boost::forward<Func>(func)
              , hpx::util::forward_as_tuple(boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ) , boost::forward<F7>( f7 ))
            ));
        p->await();
        using lcos::detail::future_access;
        return future_access::create<typename frame_type::type>(
            boost::move(p));
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
      , BOOST_FWD_REF(Func) func
      , BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5 , BOOST_FWD_REF(F6) f6 , BOOST_FWD_REF(F7) f7 , BOOST_FWD_REF(F8) f8
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
              , boost::forward<Func>(func)
              , hpx::util::forward_as_tuple(boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ) , boost::forward<F7>( f7 ) , boost::forward<F8>( f8 ))
            ));
        p->await();
        using lcos::detail::future_access;
        return future_access::create<typename frame_type::type>(
            boost::move(p));
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
      , BOOST_FWD_REF(Func) func
      , BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5 , BOOST_FWD_REF(F6) f6 , BOOST_FWD_REF(F7) f7 , BOOST_FWD_REF(F8) f8
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
              , boost::forward<Func>(func)
              , hpx::util::forward_as_tuple(boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ) , boost::forward<F7>( f7 ) , boost::forward<F8>( f8 ))
            ));
        p->await();
        using lcos::detail::future_access;
        return future_access::create<typename frame_type::type>(
            boost::move(p));
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_launch_policy<
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
    dataflow(BOOST_FWD_REF(Func) func, BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5 , BOOST_FWD_REF(F6) f6 , BOOST_FWD_REF(F7) f7 , BOOST_FWD_REF(F8) f8)
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
              , boost::forward<Func>(func)
              , hpx::util::forward_as_tuple(boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ) , boost::forward<F7>( f7 ) , boost::forward<F8>( f8 ))
            ));
        p->await();
        using lcos::detail::future_access;
        return future_access::create<typename frame_type::type>(
            boost::move(p));
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
      , BOOST_FWD_REF(Func) func
      , BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5 , BOOST_FWD_REF(F6) f6 , BOOST_FWD_REF(F7) f7 , BOOST_FWD_REF(F8) f8 , BOOST_FWD_REF(F9) f9
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
              , boost::forward<Func>(func)
              , hpx::util::forward_as_tuple(boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ) , boost::forward<F7>( f7 ) , boost::forward<F8>( f8 ) , boost::forward<F9>( f9 ))
            ));
        p->await();
        using lcos::detail::future_access;
        return future_access::create<typename frame_type::type>(
            boost::move(p));
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
      , BOOST_FWD_REF(Func) func
      , BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5 , BOOST_FWD_REF(F6) f6 , BOOST_FWD_REF(F7) f7 , BOOST_FWD_REF(F8) f8 , BOOST_FWD_REF(F9) f9
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
              , boost::forward<Func>(func)
              , hpx::util::forward_as_tuple(boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ) , boost::forward<F7>( f7 ) , boost::forward<F8>( f8 ) , boost::forward<F9>( f9 ))
            ));
        p->await();
        using lcos::detail::future_access;
        return future_access::create<typename frame_type::type>(
            boost::move(p));
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8 , typename F9>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_launch_policy<
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
    dataflow(BOOST_FWD_REF(Func) func, BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5 , BOOST_FWD_REF(F6) f6 , BOOST_FWD_REF(F7) f7 , BOOST_FWD_REF(F8) f8 , BOOST_FWD_REF(F9) f9)
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
              , boost::forward<Func>(func)
              , hpx::util::forward_as_tuple(boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ) , boost::forward<F7>( f7 ) , boost::forward<F8>( f8 ) , boost::forward<F9>( f9 ))
            ));
        p->await();
        using lcos::detail::future_access;
        return future_access::create<typename frame_type::type>(
            boost::move(p));
    }
}}}
namespace hpx { namespace lcos { namespace local
{
    
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8 , typename F9 , typename F10>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_future_or_future_range<
            typename util::decay<Func>::type
        >
      , detail::dataflow_frame<
            BOOST_SCOPED_ENUM(launch)
          , typename hpx::util::decay<Func>::type
          , hpx::util::tuple<
                typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type , typename util::decay<F5>::type , typename util::decay<F6>::type , typename util::decay<F7>::type , typename util::decay<F8>::type , typename util::decay<F9>::type , typename util::decay<F10>::type
            >
        >
    >::type
    dataflow(
        BOOST_SCOPED_ENUM(launch) policy
      , BOOST_FWD_REF(Func) func
      , BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5 , BOOST_FWD_REF(F6) f6 , BOOST_FWD_REF(F7) f7 , BOOST_FWD_REF(F8) f8 , BOOST_FWD_REF(F9) f9 , BOOST_FWD_REF(F10) f10
    )
    {
        typedef
            detail::dataflow_frame<
                BOOST_SCOPED_ENUM(launch)
              , typename hpx::util::decay<Func>::type
              , hpx::util::tuple<
                    typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type , typename util::decay<F5>::type , typename util::decay<F6>::type , typename util::decay<F7>::type , typename util::decay<F8>::type , typename util::decay<F9>::type , typename util::decay<F10>::type
                >
            >
            frame_type;
        boost::intrusive_ptr<frame_type> p(new frame_type(
                policy
              , boost::forward<Func>(func)
              , hpx::util::forward_as_tuple(boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ) , boost::forward<F7>( f7 ) , boost::forward<F8>( f8 ) , boost::forward<F9>( f9 ) , boost::forward<F10>( f10 ))
            ));
        p->await();
        using lcos::detail::future_access;
        return future_access::create<typename frame_type::type>(
            boost::move(p));
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8 , typename F9 , typename F10>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_future_or_future_range<
            typename util::decay<Func>::type
        >
      , detail::dataflow_frame<
            threads::executor
          , typename hpx::util::decay<Func>::type
          , hpx::util::tuple<
                typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type , typename util::decay<F5>::type , typename util::decay<F6>::type , typename util::decay<F7>::type , typename util::decay<F8>::type , typename util::decay<F9>::type , typename util::decay<F10>::type
            >
        >
    >::type
    dataflow(
        threads::executor& sched
      , BOOST_FWD_REF(Func) func
      , BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5 , BOOST_FWD_REF(F6) f6 , BOOST_FWD_REF(F7) f7 , BOOST_FWD_REF(F8) f8 , BOOST_FWD_REF(F9) f9 , BOOST_FWD_REF(F10) f10
    )
    {
        typedef
            detail::dataflow_frame<
                threads::executor
              , typename hpx::util::decay<Func>::type
              , hpx::util::tuple<
                    typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type , typename util::decay<F5>::type , typename util::decay<F6>::type , typename util::decay<F7>::type , typename util::decay<F8>::type , typename util::decay<F9>::type , typename util::decay<F10>::type
                >
            >
            frame_type;
        boost::intrusive_ptr<frame_type> p(new frame_type(
                sched
              , boost::forward<Func>(func)
              , hpx::util::forward_as_tuple(boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ) , boost::forward<F7>( f7 ) , boost::forward<F8>( f8 ) , boost::forward<F9>( f9 ) , boost::forward<F10>( f10 ))
            ));
        p->await();
        using lcos::detail::future_access;
        return future_access::create<typename frame_type::type>(
            boost::move(p));
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8 , typename F9 , typename F10>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_launch_policy<
            typename util::decay<Func>::type
        >
      , detail::dataflow_frame<
            BOOST_SCOPED_ENUM(launch)
          , typename hpx::util::decay<Func>::type
          , hpx::util::tuple<
                typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type , typename util::decay<F5>::type , typename util::decay<F6>::type , typename util::decay<F7>::type , typename util::decay<F8>::type , typename util::decay<F9>::type , typename util::decay<F10>::type
            >
        >
    >::type
    dataflow(BOOST_FWD_REF(Func) func, BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5 , BOOST_FWD_REF(F6) f6 , BOOST_FWD_REF(F7) f7 , BOOST_FWD_REF(F8) f8 , BOOST_FWD_REF(F9) f9 , BOOST_FWD_REF(F10) f10)
    {
        typedef
            detail::dataflow_frame<
                BOOST_SCOPED_ENUM(launch)
              , typename hpx::util::decay<Func>::type
              , hpx::util::tuple<
                    typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type , typename util::decay<F5>::type , typename util::decay<F6>::type , typename util::decay<F7>::type , typename util::decay<F8>::type , typename util::decay<F9>::type , typename util::decay<F10>::type
                >
            >
            frame_type;
        boost::intrusive_ptr<frame_type> p(new frame_type(
                launch::all
              , boost::forward<Func>(func)
              , hpx::util::forward_as_tuple(boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ) , boost::forward<F7>( f7 ) , boost::forward<F8>( f8 ) , boost::forward<F9>( f9 ) , boost::forward<F10>( f10 ))
            ));
        p->await();
        using lcos::detail::future_access;
        return future_access::create<typename frame_type::type>(
            boost::move(p));
    }
}}}
namespace hpx { namespace lcos { namespace local
{
    
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8 , typename F9 , typename F10 , typename F11>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_future_or_future_range<
            typename util::decay<Func>::type
        >
      , detail::dataflow_frame<
            BOOST_SCOPED_ENUM(launch)
          , typename hpx::util::decay<Func>::type
          , hpx::util::tuple<
                typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type , typename util::decay<F5>::type , typename util::decay<F6>::type , typename util::decay<F7>::type , typename util::decay<F8>::type , typename util::decay<F9>::type , typename util::decay<F10>::type , typename util::decay<F11>::type
            >
        >
    >::type
    dataflow(
        BOOST_SCOPED_ENUM(launch) policy
      , BOOST_FWD_REF(Func) func
      , BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5 , BOOST_FWD_REF(F6) f6 , BOOST_FWD_REF(F7) f7 , BOOST_FWD_REF(F8) f8 , BOOST_FWD_REF(F9) f9 , BOOST_FWD_REF(F10) f10 , BOOST_FWD_REF(F11) f11
    )
    {
        typedef
            detail::dataflow_frame<
                BOOST_SCOPED_ENUM(launch)
              , typename hpx::util::decay<Func>::type
              , hpx::util::tuple<
                    typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type , typename util::decay<F5>::type , typename util::decay<F6>::type , typename util::decay<F7>::type , typename util::decay<F8>::type , typename util::decay<F9>::type , typename util::decay<F10>::type , typename util::decay<F11>::type
                >
            >
            frame_type;
        boost::intrusive_ptr<frame_type> p(new frame_type(
                policy
              , boost::forward<Func>(func)
              , hpx::util::forward_as_tuple(boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ) , boost::forward<F7>( f7 ) , boost::forward<F8>( f8 ) , boost::forward<F9>( f9 ) , boost::forward<F10>( f10 ) , boost::forward<F11>( f11 ))
            ));
        p->await();
        using lcos::detail::future_access;
        return future_access::create<typename frame_type::type>(
            boost::move(p));
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8 , typename F9 , typename F10 , typename F11>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_future_or_future_range<
            typename util::decay<Func>::type
        >
      , detail::dataflow_frame<
            threads::executor
          , typename hpx::util::decay<Func>::type
          , hpx::util::tuple<
                typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type , typename util::decay<F5>::type , typename util::decay<F6>::type , typename util::decay<F7>::type , typename util::decay<F8>::type , typename util::decay<F9>::type , typename util::decay<F10>::type , typename util::decay<F11>::type
            >
        >
    >::type
    dataflow(
        threads::executor& sched
      , BOOST_FWD_REF(Func) func
      , BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5 , BOOST_FWD_REF(F6) f6 , BOOST_FWD_REF(F7) f7 , BOOST_FWD_REF(F8) f8 , BOOST_FWD_REF(F9) f9 , BOOST_FWD_REF(F10) f10 , BOOST_FWD_REF(F11) f11
    )
    {
        typedef
            detail::dataflow_frame<
                threads::executor
              , typename hpx::util::decay<Func>::type
              , hpx::util::tuple<
                    typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type , typename util::decay<F5>::type , typename util::decay<F6>::type , typename util::decay<F7>::type , typename util::decay<F8>::type , typename util::decay<F9>::type , typename util::decay<F10>::type , typename util::decay<F11>::type
                >
            >
            frame_type;
        boost::intrusive_ptr<frame_type> p(new frame_type(
                sched
              , boost::forward<Func>(func)
              , hpx::util::forward_as_tuple(boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ) , boost::forward<F7>( f7 ) , boost::forward<F8>( f8 ) , boost::forward<F9>( f9 ) , boost::forward<F10>( f10 ) , boost::forward<F11>( f11 ))
            ));
        p->await();
        using lcos::detail::future_access;
        return future_access::create<typename frame_type::type>(
            boost::move(p));
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8 , typename F9 , typename F10 , typename F11>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_launch_policy<
            typename util::decay<Func>::type
        >
      , detail::dataflow_frame<
            BOOST_SCOPED_ENUM(launch)
          , typename hpx::util::decay<Func>::type
          , hpx::util::tuple<
                typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type , typename util::decay<F5>::type , typename util::decay<F6>::type , typename util::decay<F7>::type , typename util::decay<F8>::type , typename util::decay<F9>::type , typename util::decay<F10>::type , typename util::decay<F11>::type
            >
        >
    >::type
    dataflow(BOOST_FWD_REF(Func) func, BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5 , BOOST_FWD_REF(F6) f6 , BOOST_FWD_REF(F7) f7 , BOOST_FWD_REF(F8) f8 , BOOST_FWD_REF(F9) f9 , BOOST_FWD_REF(F10) f10 , BOOST_FWD_REF(F11) f11)
    {
        typedef
            detail::dataflow_frame<
                BOOST_SCOPED_ENUM(launch)
              , typename hpx::util::decay<Func>::type
              , hpx::util::tuple<
                    typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type , typename util::decay<F5>::type , typename util::decay<F6>::type , typename util::decay<F7>::type , typename util::decay<F8>::type , typename util::decay<F9>::type , typename util::decay<F10>::type , typename util::decay<F11>::type
                >
            >
            frame_type;
        boost::intrusive_ptr<frame_type> p(new frame_type(
                launch::all
              , boost::forward<Func>(func)
              , hpx::util::forward_as_tuple(boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ) , boost::forward<F7>( f7 ) , boost::forward<F8>( f8 ) , boost::forward<F9>( f9 ) , boost::forward<F10>( f10 ) , boost::forward<F11>( f11 ))
            ));
        p->await();
        using lcos::detail::future_access;
        return future_access::create<typename frame_type::type>(
            boost::move(p));
    }
}}}
namespace hpx { namespace lcos { namespace local
{
    
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8 , typename F9 , typename F10 , typename F11 , typename F12>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_future_or_future_range<
            typename util::decay<Func>::type
        >
      , detail::dataflow_frame<
            BOOST_SCOPED_ENUM(launch)
          , typename hpx::util::decay<Func>::type
          , hpx::util::tuple<
                typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type , typename util::decay<F5>::type , typename util::decay<F6>::type , typename util::decay<F7>::type , typename util::decay<F8>::type , typename util::decay<F9>::type , typename util::decay<F10>::type , typename util::decay<F11>::type , typename util::decay<F12>::type
            >
        >
    >::type
    dataflow(
        BOOST_SCOPED_ENUM(launch) policy
      , BOOST_FWD_REF(Func) func
      , BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5 , BOOST_FWD_REF(F6) f6 , BOOST_FWD_REF(F7) f7 , BOOST_FWD_REF(F8) f8 , BOOST_FWD_REF(F9) f9 , BOOST_FWD_REF(F10) f10 , BOOST_FWD_REF(F11) f11 , BOOST_FWD_REF(F12) f12
    )
    {
        typedef
            detail::dataflow_frame<
                BOOST_SCOPED_ENUM(launch)
              , typename hpx::util::decay<Func>::type
              , hpx::util::tuple<
                    typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type , typename util::decay<F5>::type , typename util::decay<F6>::type , typename util::decay<F7>::type , typename util::decay<F8>::type , typename util::decay<F9>::type , typename util::decay<F10>::type , typename util::decay<F11>::type , typename util::decay<F12>::type
                >
            >
            frame_type;
        boost::intrusive_ptr<frame_type> p(new frame_type(
                policy
              , boost::forward<Func>(func)
              , hpx::util::forward_as_tuple(boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ) , boost::forward<F7>( f7 ) , boost::forward<F8>( f8 ) , boost::forward<F9>( f9 ) , boost::forward<F10>( f10 ) , boost::forward<F11>( f11 ) , boost::forward<F12>( f12 ))
            ));
        p->await();
        using lcos::detail::future_access;
        return future_access::create<typename frame_type::type>(
            boost::move(p));
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8 , typename F9 , typename F10 , typename F11 , typename F12>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_future_or_future_range<
            typename util::decay<Func>::type
        >
      , detail::dataflow_frame<
            threads::executor
          , typename hpx::util::decay<Func>::type
          , hpx::util::tuple<
                typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type , typename util::decay<F5>::type , typename util::decay<F6>::type , typename util::decay<F7>::type , typename util::decay<F8>::type , typename util::decay<F9>::type , typename util::decay<F10>::type , typename util::decay<F11>::type , typename util::decay<F12>::type
            >
        >
    >::type
    dataflow(
        threads::executor& sched
      , BOOST_FWD_REF(Func) func
      , BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5 , BOOST_FWD_REF(F6) f6 , BOOST_FWD_REF(F7) f7 , BOOST_FWD_REF(F8) f8 , BOOST_FWD_REF(F9) f9 , BOOST_FWD_REF(F10) f10 , BOOST_FWD_REF(F11) f11 , BOOST_FWD_REF(F12) f12
    )
    {
        typedef
            detail::dataflow_frame<
                threads::executor
              , typename hpx::util::decay<Func>::type
              , hpx::util::tuple<
                    typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type , typename util::decay<F5>::type , typename util::decay<F6>::type , typename util::decay<F7>::type , typename util::decay<F8>::type , typename util::decay<F9>::type , typename util::decay<F10>::type , typename util::decay<F11>::type , typename util::decay<F12>::type
                >
            >
            frame_type;
        boost::intrusive_ptr<frame_type> p(new frame_type(
                sched
              , boost::forward<Func>(func)
              , hpx::util::forward_as_tuple(boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ) , boost::forward<F7>( f7 ) , boost::forward<F8>( f8 ) , boost::forward<F9>( f9 ) , boost::forward<F10>( f10 ) , boost::forward<F11>( f11 ) , boost::forward<F12>( f12 ))
            ));
        p->await();
        using lcos::detail::future_access;
        return future_access::create<typename frame_type::type>(
            boost::move(p));
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8 , typename F9 , typename F10 , typename F11 , typename F12>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_launch_policy<
            typename util::decay<Func>::type
        >
      , detail::dataflow_frame<
            BOOST_SCOPED_ENUM(launch)
          , typename hpx::util::decay<Func>::type
          , hpx::util::tuple<
                typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type , typename util::decay<F5>::type , typename util::decay<F6>::type , typename util::decay<F7>::type , typename util::decay<F8>::type , typename util::decay<F9>::type , typename util::decay<F10>::type , typename util::decay<F11>::type , typename util::decay<F12>::type
            >
        >
    >::type
    dataflow(BOOST_FWD_REF(Func) func, BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5 , BOOST_FWD_REF(F6) f6 , BOOST_FWD_REF(F7) f7 , BOOST_FWD_REF(F8) f8 , BOOST_FWD_REF(F9) f9 , BOOST_FWD_REF(F10) f10 , BOOST_FWD_REF(F11) f11 , BOOST_FWD_REF(F12) f12)
    {
        typedef
            detail::dataflow_frame<
                BOOST_SCOPED_ENUM(launch)
              , typename hpx::util::decay<Func>::type
              , hpx::util::tuple<
                    typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type , typename util::decay<F5>::type , typename util::decay<F6>::type , typename util::decay<F7>::type , typename util::decay<F8>::type , typename util::decay<F9>::type , typename util::decay<F10>::type , typename util::decay<F11>::type , typename util::decay<F12>::type
                >
            >
            frame_type;
        boost::intrusive_ptr<frame_type> p(new frame_type(
                launch::all
              , boost::forward<Func>(func)
              , hpx::util::forward_as_tuple(boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ) , boost::forward<F7>( f7 ) , boost::forward<F8>( f8 ) , boost::forward<F9>( f9 ) , boost::forward<F10>( f10 ) , boost::forward<F11>( f11 ) , boost::forward<F12>( f12 ))
            ));
        p->await();
        using lcos::detail::future_access;
        return future_access::create<typename frame_type::type>(
            boost::move(p));
    }
}}}
namespace hpx { namespace lcos { namespace local
{
    
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8 , typename F9 , typename F10 , typename F11 , typename F12 , typename F13>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_future_or_future_range<
            typename util::decay<Func>::type
        >
      , detail::dataflow_frame<
            BOOST_SCOPED_ENUM(launch)
          , typename hpx::util::decay<Func>::type
          , hpx::util::tuple<
                typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type , typename util::decay<F5>::type , typename util::decay<F6>::type , typename util::decay<F7>::type , typename util::decay<F8>::type , typename util::decay<F9>::type , typename util::decay<F10>::type , typename util::decay<F11>::type , typename util::decay<F12>::type , typename util::decay<F13>::type
            >
        >
    >::type
    dataflow(
        BOOST_SCOPED_ENUM(launch) policy
      , BOOST_FWD_REF(Func) func
      , BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5 , BOOST_FWD_REF(F6) f6 , BOOST_FWD_REF(F7) f7 , BOOST_FWD_REF(F8) f8 , BOOST_FWD_REF(F9) f9 , BOOST_FWD_REF(F10) f10 , BOOST_FWD_REF(F11) f11 , BOOST_FWD_REF(F12) f12 , BOOST_FWD_REF(F13) f13
    )
    {
        typedef
            detail::dataflow_frame<
                BOOST_SCOPED_ENUM(launch)
              , typename hpx::util::decay<Func>::type
              , hpx::util::tuple<
                    typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type , typename util::decay<F5>::type , typename util::decay<F6>::type , typename util::decay<F7>::type , typename util::decay<F8>::type , typename util::decay<F9>::type , typename util::decay<F10>::type , typename util::decay<F11>::type , typename util::decay<F12>::type , typename util::decay<F13>::type
                >
            >
            frame_type;
        boost::intrusive_ptr<frame_type> p(new frame_type(
                policy
              , boost::forward<Func>(func)
              , hpx::util::forward_as_tuple(boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ) , boost::forward<F7>( f7 ) , boost::forward<F8>( f8 ) , boost::forward<F9>( f9 ) , boost::forward<F10>( f10 ) , boost::forward<F11>( f11 ) , boost::forward<F12>( f12 ) , boost::forward<F13>( f13 ))
            ));
        p->await();
        using lcos::detail::future_access;
        return future_access::create<typename frame_type::type>(
            boost::move(p));
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8 , typename F9 , typename F10 , typename F11 , typename F12 , typename F13>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_future_or_future_range<
            typename util::decay<Func>::type
        >
      , detail::dataflow_frame<
            threads::executor
          , typename hpx::util::decay<Func>::type
          , hpx::util::tuple<
                typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type , typename util::decay<F5>::type , typename util::decay<F6>::type , typename util::decay<F7>::type , typename util::decay<F8>::type , typename util::decay<F9>::type , typename util::decay<F10>::type , typename util::decay<F11>::type , typename util::decay<F12>::type , typename util::decay<F13>::type
            >
        >
    >::type
    dataflow(
        threads::executor& sched
      , BOOST_FWD_REF(Func) func
      , BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5 , BOOST_FWD_REF(F6) f6 , BOOST_FWD_REF(F7) f7 , BOOST_FWD_REF(F8) f8 , BOOST_FWD_REF(F9) f9 , BOOST_FWD_REF(F10) f10 , BOOST_FWD_REF(F11) f11 , BOOST_FWD_REF(F12) f12 , BOOST_FWD_REF(F13) f13
    )
    {
        typedef
            detail::dataflow_frame<
                threads::executor
              , typename hpx::util::decay<Func>::type
              , hpx::util::tuple<
                    typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type , typename util::decay<F5>::type , typename util::decay<F6>::type , typename util::decay<F7>::type , typename util::decay<F8>::type , typename util::decay<F9>::type , typename util::decay<F10>::type , typename util::decay<F11>::type , typename util::decay<F12>::type , typename util::decay<F13>::type
                >
            >
            frame_type;
        boost::intrusive_ptr<frame_type> p(new frame_type(
                sched
              , boost::forward<Func>(func)
              , hpx::util::forward_as_tuple(boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ) , boost::forward<F7>( f7 ) , boost::forward<F8>( f8 ) , boost::forward<F9>( f9 ) , boost::forward<F10>( f10 ) , boost::forward<F11>( f11 ) , boost::forward<F12>( f12 ) , boost::forward<F13>( f13 ))
            ));
        p->await();
        using lcos::detail::future_access;
        return future_access::create<typename frame_type::type>(
            boost::move(p));
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8 , typename F9 , typename F10 , typename F11 , typename F12 , typename F13>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_launch_policy<
            typename util::decay<Func>::type
        >
      , detail::dataflow_frame<
            BOOST_SCOPED_ENUM(launch)
          , typename hpx::util::decay<Func>::type
          , hpx::util::tuple<
                typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type , typename util::decay<F5>::type , typename util::decay<F6>::type , typename util::decay<F7>::type , typename util::decay<F8>::type , typename util::decay<F9>::type , typename util::decay<F10>::type , typename util::decay<F11>::type , typename util::decay<F12>::type , typename util::decay<F13>::type
            >
        >
    >::type
    dataflow(BOOST_FWD_REF(Func) func, BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5 , BOOST_FWD_REF(F6) f6 , BOOST_FWD_REF(F7) f7 , BOOST_FWD_REF(F8) f8 , BOOST_FWD_REF(F9) f9 , BOOST_FWD_REF(F10) f10 , BOOST_FWD_REF(F11) f11 , BOOST_FWD_REF(F12) f12 , BOOST_FWD_REF(F13) f13)
    {
        typedef
            detail::dataflow_frame<
                BOOST_SCOPED_ENUM(launch)
              , typename hpx::util::decay<Func>::type
              , hpx::util::tuple<
                    typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type , typename util::decay<F5>::type , typename util::decay<F6>::type , typename util::decay<F7>::type , typename util::decay<F8>::type , typename util::decay<F9>::type , typename util::decay<F10>::type , typename util::decay<F11>::type , typename util::decay<F12>::type , typename util::decay<F13>::type
                >
            >
            frame_type;
        boost::intrusive_ptr<frame_type> p(new frame_type(
                launch::all
              , boost::forward<Func>(func)
              , hpx::util::forward_as_tuple(boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ) , boost::forward<F7>( f7 ) , boost::forward<F8>( f8 ) , boost::forward<F9>( f9 ) , boost::forward<F10>( f10 ) , boost::forward<F11>( f11 ) , boost::forward<F12>( f12 ) , boost::forward<F13>( f13 ))
            ));
        p->await();
        using lcos::detail::future_access;
        return future_access::create<typename frame_type::type>(
            boost::move(p));
    }
}}}
namespace hpx { namespace lcos { namespace local
{
    
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8 , typename F9 , typename F10 , typename F11 , typename F12 , typename F13 , typename F14>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_future_or_future_range<
            typename util::decay<Func>::type
        >
      , detail::dataflow_frame<
            BOOST_SCOPED_ENUM(launch)
          , typename hpx::util::decay<Func>::type
          , hpx::util::tuple<
                typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type , typename util::decay<F5>::type , typename util::decay<F6>::type , typename util::decay<F7>::type , typename util::decay<F8>::type , typename util::decay<F9>::type , typename util::decay<F10>::type , typename util::decay<F11>::type , typename util::decay<F12>::type , typename util::decay<F13>::type , typename util::decay<F14>::type
            >
        >
    >::type
    dataflow(
        BOOST_SCOPED_ENUM(launch) policy
      , BOOST_FWD_REF(Func) func
      , BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5 , BOOST_FWD_REF(F6) f6 , BOOST_FWD_REF(F7) f7 , BOOST_FWD_REF(F8) f8 , BOOST_FWD_REF(F9) f9 , BOOST_FWD_REF(F10) f10 , BOOST_FWD_REF(F11) f11 , BOOST_FWD_REF(F12) f12 , BOOST_FWD_REF(F13) f13 , BOOST_FWD_REF(F14) f14
    )
    {
        typedef
            detail::dataflow_frame<
                BOOST_SCOPED_ENUM(launch)
              , typename hpx::util::decay<Func>::type
              , hpx::util::tuple<
                    typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type , typename util::decay<F5>::type , typename util::decay<F6>::type , typename util::decay<F7>::type , typename util::decay<F8>::type , typename util::decay<F9>::type , typename util::decay<F10>::type , typename util::decay<F11>::type , typename util::decay<F12>::type , typename util::decay<F13>::type , typename util::decay<F14>::type
                >
            >
            frame_type;
        boost::intrusive_ptr<frame_type> p(new frame_type(
                policy
              , boost::forward<Func>(func)
              , hpx::util::forward_as_tuple(boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ) , boost::forward<F7>( f7 ) , boost::forward<F8>( f8 ) , boost::forward<F9>( f9 ) , boost::forward<F10>( f10 ) , boost::forward<F11>( f11 ) , boost::forward<F12>( f12 ) , boost::forward<F13>( f13 ) , boost::forward<F14>( f14 ))
            ));
        p->await();
        using lcos::detail::future_access;
        return future_access::create<typename frame_type::type>(
            boost::move(p));
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8 , typename F9 , typename F10 , typename F11 , typename F12 , typename F13 , typename F14>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_future_or_future_range<
            typename util::decay<Func>::type
        >
      , detail::dataflow_frame<
            threads::executor
          , typename hpx::util::decay<Func>::type
          , hpx::util::tuple<
                typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type , typename util::decay<F5>::type , typename util::decay<F6>::type , typename util::decay<F7>::type , typename util::decay<F8>::type , typename util::decay<F9>::type , typename util::decay<F10>::type , typename util::decay<F11>::type , typename util::decay<F12>::type , typename util::decay<F13>::type , typename util::decay<F14>::type
            >
        >
    >::type
    dataflow(
        threads::executor& sched
      , BOOST_FWD_REF(Func) func
      , BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5 , BOOST_FWD_REF(F6) f6 , BOOST_FWD_REF(F7) f7 , BOOST_FWD_REF(F8) f8 , BOOST_FWD_REF(F9) f9 , BOOST_FWD_REF(F10) f10 , BOOST_FWD_REF(F11) f11 , BOOST_FWD_REF(F12) f12 , BOOST_FWD_REF(F13) f13 , BOOST_FWD_REF(F14) f14
    )
    {
        typedef
            detail::dataflow_frame<
                threads::executor
              , typename hpx::util::decay<Func>::type
              , hpx::util::tuple<
                    typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type , typename util::decay<F5>::type , typename util::decay<F6>::type , typename util::decay<F7>::type , typename util::decay<F8>::type , typename util::decay<F9>::type , typename util::decay<F10>::type , typename util::decay<F11>::type , typename util::decay<F12>::type , typename util::decay<F13>::type , typename util::decay<F14>::type
                >
            >
            frame_type;
        boost::intrusive_ptr<frame_type> p(new frame_type(
                sched
              , boost::forward<Func>(func)
              , hpx::util::forward_as_tuple(boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ) , boost::forward<F7>( f7 ) , boost::forward<F8>( f8 ) , boost::forward<F9>( f9 ) , boost::forward<F10>( f10 ) , boost::forward<F11>( f11 ) , boost::forward<F12>( f12 ) , boost::forward<F13>( f13 ) , boost::forward<F14>( f14 ))
            ));
        p->await();
        using lcos::detail::future_access;
        return future_access::create<typename frame_type::type>(
            boost::move(p));
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8 , typename F9 , typename F10 , typename F11 , typename F12 , typename F13 , typename F14>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_launch_policy<
            typename util::decay<Func>::type
        >
      , detail::dataflow_frame<
            BOOST_SCOPED_ENUM(launch)
          , typename hpx::util::decay<Func>::type
          , hpx::util::tuple<
                typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type , typename util::decay<F5>::type , typename util::decay<F6>::type , typename util::decay<F7>::type , typename util::decay<F8>::type , typename util::decay<F9>::type , typename util::decay<F10>::type , typename util::decay<F11>::type , typename util::decay<F12>::type , typename util::decay<F13>::type , typename util::decay<F14>::type
            >
        >
    >::type
    dataflow(BOOST_FWD_REF(Func) func, BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5 , BOOST_FWD_REF(F6) f6 , BOOST_FWD_REF(F7) f7 , BOOST_FWD_REF(F8) f8 , BOOST_FWD_REF(F9) f9 , BOOST_FWD_REF(F10) f10 , BOOST_FWD_REF(F11) f11 , BOOST_FWD_REF(F12) f12 , BOOST_FWD_REF(F13) f13 , BOOST_FWD_REF(F14) f14)
    {
        typedef
            detail::dataflow_frame<
                BOOST_SCOPED_ENUM(launch)
              , typename hpx::util::decay<Func>::type
              , hpx::util::tuple<
                    typename util::decay<F0>::type , typename util::decay<F1>::type , typename util::decay<F2>::type , typename util::decay<F3>::type , typename util::decay<F4>::type , typename util::decay<F5>::type , typename util::decay<F6>::type , typename util::decay<F7>::type , typename util::decay<F8>::type , typename util::decay<F9>::type , typename util::decay<F10>::type , typename util::decay<F11>::type , typename util::decay<F12>::type , typename util::decay<F13>::type , typename util::decay<F14>::type
                >
            >
            frame_type;
        boost::intrusive_ptr<frame_type> p(new frame_type(
                launch::all
              , boost::forward<Func>(func)
              , hpx::util::forward_as_tuple(boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ) , boost::forward<F7>( f7 ) , boost::forward<F8>( f8 ) , boost::forward<F9>( f9 ) , boost::forward<F10>( f10 ) , boost::forward<F11>( f11 ) , boost::forward<F12>( f12 ) , boost::forward<F13>( f13 ) , boost::forward<F14>( f14 ))
            ));
        p->await();
        using lcos::detail::future_access;
        return future_access::create<typename frame_type::type>(
            boost::move(p));
    }
}}}
