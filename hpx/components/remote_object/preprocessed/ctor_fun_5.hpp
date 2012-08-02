// Copyright (c) 2007-2012 Hartmut Kaiser
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


    
    template <
        typename T
      , typename A0 = void , typename A1 = void , typename A2 = void , typename A3 = void , typename A4 = void , typename A5 = void , typename A6 = void , typename A7 = void
      , typename Enable = void
    >
    struct ctor_fun;
    template <typename T>
    struct ctor_fun<T>
    {
        typedef void result_type;
        void operator()(void ** p) const
        {
            T * t = new T();
            *p = t;
        }
        template <typename Archive>
        void serialize(Archive &, unsigned)
        {}
    };
    template <typename T, typename A0>
    struct ctor_fun<T, A0>
    {
        typedef void result_type;
        ctor_fun() {}
        template <typename Arg0>
        ctor_fun(BOOST_FWD_REF(Arg0) arg0)
            : a0(boost::forward<Arg0>(arg0))
        {}
        ctor_fun(BOOST_COPY_ASSIGN_REF(ctor_fun) rhs)
            : a0(rhs. a0)
        {}
        ctor_fun(BOOST_RV_REF(ctor_fun) rhs)
            : a0(boost::move(rhs. a0))
        {}
        ctor_fun& operator=(BOOST_COPY_ASSIGN_REF(ctor_fun) rhs)
        {
            if (this != &rhs) {
                a0 = rhs.a0;
            }
            return *this;
        }
        ctor_fun& operator=(BOOST_RV_REF(ctor_fun) rhs)
        {
            if (this != &rhs) {
                a0 = boost::move(rhs.a0);
            }
            return *this;
        }
        void operator()(void ** p) const
        {
            T * t = new T(a0);
            *p = t;
        }
        template <typename Archive>
        void serialize(Archive & ar, unsigned)
        {
            ar & a0;
        }
        typename boost::remove_const< typename hpx::util::detail::remove_reference<A0>::type >::type a0;
    private:
        BOOST_COPYABLE_AND_MOVABLE(ctor_fun);
    };
    template <typename T, typename A0 , typename A1>
    struct ctor_fun<T, A0 , A1>
    {
        typedef void result_type;
        ctor_fun() {}
        template <typename Arg0 , typename Arg1>
        ctor_fun(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1)
            : a0(boost::forward<Arg0>(arg0)) , a1(boost::forward<Arg1>(arg1))
        {}
        ctor_fun(BOOST_COPY_ASSIGN_REF(ctor_fun) rhs)
            : a0(rhs. a0) , a1(rhs. a1)
        {}
        ctor_fun(BOOST_RV_REF(ctor_fun) rhs)
            : a0(boost::move(rhs. a0)) , a1(boost::move(rhs. a1))
        {}
        ctor_fun& operator=(BOOST_COPY_ASSIGN_REF(ctor_fun) rhs)
        {
            if (this != &rhs) {
                a0 = rhs.a0; a1 = rhs.a1;
            }
            return *this;
        }
        ctor_fun& operator=(BOOST_RV_REF(ctor_fun) rhs)
        {
            if (this != &rhs) {
                a0 = boost::move(rhs.a0); a1 = boost::move(rhs.a1);
            }
            return *this;
        }
        void operator()(void ** p) const
        {
            T * t = new T(a0 , a1);
            *p = t;
        }
        template <typename Archive>
        void serialize(Archive & ar, unsigned)
        {
            ar & a0; ar & a1;
        }
        typename boost::remove_const< typename hpx::util::detail::remove_reference<A0>::type >::type a0; typename boost::remove_const< typename hpx::util::detail::remove_reference<A1>::type >::type a1;
    private:
        BOOST_COPYABLE_AND_MOVABLE(ctor_fun);
    };
    template <typename T, typename A0 , typename A1 , typename A2>
    struct ctor_fun<T, A0 , A1 , A2>
    {
        typedef void result_type;
        ctor_fun() {}
        template <typename Arg0 , typename Arg1 , typename Arg2>
        ctor_fun(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2)
            : a0(boost::forward<Arg0>(arg0)) , a1(boost::forward<Arg1>(arg1)) , a2(boost::forward<Arg2>(arg2))
        {}
        ctor_fun(BOOST_COPY_ASSIGN_REF(ctor_fun) rhs)
            : a0(rhs. a0) , a1(rhs. a1) , a2(rhs. a2)
        {}
        ctor_fun(BOOST_RV_REF(ctor_fun) rhs)
            : a0(boost::move(rhs. a0)) , a1(boost::move(rhs. a1)) , a2(boost::move(rhs. a2))
        {}
        ctor_fun& operator=(BOOST_COPY_ASSIGN_REF(ctor_fun) rhs)
        {
            if (this != &rhs) {
                a0 = rhs.a0; a1 = rhs.a1; a2 = rhs.a2;
            }
            return *this;
        }
        ctor_fun& operator=(BOOST_RV_REF(ctor_fun) rhs)
        {
            if (this != &rhs) {
                a0 = boost::move(rhs.a0); a1 = boost::move(rhs.a1); a2 = boost::move(rhs.a2);
            }
            return *this;
        }
        void operator()(void ** p) const
        {
            T * t = new T(a0 , a1 , a2);
            *p = t;
        }
        template <typename Archive>
        void serialize(Archive & ar, unsigned)
        {
            ar & a0; ar & a1; ar & a2;
        }
        typename boost::remove_const< typename hpx::util::detail::remove_reference<A0>::type >::type a0; typename boost::remove_const< typename hpx::util::detail::remove_reference<A1>::type >::type a1; typename boost::remove_const< typename hpx::util::detail::remove_reference<A2>::type >::type a2;
    private:
        BOOST_COPYABLE_AND_MOVABLE(ctor_fun);
    };
    template <typename T, typename A0 , typename A1 , typename A2 , typename A3>
    struct ctor_fun<T, A0 , A1 , A2 , A3>
    {
        typedef void result_type;
        ctor_fun() {}
        template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
        ctor_fun(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3)
            : a0(boost::forward<Arg0>(arg0)) , a1(boost::forward<Arg1>(arg1)) , a2(boost::forward<Arg2>(arg2)) , a3(boost::forward<Arg3>(arg3))
        {}
        ctor_fun(BOOST_COPY_ASSIGN_REF(ctor_fun) rhs)
            : a0(rhs. a0) , a1(rhs. a1) , a2(rhs. a2) , a3(rhs. a3)
        {}
        ctor_fun(BOOST_RV_REF(ctor_fun) rhs)
            : a0(boost::move(rhs. a0)) , a1(boost::move(rhs. a1)) , a2(boost::move(rhs. a2)) , a3(boost::move(rhs. a3))
        {}
        ctor_fun& operator=(BOOST_COPY_ASSIGN_REF(ctor_fun) rhs)
        {
            if (this != &rhs) {
                a0 = rhs.a0; a1 = rhs.a1; a2 = rhs.a2; a3 = rhs.a3;
            }
            return *this;
        }
        ctor_fun& operator=(BOOST_RV_REF(ctor_fun) rhs)
        {
            if (this != &rhs) {
                a0 = boost::move(rhs.a0); a1 = boost::move(rhs.a1); a2 = boost::move(rhs.a2); a3 = boost::move(rhs.a3);
            }
            return *this;
        }
        void operator()(void ** p) const
        {
            T * t = new T(a0 , a1 , a2 , a3);
            *p = t;
        }
        template <typename Archive>
        void serialize(Archive & ar, unsigned)
        {
            ar & a0; ar & a1; ar & a2; ar & a3;
        }
        typename boost::remove_const< typename hpx::util::detail::remove_reference<A0>::type >::type a0; typename boost::remove_const< typename hpx::util::detail::remove_reference<A1>::type >::type a1; typename boost::remove_const< typename hpx::util::detail::remove_reference<A2>::type >::type a2; typename boost::remove_const< typename hpx::util::detail::remove_reference<A3>::type >::type a3;
    private:
        BOOST_COPYABLE_AND_MOVABLE(ctor_fun);
    };
    template <typename T, typename A0 , typename A1 , typename A2 , typename A3 , typename A4>
    struct ctor_fun<T, A0 , A1 , A2 , A3 , A4>
    {
        typedef void result_type;
        ctor_fun() {}
        template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
        ctor_fun(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4)
            : a0(boost::forward<Arg0>(arg0)) , a1(boost::forward<Arg1>(arg1)) , a2(boost::forward<Arg2>(arg2)) , a3(boost::forward<Arg3>(arg3)) , a4(boost::forward<Arg4>(arg4))
        {}
        ctor_fun(BOOST_COPY_ASSIGN_REF(ctor_fun) rhs)
            : a0(rhs. a0) , a1(rhs. a1) , a2(rhs. a2) , a3(rhs. a3) , a4(rhs. a4)
        {}
        ctor_fun(BOOST_RV_REF(ctor_fun) rhs)
            : a0(boost::move(rhs. a0)) , a1(boost::move(rhs. a1)) , a2(boost::move(rhs. a2)) , a3(boost::move(rhs. a3)) , a4(boost::move(rhs. a4))
        {}
        ctor_fun& operator=(BOOST_COPY_ASSIGN_REF(ctor_fun) rhs)
        {
            if (this != &rhs) {
                a0 = rhs.a0; a1 = rhs.a1; a2 = rhs.a2; a3 = rhs.a3; a4 = rhs.a4;
            }
            return *this;
        }
        ctor_fun& operator=(BOOST_RV_REF(ctor_fun) rhs)
        {
            if (this != &rhs) {
                a0 = boost::move(rhs.a0); a1 = boost::move(rhs.a1); a2 = boost::move(rhs.a2); a3 = boost::move(rhs.a3); a4 = boost::move(rhs.a4);
            }
            return *this;
        }
        void operator()(void ** p) const
        {
            T * t = new T(a0 , a1 , a2 , a3 , a4);
            *p = t;
        }
        template <typename Archive>
        void serialize(Archive & ar, unsigned)
        {
            ar & a0; ar & a1; ar & a2; ar & a3; ar & a4;
        }
        typename boost::remove_const< typename hpx::util::detail::remove_reference<A0>::type >::type a0; typename boost::remove_const< typename hpx::util::detail::remove_reference<A1>::type >::type a1; typename boost::remove_const< typename hpx::util::detail::remove_reference<A2>::type >::type a2; typename boost::remove_const< typename hpx::util::detail::remove_reference<A3>::type >::type a3; typename boost::remove_const< typename hpx::util::detail::remove_reference<A4>::type >::type a4;
    private:
        BOOST_COPYABLE_AND_MOVABLE(ctor_fun);
    };
    template <typename T, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5>
    struct ctor_fun<T, A0 , A1 , A2 , A3 , A4 , A5>
    {
        typedef void result_type;
        ctor_fun() {}
        template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
        ctor_fun(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5)
            : a0(boost::forward<Arg0>(arg0)) , a1(boost::forward<Arg1>(arg1)) , a2(boost::forward<Arg2>(arg2)) , a3(boost::forward<Arg3>(arg3)) , a4(boost::forward<Arg4>(arg4)) , a5(boost::forward<Arg5>(arg5))
        {}
        ctor_fun(BOOST_COPY_ASSIGN_REF(ctor_fun) rhs)
            : a0(rhs. a0) , a1(rhs. a1) , a2(rhs. a2) , a3(rhs. a3) , a4(rhs. a4) , a5(rhs. a5)
        {}
        ctor_fun(BOOST_RV_REF(ctor_fun) rhs)
            : a0(boost::move(rhs. a0)) , a1(boost::move(rhs. a1)) , a2(boost::move(rhs. a2)) , a3(boost::move(rhs. a3)) , a4(boost::move(rhs. a4)) , a5(boost::move(rhs. a5))
        {}
        ctor_fun& operator=(BOOST_COPY_ASSIGN_REF(ctor_fun) rhs)
        {
            if (this != &rhs) {
                a0 = rhs.a0; a1 = rhs.a1; a2 = rhs.a2; a3 = rhs.a3; a4 = rhs.a4; a5 = rhs.a5;
            }
            return *this;
        }
        ctor_fun& operator=(BOOST_RV_REF(ctor_fun) rhs)
        {
            if (this != &rhs) {
                a0 = boost::move(rhs.a0); a1 = boost::move(rhs.a1); a2 = boost::move(rhs.a2); a3 = boost::move(rhs.a3); a4 = boost::move(rhs.a4); a5 = boost::move(rhs.a5);
            }
            return *this;
        }
        void operator()(void ** p) const
        {
            T * t = new T(a0 , a1 , a2 , a3 , a4 , a5);
            *p = t;
        }
        template <typename Archive>
        void serialize(Archive & ar, unsigned)
        {
            ar & a0; ar & a1; ar & a2; ar & a3; ar & a4; ar & a5;
        }
        typename boost::remove_const< typename hpx::util::detail::remove_reference<A0>::type >::type a0; typename boost::remove_const< typename hpx::util::detail::remove_reference<A1>::type >::type a1; typename boost::remove_const< typename hpx::util::detail::remove_reference<A2>::type >::type a2; typename boost::remove_const< typename hpx::util::detail::remove_reference<A3>::type >::type a3; typename boost::remove_const< typename hpx::util::detail::remove_reference<A4>::type >::type a4; typename boost::remove_const< typename hpx::util::detail::remove_reference<A5>::type >::type a5;
    private:
        BOOST_COPYABLE_AND_MOVABLE(ctor_fun);
    };
    template <typename T, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6>
    struct ctor_fun<T, A0 , A1 , A2 , A3 , A4 , A5 , A6>
    {
        typedef void result_type;
        ctor_fun() {}
        template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
        ctor_fun(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6)
            : a0(boost::forward<Arg0>(arg0)) , a1(boost::forward<Arg1>(arg1)) , a2(boost::forward<Arg2>(arg2)) , a3(boost::forward<Arg3>(arg3)) , a4(boost::forward<Arg4>(arg4)) , a5(boost::forward<Arg5>(arg5)) , a6(boost::forward<Arg6>(arg6))
        {}
        ctor_fun(BOOST_COPY_ASSIGN_REF(ctor_fun) rhs)
            : a0(rhs. a0) , a1(rhs. a1) , a2(rhs. a2) , a3(rhs. a3) , a4(rhs. a4) , a5(rhs. a5) , a6(rhs. a6)
        {}
        ctor_fun(BOOST_RV_REF(ctor_fun) rhs)
            : a0(boost::move(rhs. a0)) , a1(boost::move(rhs. a1)) , a2(boost::move(rhs. a2)) , a3(boost::move(rhs. a3)) , a4(boost::move(rhs. a4)) , a5(boost::move(rhs. a5)) , a6(boost::move(rhs. a6))
        {}
        ctor_fun& operator=(BOOST_COPY_ASSIGN_REF(ctor_fun) rhs)
        {
            if (this != &rhs) {
                a0 = rhs.a0; a1 = rhs.a1; a2 = rhs.a2; a3 = rhs.a3; a4 = rhs.a4; a5 = rhs.a5; a6 = rhs.a6;
            }
            return *this;
        }
        ctor_fun& operator=(BOOST_RV_REF(ctor_fun) rhs)
        {
            if (this != &rhs) {
                a0 = boost::move(rhs.a0); a1 = boost::move(rhs.a1); a2 = boost::move(rhs.a2); a3 = boost::move(rhs.a3); a4 = boost::move(rhs.a4); a5 = boost::move(rhs.a5); a6 = boost::move(rhs.a6);
            }
            return *this;
        }
        void operator()(void ** p) const
        {
            T * t = new T(a0 , a1 , a2 , a3 , a4 , a5 , a6);
            *p = t;
        }
        template <typename Archive>
        void serialize(Archive & ar, unsigned)
        {
            ar & a0; ar & a1; ar & a2; ar & a3; ar & a4; ar & a5; ar & a6;
        }
        typename boost::remove_const< typename hpx::util::detail::remove_reference<A0>::type >::type a0; typename boost::remove_const< typename hpx::util::detail::remove_reference<A1>::type >::type a1; typename boost::remove_const< typename hpx::util::detail::remove_reference<A2>::type >::type a2; typename boost::remove_const< typename hpx::util::detail::remove_reference<A3>::type >::type a3; typename boost::remove_const< typename hpx::util::detail::remove_reference<A4>::type >::type a4; typename boost::remove_const< typename hpx::util::detail::remove_reference<A5>::type >::type a5; typename boost::remove_const< typename hpx::util::detail::remove_reference<A6>::type >::type a6;
    private:
        BOOST_COPYABLE_AND_MOVABLE(ctor_fun);
    };
    template <typename T, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7>
    struct ctor_fun<T, A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7>
    {
        typedef void result_type;
        ctor_fun() {}
        template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
        ctor_fun(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7)
            : a0(boost::forward<Arg0>(arg0)) , a1(boost::forward<Arg1>(arg1)) , a2(boost::forward<Arg2>(arg2)) , a3(boost::forward<Arg3>(arg3)) , a4(boost::forward<Arg4>(arg4)) , a5(boost::forward<Arg5>(arg5)) , a6(boost::forward<Arg6>(arg6)) , a7(boost::forward<Arg7>(arg7))
        {}
        ctor_fun(BOOST_COPY_ASSIGN_REF(ctor_fun) rhs)
            : a0(rhs. a0) , a1(rhs. a1) , a2(rhs. a2) , a3(rhs. a3) , a4(rhs. a4) , a5(rhs. a5) , a6(rhs. a6) , a7(rhs. a7)
        {}
        ctor_fun(BOOST_RV_REF(ctor_fun) rhs)
            : a0(boost::move(rhs. a0)) , a1(boost::move(rhs. a1)) , a2(boost::move(rhs. a2)) , a3(boost::move(rhs. a3)) , a4(boost::move(rhs. a4)) , a5(boost::move(rhs. a5)) , a6(boost::move(rhs. a6)) , a7(boost::move(rhs. a7))
        {}
        ctor_fun& operator=(BOOST_COPY_ASSIGN_REF(ctor_fun) rhs)
        {
            if (this != &rhs) {
                a0 = rhs.a0; a1 = rhs.a1; a2 = rhs.a2; a3 = rhs.a3; a4 = rhs.a4; a5 = rhs.a5; a6 = rhs.a6; a7 = rhs.a7;
            }
            return *this;
        }
        ctor_fun& operator=(BOOST_RV_REF(ctor_fun) rhs)
        {
            if (this != &rhs) {
                a0 = boost::move(rhs.a0); a1 = boost::move(rhs.a1); a2 = boost::move(rhs.a2); a3 = boost::move(rhs.a3); a4 = boost::move(rhs.a4); a5 = boost::move(rhs.a5); a6 = boost::move(rhs.a6); a7 = boost::move(rhs.a7);
            }
            return *this;
        }
        void operator()(void ** p) const
        {
            T * t = new T(a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7);
            *p = t;
        }
        template <typename Archive>
        void serialize(Archive & ar, unsigned)
        {
            ar & a0; ar & a1; ar & a2; ar & a3; ar & a4; ar & a5; ar & a6; ar & a7;
        }
        typename boost::remove_const< typename hpx::util::detail::remove_reference<A0>::type >::type a0; typename boost::remove_const< typename hpx::util::detail::remove_reference<A1>::type >::type a1; typename boost::remove_const< typename hpx::util::detail::remove_reference<A2>::type >::type a2; typename boost::remove_const< typename hpx::util::detail::remove_reference<A3>::type >::type a3; typename boost::remove_const< typename hpx::util::detail::remove_reference<A4>::type >::type a4; typename boost::remove_const< typename hpx::util::detail::remove_reference<A5>::type >::type a5; typename boost::remove_const< typename hpx::util::detail::remove_reference<A6>::type >::type a6; typename boost::remove_const< typename hpx::util::detail::remove_reference<A7>::type >::type a7;
    private:
        BOOST_COPYABLE_AND_MOVABLE(ctor_fun);
    };
