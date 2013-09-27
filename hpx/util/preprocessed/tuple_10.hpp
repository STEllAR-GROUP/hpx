// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


namespace hpx { namespace util { namespace detail
{
    template <typename Tuple> struct tuple_element< 0, Tuple> { typedef typename Tuple::member_type0 type; typedef typename detail::tuple_element_access<type>::type rtype; typedef typename detail::tuple_element_access<type>::ctype crtype; static BOOST_CONSTEXPR rtype get(Tuple& t) BOOST_NOEXCEPT { return t.a0; } }; template <typename Tuple> struct tuple_element< 1, Tuple> { typedef typename Tuple::member_type1 type; typedef typename detail::tuple_element_access<type>::type rtype; typedef typename detail::tuple_element_access<type>::ctype crtype; static BOOST_CONSTEXPR rtype get(Tuple& t) BOOST_NOEXCEPT { return t.a1; } }; template <typename Tuple> struct tuple_element< 2, Tuple> { typedef typename Tuple::member_type2 type; typedef typename detail::tuple_element_access<type>::type rtype; typedef typename detail::tuple_element_access<type>::ctype crtype; static BOOST_CONSTEXPR rtype get(Tuple& t) BOOST_NOEXCEPT { return t.a2; } }; template <typename Tuple> struct tuple_element< 3, Tuple> { typedef typename Tuple::member_type3 type; typedef typename detail::tuple_element_access<type>::type rtype; typedef typename detail::tuple_element_access<type>::ctype crtype; static BOOST_CONSTEXPR rtype get(Tuple& t) BOOST_NOEXCEPT { return t.a3; } }; template <typename Tuple> struct tuple_element< 4, Tuple> { typedef typename Tuple::member_type4 type; typedef typename detail::tuple_element_access<type>::type rtype; typedef typename detail::tuple_element_access<type>::ctype crtype; static BOOST_CONSTEXPR rtype get(Tuple& t) BOOST_NOEXCEPT { return t.a4; } }; template <typename Tuple> struct tuple_element< 5, Tuple> { typedef typename Tuple::member_type5 type; typedef typename detail::tuple_element_access<type>::type rtype; typedef typename detail::tuple_element_access<type>::ctype crtype; static BOOST_CONSTEXPR rtype get(Tuple& t) BOOST_NOEXCEPT { return t.a5; } }; template <typename Tuple> struct tuple_element< 6, Tuple> { typedef typename Tuple::member_type6 type; typedef typename detail::tuple_element_access<type>::type rtype; typedef typename detail::tuple_element_access<type>::ctype crtype; static BOOST_CONSTEXPR rtype get(Tuple& t) BOOST_NOEXCEPT { return t.a6; } }; template <typename Tuple> struct tuple_element< 7, Tuple> { typedef typename Tuple::member_type7 type; typedef typename detail::tuple_element_access<type>::type rtype; typedef typename detail::tuple_element_access<type>::ctype crtype; static BOOST_CONSTEXPR rtype get(Tuple& t) BOOST_NOEXCEPT { return t.a7; } }; template <typename Tuple> struct tuple_element< 8, Tuple> { typedef typename Tuple::member_type8 type; typedef typename detail::tuple_element_access<type>::type rtype; typedef typename detail::tuple_element_access<type>::ctype crtype; static BOOST_CONSTEXPR rtype get(Tuple& t) BOOST_NOEXCEPT { return t.a8; } }; template <typename Tuple> struct tuple_element< 9, Tuple> { typedef typename Tuple::member_type9 type; typedef typename detail::tuple_element_access<type>::type rtype; typedef typename detail::tuple_element_access<type>::ctype crtype; static BOOST_CONSTEXPR rtype get(Tuple& t) BOOST_NOEXCEPT { return t.a9; } }; template <typename Tuple> struct tuple_element< 10, Tuple> { typedef typename Tuple::member_type10 type; typedef typename detail::tuple_element_access<type>::type rtype; typedef typename detail::tuple_element_access<type>::ctype crtype; static BOOST_CONSTEXPR rtype get(Tuple& t) BOOST_NOEXCEPT { return t.a10; } }; template <typename Tuple> struct tuple_element< 11, Tuple> { typedef typename Tuple::member_type11 type; typedef typename detail::tuple_element_access<type>::type rtype; typedef typename detail::tuple_element_access<type>::ctype crtype; static BOOST_CONSTEXPR rtype get(Tuple& t) BOOST_NOEXCEPT { return t.a11; } }; template <typename Tuple> struct tuple_element< 12, Tuple> { typedef typename Tuple::member_type12 type; typedef typename detail::tuple_element_access<type>::type rtype; typedef typename detail::tuple_element_access<type>::ctype crtype; static BOOST_CONSTEXPR rtype get(Tuple& t) BOOST_NOEXCEPT { return t.a12; } };
    template <typename Tuple> struct tuple_element< 0, Tuple const> { typedef typename boost::add_const< typename Tuple::member_type0>::type type; typedef typename detail::tuple_element_access<type>::type rtype; typedef typename detail::tuple_element_access<type>::ctype crtype; static BOOST_CONSTEXPR crtype get(Tuple const& t) BOOST_NOEXCEPT { return t.a0; } }; template <typename Tuple> struct tuple_element< 1, Tuple const> { typedef typename boost::add_const< typename Tuple::member_type1>::type type; typedef typename detail::tuple_element_access<type>::type rtype; typedef typename detail::tuple_element_access<type>::ctype crtype; static BOOST_CONSTEXPR crtype get(Tuple const& t) BOOST_NOEXCEPT { return t.a1; } }; template <typename Tuple> struct tuple_element< 2, Tuple const> { typedef typename boost::add_const< typename Tuple::member_type2>::type type; typedef typename detail::tuple_element_access<type>::type rtype; typedef typename detail::tuple_element_access<type>::ctype crtype; static BOOST_CONSTEXPR crtype get(Tuple const& t) BOOST_NOEXCEPT { return t.a2; } }; template <typename Tuple> struct tuple_element< 3, Tuple const> { typedef typename boost::add_const< typename Tuple::member_type3>::type type; typedef typename detail::tuple_element_access<type>::type rtype; typedef typename detail::tuple_element_access<type>::ctype crtype; static BOOST_CONSTEXPR crtype get(Tuple const& t) BOOST_NOEXCEPT { return t.a3; } }; template <typename Tuple> struct tuple_element< 4, Tuple const> { typedef typename boost::add_const< typename Tuple::member_type4>::type type; typedef typename detail::tuple_element_access<type>::type rtype; typedef typename detail::tuple_element_access<type>::ctype crtype; static BOOST_CONSTEXPR crtype get(Tuple const& t) BOOST_NOEXCEPT { return t.a4; } }; template <typename Tuple> struct tuple_element< 5, Tuple const> { typedef typename boost::add_const< typename Tuple::member_type5>::type type; typedef typename detail::tuple_element_access<type>::type rtype; typedef typename detail::tuple_element_access<type>::ctype crtype; static BOOST_CONSTEXPR crtype get(Tuple const& t) BOOST_NOEXCEPT { return t.a5; } }; template <typename Tuple> struct tuple_element< 6, Tuple const> { typedef typename boost::add_const< typename Tuple::member_type6>::type type; typedef typename detail::tuple_element_access<type>::type rtype; typedef typename detail::tuple_element_access<type>::ctype crtype; static BOOST_CONSTEXPR crtype get(Tuple const& t) BOOST_NOEXCEPT { return t.a6; } }; template <typename Tuple> struct tuple_element< 7, Tuple const> { typedef typename boost::add_const< typename Tuple::member_type7>::type type; typedef typename detail::tuple_element_access<type>::type rtype; typedef typename detail::tuple_element_access<type>::ctype crtype; static BOOST_CONSTEXPR crtype get(Tuple const& t) BOOST_NOEXCEPT { return t.a7; } }; template <typename Tuple> struct tuple_element< 8, Tuple const> { typedef typename boost::add_const< typename Tuple::member_type8>::type type; typedef typename detail::tuple_element_access<type>::type rtype; typedef typename detail::tuple_element_access<type>::ctype crtype; static BOOST_CONSTEXPR crtype get(Tuple const& t) BOOST_NOEXCEPT { return t.a8; } }; template <typename Tuple> struct tuple_element< 9, Tuple const> { typedef typename boost::add_const< typename Tuple::member_type9>::type type; typedef typename detail::tuple_element_access<type>::type rtype; typedef typename detail::tuple_element_access<type>::ctype crtype; static BOOST_CONSTEXPR crtype get(Tuple const& t) BOOST_NOEXCEPT { return t.a9; } }; template <typename Tuple> struct tuple_element< 10, Tuple const> { typedef typename boost::add_const< typename Tuple::member_type10>::type type; typedef typename detail::tuple_element_access<type>::type rtype; typedef typename detail::tuple_element_access<type>::ctype crtype; static BOOST_CONSTEXPR crtype get(Tuple const& t) BOOST_NOEXCEPT { return t.a10; } }; template <typename Tuple> struct tuple_element< 11, Tuple const> { typedef typename boost::add_const< typename Tuple::member_type11>::type type; typedef typename detail::tuple_element_access<type>::type rtype; typedef typename detail::tuple_element_access<type>::ctype crtype; static BOOST_CONSTEXPR crtype get(Tuple const& t) BOOST_NOEXCEPT { return t.a11; } }; template <typename Tuple> struct tuple_element< 12, Tuple const> { typedef typename boost::add_const< typename Tuple::member_type12>::type type; typedef typename detail::tuple_element_access<type>::type rtype; typedef typename detail::tuple_element_access<type>::ctype crtype; static BOOST_CONSTEXPR crtype get(Tuple const& t) BOOST_NOEXCEPT { return t.a12; } };
}}}
namespace hpx { namespace util
{
    
    template <typename A0>
    struct tuple<A0>
    {
        typedef A0 member_type0; A0 a0;
        template <int E>
        typename detail::tuple_element<E, tuple>::rtype
        get() BOOST_NOEXCEPT
        {
            return detail::tuple_element<E, tuple>::get(*this);
        }
        template <int E>
        BOOST_CONSTEXPR
        typename detail::tuple_element<E, tuple const>::crtype
        get() const BOOST_NOEXCEPT
        {
            return detail::tuple_element<E, tuple const>::get(*this);
        }
        
        tuple()
          : a0()
        {}
        template <typename Arg0>
        tuple(BOOST_FWD_REF(Arg0) arg0
          , typename boost::disable_if<is_tuple<Arg0> >::type* = 0)
          : a0(boost::forward<Arg0>(arg0))
        {}
        template <typename Arg0>
        tuple(BOOST_FWD_REF(Arg0) arg0, detail::forwarding_tag)
          : a0(boost::forward<Arg0>(arg0))
        {}
        
        tuple(tuple const& other)
          : a0( detail::copy_construct< A0 , typename boost::add_const<A0>::type >::call(other.a0))
        {}
        tuple(BOOST_RV_REF(tuple) other)
          : a0(boost::forward<A0>( other.a0))
        {}
        template <typename T0>
        tuple(tuple<T0> const& other)
          : a0( detail::copy_construct< A0 , typename boost::add_const<T0>::type >::call(other.a0))
        {}
        template <typename T0>
        tuple(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple<T0>
            ))) other)
          : a0(boost::forward<T0>( other.a0))
        {}
        
        tuple& operator=(BOOST_COPY_ASSIGN_REF(tuple) other)
        {
            a0 = other.a0;;
            return *this;
        }
        tuple& operator=(BOOST_RV_REF(tuple) other)
        {
            a0 = boost::forward<A0>( other.a0);;
            return *this;
        }
        template <typename T0>
        tuple& operator=(BOOST_COPY_ASSIGN_REF(HPX_UTIL_STRIP((
                tuple<T0>
            ))) other)
        {
            a0 = other.a0;;
            return *this;
        }
        template <typename T0>
        tuple& operator=(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple<T0>
            ))) other)
        {
            a0 = boost::forward<T0>( other.a0);;
            return *this;
        }
        void swap(tuple& other)
        {
            boost::swap(a0, other.a0);;
        }
        typedef boost::mpl::int_<1> size_type;
        static const int size_value = 1;
    private:
        BOOST_COPYABLE_AND_MOVABLE(tuple);
    };
    template <typename A0>
    struct tuple_size<tuple<A0> >
    {
        static const std::size_t value = 1;
    };
    
    template <typename Arg0>
    BOOST_FORCEINLINE
    tuple<typename util::decay<Arg0>::type>
    make_tuple(BOOST_FWD_REF(Arg0) arg0)
    {
        typedef tuple<typename util::decay<Arg0>::type> result_type;
        return result_type(boost::forward<Arg0>(arg0), detail::forwarding_tag());
    }
    
    template <typename Arg0>
    BOOST_FORCEINLINE
    tuple<typename detail::add_rvalue_reference<Arg0>::type>
    forward_as_tuple(BOOST_FWD_REF(Arg0) arg0) BOOST_NOEXCEPT
    {
        typedef tuple<typename detail::add_rvalue_reference<Arg0>::type> result_type;
        return result_type(boost::forward<Arg0>(arg0), detail::forwarding_tag());
    }
    
    template <typename Arg0>
    BOOST_FORCEINLINE
    tuple<Arg0&>
    tie(Arg0& arg0) BOOST_NOEXCEPT
    {
        typedef tuple<Arg0&> result_type;
        return result_type(arg0, detail::forwarding_tag());
    }
    
    template <typename T0>
    BOOST_FORCEINLINE T0
    tuple_cat(BOOST_FWD_REF(T0) t0)
    {
        return boost::forward<T0>(t0);
    }
}}
namespace boost { namespace fusion { namespace traits { template< typename A0 > struct tag_of<hpx::util::tuple<A0> > { typedef struct_tag type; }; template< typename A0 > struct tag_of<hpx::util::tuple<A0> const> { typedef struct_tag type; }; } namespace extension { template< typename A0 > struct access::struct_member< hpx::util::tuple<A0> , 0 > { typedef A0 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a0; } }; }; template< typename A0 > struct struct_member_name< hpx::util::tuple<A0> , 0 > { typedef char const* type; static type call() { return "a0"; } }; template< typename A0 > struct struct_size<hpx::util::tuple<A0> > : mpl::int_<1> {}; template< typename A0 > struct struct_is_view< hpx::util::tuple<A0> > : mpl::false_ {}; } } namespace mpl { template<typename> struct sequence_tag; template< typename A0 > struct sequence_tag<hpx::util::tuple<A0> > { typedef fusion::fusion_sequence_tag type; }; template< typename A0 > struct sequence_tag< hpx::util::tuple<A0> const > { typedef fusion::fusion_sequence_tag type; }; } }
namespace boost { namespace serialization
{
    
    template <typename T0>
    struct is_bitwise_serializable<
            hpx::util::tuple<T0> >
      : hpx::util::detail::sequence_is_bitwise_serializable<
            hpx::util::tuple<T0> >
    {};
    
    template <typename Archive, typename T0>
    BOOST_FORCEINLINE void serialize(Archive& ar,
        hpx::util::tuple<T0>& t,
        unsigned int const version)
    {
        hpx::util::serialize_sequence(ar, t);
    }
}}
namespace hpx { namespace util
{
    
    template <typename A0 , typename A1>
    struct tuple<A0 , A1>
    {
        typedef A0 member_type0; A0 a0; typedef A1 member_type1; A1 a1;
        template <int E>
        typename detail::tuple_element<E, tuple>::rtype
        get() BOOST_NOEXCEPT
        {
            return detail::tuple_element<E, tuple>::get(*this);
        }
        template <int E>
        BOOST_CONSTEXPR
        typename detail::tuple_element<E, tuple const>::crtype
        get() const BOOST_NOEXCEPT
        {
            return detail::tuple_element<E, tuple const>::get(*this);
        }
        
        tuple()
          : a0() , a1()
        {}
        template <typename Arg0 , typename Arg1>
        tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1)
          : a0(boost::forward<Arg0>( arg0 )) , a1(boost::forward<Arg1>( arg1 ))
        {}
        
        tuple(tuple const& other)
          : a0( detail::copy_construct< A0 , typename boost::add_const<A0>::type >::call(other.a0)) , a1( detail::copy_construct< A1 , typename boost::add_const<A1>::type >::call(other.a1))
        {}
        tuple(BOOST_RV_REF(tuple) other)
          : a0(boost::forward<A0>( other.a0)) , a1(boost::forward<A1>( other.a1))
        {}
        template <typename T0 , typename T1>
        tuple(tuple<T0 , T1> const& other)
          : a0( detail::copy_construct< A0 , typename boost::add_const<T0>::type >::call(other.a0)) , a1( detail::copy_construct< A1 , typename boost::add_const<T1>::type >::call(other.a1))
        {}
        template <typename T0 , typename T1>
        tuple(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple<T0 , T1>
            ))) other)
          : a0(boost::forward<T0>( other.a0)) , a1(boost::forward<T1>( other.a1))
        {}
        template <typename U1, typename U2>
        tuple(std::pair<U1, U2> const& other)
          : a0(other.first)
          , a1(other.second)
        {}
        template <typename U1, typename U2>
        tuple(BOOST_RV_REF(HPX_UTIL_STRIP((std::pair<U1, U2>))) other)
          : a0(boost::move(other.first))
          , a1(boost::move(other.second))
        {}
        
        tuple& operator=(BOOST_COPY_ASSIGN_REF(tuple) other)
        {
            a0 = other.a0; a1 = other.a1;;
            return *this;
        }
        tuple& operator=(BOOST_RV_REF(tuple) other)
        {
            a0 = boost::forward<A0>( other.a0); a1 = boost::forward<A1>( other.a1);;
            return *this;
        }
        template <typename T0 , typename T1>
        tuple& operator=(BOOST_COPY_ASSIGN_REF(HPX_UTIL_STRIP((
                tuple<T0 , T1>
            ))) other)
        {
            a0 = other.a0; a1 = other.a1;;
            return *this;
        }
        template <typename T0 , typename T1>
        tuple& operator=(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple<T0 , T1>
            ))) other)
        {
            a0 = boost::forward<T0>( other.a0); a1 = boost::forward<T1>( other.a1);;
            return *this;
        }
        template <typename U1, typename U2>
        tuple& operator=(BOOST_COPY_ASSIGN_REF(HPX_UTIL_STRIP((
                std::pair<U1, U2>
            ))) other)
        {
            a0 = other.first;
            a1 = other.second;
            return *this;
        }
        template <typename U1, typename U2>
        tuple& operator=(BOOST_RV_REF(HPX_UTIL_STRIP((
                std::pair<U1, U2>
            ))) other)
        {
            a0 = boost::move(other.first);
            a1 = boost::move(other.second);
            return *this;
        }
        void swap(tuple& other)
        {
            boost::swap(a0, other.a0); boost::swap(a1, other.a1);;
        }
        typedef boost::mpl::int_<2> size_type;
        static const int size_value = 2;
    private:
        BOOST_COPYABLE_AND_MOVABLE(tuple);
    };
    template <typename A0 , typename A1>
    struct tuple_size<tuple<A0 , A1> >
    {
        static const std::size_t value = 2;
    };
    
    template <typename Arg0 , typename Arg1>
    BOOST_FORCEINLINE
    tuple<typename util::decay<Arg0>::type , typename util::decay<Arg1>::type>
    make_tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1)
    {
        return tuple<typename util::decay<Arg0>::type , typename util::decay<Arg1>::type>(
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ));
    }
    
    template <typename Arg0 , typename Arg1>
    BOOST_FORCEINLINE
    tuple<typename detail::add_rvalue_reference<Arg0>::type , typename detail::add_rvalue_reference<Arg1>::type>
    forward_as_tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1) BOOST_NOEXCEPT
    {
        return tuple<
                typename detail::add_rvalue_reference<Arg0>::type , typename detail::add_rvalue_reference<Arg1>::type>(
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ));
    }
    
    template <typename Arg0 , typename Arg1>
    BOOST_FORCEINLINE
    tuple<Arg0 & , Arg1 &>
    tie(Arg0 & arg0 , Arg1 & arg1) BOOST_NOEXCEPT
    {
        return tuple<
                Arg0 & , Arg1 &>(
            arg0 , arg1);
    }
    
    template <typename T0, typename T1> typename boost::lazy_enable_if_c< 0 == util::decay<T0>::type::size_value + util::decay<T1>::type::size_value , detail::tuple_cat_result<T0, T1> >::type tuple_cat(BOOST_FWD_REF(T0) t0, BOOST_FWD_REF(T1) t1) { typedef typename detail::tuple_cat_result<T0, T1>::type result_type; return result_type(); } template <typename T0, typename T1> typename boost::lazy_enable_if_c< 1 == util::decay<T0>::type::size_value + util::decay<T1>::type::size_value , detail::tuple_cat_result<T0, T1> >::type tuple_cat(BOOST_FWD_REF(T0) t0, BOOST_FWD_REF(T1) t1) { typedef typename detail::tuple_cat_result<T0, T1>::type result_type; return result_type(detail::tuple_cat_element< 0, T0, T1>::call(t0, t1)); } template <typename T0, typename T1> typename boost::lazy_enable_if_c< 2 == util::decay<T0>::type::size_value + util::decay<T1>::type::size_value , detail::tuple_cat_result<T0, T1> >::type tuple_cat(BOOST_FWD_REF(T0) t0, BOOST_FWD_REF(T1) t1) { typedef typename detail::tuple_cat_result<T0, T1>::type result_type; return result_type(detail::tuple_cat_element< 0, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 1, T0, T1>::call(t0, t1)); } template <typename T0, typename T1> typename boost::lazy_enable_if_c< 3 == util::decay<T0>::type::size_value + util::decay<T1>::type::size_value , detail::tuple_cat_result<T0, T1> >::type tuple_cat(BOOST_FWD_REF(T0) t0, BOOST_FWD_REF(T1) t1) { typedef typename detail::tuple_cat_result<T0, T1>::type result_type; return result_type(detail::tuple_cat_element< 0, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 1, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 2, T0, T1>::call(t0, t1)); } template <typename T0, typename T1> typename boost::lazy_enable_if_c< 4 == util::decay<T0>::type::size_value + util::decay<T1>::type::size_value , detail::tuple_cat_result<T0, T1> >::type tuple_cat(BOOST_FWD_REF(T0) t0, BOOST_FWD_REF(T1) t1) { typedef typename detail::tuple_cat_result<T0, T1>::type result_type; return result_type(detail::tuple_cat_element< 0, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 1, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 2, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 3, T0, T1>::call(t0, t1)); } template <typename T0, typename T1> typename boost::lazy_enable_if_c< 5 == util::decay<T0>::type::size_value + util::decay<T1>::type::size_value , detail::tuple_cat_result<T0, T1> >::type tuple_cat(BOOST_FWD_REF(T0) t0, BOOST_FWD_REF(T1) t1) { typedef typename detail::tuple_cat_result<T0, T1>::type result_type; return result_type(detail::tuple_cat_element< 0, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 1, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 2, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 3, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 4, T0, T1>::call(t0, t1)); } template <typename T0, typename T1> typename boost::lazy_enable_if_c< 6 == util::decay<T0>::type::size_value + util::decay<T1>::type::size_value , detail::tuple_cat_result<T0, T1> >::type tuple_cat(BOOST_FWD_REF(T0) t0, BOOST_FWD_REF(T1) t1) { typedef typename detail::tuple_cat_result<T0, T1>::type result_type; return result_type(detail::tuple_cat_element< 0, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 1, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 2, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 3, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 4, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 5, T0, T1>::call(t0, t1)); } template <typename T0, typename T1> typename boost::lazy_enable_if_c< 7 == util::decay<T0>::type::size_value + util::decay<T1>::type::size_value , detail::tuple_cat_result<T0, T1> >::type tuple_cat(BOOST_FWD_REF(T0) t0, BOOST_FWD_REF(T1) t1) { typedef typename detail::tuple_cat_result<T0, T1>::type result_type; return result_type(detail::tuple_cat_element< 0, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 1, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 2, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 3, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 4, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 5, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 6, T0, T1>::call(t0, t1)); } template <typename T0, typename T1> typename boost::lazy_enable_if_c< 8 == util::decay<T0>::type::size_value + util::decay<T1>::type::size_value , detail::tuple_cat_result<T0, T1> >::type tuple_cat(BOOST_FWD_REF(T0) t0, BOOST_FWD_REF(T1) t1) { typedef typename detail::tuple_cat_result<T0, T1>::type result_type; return result_type(detail::tuple_cat_element< 0, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 1, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 2, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 3, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 4, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 5, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 6, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 7, T0, T1>::call(t0, t1)); } template <typename T0, typename T1> typename boost::lazy_enable_if_c< 9 == util::decay<T0>::type::size_value + util::decay<T1>::type::size_value , detail::tuple_cat_result<T0, T1> >::type tuple_cat(BOOST_FWD_REF(T0) t0, BOOST_FWD_REF(T1) t1) { typedef typename detail::tuple_cat_result<T0, T1>::type result_type; return result_type(detail::tuple_cat_element< 0, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 1, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 2, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 3, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 4, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 5, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 6, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 7, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 8, T0, T1>::call(t0, t1)); } template <typename T0, typename T1> typename boost::lazy_enable_if_c< 10 == util::decay<T0>::type::size_value + util::decay<T1>::type::size_value , detail::tuple_cat_result<T0, T1> >::type tuple_cat(BOOST_FWD_REF(T0) t0, BOOST_FWD_REF(T1) t1) { typedef typename detail::tuple_cat_result<T0, T1>::type result_type; return result_type(detail::tuple_cat_element< 0, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 1, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 2, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 3, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 4, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 5, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 6, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 7, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 8, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 9, T0, T1>::call(t0, t1)); } template <typename T0, typename T1> typename boost::lazy_enable_if_c< 11 == util::decay<T0>::type::size_value + util::decay<T1>::type::size_value , detail::tuple_cat_result<T0, T1> >::type tuple_cat(BOOST_FWD_REF(T0) t0, BOOST_FWD_REF(T1) t1) { typedef typename detail::tuple_cat_result<T0, T1>::type result_type; return result_type(detail::tuple_cat_element< 0, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 1, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 2, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 3, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 4, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 5, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 6, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 7, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 8, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 9, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 10, T0, T1>::call(t0, t1)); } template <typename T0, typename T1> typename boost::lazy_enable_if_c< 12 == util::decay<T0>::type::size_value + util::decay<T1>::type::size_value , detail::tuple_cat_result<T0, T1> >::type tuple_cat(BOOST_FWD_REF(T0) t0, BOOST_FWD_REF(T1) t1) { typedef typename detail::tuple_cat_result<T0, T1>::type result_type; return result_type(detail::tuple_cat_element< 0, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 1, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 2, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 3, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 4, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 5, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 6, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 7, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 8, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 9, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 10, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 11, T0, T1>::call(t0, t1)); }
}}
namespace boost { namespace fusion { namespace traits { template< typename A0, typename A1 > struct tag_of<hpx::util::tuple<A0, A1> > { typedef struct_tag type; }; template< typename A0, typename A1 > struct tag_of<hpx::util::tuple<A0, A1> const> { typedef struct_tag type; }; } namespace extension { template< typename A0, typename A1 > struct access::struct_member< hpx::util::tuple<A0, A1> , 0 > { typedef A0 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a0; } }; }; template< typename A0, typename A1 > struct struct_member_name< hpx::util::tuple<A0, A1> , 0 > { typedef char const* type; static type call() { return "a0"; } }; template< typename A0, typename A1 > struct access::struct_member< hpx::util::tuple<A0, A1> , 1 > { typedef A1 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a1; } }; }; template< typename A0, typename A1 > struct struct_member_name< hpx::util::tuple<A0, A1> , 1 > { typedef char const* type; static type call() { return "a1"; } }; template< typename A0, typename A1 > struct struct_size<hpx::util::tuple<A0, A1> > : mpl::int_<2> {}; template< typename A0, typename A1 > struct struct_is_view< hpx::util::tuple<A0, A1> > : mpl::false_ {}; } } namespace mpl { template<typename> struct sequence_tag; template< typename A0, typename A1 > struct sequence_tag<hpx::util::tuple<A0, A1> > { typedef fusion::fusion_sequence_tag type; }; template< typename A0, typename A1 > struct sequence_tag< hpx::util::tuple<A0, A1> const > { typedef fusion::fusion_sequence_tag type; }; } }
namespace boost { namespace serialization
{
    
    template <typename T0 , typename T1>
    struct is_bitwise_serializable<
            hpx::util::tuple<T0 , T1> >
      : hpx::util::detail::sequence_is_bitwise_serializable<
            hpx::util::tuple<T0 , T1> >
    {};
    
    template <typename Archive, typename T0 , typename T1>
    BOOST_FORCEINLINE void serialize(Archive& ar,
        hpx::util::tuple<T0 , T1>& t,
        unsigned int const version)
    {
        hpx::util::serialize_sequence(ar, t);
    }
}}
namespace hpx { namespace util
{
    
    template <typename A0 , typename A1 , typename A2>
    struct tuple<A0 , A1 , A2>
    {
        typedef A0 member_type0; A0 a0; typedef A1 member_type1; A1 a1; typedef A2 member_type2; A2 a2;
        template <int E>
        typename detail::tuple_element<E, tuple>::rtype
        get() BOOST_NOEXCEPT
        {
            return detail::tuple_element<E, tuple>::get(*this);
        }
        template <int E>
        BOOST_CONSTEXPR
        typename detail::tuple_element<E, tuple const>::crtype
        get() const BOOST_NOEXCEPT
        {
            return detail::tuple_element<E, tuple const>::get(*this);
        }
        
        tuple()
          : a0() , a1() , a2()
        {}
        template <typename Arg0 , typename Arg1 , typename Arg2>
        tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2)
          : a0(boost::forward<Arg0>( arg0 )) , a1(boost::forward<Arg1>( arg1 )) , a2(boost::forward<Arg2>( arg2 ))
        {}
        
        tuple(tuple const& other)
          : a0( detail::copy_construct< A0 , typename boost::add_const<A0>::type >::call(other.a0)) , a1( detail::copy_construct< A1 , typename boost::add_const<A1>::type >::call(other.a1)) , a2( detail::copy_construct< A2 , typename boost::add_const<A2>::type >::call(other.a2))
        {}
        tuple(BOOST_RV_REF(tuple) other)
          : a0(boost::forward<A0>( other.a0)) , a1(boost::forward<A1>( other.a1)) , a2(boost::forward<A2>( other.a2))
        {}
        template <typename T0 , typename T1 , typename T2>
        tuple(tuple<T0 , T1 , T2> const& other)
          : a0( detail::copy_construct< A0 , typename boost::add_const<T0>::type >::call(other.a0)) , a1( detail::copy_construct< A1 , typename boost::add_const<T1>::type >::call(other.a1)) , a2( detail::copy_construct< A2 , typename boost::add_const<T2>::type >::call(other.a2))
        {}
        template <typename T0 , typename T1 , typename T2>
        tuple(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple<T0 , T1 , T2>
            ))) other)
          : a0(boost::forward<T0>( other.a0)) , a1(boost::forward<T1>( other.a1)) , a2(boost::forward<T2>( other.a2))
        {}
        
        tuple& operator=(BOOST_COPY_ASSIGN_REF(tuple) other)
        {
            a0 = other.a0; a1 = other.a1; a2 = other.a2;;
            return *this;
        }
        tuple& operator=(BOOST_RV_REF(tuple) other)
        {
            a0 = boost::forward<A0>( other.a0); a1 = boost::forward<A1>( other.a1); a2 = boost::forward<A2>( other.a2);;
            return *this;
        }
        template <typename T0 , typename T1 , typename T2>
        tuple& operator=(BOOST_COPY_ASSIGN_REF(HPX_UTIL_STRIP((
                tuple<T0 , T1 , T2>
            ))) other)
        {
            a0 = other.a0; a1 = other.a1; a2 = other.a2;;
            return *this;
        }
        template <typename T0 , typename T1 , typename T2>
        tuple& operator=(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple<T0 , T1 , T2>
            ))) other)
        {
            a0 = boost::forward<T0>( other.a0); a1 = boost::forward<T1>( other.a1); a2 = boost::forward<T2>( other.a2);;
            return *this;
        }
        void swap(tuple& other)
        {
            boost::swap(a0, other.a0); boost::swap(a1, other.a1); boost::swap(a2, other.a2);;
        }
        typedef boost::mpl::int_<3> size_type;
        static const int size_value = 3;
    private:
        BOOST_COPYABLE_AND_MOVABLE(tuple);
    };
    template <typename A0 , typename A1 , typename A2>
    struct tuple_size<tuple<A0 , A1 , A2> >
    {
        static const std::size_t value = 3;
    };
    
    template <typename Arg0 , typename Arg1 , typename Arg2>
    BOOST_FORCEINLINE
    tuple<typename util::decay<Arg0>::type , typename util::decay<Arg1>::type , typename util::decay<Arg2>::type>
    make_tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2)
    {
        return tuple<typename util::decay<Arg0>::type , typename util::decay<Arg1>::type , typename util::decay<Arg2>::type>(
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ));
    }
    
    template <typename Arg0 , typename Arg1 , typename Arg2>
    BOOST_FORCEINLINE
    tuple<typename detail::add_rvalue_reference<Arg0>::type , typename detail::add_rvalue_reference<Arg1>::type , typename detail::add_rvalue_reference<Arg2>::type>
    forward_as_tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2) BOOST_NOEXCEPT
    {
        return tuple<
                typename detail::add_rvalue_reference<Arg0>::type , typename detail::add_rvalue_reference<Arg1>::type , typename detail::add_rvalue_reference<Arg2>::type>(
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ));
    }
    
    template <typename Arg0 , typename Arg1 , typename Arg2>
    BOOST_FORCEINLINE
    tuple<Arg0 & , Arg1 & , Arg2 &>
    tie(Arg0 & arg0 , Arg1 & arg1 , Arg2 & arg2) BOOST_NOEXCEPT
    {
        return tuple<
                Arg0 & , Arg1 & , Arg2 &>(
            arg0 , arg1 , arg2);
    }
    
    
    template <typename T0 , typename T1 , typename T2>
    typename detail::tuple_cat_result<
        typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type
    >::type
    tuple_cat(BOOST_FWD_REF(T0) t0 , BOOST_FWD_REF(T1) t1 , BOOST_FWD_REF(T2) t2)
    {
        typedef
            typename detail::tuple_cat_result<T0, T1>::type
            head_type;
        head_type head =
            tuple_cat(boost::forward<T0>(t0), boost::forward<T1>(t1));
        return tuple_cat(boost::move(head)
                , boost::forward<T2>(t2));
    }
}}
namespace boost { namespace fusion { namespace traits { template< typename A0, typename A1, typename A2 > struct tag_of<hpx::util::tuple<A0, A1, A2> > { typedef struct_tag type; }; template< typename A0, typename A1, typename A2 > struct tag_of<hpx::util::tuple<A0, A1, A2> const> { typedef struct_tag type; }; } namespace extension { template< typename A0, typename A1, typename A2 > struct access::struct_member< hpx::util::tuple<A0, A1, A2> , 0 > { typedef A0 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a0; } }; }; template< typename A0, typename A1, typename A2 > struct struct_member_name< hpx::util::tuple<A0, A1, A2> , 0 > { typedef char const* type; static type call() { return "a0"; } }; template< typename A0, typename A1, typename A2 > struct access::struct_member< hpx::util::tuple<A0, A1, A2> , 1 > { typedef A1 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a1; } }; }; template< typename A0, typename A1, typename A2 > struct struct_member_name< hpx::util::tuple<A0, A1, A2> , 1 > { typedef char const* type; static type call() { return "a1"; } }; template< typename A0, typename A1, typename A2 > struct access::struct_member< hpx::util::tuple<A0, A1, A2> , 2 > { typedef A2 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a2; } }; }; template< typename A0, typename A1, typename A2 > struct struct_member_name< hpx::util::tuple<A0, A1, A2> , 2 > { typedef char const* type; static type call() { return "a2"; } }; template< typename A0, typename A1, typename A2 > struct struct_size<hpx::util::tuple<A0, A1, A2> > : mpl::int_<3> {}; template< typename A0, typename A1, typename A2 > struct struct_is_view< hpx::util::tuple<A0, A1, A2> > : mpl::false_ {}; } } namespace mpl { template<typename> struct sequence_tag; template< typename A0, typename A1, typename A2 > struct sequence_tag<hpx::util::tuple<A0, A1, A2> > { typedef fusion::fusion_sequence_tag type; }; template< typename A0, typename A1, typename A2 > struct sequence_tag< hpx::util::tuple<A0, A1, A2> const > { typedef fusion::fusion_sequence_tag type; }; } }
namespace boost { namespace serialization
{
    
    template <typename T0 , typename T1 , typename T2>
    struct is_bitwise_serializable<
            hpx::util::tuple<T0 , T1 , T2> >
      : hpx::util::detail::sequence_is_bitwise_serializable<
            hpx::util::tuple<T0 , T1 , T2> >
    {};
    
    template <typename Archive, typename T0 , typename T1 , typename T2>
    BOOST_FORCEINLINE void serialize(Archive& ar,
        hpx::util::tuple<T0 , T1 , T2>& t,
        unsigned int const version)
    {
        hpx::util::serialize_sequence(ar, t);
    }
}}
namespace hpx { namespace util
{
    
    template <typename A0 , typename A1 , typename A2 , typename A3>
    struct tuple<A0 , A1 , A2 , A3>
    {
        typedef A0 member_type0; A0 a0; typedef A1 member_type1; A1 a1; typedef A2 member_type2; A2 a2; typedef A3 member_type3; A3 a3;
        template <int E>
        typename detail::tuple_element<E, tuple>::rtype
        get() BOOST_NOEXCEPT
        {
            return detail::tuple_element<E, tuple>::get(*this);
        }
        template <int E>
        BOOST_CONSTEXPR
        typename detail::tuple_element<E, tuple const>::crtype
        get() const BOOST_NOEXCEPT
        {
            return detail::tuple_element<E, tuple const>::get(*this);
        }
        
        tuple()
          : a0() , a1() , a2() , a3()
        {}
        template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
        tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3)
          : a0(boost::forward<Arg0>( arg0 )) , a1(boost::forward<Arg1>( arg1 )) , a2(boost::forward<Arg2>( arg2 )) , a3(boost::forward<Arg3>( arg3 ))
        {}
        
        tuple(tuple const& other)
          : a0( detail::copy_construct< A0 , typename boost::add_const<A0>::type >::call(other.a0)) , a1( detail::copy_construct< A1 , typename boost::add_const<A1>::type >::call(other.a1)) , a2( detail::copy_construct< A2 , typename boost::add_const<A2>::type >::call(other.a2)) , a3( detail::copy_construct< A3 , typename boost::add_const<A3>::type >::call(other.a3))
        {}
        tuple(BOOST_RV_REF(tuple) other)
          : a0(boost::forward<A0>( other.a0)) , a1(boost::forward<A1>( other.a1)) , a2(boost::forward<A2>( other.a2)) , a3(boost::forward<A3>( other.a3))
        {}
        template <typename T0 , typename T1 , typename T2 , typename T3>
        tuple(tuple<T0 , T1 , T2 , T3> const& other)
          : a0( detail::copy_construct< A0 , typename boost::add_const<T0>::type >::call(other.a0)) , a1( detail::copy_construct< A1 , typename boost::add_const<T1>::type >::call(other.a1)) , a2( detail::copy_construct< A2 , typename boost::add_const<T2>::type >::call(other.a2)) , a3( detail::copy_construct< A3 , typename boost::add_const<T3>::type >::call(other.a3))
        {}
        template <typename T0 , typename T1 , typename T2 , typename T3>
        tuple(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple<T0 , T1 , T2 , T3>
            ))) other)
          : a0(boost::forward<T0>( other.a0)) , a1(boost::forward<T1>( other.a1)) , a2(boost::forward<T2>( other.a2)) , a3(boost::forward<T3>( other.a3))
        {}
        
        tuple& operator=(BOOST_COPY_ASSIGN_REF(tuple) other)
        {
            a0 = other.a0; a1 = other.a1; a2 = other.a2; a3 = other.a3;;
            return *this;
        }
        tuple& operator=(BOOST_RV_REF(tuple) other)
        {
            a0 = boost::forward<A0>( other.a0); a1 = boost::forward<A1>( other.a1); a2 = boost::forward<A2>( other.a2); a3 = boost::forward<A3>( other.a3);;
            return *this;
        }
        template <typename T0 , typename T1 , typename T2 , typename T3>
        tuple& operator=(BOOST_COPY_ASSIGN_REF(HPX_UTIL_STRIP((
                tuple<T0 , T1 , T2 , T3>
            ))) other)
        {
            a0 = other.a0; a1 = other.a1; a2 = other.a2; a3 = other.a3;;
            return *this;
        }
        template <typename T0 , typename T1 , typename T2 , typename T3>
        tuple& operator=(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple<T0 , T1 , T2 , T3>
            ))) other)
        {
            a0 = boost::forward<T0>( other.a0); a1 = boost::forward<T1>( other.a1); a2 = boost::forward<T2>( other.a2); a3 = boost::forward<T3>( other.a3);;
            return *this;
        }
        void swap(tuple& other)
        {
            boost::swap(a0, other.a0); boost::swap(a1, other.a1); boost::swap(a2, other.a2); boost::swap(a3, other.a3);;
        }
        typedef boost::mpl::int_<4> size_type;
        static const int size_value = 4;
    private:
        BOOST_COPYABLE_AND_MOVABLE(tuple);
    };
    template <typename A0 , typename A1 , typename A2 , typename A3>
    struct tuple_size<tuple<A0 , A1 , A2 , A3> >
    {
        static const std::size_t value = 4;
    };
    
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    BOOST_FORCEINLINE
    tuple<typename util::decay<Arg0>::type , typename util::decay<Arg1>::type , typename util::decay<Arg2>::type , typename util::decay<Arg3>::type>
    make_tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3)
    {
        return tuple<typename util::decay<Arg0>::type , typename util::decay<Arg1>::type , typename util::decay<Arg2>::type , typename util::decay<Arg3>::type>(
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ));
    }
    
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    BOOST_FORCEINLINE
    tuple<typename detail::add_rvalue_reference<Arg0>::type , typename detail::add_rvalue_reference<Arg1>::type , typename detail::add_rvalue_reference<Arg2>::type , typename detail::add_rvalue_reference<Arg3>::type>
    forward_as_tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3) BOOST_NOEXCEPT
    {
        return tuple<
                typename detail::add_rvalue_reference<Arg0>::type , typename detail::add_rvalue_reference<Arg1>::type , typename detail::add_rvalue_reference<Arg2>::type , typename detail::add_rvalue_reference<Arg3>::type>(
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ));
    }
    
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    BOOST_FORCEINLINE
    tuple<Arg0 & , Arg1 & , Arg2 & , Arg3 &>
    tie(Arg0 & arg0 , Arg1 & arg1 , Arg2 & arg2 , Arg3 & arg3) BOOST_NOEXCEPT
    {
        return tuple<
                Arg0 & , Arg1 & , Arg2 & , Arg3 &>(
            arg0 , arg1 , arg2 , arg3);
    }
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3>
    typename detail::tuple_cat_result<
        typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type
    >::type
    tuple_cat(BOOST_FWD_REF(T0) t0 , BOOST_FWD_REF(T1) t1 , BOOST_FWD_REF(T2) t2 , BOOST_FWD_REF(T3) t3)
    {
        typedef
            typename detail::tuple_cat_result<T0, T1>::type
            head_type;
        head_type head =
            tuple_cat(boost::forward<T0>(t0), boost::forward<T1>(t1));
        return tuple_cat(boost::move(head)
                , boost::forward<T2>(t2) , boost::forward<T3>(t3));
    }
}}
namespace boost { namespace fusion { namespace traits { template< typename A0, typename A1, typename A2, typename A3 > struct tag_of<hpx::util::tuple<A0, A1, A2, A3> > { typedef struct_tag type; }; template< typename A0, typename A1, typename A2, typename A3 > struct tag_of<hpx::util::tuple<A0, A1, A2, A3> const> { typedef struct_tag type; }; } namespace extension { template< typename A0, typename A1, typename A2, typename A3 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3> , 0 > { typedef A0 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a0; } }; }; template< typename A0, typename A1, typename A2, typename A3 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3> , 0 > { typedef char const* type; static type call() { return "a0"; } }; template< typename A0, typename A1, typename A2, typename A3 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3> , 1 > { typedef A1 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a1; } }; }; template< typename A0, typename A1, typename A2, typename A3 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3> , 1 > { typedef char const* type; static type call() { return "a1"; } }; template< typename A0, typename A1, typename A2, typename A3 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3> , 2 > { typedef A2 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a2; } }; }; template< typename A0, typename A1, typename A2, typename A3 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3> , 2 > { typedef char const* type; static type call() { return "a2"; } }; template< typename A0, typename A1, typename A2, typename A3 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3> , 3 > { typedef A3 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a3; } }; }; template< typename A0, typename A1, typename A2, typename A3 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3> , 3 > { typedef char const* type; static type call() { return "a3"; } }; template< typename A0, typename A1, typename A2, typename A3 > struct struct_size<hpx::util::tuple<A0, A1, A2, A3> > : mpl::int_<4> {}; template< typename A0, typename A1, typename A2, typename A3 > struct struct_is_view< hpx::util::tuple<A0, A1, A2, A3> > : mpl::false_ {}; } } namespace mpl { template<typename> struct sequence_tag; template< typename A0, typename A1, typename A2, typename A3 > struct sequence_tag<hpx::util::tuple<A0, A1, A2, A3> > { typedef fusion::fusion_sequence_tag type; }; template< typename A0, typename A1, typename A2, typename A3 > struct sequence_tag< hpx::util::tuple<A0, A1, A2, A3> const > { typedef fusion::fusion_sequence_tag type; }; } }
namespace boost { namespace serialization
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3>
    struct is_bitwise_serializable<
            hpx::util::tuple<T0 , T1 , T2 , T3> >
      : hpx::util::detail::sequence_is_bitwise_serializable<
            hpx::util::tuple<T0 , T1 , T2 , T3> >
    {};
    
    template <typename Archive, typename T0 , typename T1 , typename T2 , typename T3>
    BOOST_FORCEINLINE void serialize(Archive& ar,
        hpx::util::tuple<T0 , T1 , T2 , T3>& t,
        unsigned int const version)
    {
        hpx::util::serialize_sequence(ar, t);
    }
}}
namespace hpx { namespace util
{
    
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4>
    struct tuple<A0 , A1 , A2 , A3 , A4>
    {
        typedef A0 member_type0; A0 a0; typedef A1 member_type1; A1 a1; typedef A2 member_type2; A2 a2; typedef A3 member_type3; A3 a3; typedef A4 member_type4; A4 a4;
        template <int E>
        typename detail::tuple_element<E, tuple>::rtype
        get() BOOST_NOEXCEPT
        {
            return detail::tuple_element<E, tuple>::get(*this);
        }
        template <int E>
        BOOST_CONSTEXPR
        typename detail::tuple_element<E, tuple const>::crtype
        get() const BOOST_NOEXCEPT
        {
            return detail::tuple_element<E, tuple const>::get(*this);
        }
        
        tuple()
          : a0() , a1() , a2() , a3() , a4()
        {}
        template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
        tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4)
          : a0(boost::forward<Arg0>( arg0 )) , a1(boost::forward<Arg1>( arg1 )) , a2(boost::forward<Arg2>( arg2 )) , a3(boost::forward<Arg3>( arg3 )) , a4(boost::forward<Arg4>( arg4 ))
        {}
        
        tuple(tuple const& other)
          : a0( detail::copy_construct< A0 , typename boost::add_const<A0>::type >::call(other.a0)) , a1( detail::copy_construct< A1 , typename boost::add_const<A1>::type >::call(other.a1)) , a2( detail::copy_construct< A2 , typename boost::add_const<A2>::type >::call(other.a2)) , a3( detail::copy_construct< A3 , typename boost::add_const<A3>::type >::call(other.a3)) , a4( detail::copy_construct< A4 , typename boost::add_const<A4>::type >::call(other.a4))
        {}
        tuple(BOOST_RV_REF(tuple) other)
          : a0(boost::forward<A0>( other.a0)) , a1(boost::forward<A1>( other.a1)) , a2(boost::forward<A2>( other.a2)) , a3(boost::forward<A3>( other.a3)) , a4(boost::forward<A4>( other.a4))
        {}
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4>
        tuple(tuple<T0 , T1 , T2 , T3 , T4> const& other)
          : a0( detail::copy_construct< A0 , typename boost::add_const<T0>::type >::call(other.a0)) , a1( detail::copy_construct< A1 , typename boost::add_const<T1>::type >::call(other.a1)) , a2( detail::copy_construct< A2 , typename boost::add_const<T2>::type >::call(other.a2)) , a3( detail::copy_construct< A3 , typename boost::add_const<T3>::type >::call(other.a3)) , a4( detail::copy_construct< A4 , typename boost::add_const<T4>::type >::call(other.a4))
        {}
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4>
        tuple(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple<T0 , T1 , T2 , T3 , T4>
            ))) other)
          : a0(boost::forward<T0>( other.a0)) , a1(boost::forward<T1>( other.a1)) , a2(boost::forward<T2>( other.a2)) , a3(boost::forward<T3>( other.a3)) , a4(boost::forward<T4>( other.a4))
        {}
        
        tuple& operator=(BOOST_COPY_ASSIGN_REF(tuple) other)
        {
            a0 = other.a0; a1 = other.a1; a2 = other.a2; a3 = other.a3; a4 = other.a4;;
            return *this;
        }
        tuple& operator=(BOOST_RV_REF(tuple) other)
        {
            a0 = boost::forward<A0>( other.a0); a1 = boost::forward<A1>( other.a1); a2 = boost::forward<A2>( other.a2); a3 = boost::forward<A3>( other.a3); a4 = boost::forward<A4>( other.a4);;
            return *this;
        }
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4>
        tuple& operator=(BOOST_COPY_ASSIGN_REF(HPX_UTIL_STRIP((
                tuple<T0 , T1 , T2 , T3 , T4>
            ))) other)
        {
            a0 = other.a0; a1 = other.a1; a2 = other.a2; a3 = other.a3; a4 = other.a4;;
            return *this;
        }
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4>
        tuple& operator=(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple<T0 , T1 , T2 , T3 , T4>
            ))) other)
        {
            a0 = boost::forward<T0>( other.a0); a1 = boost::forward<T1>( other.a1); a2 = boost::forward<T2>( other.a2); a3 = boost::forward<T3>( other.a3); a4 = boost::forward<T4>( other.a4);;
            return *this;
        }
        void swap(tuple& other)
        {
            boost::swap(a0, other.a0); boost::swap(a1, other.a1); boost::swap(a2, other.a2); boost::swap(a3, other.a3); boost::swap(a4, other.a4);;
        }
        typedef boost::mpl::int_<5> size_type;
        static const int size_value = 5;
    private:
        BOOST_COPYABLE_AND_MOVABLE(tuple);
    };
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4>
    struct tuple_size<tuple<A0 , A1 , A2 , A3 , A4> >
    {
        static const std::size_t value = 5;
    };
    
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    BOOST_FORCEINLINE
    tuple<typename util::decay<Arg0>::type , typename util::decay<Arg1>::type , typename util::decay<Arg2>::type , typename util::decay<Arg3>::type , typename util::decay<Arg4>::type>
    make_tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4)
    {
        return tuple<typename util::decay<Arg0>::type , typename util::decay<Arg1>::type , typename util::decay<Arg2>::type , typename util::decay<Arg3>::type , typename util::decay<Arg4>::type>(
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ));
    }
    
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    BOOST_FORCEINLINE
    tuple<typename detail::add_rvalue_reference<Arg0>::type , typename detail::add_rvalue_reference<Arg1>::type , typename detail::add_rvalue_reference<Arg2>::type , typename detail::add_rvalue_reference<Arg3>::type , typename detail::add_rvalue_reference<Arg4>::type>
    forward_as_tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4) BOOST_NOEXCEPT
    {
        return tuple<
                typename detail::add_rvalue_reference<Arg0>::type , typename detail::add_rvalue_reference<Arg1>::type , typename detail::add_rvalue_reference<Arg2>::type , typename detail::add_rvalue_reference<Arg3>::type , typename detail::add_rvalue_reference<Arg4>::type>(
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ));
    }
    
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    BOOST_FORCEINLINE
    tuple<Arg0 & , Arg1 & , Arg2 & , Arg3 & , Arg4 &>
    tie(Arg0 & arg0 , Arg1 & arg1 , Arg2 & arg2 , Arg3 & arg3 , Arg4 & arg4) BOOST_NOEXCEPT
    {
        return tuple<
                Arg0 & , Arg1 & , Arg2 & , Arg3 & , Arg4 &>(
            arg0 , arg1 , arg2 , arg3 , arg4);
    }
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4>
    typename detail::tuple_cat_result<
        typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type
    >::type
    tuple_cat(BOOST_FWD_REF(T0) t0 , BOOST_FWD_REF(T1) t1 , BOOST_FWD_REF(T2) t2 , BOOST_FWD_REF(T3) t3 , BOOST_FWD_REF(T4) t4)
    {
        typedef
            typename detail::tuple_cat_result<T0, T1>::type
            head_type;
        head_type head =
            tuple_cat(boost::forward<T0>(t0), boost::forward<T1>(t1));
        return tuple_cat(boost::move(head)
                , boost::forward<T2>(t2) , boost::forward<T3>(t3) , boost::forward<T4>(t4));
    }
}}
namespace boost { namespace fusion { namespace traits { template< typename A0, typename A1, typename A2, typename A3, typename A4 > struct tag_of<hpx::util::tuple<A0, A1, A2, A3, A4> > { typedef struct_tag type; }; template< typename A0, typename A1, typename A2, typename A3, typename A4 > struct tag_of<hpx::util::tuple<A0, A1, A2, A3, A4> const> { typedef struct_tag type; }; } namespace extension { template< typename A0, typename A1, typename A2, typename A3, typename A4 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4> , 0 > { typedef A0 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a0; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4> , 0 > { typedef char const* type; static type call() { return "a0"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4> , 1 > { typedef A1 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a1; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4> , 1 > { typedef char const* type; static type call() { return "a1"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4> , 2 > { typedef A2 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a2; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4> , 2 > { typedef char const* type; static type call() { return "a2"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4> , 3 > { typedef A3 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a3; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4> , 3 > { typedef char const* type; static type call() { return "a3"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4> , 4 > { typedef A4 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a4; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4> , 4 > { typedef char const* type; static type call() { return "a4"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4 > struct struct_size<hpx::util::tuple<A0, A1, A2, A3, A4> > : mpl::int_<5> {}; template< typename A0, typename A1, typename A2, typename A3, typename A4 > struct struct_is_view< hpx::util::tuple<A0, A1, A2, A3, A4> > : mpl::false_ {}; } } namespace mpl { template<typename> struct sequence_tag; template< typename A0, typename A1, typename A2, typename A3, typename A4 > struct sequence_tag<hpx::util::tuple<A0, A1, A2, A3, A4> > { typedef fusion::fusion_sequence_tag type; }; template< typename A0, typename A1, typename A2, typename A3, typename A4 > struct sequence_tag< hpx::util::tuple<A0, A1, A2, A3, A4> const > { typedef fusion::fusion_sequence_tag type; }; } }
namespace boost { namespace serialization
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4>
    struct is_bitwise_serializable<
            hpx::util::tuple<T0 , T1 , T2 , T3 , T4> >
      : hpx::util::detail::sequence_is_bitwise_serializable<
            hpx::util::tuple<T0 , T1 , T2 , T3 , T4> >
    {};
    
    template <typename Archive, typename T0 , typename T1 , typename T2 , typename T3 , typename T4>
    BOOST_FORCEINLINE void serialize(Archive& ar,
        hpx::util::tuple<T0 , T1 , T2 , T3 , T4>& t,
        unsigned int const version)
    {
        hpx::util::serialize_sequence(ar, t);
    }
}}
namespace hpx { namespace util
{
    
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5>
    struct tuple<A0 , A1 , A2 , A3 , A4 , A5>
    {
        typedef A0 member_type0; A0 a0; typedef A1 member_type1; A1 a1; typedef A2 member_type2; A2 a2; typedef A3 member_type3; A3 a3; typedef A4 member_type4; A4 a4; typedef A5 member_type5; A5 a5;
        template <int E>
        typename detail::tuple_element<E, tuple>::rtype
        get() BOOST_NOEXCEPT
        {
            return detail::tuple_element<E, tuple>::get(*this);
        }
        template <int E>
        BOOST_CONSTEXPR
        typename detail::tuple_element<E, tuple const>::crtype
        get() const BOOST_NOEXCEPT
        {
            return detail::tuple_element<E, tuple const>::get(*this);
        }
        
        tuple()
          : a0() , a1() , a2() , a3() , a4() , a5()
        {}
        template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
        tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5)
          : a0(boost::forward<Arg0>( arg0 )) , a1(boost::forward<Arg1>( arg1 )) , a2(boost::forward<Arg2>( arg2 )) , a3(boost::forward<Arg3>( arg3 )) , a4(boost::forward<Arg4>( arg4 )) , a5(boost::forward<Arg5>( arg5 ))
        {}
        
        tuple(tuple const& other)
          : a0( detail::copy_construct< A0 , typename boost::add_const<A0>::type >::call(other.a0)) , a1( detail::copy_construct< A1 , typename boost::add_const<A1>::type >::call(other.a1)) , a2( detail::copy_construct< A2 , typename boost::add_const<A2>::type >::call(other.a2)) , a3( detail::copy_construct< A3 , typename boost::add_const<A3>::type >::call(other.a3)) , a4( detail::copy_construct< A4 , typename boost::add_const<A4>::type >::call(other.a4)) , a5( detail::copy_construct< A5 , typename boost::add_const<A5>::type >::call(other.a5))
        {}
        tuple(BOOST_RV_REF(tuple) other)
          : a0(boost::forward<A0>( other.a0)) , a1(boost::forward<A1>( other.a1)) , a2(boost::forward<A2>( other.a2)) , a3(boost::forward<A3>( other.a3)) , a4(boost::forward<A4>( other.a4)) , a5(boost::forward<A5>( other.a5))
        {}
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5>
        tuple(tuple<T0 , T1 , T2 , T3 , T4 , T5> const& other)
          : a0( detail::copy_construct< A0 , typename boost::add_const<T0>::type >::call(other.a0)) , a1( detail::copy_construct< A1 , typename boost::add_const<T1>::type >::call(other.a1)) , a2( detail::copy_construct< A2 , typename boost::add_const<T2>::type >::call(other.a2)) , a3( detail::copy_construct< A3 , typename boost::add_const<T3>::type >::call(other.a3)) , a4( detail::copy_construct< A4 , typename boost::add_const<T4>::type >::call(other.a4)) , a5( detail::copy_construct< A5 , typename boost::add_const<T5>::type >::call(other.a5))
        {}
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5>
        tuple(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple<T0 , T1 , T2 , T3 , T4 , T5>
            ))) other)
          : a0(boost::forward<T0>( other.a0)) , a1(boost::forward<T1>( other.a1)) , a2(boost::forward<T2>( other.a2)) , a3(boost::forward<T3>( other.a3)) , a4(boost::forward<T4>( other.a4)) , a5(boost::forward<T5>( other.a5))
        {}
        
        tuple& operator=(BOOST_COPY_ASSIGN_REF(tuple) other)
        {
            a0 = other.a0; a1 = other.a1; a2 = other.a2; a3 = other.a3; a4 = other.a4; a5 = other.a5;;
            return *this;
        }
        tuple& operator=(BOOST_RV_REF(tuple) other)
        {
            a0 = boost::forward<A0>( other.a0); a1 = boost::forward<A1>( other.a1); a2 = boost::forward<A2>( other.a2); a3 = boost::forward<A3>( other.a3); a4 = boost::forward<A4>( other.a4); a5 = boost::forward<A5>( other.a5);;
            return *this;
        }
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5>
        tuple& operator=(BOOST_COPY_ASSIGN_REF(HPX_UTIL_STRIP((
                tuple<T0 , T1 , T2 , T3 , T4 , T5>
            ))) other)
        {
            a0 = other.a0; a1 = other.a1; a2 = other.a2; a3 = other.a3; a4 = other.a4; a5 = other.a5;;
            return *this;
        }
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5>
        tuple& operator=(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple<T0 , T1 , T2 , T3 , T4 , T5>
            ))) other)
        {
            a0 = boost::forward<T0>( other.a0); a1 = boost::forward<T1>( other.a1); a2 = boost::forward<T2>( other.a2); a3 = boost::forward<T3>( other.a3); a4 = boost::forward<T4>( other.a4); a5 = boost::forward<T5>( other.a5);;
            return *this;
        }
        void swap(tuple& other)
        {
            boost::swap(a0, other.a0); boost::swap(a1, other.a1); boost::swap(a2, other.a2); boost::swap(a3, other.a3); boost::swap(a4, other.a4); boost::swap(a5, other.a5);;
        }
        typedef boost::mpl::int_<6> size_type;
        static const int size_value = 6;
    private:
        BOOST_COPYABLE_AND_MOVABLE(tuple);
    };
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5>
    struct tuple_size<tuple<A0 , A1 , A2 , A3 , A4 , A5> >
    {
        static const std::size_t value = 6;
    };
    
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    BOOST_FORCEINLINE
    tuple<typename util::decay<Arg0>::type , typename util::decay<Arg1>::type , typename util::decay<Arg2>::type , typename util::decay<Arg3>::type , typename util::decay<Arg4>::type , typename util::decay<Arg5>::type>
    make_tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5)
    {
        return tuple<typename util::decay<Arg0>::type , typename util::decay<Arg1>::type , typename util::decay<Arg2>::type , typename util::decay<Arg3>::type , typename util::decay<Arg4>::type , typename util::decay<Arg5>::type>(
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ));
    }
    
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    BOOST_FORCEINLINE
    tuple<typename detail::add_rvalue_reference<Arg0>::type , typename detail::add_rvalue_reference<Arg1>::type , typename detail::add_rvalue_reference<Arg2>::type , typename detail::add_rvalue_reference<Arg3>::type , typename detail::add_rvalue_reference<Arg4>::type , typename detail::add_rvalue_reference<Arg5>::type>
    forward_as_tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5) BOOST_NOEXCEPT
    {
        return tuple<
                typename detail::add_rvalue_reference<Arg0>::type , typename detail::add_rvalue_reference<Arg1>::type , typename detail::add_rvalue_reference<Arg2>::type , typename detail::add_rvalue_reference<Arg3>::type , typename detail::add_rvalue_reference<Arg4>::type , typename detail::add_rvalue_reference<Arg5>::type>(
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ));
    }
    
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    BOOST_FORCEINLINE
    tuple<Arg0 & , Arg1 & , Arg2 & , Arg3 & , Arg4 & , Arg5 &>
    tie(Arg0 & arg0 , Arg1 & arg1 , Arg2 & arg2 , Arg3 & arg3 , Arg4 & arg4 , Arg5 & arg5) BOOST_NOEXCEPT
    {
        return tuple<
                Arg0 & , Arg1 & , Arg2 & , Arg3 & , Arg4 & , Arg5 &>(
            arg0 , arg1 , arg2 , arg3 , arg4 , arg5);
    }
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5>
    typename detail::tuple_cat_result<
        typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type
    >::type
    tuple_cat(BOOST_FWD_REF(T0) t0 , BOOST_FWD_REF(T1) t1 , BOOST_FWD_REF(T2) t2 , BOOST_FWD_REF(T3) t3 , BOOST_FWD_REF(T4) t4 , BOOST_FWD_REF(T5) t5)
    {
        typedef
            typename detail::tuple_cat_result<T0, T1>::type
            head_type;
        head_type head =
            tuple_cat(boost::forward<T0>(t0), boost::forward<T1>(t1));
        return tuple_cat(boost::move(head)
                , boost::forward<T2>(t2) , boost::forward<T3>(t3) , boost::forward<T4>(t4) , boost::forward<T5>(t5));
    }
}}
namespace boost { namespace fusion { namespace traits { template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5 > struct tag_of<hpx::util::tuple<A0, A1, A2, A3, A4, A5> > { typedef struct_tag type; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5 > struct tag_of<hpx::util::tuple<A0, A1, A2, A3, A4, A5> const> { typedef struct_tag type; }; } namespace extension { template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5> , 0 > { typedef A0 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a0; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5> , 0 > { typedef char const* type; static type call() { return "a0"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5> , 1 > { typedef A1 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a1; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5> , 1 > { typedef char const* type; static type call() { return "a1"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5> , 2 > { typedef A2 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a2; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5> , 2 > { typedef char const* type; static type call() { return "a2"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5> , 3 > { typedef A3 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a3; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5> , 3 > { typedef char const* type; static type call() { return "a3"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5> , 4 > { typedef A4 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a4; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5> , 4 > { typedef char const* type; static type call() { return "a4"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5> , 5 > { typedef A5 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a5; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5> , 5 > { typedef char const* type; static type call() { return "a5"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5 > struct struct_size<hpx::util::tuple<A0, A1, A2, A3, A4, A5> > : mpl::int_<6> {}; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5 > struct struct_is_view< hpx::util::tuple<A0, A1, A2, A3, A4, A5> > : mpl::false_ {}; } } namespace mpl { template<typename> struct sequence_tag; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5 > struct sequence_tag<hpx::util::tuple<A0, A1, A2, A3, A4, A5> > { typedef fusion::fusion_sequence_tag type; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5 > struct sequence_tag< hpx::util::tuple<A0, A1, A2, A3, A4, A5> const > { typedef fusion::fusion_sequence_tag type; }; } }
namespace boost { namespace serialization
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5>
    struct is_bitwise_serializable<
            hpx::util::tuple<T0 , T1 , T2 , T3 , T4 , T5> >
      : hpx::util::detail::sequence_is_bitwise_serializable<
            hpx::util::tuple<T0 , T1 , T2 , T3 , T4 , T5> >
    {};
    
    template <typename Archive, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5>
    BOOST_FORCEINLINE void serialize(Archive& ar,
        hpx::util::tuple<T0 , T1 , T2 , T3 , T4 , T5>& t,
        unsigned int const version)
    {
        hpx::util::serialize_sequence(ar, t);
    }
}}
namespace hpx { namespace util
{
    
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6>
    struct tuple<A0 , A1 , A2 , A3 , A4 , A5 , A6>
    {
        typedef A0 member_type0; A0 a0; typedef A1 member_type1; A1 a1; typedef A2 member_type2; A2 a2; typedef A3 member_type3; A3 a3; typedef A4 member_type4; A4 a4; typedef A5 member_type5; A5 a5; typedef A6 member_type6; A6 a6;
        template <int E>
        typename detail::tuple_element<E, tuple>::rtype
        get() BOOST_NOEXCEPT
        {
            return detail::tuple_element<E, tuple>::get(*this);
        }
        template <int E>
        BOOST_CONSTEXPR
        typename detail::tuple_element<E, tuple const>::crtype
        get() const BOOST_NOEXCEPT
        {
            return detail::tuple_element<E, tuple const>::get(*this);
        }
        
        tuple()
          : a0() , a1() , a2() , a3() , a4() , a5() , a6()
        {}
        template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
        tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6)
          : a0(boost::forward<Arg0>( arg0 )) , a1(boost::forward<Arg1>( arg1 )) , a2(boost::forward<Arg2>( arg2 )) , a3(boost::forward<Arg3>( arg3 )) , a4(boost::forward<Arg4>( arg4 )) , a5(boost::forward<Arg5>( arg5 )) , a6(boost::forward<Arg6>( arg6 ))
        {}
        
        tuple(tuple const& other)
          : a0( detail::copy_construct< A0 , typename boost::add_const<A0>::type >::call(other.a0)) , a1( detail::copy_construct< A1 , typename boost::add_const<A1>::type >::call(other.a1)) , a2( detail::copy_construct< A2 , typename boost::add_const<A2>::type >::call(other.a2)) , a3( detail::copy_construct< A3 , typename boost::add_const<A3>::type >::call(other.a3)) , a4( detail::copy_construct< A4 , typename boost::add_const<A4>::type >::call(other.a4)) , a5( detail::copy_construct< A5 , typename boost::add_const<A5>::type >::call(other.a5)) , a6( detail::copy_construct< A6 , typename boost::add_const<A6>::type >::call(other.a6))
        {}
        tuple(BOOST_RV_REF(tuple) other)
          : a0(boost::forward<A0>( other.a0)) , a1(boost::forward<A1>( other.a1)) , a2(boost::forward<A2>( other.a2)) , a3(boost::forward<A3>( other.a3)) , a4(boost::forward<A4>( other.a4)) , a5(boost::forward<A5>( other.a5)) , a6(boost::forward<A6>( other.a6))
        {}
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6>
        tuple(tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6> const& other)
          : a0( detail::copy_construct< A0 , typename boost::add_const<T0>::type >::call(other.a0)) , a1( detail::copy_construct< A1 , typename boost::add_const<T1>::type >::call(other.a1)) , a2( detail::copy_construct< A2 , typename boost::add_const<T2>::type >::call(other.a2)) , a3( detail::copy_construct< A3 , typename boost::add_const<T3>::type >::call(other.a3)) , a4( detail::copy_construct< A4 , typename boost::add_const<T4>::type >::call(other.a4)) , a5( detail::copy_construct< A5 , typename boost::add_const<T5>::type >::call(other.a5)) , a6( detail::copy_construct< A6 , typename boost::add_const<T6>::type >::call(other.a6))
        {}
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6>
        tuple(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6>
            ))) other)
          : a0(boost::forward<T0>( other.a0)) , a1(boost::forward<T1>( other.a1)) , a2(boost::forward<T2>( other.a2)) , a3(boost::forward<T3>( other.a3)) , a4(boost::forward<T4>( other.a4)) , a5(boost::forward<T5>( other.a5)) , a6(boost::forward<T6>( other.a6))
        {}
        
        tuple& operator=(BOOST_COPY_ASSIGN_REF(tuple) other)
        {
            a0 = other.a0; a1 = other.a1; a2 = other.a2; a3 = other.a3; a4 = other.a4; a5 = other.a5; a6 = other.a6;;
            return *this;
        }
        tuple& operator=(BOOST_RV_REF(tuple) other)
        {
            a0 = boost::forward<A0>( other.a0); a1 = boost::forward<A1>( other.a1); a2 = boost::forward<A2>( other.a2); a3 = boost::forward<A3>( other.a3); a4 = boost::forward<A4>( other.a4); a5 = boost::forward<A5>( other.a5); a6 = boost::forward<A6>( other.a6);;
            return *this;
        }
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6>
        tuple& operator=(BOOST_COPY_ASSIGN_REF(HPX_UTIL_STRIP((
                tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6>
            ))) other)
        {
            a0 = other.a0; a1 = other.a1; a2 = other.a2; a3 = other.a3; a4 = other.a4; a5 = other.a5; a6 = other.a6;;
            return *this;
        }
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6>
        tuple& operator=(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6>
            ))) other)
        {
            a0 = boost::forward<T0>( other.a0); a1 = boost::forward<T1>( other.a1); a2 = boost::forward<T2>( other.a2); a3 = boost::forward<T3>( other.a3); a4 = boost::forward<T4>( other.a4); a5 = boost::forward<T5>( other.a5); a6 = boost::forward<T6>( other.a6);;
            return *this;
        }
        void swap(tuple& other)
        {
            boost::swap(a0, other.a0); boost::swap(a1, other.a1); boost::swap(a2, other.a2); boost::swap(a3, other.a3); boost::swap(a4, other.a4); boost::swap(a5, other.a5); boost::swap(a6, other.a6);;
        }
        typedef boost::mpl::int_<7> size_type;
        static const int size_value = 7;
    private:
        BOOST_COPYABLE_AND_MOVABLE(tuple);
    };
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6>
    struct tuple_size<tuple<A0 , A1 , A2 , A3 , A4 , A5 , A6> >
    {
        static const std::size_t value = 7;
    };
    
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    BOOST_FORCEINLINE
    tuple<typename util::decay<Arg0>::type , typename util::decay<Arg1>::type , typename util::decay<Arg2>::type , typename util::decay<Arg3>::type , typename util::decay<Arg4>::type , typename util::decay<Arg5>::type , typename util::decay<Arg6>::type>
    make_tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6)
    {
        return tuple<typename util::decay<Arg0>::type , typename util::decay<Arg1>::type , typename util::decay<Arg2>::type , typename util::decay<Arg3>::type , typename util::decay<Arg4>::type , typename util::decay<Arg5>::type , typename util::decay<Arg6>::type>(
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ));
    }
    
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    BOOST_FORCEINLINE
    tuple<typename detail::add_rvalue_reference<Arg0>::type , typename detail::add_rvalue_reference<Arg1>::type , typename detail::add_rvalue_reference<Arg2>::type , typename detail::add_rvalue_reference<Arg3>::type , typename detail::add_rvalue_reference<Arg4>::type , typename detail::add_rvalue_reference<Arg5>::type , typename detail::add_rvalue_reference<Arg6>::type>
    forward_as_tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6) BOOST_NOEXCEPT
    {
        return tuple<
                typename detail::add_rvalue_reference<Arg0>::type , typename detail::add_rvalue_reference<Arg1>::type , typename detail::add_rvalue_reference<Arg2>::type , typename detail::add_rvalue_reference<Arg3>::type , typename detail::add_rvalue_reference<Arg4>::type , typename detail::add_rvalue_reference<Arg5>::type , typename detail::add_rvalue_reference<Arg6>::type>(
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ));
    }
    
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    BOOST_FORCEINLINE
    tuple<Arg0 & , Arg1 & , Arg2 & , Arg3 & , Arg4 & , Arg5 & , Arg6 &>
    tie(Arg0 & arg0 , Arg1 & arg1 , Arg2 & arg2 , Arg3 & arg3 , Arg4 & arg4 , Arg5 & arg5 , Arg6 & arg6) BOOST_NOEXCEPT
    {
        return tuple<
                Arg0 & , Arg1 & , Arg2 & , Arg3 & , Arg4 & , Arg5 & , Arg6 &>(
            arg0 , arg1 , arg2 , arg3 , arg4 , arg5 , arg6);
    }
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6>
    typename detail::tuple_cat_result<
        typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type
    >::type
    tuple_cat(BOOST_FWD_REF(T0) t0 , BOOST_FWD_REF(T1) t1 , BOOST_FWD_REF(T2) t2 , BOOST_FWD_REF(T3) t3 , BOOST_FWD_REF(T4) t4 , BOOST_FWD_REF(T5) t5 , BOOST_FWD_REF(T6) t6)
    {
        typedef
            typename detail::tuple_cat_result<T0, T1>::type
            head_type;
        head_type head =
            tuple_cat(boost::forward<T0>(t0), boost::forward<T1>(t1));
        return tuple_cat(boost::move(head)
                , boost::forward<T2>(t2) , boost::forward<T3>(t3) , boost::forward<T4>(t4) , boost::forward<T5>(t5) , boost::forward<T6>(t6));
    }
}}
namespace boost { namespace fusion { namespace traits { template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6 > struct tag_of<hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6> > { typedef struct_tag type; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6 > struct tag_of<hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6> const> { typedef struct_tag type; }; } namespace extension { template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6> , 0 > { typedef A0 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a0; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6> , 0 > { typedef char const* type; static type call() { return "a0"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6> , 1 > { typedef A1 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a1; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6> , 1 > { typedef char const* type; static type call() { return "a1"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6> , 2 > { typedef A2 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a2; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6> , 2 > { typedef char const* type; static type call() { return "a2"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6> , 3 > { typedef A3 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a3; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6> , 3 > { typedef char const* type; static type call() { return "a3"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6> , 4 > { typedef A4 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a4; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6> , 4 > { typedef char const* type; static type call() { return "a4"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6> , 5 > { typedef A5 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a5; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6> , 5 > { typedef char const* type; static type call() { return "a5"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6> , 6 > { typedef A6 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a6; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6> , 6 > { typedef char const* type; static type call() { return "a6"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6 > struct struct_size<hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6> > : mpl::int_<7> {}; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6 > struct struct_is_view< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6> > : mpl::false_ {}; } } namespace mpl { template<typename> struct sequence_tag; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6 > struct sequence_tag<hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6> > { typedef fusion::fusion_sequence_tag type; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6 > struct sequence_tag< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6> const > { typedef fusion::fusion_sequence_tag type; }; } }
namespace boost { namespace serialization
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6>
    struct is_bitwise_serializable<
            hpx::util::tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6> >
      : hpx::util::detail::sequence_is_bitwise_serializable<
            hpx::util::tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6> >
    {};
    
    template <typename Archive, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6>
    BOOST_FORCEINLINE void serialize(Archive& ar,
        hpx::util::tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6>& t,
        unsigned int const version)
    {
        hpx::util::serialize_sequence(ar, t);
    }
}}
namespace hpx { namespace util
{
    
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7>
    struct tuple<A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7>
    {
        typedef A0 member_type0; A0 a0; typedef A1 member_type1; A1 a1; typedef A2 member_type2; A2 a2; typedef A3 member_type3; A3 a3; typedef A4 member_type4; A4 a4; typedef A5 member_type5; A5 a5; typedef A6 member_type6; A6 a6; typedef A7 member_type7; A7 a7;
        template <int E>
        typename detail::tuple_element<E, tuple>::rtype
        get() BOOST_NOEXCEPT
        {
            return detail::tuple_element<E, tuple>::get(*this);
        }
        template <int E>
        BOOST_CONSTEXPR
        typename detail::tuple_element<E, tuple const>::crtype
        get() const BOOST_NOEXCEPT
        {
            return detail::tuple_element<E, tuple const>::get(*this);
        }
        
        tuple()
          : a0() , a1() , a2() , a3() , a4() , a5() , a6() , a7()
        {}
        template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
        tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7)
          : a0(boost::forward<Arg0>( arg0 )) , a1(boost::forward<Arg1>( arg1 )) , a2(boost::forward<Arg2>( arg2 )) , a3(boost::forward<Arg3>( arg3 )) , a4(boost::forward<Arg4>( arg4 )) , a5(boost::forward<Arg5>( arg5 )) , a6(boost::forward<Arg6>( arg6 )) , a7(boost::forward<Arg7>( arg7 ))
        {}
        
        tuple(tuple const& other)
          : a0( detail::copy_construct< A0 , typename boost::add_const<A0>::type >::call(other.a0)) , a1( detail::copy_construct< A1 , typename boost::add_const<A1>::type >::call(other.a1)) , a2( detail::copy_construct< A2 , typename boost::add_const<A2>::type >::call(other.a2)) , a3( detail::copy_construct< A3 , typename boost::add_const<A3>::type >::call(other.a3)) , a4( detail::copy_construct< A4 , typename boost::add_const<A4>::type >::call(other.a4)) , a5( detail::copy_construct< A5 , typename boost::add_const<A5>::type >::call(other.a5)) , a6( detail::copy_construct< A6 , typename boost::add_const<A6>::type >::call(other.a6)) , a7( detail::copy_construct< A7 , typename boost::add_const<A7>::type >::call(other.a7))
        {}
        tuple(BOOST_RV_REF(tuple) other)
          : a0(boost::forward<A0>( other.a0)) , a1(boost::forward<A1>( other.a1)) , a2(boost::forward<A2>( other.a2)) , a3(boost::forward<A3>( other.a3)) , a4(boost::forward<A4>( other.a4)) , a5(boost::forward<A5>( other.a5)) , a6(boost::forward<A6>( other.a6)) , a7(boost::forward<A7>( other.a7))
        {}
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
        tuple(tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7> const& other)
          : a0( detail::copy_construct< A0 , typename boost::add_const<T0>::type >::call(other.a0)) , a1( detail::copy_construct< A1 , typename boost::add_const<T1>::type >::call(other.a1)) , a2( detail::copy_construct< A2 , typename boost::add_const<T2>::type >::call(other.a2)) , a3( detail::copy_construct< A3 , typename boost::add_const<T3>::type >::call(other.a3)) , a4( detail::copy_construct< A4 , typename boost::add_const<T4>::type >::call(other.a4)) , a5( detail::copy_construct< A5 , typename boost::add_const<T5>::type >::call(other.a5)) , a6( detail::copy_construct< A6 , typename boost::add_const<T6>::type >::call(other.a6)) , a7( detail::copy_construct< A7 , typename boost::add_const<T7>::type >::call(other.a7))
        {}
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
        tuple(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7>
            ))) other)
          : a0(boost::forward<T0>( other.a0)) , a1(boost::forward<T1>( other.a1)) , a2(boost::forward<T2>( other.a2)) , a3(boost::forward<T3>( other.a3)) , a4(boost::forward<T4>( other.a4)) , a5(boost::forward<T5>( other.a5)) , a6(boost::forward<T6>( other.a6)) , a7(boost::forward<T7>( other.a7))
        {}
        
        tuple& operator=(BOOST_COPY_ASSIGN_REF(tuple) other)
        {
            a0 = other.a0; a1 = other.a1; a2 = other.a2; a3 = other.a3; a4 = other.a4; a5 = other.a5; a6 = other.a6; a7 = other.a7;;
            return *this;
        }
        tuple& operator=(BOOST_RV_REF(tuple) other)
        {
            a0 = boost::forward<A0>( other.a0); a1 = boost::forward<A1>( other.a1); a2 = boost::forward<A2>( other.a2); a3 = boost::forward<A3>( other.a3); a4 = boost::forward<A4>( other.a4); a5 = boost::forward<A5>( other.a5); a6 = boost::forward<A6>( other.a6); a7 = boost::forward<A7>( other.a7);;
            return *this;
        }
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
        tuple& operator=(BOOST_COPY_ASSIGN_REF(HPX_UTIL_STRIP((
                tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7>
            ))) other)
        {
            a0 = other.a0; a1 = other.a1; a2 = other.a2; a3 = other.a3; a4 = other.a4; a5 = other.a5; a6 = other.a6; a7 = other.a7;;
            return *this;
        }
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
        tuple& operator=(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7>
            ))) other)
        {
            a0 = boost::forward<T0>( other.a0); a1 = boost::forward<T1>( other.a1); a2 = boost::forward<T2>( other.a2); a3 = boost::forward<T3>( other.a3); a4 = boost::forward<T4>( other.a4); a5 = boost::forward<T5>( other.a5); a6 = boost::forward<T6>( other.a6); a7 = boost::forward<T7>( other.a7);;
            return *this;
        }
        void swap(tuple& other)
        {
            boost::swap(a0, other.a0); boost::swap(a1, other.a1); boost::swap(a2, other.a2); boost::swap(a3, other.a3); boost::swap(a4, other.a4); boost::swap(a5, other.a5); boost::swap(a6, other.a6); boost::swap(a7, other.a7);;
        }
        typedef boost::mpl::int_<8> size_type;
        static const int size_value = 8;
    private:
        BOOST_COPYABLE_AND_MOVABLE(tuple);
    };
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7>
    struct tuple_size<tuple<A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7> >
    {
        static const std::size_t value = 8;
    };
    
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    BOOST_FORCEINLINE
    tuple<typename util::decay<Arg0>::type , typename util::decay<Arg1>::type , typename util::decay<Arg2>::type , typename util::decay<Arg3>::type , typename util::decay<Arg4>::type , typename util::decay<Arg5>::type , typename util::decay<Arg6>::type , typename util::decay<Arg7>::type>
    make_tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7)
    {
        return tuple<typename util::decay<Arg0>::type , typename util::decay<Arg1>::type , typename util::decay<Arg2>::type , typename util::decay<Arg3>::type , typename util::decay<Arg4>::type , typename util::decay<Arg5>::type , typename util::decay<Arg6>::type , typename util::decay<Arg7>::type>(
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ));
    }
    
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    BOOST_FORCEINLINE
    tuple<typename detail::add_rvalue_reference<Arg0>::type , typename detail::add_rvalue_reference<Arg1>::type , typename detail::add_rvalue_reference<Arg2>::type , typename detail::add_rvalue_reference<Arg3>::type , typename detail::add_rvalue_reference<Arg4>::type , typename detail::add_rvalue_reference<Arg5>::type , typename detail::add_rvalue_reference<Arg6>::type , typename detail::add_rvalue_reference<Arg7>::type>
    forward_as_tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7) BOOST_NOEXCEPT
    {
        return tuple<
                typename detail::add_rvalue_reference<Arg0>::type , typename detail::add_rvalue_reference<Arg1>::type , typename detail::add_rvalue_reference<Arg2>::type , typename detail::add_rvalue_reference<Arg3>::type , typename detail::add_rvalue_reference<Arg4>::type , typename detail::add_rvalue_reference<Arg5>::type , typename detail::add_rvalue_reference<Arg6>::type , typename detail::add_rvalue_reference<Arg7>::type>(
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ));
    }
    
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    BOOST_FORCEINLINE
    tuple<Arg0 & , Arg1 & , Arg2 & , Arg3 & , Arg4 & , Arg5 & , Arg6 & , Arg7 &>
    tie(Arg0 & arg0 , Arg1 & arg1 , Arg2 & arg2 , Arg3 & arg3 , Arg4 & arg4 , Arg5 & arg5 , Arg6 & arg6 , Arg7 & arg7) BOOST_NOEXCEPT
    {
        return tuple<
                Arg0 & , Arg1 & , Arg2 & , Arg3 & , Arg4 & , Arg5 & , Arg6 & , Arg7 &>(
            arg0 , arg1 , arg2 , arg3 , arg4 , arg5 , arg6 , arg7);
    }
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
    typename detail::tuple_cat_result<
        typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type
    >::type
    tuple_cat(BOOST_FWD_REF(T0) t0 , BOOST_FWD_REF(T1) t1 , BOOST_FWD_REF(T2) t2 , BOOST_FWD_REF(T3) t3 , BOOST_FWD_REF(T4) t4 , BOOST_FWD_REF(T5) t5 , BOOST_FWD_REF(T6) t6 , BOOST_FWD_REF(T7) t7)
    {
        typedef
            typename detail::tuple_cat_result<T0, T1>::type
            head_type;
        head_type head =
            tuple_cat(boost::forward<T0>(t0), boost::forward<T1>(t1));
        return tuple_cat(boost::move(head)
                , boost::forward<T2>(t2) , boost::forward<T3>(t3) , boost::forward<T4>(t4) , boost::forward<T5>(t5) , boost::forward<T6>(t6) , boost::forward<T7>(t7));
    }
}}
namespace boost { namespace fusion { namespace traits { template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct tag_of<hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7> > { typedef struct_tag type; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct tag_of<hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7> const> { typedef struct_tag type; }; } namespace extension { template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7> , 0 > { typedef A0 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a0; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7> , 0 > { typedef char const* type; static type call() { return "a0"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7> , 1 > { typedef A1 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a1; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7> , 1 > { typedef char const* type; static type call() { return "a1"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7> , 2 > { typedef A2 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a2; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7> , 2 > { typedef char const* type; static type call() { return "a2"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7> , 3 > { typedef A3 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a3; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7> , 3 > { typedef char const* type; static type call() { return "a3"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7> , 4 > { typedef A4 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a4; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7> , 4 > { typedef char const* type; static type call() { return "a4"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7> , 5 > { typedef A5 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a5; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7> , 5 > { typedef char const* type; static type call() { return "a5"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7> , 6 > { typedef A6 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a6; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7> , 6 > { typedef char const* type; static type call() { return "a6"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7> , 7 > { typedef A7 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a7; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7> , 7 > { typedef char const* type; static type call() { return "a7"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct struct_size<hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7> > : mpl::int_<8> {}; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct struct_is_view< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7> > : mpl::false_ {}; } } namespace mpl { template<typename> struct sequence_tag; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct sequence_tag<hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7> > { typedef fusion::fusion_sequence_tag type; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct sequence_tag< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7> const > { typedef fusion::fusion_sequence_tag type; }; } }
namespace boost { namespace serialization
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
    struct is_bitwise_serializable<
            hpx::util::tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7> >
      : hpx::util::detail::sequence_is_bitwise_serializable<
            hpx::util::tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7> >
    {};
    
    template <typename Archive, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
    BOOST_FORCEINLINE void serialize(Archive& ar,
        hpx::util::tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7>& t,
        unsigned int const version)
    {
        hpx::util::serialize_sequence(ar, t);
    }
}}
namespace hpx { namespace util
{
    
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8>
    struct tuple<A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8>
    {
        typedef A0 member_type0; A0 a0; typedef A1 member_type1; A1 a1; typedef A2 member_type2; A2 a2; typedef A3 member_type3; A3 a3; typedef A4 member_type4; A4 a4; typedef A5 member_type5; A5 a5; typedef A6 member_type6; A6 a6; typedef A7 member_type7; A7 a7; typedef A8 member_type8; A8 a8;
        template <int E>
        typename detail::tuple_element<E, tuple>::rtype
        get() BOOST_NOEXCEPT
        {
            return detail::tuple_element<E, tuple>::get(*this);
        }
        template <int E>
        BOOST_CONSTEXPR
        typename detail::tuple_element<E, tuple const>::crtype
        get() const BOOST_NOEXCEPT
        {
            return detail::tuple_element<E, tuple const>::get(*this);
        }
        
        tuple()
          : a0() , a1() , a2() , a3() , a4() , a5() , a6() , a7() , a8()
        {}
        template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
        tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8)
          : a0(boost::forward<Arg0>( arg0 )) , a1(boost::forward<Arg1>( arg1 )) , a2(boost::forward<Arg2>( arg2 )) , a3(boost::forward<Arg3>( arg3 )) , a4(boost::forward<Arg4>( arg4 )) , a5(boost::forward<Arg5>( arg5 )) , a6(boost::forward<Arg6>( arg6 )) , a7(boost::forward<Arg7>( arg7 )) , a8(boost::forward<Arg8>( arg8 ))
        {}
        
        tuple(tuple const& other)
          : a0( detail::copy_construct< A0 , typename boost::add_const<A0>::type >::call(other.a0)) , a1( detail::copy_construct< A1 , typename boost::add_const<A1>::type >::call(other.a1)) , a2( detail::copy_construct< A2 , typename boost::add_const<A2>::type >::call(other.a2)) , a3( detail::copy_construct< A3 , typename boost::add_const<A3>::type >::call(other.a3)) , a4( detail::copy_construct< A4 , typename boost::add_const<A4>::type >::call(other.a4)) , a5( detail::copy_construct< A5 , typename boost::add_const<A5>::type >::call(other.a5)) , a6( detail::copy_construct< A6 , typename boost::add_const<A6>::type >::call(other.a6)) , a7( detail::copy_construct< A7 , typename boost::add_const<A7>::type >::call(other.a7)) , a8( detail::copy_construct< A8 , typename boost::add_const<A8>::type >::call(other.a8))
        {}
        tuple(BOOST_RV_REF(tuple) other)
          : a0(boost::forward<A0>( other.a0)) , a1(boost::forward<A1>( other.a1)) , a2(boost::forward<A2>( other.a2)) , a3(boost::forward<A3>( other.a3)) , a4(boost::forward<A4>( other.a4)) , a5(boost::forward<A5>( other.a5)) , a6(boost::forward<A6>( other.a6)) , a7(boost::forward<A7>( other.a7)) , a8(boost::forward<A8>( other.a8))
        {}
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8>
        tuple(tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8> const& other)
          : a0( detail::copy_construct< A0 , typename boost::add_const<T0>::type >::call(other.a0)) , a1( detail::copy_construct< A1 , typename boost::add_const<T1>::type >::call(other.a1)) , a2( detail::copy_construct< A2 , typename boost::add_const<T2>::type >::call(other.a2)) , a3( detail::copy_construct< A3 , typename boost::add_const<T3>::type >::call(other.a3)) , a4( detail::copy_construct< A4 , typename boost::add_const<T4>::type >::call(other.a4)) , a5( detail::copy_construct< A5 , typename boost::add_const<T5>::type >::call(other.a5)) , a6( detail::copy_construct< A6 , typename boost::add_const<T6>::type >::call(other.a6)) , a7( detail::copy_construct< A7 , typename boost::add_const<T7>::type >::call(other.a7)) , a8( detail::copy_construct< A8 , typename boost::add_const<T8>::type >::call(other.a8))
        {}
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8>
        tuple(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8>
            ))) other)
          : a0(boost::forward<T0>( other.a0)) , a1(boost::forward<T1>( other.a1)) , a2(boost::forward<T2>( other.a2)) , a3(boost::forward<T3>( other.a3)) , a4(boost::forward<T4>( other.a4)) , a5(boost::forward<T5>( other.a5)) , a6(boost::forward<T6>( other.a6)) , a7(boost::forward<T7>( other.a7)) , a8(boost::forward<T8>( other.a8))
        {}
        
        tuple& operator=(BOOST_COPY_ASSIGN_REF(tuple) other)
        {
            a0 = other.a0; a1 = other.a1; a2 = other.a2; a3 = other.a3; a4 = other.a4; a5 = other.a5; a6 = other.a6; a7 = other.a7; a8 = other.a8;;
            return *this;
        }
        tuple& operator=(BOOST_RV_REF(tuple) other)
        {
            a0 = boost::forward<A0>( other.a0); a1 = boost::forward<A1>( other.a1); a2 = boost::forward<A2>( other.a2); a3 = boost::forward<A3>( other.a3); a4 = boost::forward<A4>( other.a4); a5 = boost::forward<A5>( other.a5); a6 = boost::forward<A6>( other.a6); a7 = boost::forward<A7>( other.a7); a8 = boost::forward<A8>( other.a8);;
            return *this;
        }
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8>
        tuple& operator=(BOOST_COPY_ASSIGN_REF(HPX_UTIL_STRIP((
                tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8>
            ))) other)
        {
            a0 = other.a0; a1 = other.a1; a2 = other.a2; a3 = other.a3; a4 = other.a4; a5 = other.a5; a6 = other.a6; a7 = other.a7; a8 = other.a8;;
            return *this;
        }
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8>
        tuple& operator=(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8>
            ))) other)
        {
            a0 = boost::forward<T0>( other.a0); a1 = boost::forward<T1>( other.a1); a2 = boost::forward<T2>( other.a2); a3 = boost::forward<T3>( other.a3); a4 = boost::forward<T4>( other.a4); a5 = boost::forward<T5>( other.a5); a6 = boost::forward<T6>( other.a6); a7 = boost::forward<T7>( other.a7); a8 = boost::forward<T8>( other.a8);;
            return *this;
        }
        void swap(tuple& other)
        {
            boost::swap(a0, other.a0); boost::swap(a1, other.a1); boost::swap(a2, other.a2); boost::swap(a3, other.a3); boost::swap(a4, other.a4); boost::swap(a5, other.a5); boost::swap(a6, other.a6); boost::swap(a7, other.a7); boost::swap(a8, other.a8);;
        }
        typedef boost::mpl::int_<9> size_type;
        static const int size_value = 9;
    private:
        BOOST_COPYABLE_AND_MOVABLE(tuple);
    };
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8>
    struct tuple_size<tuple<A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8> >
    {
        static const std::size_t value = 9;
    };
    
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
    BOOST_FORCEINLINE
    tuple<typename util::decay<Arg0>::type , typename util::decay<Arg1>::type , typename util::decay<Arg2>::type , typename util::decay<Arg3>::type , typename util::decay<Arg4>::type , typename util::decay<Arg5>::type , typename util::decay<Arg6>::type , typename util::decay<Arg7>::type , typename util::decay<Arg8>::type>
    make_tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8)
    {
        return tuple<typename util::decay<Arg0>::type , typename util::decay<Arg1>::type , typename util::decay<Arg2>::type , typename util::decay<Arg3>::type , typename util::decay<Arg4>::type , typename util::decay<Arg5>::type , typename util::decay<Arg6>::type , typename util::decay<Arg7>::type , typename util::decay<Arg8>::type>(
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ));
    }
    
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
    BOOST_FORCEINLINE
    tuple<typename detail::add_rvalue_reference<Arg0>::type , typename detail::add_rvalue_reference<Arg1>::type , typename detail::add_rvalue_reference<Arg2>::type , typename detail::add_rvalue_reference<Arg3>::type , typename detail::add_rvalue_reference<Arg4>::type , typename detail::add_rvalue_reference<Arg5>::type , typename detail::add_rvalue_reference<Arg6>::type , typename detail::add_rvalue_reference<Arg7>::type , typename detail::add_rvalue_reference<Arg8>::type>
    forward_as_tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8) BOOST_NOEXCEPT
    {
        return tuple<
                typename detail::add_rvalue_reference<Arg0>::type , typename detail::add_rvalue_reference<Arg1>::type , typename detail::add_rvalue_reference<Arg2>::type , typename detail::add_rvalue_reference<Arg3>::type , typename detail::add_rvalue_reference<Arg4>::type , typename detail::add_rvalue_reference<Arg5>::type , typename detail::add_rvalue_reference<Arg6>::type , typename detail::add_rvalue_reference<Arg7>::type , typename detail::add_rvalue_reference<Arg8>::type>(
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ));
    }
    
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
    BOOST_FORCEINLINE
    tuple<Arg0 & , Arg1 & , Arg2 & , Arg3 & , Arg4 & , Arg5 & , Arg6 & , Arg7 & , Arg8 &>
    tie(Arg0 & arg0 , Arg1 & arg1 , Arg2 & arg2 , Arg3 & arg3 , Arg4 & arg4 , Arg5 & arg5 , Arg6 & arg6 , Arg7 & arg7 , Arg8 & arg8) BOOST_NOEXCEPT
    {
        return tuple<
                Arg0 & , Arg1 & , Arg2 & , Arg3 & , Arg4 & , Arg5 & , Arg6 & , Arg7 & , Arg8 &>(
            arg0 , arg1 , arg2 , arg3 , arg4 , arg5 , arg6 , arg7 , arg8);
    }
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8>
    typename detail::tuple_cat_result<
        typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type
    >::type
    tuple_cat(BOOST_FWD_REF(T0) t0 , BOOST_FWD_REF(T1) t1 , BOOST_FWD_REF(T2) t2 , BOOST_FWD_REF(T3) t3 , BOOST_FWD_REF(T4) t4 , BOOST_FWD_REF(T5) t5 , BOOST_FWD_REF(T6) t6 , BOOST_FWD_REF(T7) t7 , BOOST_FWD_REF(T8) t8)
    {
        typedef
            typename detail::tuple_cat_result<T0, T1>::type
            head_type;
        head_type head =
            tuple_cat(boost::forward<T0>(t0), boost::forward<T1>(t1));
        return tuple_cat(boost::move(head)
                , boost::forward<T2>(t2) , boost::forward<T3>(t3) , boost::forward<T4>(t4) , boost::forward<T5>(t5) , boost::forward<T6>(t6) , boost::forward<T7>(t7) , boost::forward<T8>(t8));
    }
}}
namespace boost { namespace fusion { namespace traits { template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8 > struct tag_of<hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8> > { typedef struct_tag type; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8 > struct tag_of<hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8> const> { typedef struct_tag type; }; } namespace extension { template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8> , 0 > { typedef A0 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7) (A8)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a0; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8> , 0 > { typedef char const* type; static type call() { return "a0"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8> , 1 > { typedef A1 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7) (A8)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a1; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8> , 1 > { typedef char const* type; static type call() { return "a1"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8> , 2 > { typedef A2 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7) (A8)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a2; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8> , 2 > { typedef char const* type; static type call() { return "a2"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8> , 3 > { typedef A3 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7) (A8)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a3; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8> , 3 > { typedef char const* type; static type call() { return "a3"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8> , 4 > { typedef A4 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7) (A8)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a4; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8> , 4 > { typedef char const* type; static type call() { return "a4"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8> , 5 > { typedef A5 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7) (A8)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a5; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8> , 5 > { typedef char const* type; static type call() { return "a5"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8> , 6 > { typedef A6 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7) (A8)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a6; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8> , 6 > { typedef char const* type; static type call() { return "a6"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8> , 7 > { typedef A7 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7) (A8)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a7; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8> , 7 > { typedef char const* type; static type call() { return "a7"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8> , 8 > { typedef A8 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7) (A8)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a8; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8> , 8 > { typedef char const* type; static type call() { return "a8"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8 > struct struct_size<hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8> > : mpl::int_<9> {}; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8 > struct struct_is_view< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8> > : mpl::false_ {}; } } namespace mpl { template<typename> struct sequence_tag; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8 > struct sequence_tag<hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8> > { typedef fusion::fusion_sequence_tag type; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8 > struct sequence_tag< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8> const > { typedef fusion::fusion_sequence_tag type; }; } }
namespace boost { namespace serialization
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8>
    struct is_bitwise_serializable<
            hpx::util::tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8> >
      : hpx::util::detail::sequence_is_bitwise_serializable<
            hpx::util::tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8> >
    {};
    
    template <typename Archive, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8>
    BOOST_FORCEINLINE void serialize(Archive& ar,
        hpx::util::tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8>& t,
        unsigned int const version)
    {
        hpx::util::serialize_sequence(ar, t);
    }
}}
namespace hpx { namespace util
{
    
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9>
    struct tuple<A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9>
    {
        typedef A0 member_type0; A0 a0; typedef A1 member_type1; A1 a1; typedef A2 member_type2; A2 a2; typedef A3 member_type3; A3 a3; typedef A4 member_type4; A4 a4; typedef A5 member_type5; A5 a5; typedef A6 member_type6; A6 a6; typedef A7 member_type7; A7 a7; typedef A8 member_type8; A8 a8; typedef A9 member_type9; A9 a9;
        template <int E>
        typename detail::tuple_element<E, tuple>::rtype
        get() BOOST_NOEXCEPT
        {
            return detail::tuple_element<E, tuple>::get(*this);
        }
        template <int E>
        BOOST_CONSTEXPR
        typename detail::tuple_element<E, tuple const>::crtype
        get() const BOOST_NOEXCEPT
        {
            return detail::tuple_element<E, tuple const>::get(*this);
        }
        
        tuple()
          : a0() , a1() , a2() , a3() , a4() , a5() , a6() , a7() , a8() , a9()
        {}
        template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
        tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9)
          : a0(boost::forward<Arg0>( arg0 )) , a1(boost::forward<Arg1>( arg1 )) , a2(boost::forward<Arg2>( arg2 )) , a3(boost::forward<Arg3>( arg3 )) , a4(boost::forward<Arg4>( arg4 )) , a5(boost::forward<Arg5>( arg5 )) , a6(boost::forward<Arg6>( arg6 )) , a7(boost::forward<Arg7>( arg7 )) , a8(boost::forward<Arg8>( arg8 )) , a9(boost::forward<Arg9>( arg9 ))
        {}
        
        tuple(tuple const& other)
          : a0( detail::copy_construct< A0 , typename boost::add_const<A0>::type >::call(other.a0)) , a1( detail::copy_construct< A1 , typename boost::add_const<A1>::type >::call(other.a1)) , a2( detail::copy_construct< A2 , typename boost::add_const<A2>::type >::call(other.a2)) , a3( detail::copy_construct< A3 , typename boost::add_const<A3>::type >::call(other.a3)) , a4( detail::copy_construct< A4 , typename boost::add_const<A4>::type >::call(other.a4)) , a5( detail::copy_construct< A5 , typename boost::add_const<A5>::type >::call(other.a5)) , a6( detail::copy_construct< A6 , typename boost::add_const<A6>::type >::call(other.a6)) , a7( detail::copy_construct< A7 , typename boost::add_const<A7>::type >::call(other.a7)) , a8( detail::copy_construct< A8 , typename boost::add_const<A8>::type >::call(other.a8)) , a9( detail::copy_construct< A9 , typename boost::add_const<A9>::type >::call(other.a9))
        {}
        tuple(BOOST_RV_REF(tuple) other)
          : a0(boost::forward<A0>( other.a0)) , a1(boost::forward<A1>( other.a1)) , a2(boost::forward<A2>( other.a2)) , a3(boost::forward<A3>( other.a3)) , a4(boost::forward<A4>( other.a4)) , a5(boost::forward<A5>( other.a5)) , a6(boost::forward<A6>( other.a6)) , a7(boost::forward<A7>( other.a7)) , a8(boost::forward<A8>( other.a8)) , a9(boost::forward<A9>( other.a9))
        {}
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9>
        tuple(tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9> const& other)
          : a0( detail::copy_construct< A0 , typename boost::add_const<T0>::type >::call(other.a0)) , a1( detail::copy_construct< A1 , typename boost::add_const<T1>::type >::call(other.a1)) , a2( detail::copy_construct< A2 , typename boost::add_const<T2>::type >::call(other.a2)) , a3( detail::copy_construct< A3 , typename boost::add_const<T3>::type >::call(other.a3)) , a4( detail::copy_construct< A4 , typename boost::add_const<T4>::type >::call(other.a4)) , a5( detail::copy_construct< A5 , typename boost::add_const<T5>::type >::call(other.a5)) , a6( detail::copy_construct< A6 , typename boost::add_const<T6>::type >::call(other.a6)) , a7( detail::copy_construct< A7 , typename boost::add_const<T7>::type >::call(other.a7)) , a8( detail::copy_construct< A8 , typename boost::add_const<T8>::type >::call(other.a8)) , a9( detail::copy_construct< A9 , typename boost::add_const<T9>::type >::call(other.a9))
        {}
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9>
        tuple(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9>
            ))) other)
          : a0(boost::forward<T0>( other.a0)) , a1(boost::forward<T1>( other.a1)) , a2(boost::forward<T2>( other.a2)) , a3(boost::forward<T3>( other.a3)) , a4(boost::forward<T4>( other.a4)) , a5(boost::forward<T5>( other.a5)) , a6(boost::forward<T6>( other.a6)) , a7(boost::forward<T7>( other.a7)) , a8(boost::forward<T8>( other.a8)) , a9(boost::forward<T9>( other.a9))
        {}
        
        tuple& operator=(BOOST_COPY_ASSIGN_REF(tuple) other)
        {
            a0 = other.a0; a1 = other.a1; a2 = other.a2; a3 = other.a3; a4 = other.a4; a5 = other.a5; a6 = other.a6; a7 = other.a7; a8 = other.a8; a9 = other.a9;;
            return *this;
        }
        tuple& operator=(BOOST_RV_REF(tuple) other)
        {
            a0 = boost::forward<A0>( other.a0); a1 = boost::forward<A1>( other.a1); a2 = boost::forward<A2>( other.a2); a3 = boost::forward<A3>( other.a3); a4 = boost::forward<A4>( other.a4); a5 = boost::forward<A5>( other.a5); a6 = boost::forward<A6>( other.a6); a7 = boost::forward<A7>( other.a7); a8 = boost::forward<A8>( other.a8); a9 = boost::forward<A9>( other.a9);;
            return *this;
        }
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9>
        tuple& operator=(BOOST_COPY_ASSIGN_REF(HPX_UTIL_STRIP((
                tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9>
            ))) other)
        {
            a0 = other.a0; a1 = other.a1; a2 = other.a2; a3 = other.a3; a4 = other.a4; a5 = other.a5; a6 = other.a6; a7 = other.a7; a8 = other.a8; a9 = other.a9;;
            return *this;
        }
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9>
        tuple& operator=(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9>
            ))) other)
        {
            a0 = boost::forward<T0>( other.a0); a1 = boost::forward<T1>( other.a1); a2 = boost::forward<T2>( other.a2); a3 = boost::forward<T3>( other.a3); a4 = boost::forward<T4>( other.a4); a5 = boost::forward<T5>( other.a5); a6 = boost::forward<T6>( other.a6); a7 = boost::forward<T7>( other.a7); a8 = boost::forward<T8>( other.a8); a9 = boost::forward<T9>( other.a9);;
            return *this;
        }
        void swap(tuple& other)
        {
            boost::swap(a0, other.a0); boost::swap(a1, other.a1); boost::swap(a2, other.a2); boost::swap(a3, other.a3); boost::swap(a4, other.a4); boost::swap(a5, other.a5); boost::swap(a6, other.a6); boost::swap(a7, other.a7); boost::swap(a8, other.a8); boost::swap(a9, other.a9);;
        }
        typedef boost::mpl::int_<10> size_type;
        static const int size_value = 10;
    private:
        BOOST_COPYABLE_AND_MOVABLE(tuple);
    };
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9>
    struct tuple_size<tuple<A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9> >
    {
        static const std::size_t value = 10;
    };
    
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
    BOOST_FORCEINLINE
    tuple<typename util::decay<Arg0>::type , typename util::decay<Arg1>::type , typename util::decay<Arg2>::type , typename util::decay<Arg3>::type , typename util::decay<Arg4>::type , typename util::decay<Arg5>::type , typename util::decay<Arg6>::type , typename util::decay<Arg7>::type , typename util::decay<Arg8>::type , typename util::decay<Arg9>::type>
    make_tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9)
    {
        return tuple<typename util::decay<Arg0>::type , typename util::decay<Arg1>::type , typename util::decay<Arg2>::type , typename util::decay<Arg3>::type , typename util::decay<Arg4>::type , typename util::decay<Arg5>::type , typename util::decay<Arg6>::type , typename util::decay<Arg7>::type , typename util::decay<Arg8>::type , typename util::decay<Arg9>::type>(
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ));
    }
    
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
    BOOST_FORCEINLINE
    tuple<typename detail::add_rvalue_reference<Arg0>::type , typename detail::add_rvalue_reference<Arg1>::type , typename detail::add_rvalue_reference<Arg2>::type , typename detail::add_rvalue_reference<Arg3>::type , typename detail::add_rvalue_reference<Arg4>::type , typename detail::add_rvalue_reference<Arg5>::type , typename detail::add_rvalue_reference<Arg6>::type , typename detail::add_rvalue_reference<Arg7>::type , typename detail::add_rvalue_reference<Arg8>::type , typename detail::add_rvalue_reference<Arg9>::type>
    forward_as_tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9) BOOST_NOEXCEPT
    {
        return tuple<
                typename detail::add_rvalue_reference<Arg0>::type , typename detail::add_rvalue_reference<Arg1>::type , typename detail::add_rvalue_reference<Arg2>::type , typename detail::add_rvalue_reference<Arg3>::type , typename detail::add_rvalue_reference<Arg4>::type , typename detail::add_rvalue_reference<Arg5>::type , typename detail::add_rvalue_reference<Arg6>::type , typename detail::add_rvalue_reference<Arg7>::type , typename detail::add_rvalue_reference<Arg8>::type , typename detail::add_rvalue_reference<Arg9>::type>(
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ));
    }
    
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
    BOOST_FORCEINLINE
    tuple<Arg0 & , Arg1 & , Arg2 & , Arg3 & , Arg4 & , Arg5 & , Arg6 & , Arg7 & , Arg8 & , Arg9 &>
    tie(Arg0 & arg0 , Arg1 & arg1 , Arg2 & arg2 , Arg3 & arg3 , Arg4 & arg4 , Arg5 & arg5 , Arg6 & arg6 , Arg7 & arg7 , Arg8 & arg8 , Arg9 & arg9) BOOST_NOEXCEPT
    {
        return tuple<
                Arg0 & , Arg1 & , Arg2 & , Arg3 & , Arg4 & , Arg5 & , Arg6 & , Arg7 & , Arg8 & , Arg9 &>(
            arg0 , arg1 , arg2 , arg3 , arg4 , arg5 , arg6 , arg7 , arg8 , arg9);
    }
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9>
    typename detail::tuple_cat_result<
        typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type
    >::type
    tuple_cat(BOOST_FWD_REF(T0) t0 , BOOST_FWD_REF(T1) t1 , BOOST_FWD_REF(T2) t2 , BOOST_FWD_REF(T3) t3 , BOOST_FWD_REF(T4) t4 , BOOST_FWD_REF(T5) t5 , BOOST_FWD_REF(T6) t6 , BOOST_FWD_REF(T7) t7 , BOOST_FWD_REF(T8) t8 , BOOST_FWD_REF(T9) t9)
    {
        typedef
            typename detail::tuple_cat_result<T0, T1>::type
            head_type;
        head_type head =
            tuple_cat(boost::forward<T0>(t0), boost::forward<T1>(t1));
        return tuple_cat(boost::move(head)
                , boost::forward<T2>(t2) , boost::forward<T3>(t3) , boost::forward<T4>(t4) , boost::forward<T5>(t5) , boost::forward<T6>(t6) , boost::forward<T7>(t7) , boost::forward<T8>(t8) , boost::forward<T9>(t9));
    }
}}
namespace boost { namespace fusion { namespace traits { template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9 > struct tag_of<hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9> > { typedef struct_tag type; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9 > struct tag_of<hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9> const> { typedef struct_tag type; }; } namespace extension { template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9> , 0 > { typedef A0 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7) (A8) (A9)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a0; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9> , 0 > { typedef char const* type; static type call() { return "a0"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9> , 1 > { typedef A1 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7) (A8) (A9)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a1; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9> , 1 > { typedef char const* type; static type call() { return "a1"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9> , 2 > { typedef A2 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7) (A8) (A9)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a2; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9> , 2 > { typedef char const* type; static type call() { return "a2"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9> , 3 > { typedef A3 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7) (A8) (A9)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a3; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9> , 3 > { typedef char const* type; static type call() { return "a3"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9> , 4 > { typedef A4 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7) (A8) (A9)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a4; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9> , 4 > { typedef char const* type; static type call() { return "a4"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9> , 5 > { typedef A5 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7) (A8) (A9)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a5; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9> , 5 > { typedef char const* type; static type call() { return "a5"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9> , 6 > { typedef A6 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7) (A8) (A9)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a6; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9> , 6 > { typedef char const* type; static type call() { return "a6"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9> , 7 > { typedef A7 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7) (A8) (A9)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a7; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9> , 7 > { typedef char const* type; static type call() { return "a7"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9> , 8 > { typedef A8 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7) (A8) (A9)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a8; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9> , 8 > { typedef char const* type; static type call() { return "a8"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9> , 9 > { typedef A9 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7) (A8) (A9)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a9; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9> , 9 > { typedef char const* type; static type call() { return "a9"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9 > struct struct_size<hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9> > : mpl::int_<10> {}; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9 > struct struct_is_view< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9> > : mpl::false_ {}; } } namespace mpl { template<typename> struct sequence_tag; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9 > struct sequence_tag<hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9> > { typedef fusion::fusion_sequence_tag type; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9 > struct sequence_tag< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9> const > { typedef fusion::fusion_sequence_tag type; }; } }
namespace boost { namespace serialization
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9>
    struct is_bitwise_serializable<
            hpx::util::tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9> >
      : hpx::util::detail::sequence_is_bitwise_serializable<
            hpx::util::tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9> >
    {};
    
    template <typename Archive, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9>
    BOOST_FORCEINLINE void serialize(Archive& ar,
        hpx::util::tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9>& t,
        unsigned int const version)
    {
        hpx::util::serialize_sequence(ar, t);
    }
}}
namespace hpx { namespace util
{
    
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10>
    struct tuple<A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10>
    {
        typedef A0 member_type0; A0 a0; typedef A1 member_type1; A1 a1; typedef A2 member_type2; A2 a2; typedef A3 member_type3; A3 a3; typedef A4 member_type4; A4 a4; typedef A5 member_type5; A5 a5; typedef A6 member_type6; A6 a6; typedef A7 member_type7; A7 a7; typedef A8 member_type8; A8 a8; typedef A9 member_type9; A9 a9; typedef A10 member_type10; A10 a10;
        template <int E>
        typename detail::tuple_element<E, tuple>::rtype
        get() BOOST_NOEXCEPT
        {
            return detail::tuple_element<E, tuple>::get(*this);
        }
        template <int E>
        BOOST_CONSTEXPR
        typename detail::tuple_element<E, tuple const>::crtype
        get() const BOOST_NOEXCEPT
        {
            return detail::tuple_element<E, tuple const>::get(*this);
        }
        
        tuple()
          : a0() , a1() , a2() , a3() , a4() , a5() , a6() , a7() , a8() , a9() , a10()
        {}
        template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10>
        tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10)
          : a0(boost::forward<Arg0>( arg0 )) , a1(boost::forward<Arg1>( arg1 )) , a2(boost::forward<Arg2>( arg2 )) , a3(boost::forward<Arg3>( arg3 )) , a4(boost::forward<Arg4>( arg4 )) , a5(boost::forward<Arg5>( arg5 )) , a6(boost::forward<Arg6>( arg6 )) , a7(boost::forward<Arg7>( arg7 )) , a8(boost::forward<Arg8>( arg8 )) , a9(boost::forward<Arg9>( arg9 )) , a10(boost::forward<Arg10>( arg10 ))
        {}
        
        tuple(tuple const& other)
          : a0( detail::copy_construct< A0 , typename boost::add_const<A0>::type >::call(other.a0)) , a1( detail::copy_construct< A1 , typename boost::add_const<A1>::type >::call(other.a1)) , a2( detail::copy_construct< A2 , typename boost::add_const<A2>::type >::call(other.a2)) , a3( detail::copy_construct< A3 , typename boost::add_const<A3>::type >::call(other.a3)) , a4( detail::copy_construct< A4 , typename boost::add_const<A4>::type >::call(other.a4)) , a5( detail::copy_construct< A5 , typename boost::add_const<A5>::type >::call(other.a5)) , a6( detail::copy_construct< A6 , typename boost::add_const<A6>::type >::call(other.a6)) , a7( detail::copy_construct< A7 , typename boost::add_const<A7>::type >::call(other.a7)) , a8( detail::copy_construct< A8 , typename boost::add_const<A8>::type >::call(other.a8)) , a9( detail::copy_construct< A9 , typename boost::add_const<A9>::type >::call(other.a9)) , a10( detail::copy_construct< A10 , typename boost::add_const<A10>::type >::call(other.a10))
        {}
        tuple(BOOST_RV_REF(tuple) other)
          : a0(boost::forward<A0>( other.a0)) , a1(boost::forward<A1>( other.a1)) , a2(boost::forward<A2>( other.a2)) , a3(boost::forward<A3>( other.a3)) , a4(boost::forward<A4>( other.a4)) , a5(boost::forward<A5>( other.a5)) , a6(boost::forward<A6>( other.a6)) , a7(boost::forward<A7>( other.a7)) , a8(boost::forward<A8>( other.a8)) , a9(boost::forward<A9>( other.a9)) , a10(boost::forward<A10>( other.a10))
        {}
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10>
        tuple(tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10> const& other)
          : a0( detail::copy_construct< A0 , typename boost::add_const<T0>::type >::call(other.a0)) , a1( detail::copy_construct< A1 , typename boost::add_const<T1>::type >::call(other.a1)) , a2( detail::copy_construct< A2 , typename boost::add_const<T2>::type >::call(other.a2)) , a3( detail::copy_construct< A3 , typename boost::add_const<T3>::type >::call(other.a3)) , a4( detail::copy_construct< A4 , typename boost::add_const<T4>::type >::call(other.a4)) , a5( detail::copy_construct< A5 , typename boost::add_const<T5>::type >::call(other.a5)) , a6( detail::copy_construct< A6 , typename boost::add_const<T6>::type >::call(other.a6)) , a7( detail::copy_construct< A7 , typename boost::add_const<T7>::type >::call(other.a7)) , a8( detail::copy_construct< A8 , typename boost::add_const<T8>::type >::call(other.a8)) , a9( detail::copy_construct< A9 , typename boost::add_const<T9>::type >::call(other.a9)) , a10( detail::copy_construct< A10 , typename boost::add_const<T10>::type >::call(other.a10))
        {}
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10>
        tuple(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10>
            ))) other)
          : a0(boost::forward<T0>( other.a0)) , a1(boost::forward<T1>( other.a1)) , a2(boost::forward<T2>( other.a2)) , a3(boost::forward<T3>( other.a3)) , a4(boost::forward<T4>( other.a4)) , a5(boost::forward<T5>( other.a5)) , a6(boost::forward<T6>( other.a6)) , a7(boost::forward<T7>( other.a7)) , a8(boost::forward<T8>( other.a8)) , a9(boost::forward<T9>( other.a9)) , a10(boost::forward<T10>( other.a10))
        {}
        
        tuple& operator=(BOOST_COPY_ASSIGN_REF(tuple) other)
        {
            a0 = other.a0; a1 = other.a1; a2 = other.a2; a3 = other.a3; a4 = other.a4; a5 = other.a5; a6 = other.a6; a7 = other.a7; a8 = other.a8; a9 = other.a9; a10 = other.a10;;
            return *this;
        }
        tuple& operator=(BOOST_RV_REF(tuple) other)
        {
            a0 = boost::forward<A0>( other.a0); a1 = boost::forward<A1>( other.a1); a2 = boost::forward<A2>( other.a2); a3 = boost::forward<A3>( other.a3); a4 = boost::forward<A4>( other.a4); a5 = boost::forward<A5>( other.a5); a6 = boost::forward<A6>( other.a6); a7 = boost::forward<A7>( other.a7); a8 = boost::forward<A8>( other.a8); a9 = boost::forward<A9>( other.a9); a10 = boost::forward<A10>( other.a10);;
            return *this;
        }
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10>
        tuple& operator=(BOOST_COPY_ASSIGN_REF(HPX_UTIL_STRIP((
                tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10>
            ))) other)
        {
            a0 = other.a0; a1 = other.a1; a2 = other.a2; a3 = other.a3; a4 = other.a4; a5 = other.a5; a6 = other.a6; a7 = other.a7; a8 = other.a8; a9 = other.a9; a10 = other.a10;;
            return *this;
        }
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10>
        tuple& operator=(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10>
            ))) other)
        {
            a0 = boost::forward<T0>( other.a0); a1 = boost::forward<T1>( other.a1); a2 = boost::forward<T2>( other.a2); a3 = boost::forward<T3>( other.a3); a4 = boost::forward<T4>( other.a4); a5 = boost::forward<T5>( other.a5); a6 = boost::forward<T6>( other.a6); a7 = boost::forward<T7>( other.a7); a8 = boost::forward<T8>( other.a8); a9 = boost::forward<T9>( other.a9); a10 = boost::forward<T10>( other.a10);;
            return *this;
        }
        void swap(tuple& other)
        {
            boost::swap(a0, other.a0); boost::swap(a1, other.a1); boost::swap(a2, other.a2); boost::swap(a3, other.a3); boost::swap(a4, other.a4); boost::swap(a5, other.a5); boost::swap(a6, other.a6); boost::swap(a7, other.a7); boost::swap(a8, other.a8); boost::swap(a9, other.a9); boost::swap(a10, other.a10);;
        }
        typedef boost::mpl::int_<11> size_type;
        static const int size_value = 11;
    private:
        BOOST_COPYABLE_AND_MOVABLE(tuple);
    };
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10>
    struct tuple_size<tuple<A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10> >
    {
        static const std::size_t value = 11;
    };
    
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10>
    BOOST_FORCEINLINE
    tuple<typename util::decay<Arg0>::type , typename util::decay<Arg1>::type , typename util::decay<Arg2>::type , typename util::decay<Arg3>::type , typename util::decay<Arg4>::type , typename util::decay<Arg5>::type , typename util::decay<Arg6>::type , typename util::decay<Arg7>::type , typename util::decay<Arg8>::type , typename util::decay<Arg9>::type , typename util::decay<Arg10>::type>
    make_tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10)
    {
        return tuple<typename util::decay<Arg0>::type , typename util::decay<Arg1>::type , typename util::decay<Arg2>::type , typename util::decay<Arg3>::type , typename util::decay<Arg4>::type , typename util::decay<Arg5>::type , typename util::decay<Arg6>::type , typename util::decay<Arg7>::type , typename util::decay<Arg8>::type , typename util::decay<Arg9>::type , typename util::decay<Arg10>::type>(
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ));
    }
    
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10>
    BOOST_FORCEINLINE
    tuple<typename detail::add_rvalue_reference<Arg0>::type , typename detail::add_rvalue_reference<Arg1>::type , typename detail::add_rvalue_reference<Arg2>::type , typename detail::add_rvalue_reference<Arg3>::type , typename detail::add_rvalue_reference<Arg4>::type , typename detail::add_rvalue_reference<Arg5>::type , typename detail::add_rvalue_reference<Arg6>::type , typename detail::add_rvalue_reference<Arg7>::type , typename detail::add_rvalue_reference<Arg8>::type , typename detail::add_rvalue_reference<Arg9>::type , typename detail::add_rvalue_reference<Arg10>::type>
    forward_as_tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10) BOOST_NOEXCEPT
    {
        return tuple<
                typename detail::add_rvalue_reference<Arg0>::type , typename detail::add_rvalue_reference<Arg1>::type , typename detail::add_rvalue_reference<Arg2>::type , typename detail::add_rvalue_reference<Arg3>::type , typename detail::add_rvalue_reference<Arg4>::type , typename detail::add_rvalue_reference<Arg5>::type , typename detail::add_rvalue_reference<Arg6>::type , typename detail::add_rvalue_reference<Arg7>::type , typename detail::add_rvalue_reference<Arg8>::type , typename detail::add_rvalue_reference<Arg9>::type , typename detail::add_rvalue_reference<Arg10>::type>(
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ));
    }
    
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10>
    BOOST_FORCEINLINE
    tuple<Arg0 & , Arg1 & , Arg2 & , Arg3 & , Arg4 & , Arg5 & , Arg6 & , Arg7 & , Arg8 & , Arg9 & , Arg10 &>
    tie(Arg0 & arg0 , Arg1 & arg1 , Arg2 & arg2 , Arg3 & arg3 , Arg4 & arg4 , Arg5 & arg5 , Arg6 & arg6 , Arg7 & arg7 , Arg8 & arg8 , Arg9 & arg9 , Arg10 & arg10) BOOST_NOEXCEPT
    {
        return tuple<
                Arg0 & , Arg1 & , Arg2 & , Arg3 & , Arg4 & , Arg5 & , Arg6 & , Arg7 & , Arg8 & , Arg9 & , Arg10 &>(
            arg0 , arg1 , arg2 , arg3 , arg4 , arg5 , arg6 , arg7 , arg8 , arg9 , arg10);
    }
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10>
    typename detail::tuple_cat_result<
        typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type
    >::type
    tuple_cat(BOOST_FWD_REF(T0) t0 , BOOST_FWD_REF(T1) t1 , BOOST_FWD_REF(T2) t2 , BOOST_FWD_REF(T3) t3 , BOOST_FWD_REF(T4) t4 , BOOST_FWD_REF(T5) t5 , BOOST_FWD_REF(T6) t6 , BOOST_FWD_REF(T7) t7 , BOOST_FWD_REF(T8) t8 , BOOST_FWD_REF(T9) t9 , BOOST_FWD_REF(T10) t10)
    {
        typedef
            typename detail::tuple_cat_result<T0, T1>::type
            head_type;
        head_type head =
            tuple_cat(boost::forward<T0>(t0), boost::forward<T1>(t1));
        return tuple_cat(boost::move(head)
                , boost::forward<T2>(t2) , boost::forward<T3>(t3) , boost::forward<T4>(t4) , boost::forward<T5>(t5) , boost::forward<T6>(t6) , boost::forward<T7>(t7) , boost::forward<T8>(t8) , boost::forward<T9>(t9) , boost::forward<T10>(t10));
    }
}}
namespace boost { namespace fusion { namespace traits { template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10 > struct tag_of<hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10> > { typedef struct_tag type; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10 > struct tag_of<hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10> const> { typedef struct_tag type; }; } namespace extension { template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10> , 0 > { typedef A0 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7) (A8) (A9) (A10)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a0; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10> , 0 > { typedef char const* type; static type call() { return "a0"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10> , 1 > { typedef A1 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7) (A8) (A9) (A10)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a1; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10> , 1 > { typedef char const* type; static type call() { return "a1"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10> , 2 > { typedef A2 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7) (A8) (A9) (A10)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a2; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10> , 2 > { typedef char const* type; static type call() { return "a2"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10> , 3 > { typedef A3 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7) (A8) (A9) (A10)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a3; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10> , 3 > { typedef char const* type; static type call() { return "a3"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10> , 4 > { typedef A4 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7) (A8) (A9) (A10)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a4; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10> , 4 > { typedef char const* type; static type call() { return "a4"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10> , 5 > { typedef A5 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7) (A8) (A9) (A10)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a5; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10> , 5 > { typedef char const* type; static type call() { return "a5"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10> , 6 > { typedef A6 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7) (A8) (A9) (A10)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a6; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10> , 6 > { typedef char const* type; static type call() { return "a6"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10> , 7 > { typedef A7 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7) (A8) (A9) (A10)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a7; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10> , 7 > { typedef char const* type; static type call() { return "a7"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10> , 8 > { typedef A8 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7) (A8) (A9) (A10)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a8; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10> , 8 > { typedef char const* type; static type call() { return "a8"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10> , 9 > { typedef A9 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7) (A8) (A9) (A10)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a9; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10> , 9 > { typedef char const* type; static type call() { return "a9"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10> , 10 > { typedef A10 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7) (A8) (A9) (A10)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a10; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10> , 10 > { typedef char const* type; static type call() { return "a10"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10 > struct struct_size<hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10> > : mpl::int_<11> {}; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10 > struct struct_is_view< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10> > : mpl::false_ {}; } } namespace mpl { template<typename> struct sequence_tag; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10 > struct sequence_tag<hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10> > { typedef fusion::fusion_sequence_tag type; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10 > struct sequence_tag< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10> const > { typedef fusion::fusion_sequence_tag type; }; } }
namespace boost { namespace serialization
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10>
    struct is_bitwise_serializable<
            hpx::util::tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10> >
      : hpx::util::detail::sequence_is_bitwise_serializable<
            hpx::util::tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10> >
    {};
    
    template <typename Archive, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10>
    BOOST_FORCEINLINE void serialize(Archive& ar,
        hpx::util::tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10>& t,
        unsigned int const version)
    {
        hpx::util::serialize_sequence(ar, t);
    }
}}
namespace hpx { namespace util
{
    
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11>
    struct tuple<A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11>
    {
        typedef A0 member_type0; A0 a0; typedef A1 member_type1; A1 a1; typedef A2 member_type2; A2 a2; typedef A3 member_type3; A3 a3; typedef A4 member_type4; A4 a4; typedef A5 member_type5; A5 a5; typedef A6 member_type6; A6 a6; typedef A7 member_type7; A7 a7; typedef A8 member_type8; A8 a8; typedef A9 member_type9; A9 a9; typedef A10 member_type10; A10 a10; typedef A11 member_type11; A11 a11;
        template <int E>
        typename detail::tuple_element<E, tuple>::rtype
        get() BOOST_NOEXCEPT
        {
            return detail::tuple_element<E, tuple>::get(*this);
        }
        template <int E>
        BOOST_CONSTEXPR
        typename detail::tuple_element<E, tuple const>::crtype
        get() const BOOST_NOEXCEPT
        {
            return detail::tuple_element<E, tuple const>::get(*this);
        }
        
        tuple()
          : a0() , a1() , a2() , a3() , a4() , a5() , a6() , a7() , a8() , a9() , a10() , a11()
        {}
        template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11>
        tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11)
          : a0(boost::forward<Arg0>( arg0 )) , a1(boost::forward<Arg1>( arg1 )) , a2(boost::forward<Arg2>( arg2 )) , a3(boost::forward<Arg3>( arg3 )) , a4(boost::forward<Arg4>( arg4 )) , a5(boost::forward<Arg5>( arg5 )) , a6(boost::forward<Arg6>( arg6 )) , a7(boost::forward<Arg7>( arg7 )) , a8(boost::forward<Arg8>( arg8 )) , a9(boost::forward<Arg9>( arg9 )) , a10(boost::forward<Arg10>( arg10 )) , a11(boost::forward<Arg11>( arg11 ))
        {}
        
        tuple(tuple const& other)
          : a0( detail::copy_construct< A0 , typename boost::add_const<A0>::type >::call(other.a0)) , a1( detail::copy_construct< A1 , typename boost::add_const<A1>::type >::call(other.a1)) , a2( detail::copy_construct< A2 , typename boost::add_const<A2>::type >::call(other.a2)) , a3( detail::copy_construct< A3 , typename boost::add_const<A3>::type >::call(other.a3)) , a4( detail::copy_construct< A4 , typename boost::add_const<A4>::type >::call(other.a4)) , a5( detail::copy_construct< A5 , typename boost::add_const<A5>::type >::call(other.a5)) , a6( detail::copy_construct< A6 , typename boost::add_const<A6>::type >::call(other.a6)) , a7( detail::copy_construct< A7 , typename boost::add_const<A7>::type >::call(other.a7)) , a8( detail::copy_construct< A8 , typename boost::add_const<A8>::type >::call(other.a8)) , a9( detail::copy_construct< A9 , typename boost::add_const<A9>::type >::call(other.a9)) , a10( detail::copy_construct< A10 , typename boost::add_const<A10>::type >::call(other.a10)) , a11( detail::copy_construct< A11 , typename boost::add_const<A11>::type >::call(other.a11))
        {}
        tuple(BOOST_RV_REF(tuple) other)
          : a0(boost::forward<A0>( other.a0)) , a1(boost::forward<A1>( other.a1)) , a2(boost::forward<A2>( other.a2)) , a3(boost::forward<A3>( other.a3)) , a4(boost::forward<A4>( other.a4)) , a5(boost::forward<A5>( other.a5)) , a6(boost::forward<A6>( other.a6)) , a7(boost::forward<A7>( other.a7)) , a8(boost::forward<A8>( other.a8)) , a9(boost::forward<A9>( other.a9)) , a10(boost::forward<A10>( other.a10)) , a11(boost::forward<A11>( other.a11))
        {}
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11>
        tuple(tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11> const& other)
          : a0( detail::copy_construct< A0 , typename boost::add_const<T0>::type >::call(other.a0)) , a1( detail::copy_construct< A1 , typename boost::add_const<T1>::type >::call(other.a1)) , a2( detail::copy_construct< A2 , typename boost::add_const<T2>::type >::call(other.a2)) , a3( detail::copy_construct< A3 , typename boost::add_const<T3>::type >::call(other.a3)) , a4( detail::copy_construct< A4 , typename boost::add_const<T4>::type >::call(other.a4)) , a5( detail::copy_construct< A5 , typename boost::add_const<T5>::type >::call(other.a5)) , a6( detail::copy_construct< A6 , typename boost::add_const<T6>::type >::call(other.a6)) , a7( detail::copy_construct< A7 , typename boost::add_const<T7>::type >::call(other.a7)) , a8( detail::copy_construct< A8 , typename boost::add_const<T8>::type >::call(other.a8)) , a9( detail::copy_construct< A9 , typename boost::add_const<T9>::type >::call(other.a9)) , a10( detail::copy_construct< A10 , typename boost::add_const<T10>::type >::call(other.a10)) , a11( detail::copy_construct< A11 , typename boost::add_const<T11>::type >::call(other.a11))
        {}
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11>
        tuple(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11>
            ))) other)
          : a0(boost::forward<T0>( other.a0)) , a1(boost::forward<T1>( other.a1)) , a2(boost::forward<T2>( other.a2)) , a3(boost::forward<T3>( other.a3)) , a4(boost::forward<T4>( other.a4)) , a5(boost::forward<T5>( other.a5)) , a6(boost::forward<T6>( other.a6)) , a7(boost::forward<T7>( other.a7)) , a8(boost::forward<T8>( other.a8)) , a9(boost::forward<T9>( other.a9)) , a10(boost::forward<T10>( other.a10)) , a11(boost::forward<T11>( other.a11))
        {}
        
        tuple& operator=(BOOST_COPY_ASSIGN_REF(tuple) other)
        {
            a0 = other.a0; a1 = other.a1; a2 = other.a2; a3 = other.a3; a4 = other.a4; a5 = other.a5; a6 = other.a6; a7 = other.a7; a8 = other.a8; a9 = other.a9; a10 = other.a10; a11 = other.a11;;
            return *this;
        }
        tuple& operator=(BOOST_RV_REF(tuple) other)
        {
            a0 = boost::forward<A0>( other.a0); a1 = boost::forward<A1>( other.a1); a2 = boost::forward<A2>( other.a2); a3 = boost::forward<A3>( other.a3); a4 = boost::forward<A4>( other.a4); a5 = boost::forward<A5>( other.a5); a6 = boost::forward<A6>( other.a6); a7 = boost::forward<A7>( other.a7); a8 = boost::forward<A8>( other.a8); a9 = boost::forward<A9>( other.a9); a10 = boost::forward<A10>( other.a10); a11 = boost::forward<A11>( other.a11);;
            return *this;
        }
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11>
        tuple& operator=(BOOST_COPY_ASSIGN_REF(HPX_UTIL_STRIP((
                tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11>
            ))) other)
        {
            a0 = other.a0; a1 = other.a1; a2 = other.a2; a3 = other.a3; a4 = other.a4; a5 = other.a5; a6 = other.a6; a7 = other.a7; a8 = other.a8; a9 = other.a9; a10 = other.a10; a11 = other.a11;;
            return *this;
        }
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11>
        tuple& operator=(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11>
            ))) other)
        {
            a0 = boost::forward<T0>( other.a0); a1 = boost::forward<T1>( other.a1); a2 = boost::forward<T2>( other.a2); a3 = boost::forward<T3>( other.a3); a4 = boost::forward<T4>( other.a4); a5 = boost::forward<T5>( other.a5); a6 = boost::forward<T6>( other.a6); a7 = boost::forward<T7>( other.a7); a8 = boost::forward<T8>( other.a8); a9 = boost::forward<T9>( other.a9); a10 = boost::forward<T10>( other.a10); a11 = boost::forward<T11>( other.a11);;
            return *this;
        }
        void swap(tuple& other)
        {
            boost::swap(a0, other.a0); boost::swap(a1, other.a1); boost::swap(a2, other.a2); boost::swap(a3, other.a3); boost::swap(a4, other.a4); boost::swap(a5, other.a5); boost::swap(a6, other.a6); boost::swap(a7, other.a7); boost::swap(a8, other.a8); boost::swap(a9, other.a9); boost::swap(a10, other.a10); boost::swap(a11, other.a11);;
        }
        typedef boost::mpl::int_<12> size_type;
        static const int size_value = 12;
    private:
        BOOST_COPYABLE_AND_MOVABLE(tuple);
    };
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11>
    struct tuple_size<tuple<A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11> >
    {
        static const std::size_t value = 12;
    };
    
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11>
    BOOST_FORCEINLINE
    tuple<typename util::decay<Arg0>::type , typename util::decay<Arg1>::type , typename util::decay<Arg2>::type , typename util::decay<Arg3>::type , typename util::decay<Arg4>::type , typename util::decay<Arg5>::type , typename util::decay<Arg6>::type , typename util::decay<Arg7>::type , typename util::decay<Arg8>::type , typename util::decay<Arg9>::type , typename util::decay<Arg10>::type , typename util::decay<Arg11>::type>
    make_tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11)
    {
        return tuple<typename util::decay<Arg0>::type , typename util::decay<Arg1>::type , typename util::decay<Arg2>::type , typename util::decay<Arg3>::type , typename util::decay<Arg4>::type , typename util::decay<Arg5>::type , typename util::decay<Arg6>::type , typename util::decay<Arg7>::type , typename util::decay<Arg8>::type , typename util::decay<Arg9>::type , typename util::decay<Arg10>::type , typename util::decay<Arg11>::type>(
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ));
    }
    
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11>
    BOOST_FORCEINLINE
    tuple<typename detail::add_rvalue_reference<Arg0>::type , typename detail::add_rvalue_reference<Arg1>::type , typename detail::add_rvalue_reference<Arg2>::type , typename detail::add_rvalue_reference<Arg3>::type , typename detail::add_rvalue_reference<Arg4>::type , typename detail::add_rvalue_reference<Arg5>::type , typename detail::add_rvalue_reference<Arg6>::type , typename detail::add_rvalue_reference<Arg7>::type , typename detail::add_rvalue_reference<Arg8>::type , typename detail::add_rvalue_reference<Arg9>::type , typename detail::add_rvalue_reference<Arg10>::type , typename detail::add_rvalue_reference<Arg11>::type>
    forward_as_tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11) BOOST_NOEXCEPT
    {
        return tuple<
                typename detail::add_rvalue_reference<Arg0>::type , typename detail::add_rvalue_reference<Arg1>::type , typename detail::add_rvalue_reference<Arg2>::type , typename detail::add_rvalue_reference<Arg3>::type , typename detail::add_rvalue_reference<Arg4>::type , typename detail::add_rvalue_reference<Arg5>::type , typename detail::add_rvalue_reference<Arg6>::type , typename detail::add_rvalue_reference<Arg7>::type , typename detail::add_rvalue_reference<Arg8>::type , typename detail::add_rvalue_reference<Arg9>::type , typename detail::add_rvalue_reference<Arg10>::type , typename detail::add_rvalue_reference<Arg11>::type>(
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ));
    }
    
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11>
    BOOST_FORCEINLINE
    tuple<Arg0 & , Arg1 & , Arg2 & , Arg3 & , Arg4 & , Arg5 & , Arg6 & , Arg7 & , Arg8 & , Arg9 & , Arg10 & , Arg11 &>
    tie(Arg0 & arg0 , Arg1 & arg1 , Arg2 & arg2 , Arg3 & arg3 , Arg4 & arg4 , Arg5 & arg5 , Arg6 & arg6 , Arg7 & arg7 , Arg8 & arg8 , Arg9 & arg9 , Arg10 & arg10 , Arg11 & arg11) BOOST_NOEXCEPT
    {
        return tuple<
                Arg0 & , Arg1 & , Arg2 & , Arg3 & , Arg4 & , Arg5 & , Arg6 & , Arg7 & , Arg8 & , Arg9 & , Arg10 & , Arg11 &>(
            arg0 , arg1 , arg2 , arg3 , arg4 , arg5 , arg6 , arg7 , arg8 , arg9 , arg10 , arg11);
    }
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11>
    typename detail::tuple_cat_result<
        typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type
    >::type
    tuple_cat(BOOST_FWD_REF(T0) t0 , BOOST_FWD_REF(T1) t1 , BOOST_FWD_REF(T2) t2 , BOOST_FWD_REF(T3) t3 , BOOST_FWD_REF(T4) t4 , BOOST_FWD_REF(T5) t5 , BOOST_FWD_REF(T6) t6 , BOOST_FWD_REF(T7) t7 , BOOST_FWD_REF(T8) t8 , BOOST_FWD_REF(T9) t9 , BOOST_FWD_REF(T10) t10 , BOOST_FWD_REF(T11) t11)
    {
        typedef
            typename detail::tuple_cat_result<T0, T1>::type
            head_type;
        head_type head =
            tuple_cat(boost::forward<T0>(t0), boost::forward<T1>(t1));
        return tuple_cat(boost::move(head)
                , boost::forward<T2>(t2) , boost::forward<T3>(t3) , boost::forward<T4>(t4) , boost::forward<T5>(t5) , boost::forward<T6>(t6) , boost::forward<T7>(t7) , boost::forward<T8>(t8) , boost::forward<T9>(t9) , boost::forward<T10>(t10) , boost::forward<T11>(t11));
    }
}}
namespace boost { namespace fusion { namespace traits { template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11 > struct tag_of<hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11> > { typedef struct_tag type; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11 > struct tag_of<hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11> const> { typedef struct_tag type; }; } namespace extension { template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11> , 0 > { typedef A0 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7) (A8) (A9) (A10) (A11)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a0; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11> , 0 > { typedef char const* type; static type call() { return "a0"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11> , 1 > { typedef A1 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7) (A8) (A9) (A10) (A11)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a1; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11> , 1 > { typedef char const* type; static type call() { return "a1"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11> , 2 > { typedef A2 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7) (A8) (A9) (A10) (A11)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a2; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11> , 2 > { typedef char const* type; static type call() { return "a2"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11> , 3 > { typedef A3 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7) (A8) (A9) (A10) (A11)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a3; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11> , 3 > { typedef char const* type; static type call() { return "a3"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11> , 4 > { typedef A4 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7) (A8) (A9) (A10) (A11)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a4; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11> , 4 > { typedef char const* type; static type call() { return "a4"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11> , 5 > { typedef A5 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7) (A8) (A9) (A10) (A11)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a5; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11> , 5 > { typedef char const* type; static type call() { return "a5"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11> , 6 > { typedef A6 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7) (A8) (A9) (A10) (A11)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a6; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11> , 6 > { typedef char const* type; static type call() { return "a6"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11> , 7 > { typedef A7 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7) (A8) (A9) (A10) (A11)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a7; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11> , 7 > { typedef char const* type; static type call() { return "a7"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11> , 8 > { typedef A8 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7) (A8) (A9) (A10) (A11)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a8; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11> , 8 > { typedef char const* type; static type call() { return "a8"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11> , 9 > { typedef A9 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7) (A8) (A9) (A10) (A11)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a9; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11> , 9 > { typedef char const* type; static type call() { return "a9"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11> , 10 > { typedef A10 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7) (A8) (A9) (A10) (A11)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a10; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11> , 10 > { typedef char const* type; static type call() { return "a10"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11> , 11 > { typedef A11 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7) (A8) (A9) (A10) (A11)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a11; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11> , 11 > { typedef char const* type; static type call() { return "a11"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11 > struct struct_size<hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11> > : mpl::int_<12> {}; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11 > struct struct_is_view< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11> > : mpl::false_ {}; } } namespace mpl { template<typename> struct sequence_tag; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11 > struct sequence_tag<hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11> > { typedef fusion::fusion_sequence_tag type; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11 > struct sequence_tag< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11> const > { typedef fusion::fusion_sequence_tag type; }; } }
namespace boost { namespace serialization
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11>
    struct is_bitwise_serializable<
            hpx::util::tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11> >
      : hpx::util::detail::sequence_is_bitwise_serializable<
            hpx::util::tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11> >
    {};
    
    template <typename Archive, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11>
    BOOST_FORCEINLINE void serialize(Archive& ar,
        hpx::util::tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11>& t,
        unsigned int const version)
    {
        hpx::util::serialize_sequence(ar, t);
    }
}}
namespace hpx { namespace util
{
    
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12>
    struct tuple<A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12>
    {
        typedef A0 member_type0; A0 a0; typedef A1 member_type1; A1 a1; typedef A2 member_type2; A2 a2; typedef A3 member_type3; A3 a3; typedef A4 member_type4; A4 a4; typedef A5 member_type5; A5 a5; typedef A6 member_type6; A6 a6; typedef A7 member_type7; A7 a7; typedef A8 member_type8; A8 a8; typedef A9 member_type9; A9 a9; typedef A10 member_type10; A10 a10; typedef A11 member_type11; A11 a11; typedef A12 member_type12; A12 a12;
        template <int E>
        typename detail::tuple_element<E, tuple>::rtype
        get() BOOST_NOEXCEPT
        {
            return detail::tuple_element<E, tuple>::get(*this);
        }
        template <int E>
        BOOST_CONSTEXPR
        typename detail::tuple_element<E, tuple const>::crtype
        get() const BOOST_NOEXCEPT
        {
            return detail::tuple_element<E, tuple const>::get(*this);
        }
        
        tuple()
          : a0() , a1() , a2() , a3() , a4() , a5() , a6() , a7() , a8() , a9() , a10() , a11() , a12()
        {}
        template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12>
        tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12)
          : a0(boost::forward<Arg0>( arg0 )) , a1(boost::forward<Arg1>( arg1 )) , a2(boost::forward<Arg2>( arg2 )) , a3(boost::forward<Arg3>( arg3 )) , a4(boost::forward<Arg4>( arg4 )) , a5(boost::forward<Arg5>( arg5 )) , a6(boost::forward<Arg6>( arg6 )) , a7(boost::forward<Arg7>( arg7 )) , a8(boost::forward<Arg8>( arg8 )) , a9(boost::forward<Arg9>( arg9 )) , a10(boost::forward<Arg10>( arg10 )) , a11(boost::forward<Arg11>( arg11 )) , a12(boost::forward<Arg12>( arg12 ))
        {}
        
        tuple(tuple const& other)
          : a0( detail::copy_construct< A0 , typename boost::add_const<A0>::type >::call(other.a0)) , a1( detail::copy_construct< A1 , typename boost::add_const<A1>::type >::call(other.a1)) , a2( detail::copy_construct< A2 , typename boost::add_const<A2>::type >::call(other.a2)) , a3( detail::copy_construct< A3 , typename boost::add_const<A3>::type >::call(other.a3)) , a4( detail::copy_construct< A4 , typename boost::add_const<A4>::type >::call(other.a4)) , a5( detail::copy_construct< A5 , typename boost::add_const<A5>::type >::call(other.a5)) , a6( detail::copy_construct< A6 , typename boost::add_const<A6>::type >::call(other.a6)) , a7( detail::copy_construct< A7 , typename boost::add_const<A7>::type >::call(other.a7)) , a8( detail::copy_construct< A8 , typename boost::add_const<A8>::type >::call(other.a8)) , a9( detail::copy_construct< A9 , typename boost::add_const<A9>::type >::call(other.a9)) , a10( detail::copy_construct< A10 , typename boost::add_const<A10>::type >::call(other.a10)) , a11( detail::copy_construct< A11 , typename boost::add_const<A11>::type >::call(other.a11)) , a12( detail::copy_construct< A12 , typename boost::add_const<A12>::type >::call(other.a12))
        {}
        tuple(BOOST_RV_REF(tuple) other)
          : a0(boost::forward<A0>( other.a0)) , a1(boost::forward<A1>( other.a1)) , a2(boost::forward<A2>( other.a2)) , a3(boost::forward<A3>( other.a3)) , a4(boost::forward<A4>( other.a4)) , a5(boost::forward<A5>( other.a5)) , a6(boost::forward<A6>( other.a6)) , a7(boost::forward<A7>( other.a7)) , a8(boost::forward<A8>( other.a8)) , a9(boost::forward<A9>( other.a9)) , a10(boost::forward<A10>( other.a10)) , a11(boost::forward<A11>( other.a11)) , a12(boost::forward<A12>( other.a12))
        {}
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12>
        tuple(tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12> const& other)
          : a0( detail::copy_construct< A0 , typename boost::add_const<T0>::type >::call(other.a0)) , a1( detail::copy_construct< A1 , typename boost::add_const<T1>::type >::call(other.a1)) , a2( detail::copy_construct< A2 , typename boost::add_const<T2>::type >::call(other.a2)) , a3( detail::copy_construct< A3 , typename boost::add_const<T3>::type >::call(other.a3)) , a4( detail::copy_construct< A4 , typename boost::add_const<T4>::type >::call(other.a4)) , a5( detail::copy_construct< A5 , typename boost::add_const<T5>::type >::call(other.a5)) , a6( detail::copy_construct< A6 , typename boost::add_const<T6>::type >::call(other.a6)) , a7( detail::copy_construct< A7 , typename boost::add_const<T7>::type >::call(other.a7)) , a8( detail::copy_construct< A8 , typename boost::add_const<T8>::type >::call(other.a8)) , a9( detail::copy_construct< A9 , typename boost::add_const<T9>::type >::call(other.a9)) , a10( detail::copy_construct< A10 , typename boost::add_const<T10>::type >::call(other.a10)) , a11( detail::copy_construct< A11 , typename boost::add_const<T11>::type >::call(other.a11)) , a12( detail::copy_construct< A12 , typename boost::add_const<T12>::type >::call(other.a12))
        {}
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12>
        tuple(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12>
            ))) other)
          : a0(boost::forward<T0>( other.a0)) , a1(boost::forward<T1>( other.a1)) , a2(boost::forward<T2>( other.a2)) , a3(boost::forward<T3>( other.a3)) , a4(boost::forward<T4>( other.a4)) , a5(boost::forward<T5>( other.a5)) , a6(boost::forward<T6>( other.a6)) , a7(boost::forward<T7>( other.a7)) , a8(boost::forward<T8>( other.a8)) , a9(boost::forward<T9>( other.a9)) , a10(boost::forward<T10>( other.a10)) , a11(boost::forward<T11>( other.a11)) , a12(boost::forward<T12>( other.a12))
        {}
        
        tuple& operator=(BOOST_COPY_ASSIGN_REF(tuple) other)
        {
            a0 = other.a0; a1 = other.a1; a2 = other.a2; a3 = other.a3; a4 = other.a4; a5 = other.a5; a6 = other.a6; a7 = other.a7; a8 = other.a8; a9 = other.a9; a10 = other.a10; a11 = other.a11; a12 = other.a12;;
            return *this;
        }
        tuple& operator=(BOOST_RV_REF(tuple) other)
        {
            a0 = boost::forward<A0>( other.a0); a1 = boost::forward<A1>( other.a1); a2 = boost::forward<A2>( other.a2); a3 = boost::forward<A3>( other.a3); a4 = boost::forward<A4>( other.a4); a5 = boost::forward<A5>( other.a5); a6 = boost::forward<A6>( other.a6); a7 = boost::forward<A7>( other.a7); a8 = boost::forward<A8>( other.a8); a9 = boost::forward<A9>( other.a9); a10 = boost::forward<A10>( other.a10); a11 = boost::forward<A11>( other.a11); a12 = boost::forward<A12>( other.a12);;
            return *this;
        }
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12>
        tuple& operator=(BOOST_COPY_ASSIGN_REF(HPX_UTIL_STRIP((
                tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12>
            ))) other)
        {
            a0 = other.a0; a1 = other.a1; a2 = other.a2; a3 = other.a3; a4 = other.a4; a5 = other.a5; a6 = other.a6; a7 = other.a7; a8 = other.a8; a9 = other.a9; a10 = other.a10; a11 = other.a11; a12 = other.a12;;
            return *this;
        }
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12>
        tuple& operator=(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12>
            ))) other)
        {
            a0 = boost::forward<T0>( other.a0); a1 = boost::forward<T1>( other.a1); a2 = boost::forward<T2>( other.a2); a3 = boost::forward<T3>( other.a3); a4 = boost::forward<T4>( other.a4); a5 = boost::forward<T5>( other.a5); a6 = boost::forward<T6>( other.a6); a7 = boost::forward<T7>( other.a7); a8 = boost::forward<T8>( other.a8); a9 = boost::forward<T9>( other.a9); a10 = boost::forward<T10>( other.a10); a11 = boost::forward<T11>( other.a11); a12 = boost::forward<T12>( other.a12);;
            return *this;
        }
        void swap(tuple& other)
        {
            boost::swap(a0, other.a0); boost::swap(a1, other.a1); boost::swap(a2, other.a2); boost::swap(a3, other.a3); boost::swap(a4, other.a4); boost::swap(a5, other.a5); boost::swap(a6, other.a6); boost::swap(a7, other.a7); boost::swap(a8, other.a8); boost::swap(a9, other.a9); boost::swap(a10, other.a10); boost::swap(a11, other.a11); boost::swap(a12, other.a12);;
        }
        typedef boost::mpl::int_<13> size_type;
        static const int size_value = 13;
    private:
        BOOST_COPYABLE_AND_MOVABLE(tuple);
    };
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12>
    struct tuple_size<tuple<A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12> >
    {
        static const std::size_t value = 13;
    };
    
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12>
    BOOST_FORCEINLINE
    tuple<typename util::decay<Arg0>::type , typename util::decay<Arg1>::type , typename util::decay<Arg2>::type , typename util::decay<Arg3>::type , typename util::decay<Arg4>::type , typename util::decay<Arg5>::type , typename util::decay<Arg6>::type , typename util::decay<Arg7>::type , typename util::decay<Arg8>::type , typename util::decay<Arg9>::type , typename util::decay<Arg10>::type , typename util::decay<Arg11>::type , typename util::decay<Arg12>::type>
    make_tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12)
    {
        return tuple<typename util::decay<Arg0>::type , typename util::decay<Arg1>::type , typename util::decay<Arg2>::type , typename util::decay<Arg3>::type , typename util::decay<Arg4>::type , typename util::decay<Arg5>::type , typename util::decay<Arg6>::type , typename util::decay<Arg7>::type , typename util::decay<Arg8>::type , typename util::decay<Arg9>::type , typename util::decay<Arg10>::type , typename util::decay<Arg11>::type , typename util::decay<Arg12>::type>(
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ));
    }
    
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12>
    BOOST_FORCEINLINE
    tuple<typename detail::add_rvalue_reference<Arg0>::type , typename detail::add_rvalue_reference<Arg1>::type , typename detail::add_rvalue_reference<Arg2>::type , typename detail::add_rvalue_reference<Arg3>::type , typename detail::add_rvalue_reference<Arg4>::type , typename detail::add_rvalue_reference<Arg5>::type , typename detail::add_rvalue_reference<Arg6>::type , typename detail::add_rvalue_reference<Arg7>::type , typename detail::add_rvalue_reference<Arg8>::type , typename detail::add_rvalue_reference<Arg9>::type , typename detail::add_rvalue_reference<Arg10>::type , typename detail::add_rvalue_reference<Arg11>::type , typename detail::add_rvalue_reference<Arg12>::type>
    forward_as_tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12) BOOST_NOEXCEPT
    {
        return tuple<
                typename detail::add_rvalue_reference<Arg0>::type , typename detail::add_rvalue_reference<Arg1>::type , typename detail::add_rvalue_reference<Arg2>::type , typename detail::add_rvalue_reference<Arg3>::type , typename detail::add_rvalue_reference<Arg4>::type , typename detail::add_rvalue_reference<Arg5>::type , typename detail::add_rvalue_reference<Arg6>::type , typename detail::add_rvalue_reference<Arg7>::type , typename detail::add_rvalue_reference<Arg8>::type , typename detail::add_rvalue_reference<Arg9>::type , typename detail::add_rvalue_reference<Arg10>::type , typename detail::add_rvalue_reference<Arg11>::type , typename detail::add_rvalue_reference<Arg12>::type>(
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ));
    }
    
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12>
    BOOST_FORCEINLINE
    tuple<Arg0 & , Arg1 & , Arg2 & , Arg3 & , Arg4 & , Arg5 & , Arg6 & , Arg7 & , Arg8 & , Arg9 & , Arg10 & , Arg11 & , Arg12 &>
    tie(Arg0 & arg0 , Arg1 & arg1 , Arg2 & arg2 , Arg3 & arg3 , Arg4 & arg4 , Arg5 & arg5 , Arg6 & arg6 , Arg7 & arg7 , Arg8 & arg8 , Arg9 & arg9 , Arg10 & arg10 , Arg11 & arg11 , Arg12 & arg12) BOOST_NOEXCEPT
    {
        return tuple<
                Arg0 & , Arg1 & , Arg2 & , Arg3 & , Arg4 & , Arg5 & , Arg6 & , Arg7 & , Arg8 & , Arg9 & , Arg10 & , Arg11 & , Arg12 &>(
            arg0 , arg1 , arg2 , arg3 , arg4 , arg5 , arg6 , arg7 , arg8 , arg9 , arg10 , arg11 , arg12);
    }
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12>
    typename detail::tuple_cat_result<
        typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type
    >::type
    tuple_cat(BOOST_FWD_REF(T0) t0 , BOOST_FWD_REF(T1) t1 , BOOST_FWD_REF(T2) t2 , BOOST_FWD_REF(T3) t3 , BOOST_FWD_REF(T4) t4 , BOOST_FWD_REF(T5) t5 , BOOST_FWD_REF(T6) t6 , BOOST_FWD_REF(T7) t7 , BOOST_FWD_REF(T8) t8 , BOOST_FWD_REF(T9) t9 , BOOST_FWD_REF(T10) t10 , BOOST_FWD_REF(T11) t11 , BOOST_FWD_REF(T12) t12)
    {
        typedef
            typename detail::tuple_cat_result<T0, T1>::type
            head_type;
        head_type head =
            tuple_cat(boost::forward<T0>(t0), boost::forward<T1>(t1));
        return tuple_cat(boost::move(head)
                , boost::forward<T2>(t2) , boost::forward<T3>(t3) , boost::forward<T4>(t4) , boost::forward<T5>(t5) , boost::forward<T6>(t6) , boost::forward<T7>(t7) , boost::forward<T8>(t8) , boost::forward<T9>(t9) , boost::forward<T10>(t10) , boost::forward<T11>(t11) , boost::forward<T12>(t12));
    }
}}
namespace boost { namespace fusion { namespace traits { template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11, typename A12 > struct tag_of<hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12> > { typedef struct_tag type; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11, typename A12 > struct tag_of<hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12> const> { typedef struct_tag type; }; } namespace extension { template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11, typename A12 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12> , 0 > { typedef A0 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7) (A8) (A9) (A10) (A11) (A12)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a0; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11, typename A12 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12> , 0 > { typedef char const* type; static type call() { return "a0"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11, typename A12 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12> , 1 > { typedef A1 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7) (A8) (A9) (A10) (A11) (A12)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a1; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11, typename A12 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12> , 1 > { typedef char const* type; static type call() { return "a1"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11, typename A12 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12> , 2 > { typedef A2 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7) (A8) (A9) (A10) (A11) (A12)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a2; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11, typename A12 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12> , 2 > { typedef char const* type; static type call() { return "a2"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11, typename A12 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12> , 3 > { typedef A3 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7) (A8) (A9) (A10) (A11) (A12)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a3; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11, typename A12 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12> , 3 > { typedef char const* type; static type call() { return "a3"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11, typename A12 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12> , 4 > { typedef A4 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7) (A8) (A9) (A10) (A11) (A12)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a4; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11, typename A12 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12> , 4 > { typedef char const* type; static type call() { return "a4"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11, typename A12 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12> , 5 > { typedef A5 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7) (A8) (A9) (A10) (A11) (A12)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a5; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11, typename A12 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12> , 5 > { typedef char const* type; static type call() { return "a5"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11, typename A12 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12> , 6 > { typedef A6 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7) (A8) (A9) (A10) (A11) (A12)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a6; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11, typename A12 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12> , 6 > { typedef char const* type; static type call() { return "a6"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11, typename A12 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12> , 7 > { typedef A7 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7) (A8) (A9) (A10) (A11) (A12)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a7; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11, typename A12 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12> , 7 > { typedef char const* type; static type call() { return "a7"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11, typename A12 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12> , 8 > { typedef A8 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7) (A8) (A9) (A10) (A11) (A12)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a8; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11, typename A12 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12> , 8 > { typedef char const* type; static type call() { return "a8"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11, typename A12 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12> , 9 > { typedef A9 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7) (A8) (A9) (A10) (A11) (A12)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a9; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11, typename A12 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12> , 9 > { typedef char const* type; static type call() { return "a9"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11, typename A12 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12> , 10 > { typedef A10 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7) (A8) (A9) (A10) (A11) (A12)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a10; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11, typename A12 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12> , 10 > { typedef char const* type; static type call() { return "a10"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11, typename A12 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12> , 11 > { typedef A11 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7) (A8) (A9) (A10) (A11) (A12)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a11; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11, typename A12 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12> , 11 > { typedef char const* type; static type call() { return "a11"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11, typename A12 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12> , 12 > { typedef A12 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7) (A8) (A9) (A10) (A11) (A12)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a12; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11, typename A12 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12> , 12 > { typedef char const* type; static type call() { return "a12"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11, typename A12 > struct struct_size<hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12> > : mpl::int_<13> {}; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11, typename A12 > struct struct_is_view< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12> > : mpl::false_ {}; } } namespace mpl { template<typename> struct sequence_tag; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11, typename A12 > struct sequence_tag<hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12> > { typedef fusion::fusion_sequence_tag type; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11, typename A12 > struct sequence_tag< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12> const > { typedef fusion::fusion_sequence_tag type; }; } }
namespace boost { namespace serialization
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12>
    struct is_bitwise_serializable<
            hpx::util::tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12> >
      : hpx::util::detail::sequence_is_bitwise_serializable<
            hpx::util::tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12> >
    {};
    
    template <typename Archive, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12>
    BOOST_FORCEINLINE void serialize(Archive& ar,
        hpx::util::tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12>& t,
        unsigned int const version)
    {
        hpx::util::serialize_sequence(ar, t);
    }
}}
