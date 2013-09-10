// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


namespace hpx { namespace util { namespace detail
{
    template <typename Tuple> struct tuple_element< 0, Tuple> { typedef typename Tuple::member_type0 type; typedef typename detail::tuple_element_access<type>::type rtype; typedef typename detail::tuple_element_access<type>::ctype crtype; static BOOST_CONSTEXPR rtype get(Tuple& t) BOOST_NOEXCEPT { return t.a0; } }; template <typename Tuple> struct tuple_element< 1, Tuple> { typedef typename Tuple::member_type1 type; typedef typename detail::tuple_element_access<type>::type rtype; typedef typename detail::tuple_element_access<type>::ctype crtype; static BOOST_CONSTEXPR rtype get(Tuple& t) BOOST_NOEXCEPT { return t.a1; } }; template <typename Tuple> struct tuple_element< 2, Tuple> { typedef typename Tuple::member_type2 type; typedef typename detail::tuple_element_access<type>::type rtype; typedef typename detail::tuple_element_access<type>::ctype crtype; static BOOST_CONSTEXPR rtype get(Tuple& t) BOOST_NOEXCEPT { return t.a2; } }; template <typename Tuple> struct tuple_element< 3, Tuple> { typedef typename Tuple::member_type3 type; typedef typename detail::tuple_element_access<type>::type rtype; typedef typename detail::tuple_element_access<type>::ctype crtype; static BOOST_CONSTEXPR rtype get(Tuple& t) BOOST_NOEXCEPT { return t.a3; } }; template <typename Tuple> struct tuple_element< 4, Tuple> { typedef typename Tuple::member_type4 type; typedef typename detail::tuple_element_access<type>::type rtype; typedef typename detail::tuple_element_access<type>::ctype crtype; static BOOST_CONSTEXPR rtype get(Tuple& t) BOOST_NOEXCEPT { return t.a4; } }; template <typename Tuple> struct tuple_element< 5, Tuple> { typedef typename Tuple::member_type5 type; typedef typename detail::tuple_element_access<type>::type rtype; typedef typename detail::tuple_element_access<type>::ctype crtype; static BOOST_CONSTEXPR rtype get(Tuple& t) BOOST_NOEXCEPT { return t.a5; } }; template <typename Tuple> struct tuple_element< 6, Tuple> { typedef typename Tuple::member_type6 type; typedef typename detail::tuple_element_access<type>::type rtype; typedef typename detail::tuple_element_access<type>::ctype crtype; static BOOST_CONSTEXPR rtype get(Tuple& t) BOOST_NOEXCEPT { return t.a6; } }; template <typename Tuple> struct tuple_element< 7, Tuple> { typedef typename Tuple::member_type7 type; typedef typename detail::tuple_element_access<type>::type rtype; typedef typename detail::tuple_element_access<type>::ctype crtype; static BOOST_CONSTEXPR rtype get(Tuple& t) BOOST_NOEXCEPT { return t.a7; } };
    template <typename Tuple> struct tuple_element< 0, Tuple const> { typedef typename boost::add_const< typename Tuple::member_type0>::type type; typedef typename detail::tuple_element_access<type>::type rtype; typedef typename detail::tuple_element_access<type>::ctype crtype; static BOOST_CONSTEXPR crtype get(Tuple const& t) BOOST_NOEXCEPT { return t.a0; } }; template <typename Tuple> struct tuple_element< 1, Tuple const> { typedef typename boost::add_const< typename Tuple::member_type1>::type type; typedef typename detail::tuple_element_access<type>::type rtype; typedef typename detail::tuple_element_access<type>::ctype crtype; static BOOST_CONSTEXPR crtype get(Tuple const& t) BOOST_NOEXCEPT { return t.a1; } }; template <typename Tuple> struct tuple_element< 2, Tuple const> { typedef typename boost::add_const< typename Tuple::member_type2>::type type; typedef typename detail::tuple_element_access<type>::type rtype; typedef typename detail::tuple_element_access<type>::ctype crtype; static BOOST_CONSTEXPR crtype get(Tuple const& t) BOOST_NOEXCEPT { return t.a2; } }; template <typename Tuple> struct tuple_element< 3, Tuple const> { typedef typename boost::add_const< typename Tuple::member_type3>::type type; typedef typename detail::tuple_element_access<type>::type rtype; typedef typename detail::tuple_element_access<type>::ctype crtype; static BOOST_CONSTEXPR crtype get(Tuple const& t) BOOST_NOEXCEPT { return t.a3; } }; template <typename Tuple> struct tuple_element< 4, Tuple const> { typedef typename boost::add_const< typename Tuple::member_type4>::type type; typedef typename detail::tuple_element_access<type>::type rtype; typedef typename detail::tuple_element_access<type>::ctype crtype; static BOOST_CONSTEXPR crtype get(Tuple const& t) BOOST_NOEXCEPT { return t.a4; } }; template <typename Tuple> struct tuple_element< 5, Tuple const> { typedef typename boost::add_const< typename Tuple::member_type5>::type type; typedef typename detail::tuple_element_access<type>::type rtype; typedef typename detail::tuple_element_access<type>::ctype crtype; static BOOST_CONSTEXPR crtype get(Tuple const& t) BOOST_NOEXCEPT { return t.a5; } }; template <typename Tuple> struct tuple_element< 6, Tuple const> { typedef typename boost::add_const< typename Tuple::member_type6>::type type; typedef typename detail::tuple_element_access<type>::type rtype; typedef typename detail::tuple_element_access<type>::ctype crtype; static BOOST_CONSTEXPR crtype get(Tuple const& t) BOOST_NOEXCEPT { return t.a6; } }; template <typename Tuple> struct tuple_element< 7, Tuple const> { typedef typename boost::add_const< typename Tuple::member_type7>::type type; typedef typename detail::tuple_element_access<type>::type rtype; typedef typename detail::tuple_element_access<type>::ctype crtype; static BOOST_CONSTEXPR crtype get(Tuple const& t) BOOST_NOEXCEPT { return t.a7; } };
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
          , typename boost::disable_if<is_tuple<Arg0>>::type* = 0)
          : a0(boost::forward<Arg0>(arg0))
        {}
        template <typename Arg0>
        tuple(BOOST_FWD_REF(Arg0) arg0, detail::forwarding_tag)
          : a0(boost::forward<Arg0>(arg0))
        {}
        
        tuple(tuple const& other)
          : a0(other.a0)
        {}
        tuple(BOOST_RV_REF(tuple) other)
          : a0(detail::move_if_no_ref<A0>::call( other.a0))
        {}
        template <typename T0>
        tuple(tuple<T0> const& other)
          : a0(other.a0)
        {}
        template <typename T0>
        tuple(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple<T0>
            ))) other)
          : a0(detail::move_if_no_ref<T0>::call( other.a0))
        {}
        
        tuple& operator=(BOOST_COPY_ASSIGN_REF(tuple) other)
        {
            a0 = other.a0;;
            return *this;
        }
        tuple& operator=(BOOST_RV_REF(tuple) other)
        {
            a0 = detail::move_if_no_ref<A0>::call( other.a0);;
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
            a0 = detail::move_if_no_ref<T0>::call( other.a0);;
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
    tuple<typename detail::env_value_type<Arg0>::type>
    forward_as_tuple(BOOST_FWD_REF(Arg0) arg0) BOOST_NOEXCEPT
    {
        typedef tuple<typename detail::env_value_type<Arg0>::type> result_type;
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
          : a0(other.a0) , a1(other.a1)
        {}
        tuple(BOOST_RV_REF(tuple) other)
          : a0(detail::move_if_no_ref<A0>::call( other.a0)) , a1(detail::move_if_no_ref<A1>::call( other.a1))
        {}
        template <typename T0 , typename T1>
        tuple(tuple<T0 , T1> const& other)
          : a0(other.a0) , a1(other.a1)
        {}
        template <typename T0 , typename T1>
        tuple(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple<T0 , T1>
            ))) other)
          : a0(detail::move_if_no_ref<T0>::call( other.a0)) , a1(detail::move_if_no_ref<T1>::call( other.a1))
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
            a0 = detail::move_if_no_ref<A0>::call( other.a0); a1 = detail::move_if_no_ref<A1>::call( other.a1);;
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
            a0 = detail::move_if_no_ref<T0>::call( other.a0); a1 = detail::move_if_no_ref<T1>::call( other.a1);;
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
    tuple<typename detail::env_value_type<Arg0>::type , typename detail::env_value_type<Arg1>::type>
    forward_as_tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1) BOOST_NOEXCEPT
    {
        return tuple<
                typename detail::env_value_type<Arg0>::type , typename detail::env_value_type<Arg1>::type>(
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
    
    template <typename T0, typename T1> typename boost::lazy_enable_if_c< 0 == util::decay<T0>::type::size_value + util::decay<T1>::type::size_value , detail::tuple_cat_result< typename util::decay<T0>::type , typename util::decay<T1>::type > >::type tuple_cat(BOOST_FWD_REF(T0) t0, BOOST_FWD_REF(T1) t1) { return make_tuple(); } template <typename T0, typename T1> typename boost::lazy_enable_if_c< 1 == util::decay<T0>::type::size_value + util::decay<T1>::type::size_value , detail::tuple_cat_result< typename util::decay<T0>::type , typename util::decay<T1>::type > >::type tuple_cat(BOOST_FWD_REF(T0) t0, BOOST_FWD_REF(T1) t1) { return make_tuple(detail::tuple_cat_element< 0, T0, T1>::call(t0, t1)); } template <typename T0, typename T1> typename boost::lazy_enable_if_c< 2 == util::decay<T0>::type::size_value + util::decay<T1>::type::size_value , detail::tuple_cat_result< typename util::decay<T0>::type , typename util::decay<T1>::type > >::type tuple_cat(BOOST_FWD_REF(T0) t0, BOOST_FWD_REF(T1) t1) { return make_tuple(detail::tuple_cat_element< 0, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 1, T0, T1>::call(t0, t1)); } template <typename T0, typename T1> typename boost::lazy_enable_if_c< 3 == util::decay<T0>::type::size_value + util::decay<T1>::type::size_value , detail::tuple_cat_result< typename util::decay<T0>::type , typename util::decay<T1>::type > >::type tuple_cat(BOOST_FWD_REF(T0) t0, BOOST_FWD_REF(T1) t1) { return make_tuple(detail::tuple_cat_element< 0, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 1, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 2, T0, T1>::call(t0, t1)); } template <typename T0, typename T1> typename boost::lazy_enable_if_c< 4 == util::decay<T0>::type::size_value + util::decay<T1>::type::size_value , detail::tuple_cat_result< typename util::decay<T0>::type , typename util::decay<T1>::type > >::type tuple_cat(BOOST_FWD_REF(T0) t0, BOOST_FWD_REF(T1) t1) { return make_tuple(detail::tuple_cat_element< 0, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 1, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 2, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 3, T0, T1>::call(t0, t1)); } template <typename T0, typename T1> typename boost::lazy_enable_if_c< 5 == util::decay<T0>::type::size_value + util::decay<T1>::type::size_value , detail::tuple_cat_result< typename util::decay<T0>::type , typename util::decay<T1>::type > >::type tuple_cat(BOOST_FWD_REF(T0) t0, BOOST_FWD_REF(T1) t1) { return make_tuple(detail::tuple_cat_element< 0, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 1, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 2, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 3, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 4, T0, T1>::call(t0, t1)); } template <typename T0, typename T1> typename boost::lazy_enable_if_c< 6 == util::decay<T0>::type::size_value + util::decay<T1>::type::size_value , detail::tuple_cat_result< typename util::decay<T0>::type , typename util::decay<T1>::type > >::type tuple_cat(BOOST_FWD_REF(T0) t0, BOOST_FWD_REF(T1) t1) { return make_tuple(detail::tuple_cat_element< 0, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 1, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 2, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 3, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 4, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 5, T0, T1>::call(t0, t1)); } template <typename T0, typename T1> typename boost::lazy_enable_if_c< 7 == util::decay<T0>::type::size_value + util::decay<T1>::type::size_value , detail::tuple_cat_result< typename util::decay<T0>::type , typename util::decay<T1>::type > >::type tuple_cat(BOOST_FWD_REF(T0) t0, BOOST_FWD_REF(T1) t1) { return make_tuple(detail::tuple_cat_element< 0, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 1, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 2, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 3, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 4, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 5, T0, T1>::call(t0, t1) , detail::tuple_cat_element< 6, T0, T1>::call(t0, t1)); }
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
          : a0(other.a0) , a1(other.a1) , a2(other.a2)
        {}
        tuple(BOOST_RV_REF(tuple) other)
          : a0(detail::move_if_no_ref<A0>::call( other.a0)) , a1(detail::move_if_no_ref<A1>::call( other.a1)) , a2(detail::move_if_no_ref<A2>::call( other.a2))
        {}
        template <typename T0 , typename T1 , typename T2>
        tuple(tuple<T0 , T1 , T2> const& other)
          : a0(other.a0) , a1(other.a1) , a2(other.a2)
        {}
        template <typename T0 , typename T1 , typename T2>
        tuple(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple<T0 , T1 , T2>
            ))) other)
          : a0(detail::move_if_no_ref<T0>::call( other.a0)) , a1(detail::move_if_no_ref<T1>::call( other.a1)) , a2(detail::move_if_no_ref<T2>::call( other.a2))
        {}
        
        tuple& operator=(BOOST_COPY_ASSIGN_REF(tuple) other)
        {
            a0 = other.a0; a1 = other.a1; a2 = other.a2;;
            return *this;
        }
        tuple& operator=(BOOST_RV_REF(tuple) other)
        {
            a0 = detail::move_if_no_ref<A0>::call( other.a0); a1 = detail::move_if_no_ref<A1>::call( other.a1); a2 = detail::move_if_no_ref<A2>::call( other.a2);;
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
            a0 = detail::move_if_no_ref<T0>::call( other.a0); a1 = detail::move_if_no_ref<T1>::call( other.a1); a2 = detail::move_if_no_ref<T2>::call( other.a2);;
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
    tuple<typename detail::env_value_type<Arg0>::type , typename detail::env_value_type<Arg1>::type , typename detail::env_value_type<Arg2>::type>
    forward_as_tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2) BOOST_NOEXCEPT
    {
        return tuple<
                typename detail::env_value_type<Arg0>::type , typename detail::env_value_type<Arg1>::type , typename detail::env_value_type<Arg2>::type>(
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
            typename detail::tuple_cat_result<
                typename util::decay<T0>::type, typename util::decay<T1>::type
            >::type
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
          : a0(other.a0) , a1(other.a1) , a2(other.a2) , a3(other.a3)
        {}
        tuple(BOOST_RV_REF(tuple) other)
          : a0(detail::move_if_no_ref<A0>::call( other.a0)) , a1(detail::move_if_no_ref<A1>::call( other.a1)) , a2(detail::move_if_no_ref<A2>::call( other.a2)) , a3(detail::move_if_no_ref<A3>::call( other.a3))
        {}
        template <typename T0 , typename T1 , typename T2 , typename T3>
        tuple(tuple<T0 , T1 , T2 , T3> const& other)
          : a0(other.a0) , a1(other.a1) , a2(other.a2) , a3(other.a3)
        {}
        template <typename T0 , typename T1 , typename T2 , typename T3>
        tuple(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple<T0 , T1 , T2 , T3>
            ))) other)
          : a0(detail::move_if_no_ref<T0>::call( other.a0)) , a1(detail::move_if_no_ref<T1>::call( other.a1)) , a2(detail::move_if_no_ref<T2>::call( other.a2)) , a3(detail::move_if_no_ref<T3>::call( other.a3))
        {}
        
        tuple& operator=(BOOST_COPY_ASSIGN_REF(tuple) other)
        {
            a0 = other.a0; a1 = other.a1; a2 = other.a2; a3 = other.a3;;
            return *this;
        }
        tuple& operator=(BOOST_RV_REF(tuple) other)
        {
            a0 = detail::move_if_no_ref<A0>::call( other.a0); a1 = detail::move_if_no_ref<A1>::call( other.a1); a2 = detail::move_if_no_ref<A2>::call( other.a2); a3 = detail::move_if_no_ref<A3>::call( other.a3);;
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
            a0 = detail::move_if_no_ref<T0>::call( other.a0); a1 = detail::move_if_no_ref<T1>::call( other.a1); a2 = detail::move_if_no_ref<T2>::call( other.a2); a3 = detail::move_if_no_ref<T3>::call( other.a3);;
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
    tuple<typename detail::env_value_type<Arg0>::type , typename detail::env_value_type<Arg1>::type , typename detail::env_value_type<Arg2>::type , typename detail::env_value_type<Arg3>::type>
    forward_as_tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3) BOOST_NOEXCEPT
    {
        return tuple<
                typename detail::env_value_type<Arg0>::type , typename detail::env_value_type<Arg1>::type , typename detail::env_value_type<Arg2>::type , typename detail::env_value_type<Arg3>::type>(
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
            typename detail::tuple_cat_result<
                typename util::decay<T0>::type, typename util::decay<T1>::type
            >::type
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
          : a0(other.a0) , a1(other.a1) , a2(other.a2) , a3(other.a3) , a4(other.a4)
        {}
        tuple(BOOST_RV_REF(tuple) other)
          : a0(detail::move_if_no_ref<A0>::call( other.a0)) , a1(detail::move_if_no_ref<A1>::call( other.a1)) , a2(detail::move_if_no_ref<A2>::call( other.a2)) , a3(detail::move_if_no_ref<A3>::call( other.a3)) , a4(detail::move_if_no_ref<A4>::call( other.a4))
        {}
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4>
        tuple(tuple<T0 , T1 , T2 , T3 , T4> const& other)
          : a0(other.a0) , a1(other.a1) , a2(other.a2) , a3(other.a3) , a4(other.a4)
        {}
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4>
        tuple(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple<T0 , T1 , T2 , T3 , T4>
            ))) other)
          : a0(detail::move_if_no_ref<T0>::call( other.a0)) , a1(detail::move_if_no_ref<T1>::call( other.a1)) , a2(detail::move_if_no_ref<T2>::call( other.a2)) , a3(detail::move_if_no_ref<T3>::call( other.a3)) , a4(detail::move_if_no_ref<T4>::call( other.a4))
        {}
        
        tuple& operator=(BOOST_COPY_ASSIGN_REF(tuple) other)
        {
            a0 = other.a0; a1 = other.a1; a2 = other.a2; a3 = other.a3; a4 = other.a4;;
            return *this;
        }
        tuple& operator=(BOOST_RV_REF(tuple) other)
        {
            a0 = detail::move_if_no_ref<A0>::call( other.a0); a1 = detail::move_if_no_ref<A1>::call( other.a1); a2 = detail::move_if_no_ref<A2>::call( other.a2); a3 = detail::move_if_no_ref<A3>::call( other.a3); a4 = detail::move_if_no_ref<A4>::call( other.a4);;
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
            a0 = detail::move_if_no_ref<T0>::call( other.a0); a1 = detail::move_if_no_ref<T1>::call( other.a1); a2 = detail::move_if_no_ref<T2>::call( other.a2); a3 = detail::move_if_no_ref<T3>::call( other.a3); a4 = detail::move_if_no_ref<T4>::call( other.a4);;
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
    tuple<typename detail::env_value_type<Arg0>::type , typename detail::env_value_type<Arg1>::type , typename detail::env_value_type<Arg2>::type , typename detail::env_value_type<Arg3>::type , typename detail::env_value_type<Arg4>::type>
    forward_as_tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4) BOOST_NOEXCEPT
    {
        return tuple<
                typename detail::env_value_type<Arg0>::type , typename detail::env_value_type<Arg1>::type , typename detail::env_value_type<Arg2>::type , typename detail::env_value_type<Arg3>::type , typename detail::env_value_type<Arg4>::type>(
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
            typename detail::tuple_cat_result<
                typename util::decay<T0>::type, typename util::decay<T1>::type
            >::type
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
          : a0(other.a0) , a1(other.a1) , a2(other.a2) , a3(other.a3) , a4(other.a4) , a5(other.a5)
        {}
        tuple(BOOST_RV_REF(tuple) other)
          : a0(detail::move_if_no_ref<A0>::call( other.a0)) , a1(detail::move_if_no_ref<A1>::call( other.a1)) , a2(detail::move_if_no_ref<A2>::call( other.a2)) , a3(detail::move_if_no_ref<A3>::call( other.a3)) , a4(detail::move_if_no_ref<A4>::call( other.a4)) , a5(detail::move_if_no_ref<A5>::call( other.a5))
        {}
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5>
        tuple(tuple<T0 , T1 , T2 , T3 , T4 , T5> const& other)
          : a0(other.a0) , a1(other.a1) , a2(other.a2) , a3(other.a3) , a4(other.a4) , a5(other.a5)
        {}
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5>
        tuple(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple<T0 , T1 , T2 , T3 , T4 , T5>
            ))) other)
          : a0(detail::move_if_no_ref<T0>::call( other.a0)) , a1(detail::move_if_no_ref<T1>::call( other.a1)) , a2(detail::move_if_no_ref<T2>::call( other.a2)) , a3(detail::move_if_no_ref<T3>::call( other.a3)) , a4(detail::move_if_no_ref<T4>::call( other.a4)) , a5(detail::move_if_no_ref<T5>::call( other.a5))
        {}
        
        tuple& operator=(BOOST_COPY_ASSIGN_REF(tuple) other)
        {
            a0 = other.a0; a1 = other.a1; a2 = other.a2; a3 = other.a3; a4 = other.a4; a5 = other.a5;;
            return *this;
        }
        tuple& operator=(BOOST_RV_REF(tuple) other)
        {
            a0 = detail::move_if_no_ref<A0>::call( other.a0); a1 = detail::move_if_no_ref<A1>::call( other.a1); a2 = detail::move_if_no_ref<A2>::call( other.a2); a3 = detail::move_if_no_ref<A3>::call( other.a3); a4 = detail::move_if_no_ref<A4>::call( other.a4); a5 = detail::move_if_no_ref<A5>::call( other.a5);;
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
            a0 = detail::move_if_no_ref<T0>::call( other.a0); a1 = detail::move_if_no_ref<T1>::call( other.a1); a2 = detail::move_if_no_ref<T2>::call( other.a2); a3 = detail::move_if_no_ref<T3>::call( other.a3); a4 = detail::move_if_no_ref<T4>::call( other.a4); a5 = detail::move_if_no_ref<T5>::call( other.a5);;
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
    tuple<typename detail::env_value_type<Arg0>::type , typename detail::env_value_type<Arg1>::type , typename detail::env_value_type<Arg2>::type , typename detail::env_value_type<Arg3>::type , typename detail::env_value_type<Arg4>::type , typename detail::env_value_type<Arg5>::type>
    forward_as_tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5) BOOST_NOEXCEPT
    {
        return tuple<
                typename detail::env_value_type<Arg0>::type , typename detail::env_value_type<Arg1>::type , typename detail::env_value_type<Arg2>::type , typename detail::env_value_type<Arg3>::type , typename detail::env_value_type<Arg4>::type , typename detail::env_value_type<Arg5>::type>(
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
            typename detail::tuple_cat_result<
                typename util::decay<T0>::type, typename util::decay<T1>::type
            >::type
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
          : a0(other.a0) , a1(other.a1) , a2(other.a2) , a3(other.a3) , a4(other.a4) , a5(other.a5) , a6(other.a6)
        {}
        tuple(BOOST_RV_REF(tuple) other)
          : a0(detail::move_if_no_ref<A0>::call( other.a0)) , a1(detail::move_if_no_ref<A1>::call( other.a1)) , a2(detail::move_if_no_ref<A2>::call( other.a2)) , a3(detail::move_if_no_ref<A3>::call( other.a3)) , a4(detail::move_if_no_ref<A4>::call( other.a4)) , a5(detail::move_if_no_ref<A5>::call( other.a5)) , a6(detail::move_if_no_ref<A6>::call( other.a6))
        {}
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6>
        tuple(tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6> const& other)
          : a0(other.a0) , a1(other.a1) , a2(other.a2) , a3(other.a3) , a4(other.a4) , a5(other.a5) , a6(other.a6)
        {}
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6>
        tuple(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6>
            ))) other)
          : a0(detail::move_if_no_ref<T0>::call( other.a0)) , a1(detail::move_if_no_ref<T1>::call( other.a1)) , a2(detail::move_if_no_ref<T2>::call( other.a2)) , a3(detail::move_if_no_ref<T3>::call( other.a3)) , a4(detail::move_if_no_ref<T4>::call( other.a4)) , a5(detail::move_if_no_ref<T5>::call( other.a5)) , a6(detail::move_if_no_ref<T6>::call( other.a6))
        {}
        
        tuple& operator=(BOOST_COPY_ASSIGN_REF(tuple) other)
        {
            a0 = other.a0; a1 = other.a1; a2 = other.a2; a3 = other.a3; a4 = other.a4; a5 = other.a5; a6 = other.a6;;
            return *this;
        }
        tuple& operator=(BOOST_RV_REF(tuple) other)
        {
            a0 = detail::move_if_no_ref<A0>::call( other.a0); a1 = detail::move_if_no_ref<A1>::call( other.a1); a2 = detail::move_if_no_ref<A2>::call( other.a2); a3 = detail::move_if_no_ref<A3>::call( other.a3); a4 = detail::move_if_no_ref<A4>::call( other.a4); a5 = detail::move_if_no_ref<A5>::call( other.a5); a6 = detail::move_if_no_ref<A6>::call( other.a6);;
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
            a0 = detail::move_if_no_ref<T0>::call( other.a0); a1 = detail::move_if_no_ref<T1>::call( other.a1); a2 = detail::move_if_no_ref<T2>::call( other.a2); a3 = detail::move_if_no_ref<T3>::call( other.a3); a4 = detail::move_if_no_ref<T4>::call( other.a4); a5 = detail::move_if_no_ref<T5>::call( other.a5); a6 = detail::move_if_no_ref<T6>::call( other.a6);;
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
    tuple<typename detail::env_value_type<Arg0>::type , typename detail::env_value_type<Arg1>::type , typename detail::env_value_type<Arg2>::type , typename detail::env_value_type<Arg3>::type , typename detail::env_value_type<Arg4>::type , typename detail::env_value_type<Arg5>::type , typename detail::env_value_type<Arg6>::type>
    forward_as_tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6) BOOST_NOEXCEPT
    {
        return tuple<
                typename detail::env_value_type<Arg0>::type , typename detail::env_value_type<Arg1>::type , typename detail::env_value_type<Arg2>::type , typename detail::env_value_type<Arg3>::type , typename detail::env_value_type<Arg4>::type , typename detail::env_value_type<Arg5>::type , typename detail::env_value_type<Arg6>::type>(
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
            typename detail::tuple_cat_result<
                typename util::decay<T0>::type, typename util::decay<T1>::type
            >::type
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
          : a0(other.a0) , a1(other.a1) , a2(other.a2) , a3(other.a3) , a4(other.a4) , a5(other.a5) , a6(other.a6) , a7(other.a7)
        {}
        tuple(BOOST_RV_REF(tuple) other)
          : a0(detail::move_if_no_ref<A0>::call( other.a0)) , a1(detail::move_if_no_ref<A1>::call( other.a1)) , a2(detail::move_if_no_ref<A2>::call( other.a2)) , a3(detail::move_if_no_ref<A3>::call( other.a3)) , a4(detail::move_if_no_ref<A4>::call( other.a4)) , a5(detail::move_if_no_ref<A5>::call( other.a5)) , a6(detail::move_if_no_ref<A6>::call( other.a6)) , a7(detail::move_if_no_ref<A7>::call( other.a7))
        {}
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
        tuple(tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7> const& other)
          : a0(other.a0) , a1(other.a1) , a2(other.a2) , a3(other.a3) , a4(other.a4) , a5(other.a5) , a6(other.a6) , a7(other.a7)
        {}
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
        tuple(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7>
            ))) other)
          : a0(detail::move_if_no_ref<T0>::call( other.a0)) , a1(detail::move_if_no_ref<T1>::call( other.a1)) , a2(detail::move_if_no_ref<T2>::call( other.a2)) , a3(detail::move_if_no_ref<T3>::call( other.a3)) , a4(detail::move_if_no_ref<T4>::call( other.a4)) , a5(detail::move_if_no_ref<T5>::call( other.a5)) , a6(detail::move_if_no_ref<T6>::call( other.a6)) , a7(detail::move_if_no_ref<T7>::call( other.a7))
        {}
        
        tuple& operator=(BOOST_COPY_ASSIGN_REF(tuple) other)
        {
            a0 = other.a0; a1 = other.a1; a2 = other.a2; a3 = other.a3; a4 = other.a4; a5 = other.a5; a6 = other.a6; a7 = other.a7;;
            return *this;
        }
        tuple& operator=(BOOST_RV_REF(tuple) other)
        {
            a0 = detail::move_if_no_ref<A0>::call( other.a0); a1 = detail::move_if_no_ref<A1>::call( other.a1); a2 = detail::move_if_no_ref<A2>::call( other.a2); a3 = detail::move_if_no_ref<A3>::call( other.a3); a4 = detail::move_if_no_ref<A4>::call( other.a4); a5 = detail::move_if_no_ref<A5>::call( other.a5); a6 = detail::move_if_no_ref<A6>::call( other.a6); a7 = detail::move_if_no_ref<A7>::call( other.a7);;
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
            a0 = detail::move_if_no_ref<T0>::call( other.a0); a1 = detail::move_if_no_ref<T1>::call( other.a1); a2 = detail::move_if_no_ref<T2>::call( other.a2); a3 = detail::move_if_no_ref<T3>::call( other.a3); a4 = detail::move_if_no_ref<T4>::call( other.a4); a5 = detail::move_if_no_ref<T5>::call( other.a5); a6 = detail::move_if_no_ref<T6>::call( other.a6); a7 = detail::move_if_no_ref<T7>::call( other.a7);;
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
    tuple<typename detail::env_value_type<Arg0>::type , typename detail::env_value_type<Arg1>::type , typename detail::env_value_type<Arg2>::type , typename detail::env_value_type<Arg3>::type , typename detail::env_value_type<Arg4>::type , typename detail::env_value_type<Arg5>::type , typename detail::env_value_type<Arg6>::type , typename detail::env_value_type<Arg7>::type>
    forward_as_tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7) BOOST_NOEXCEPT
    {
        return tuple<
                typename detail::env_value_type<Arg0>::type , typename detail::env_value_type<Arg1>::type , typename detail::env_value_type<Arg2>::type , typename detail::env_value_type<Arg3>::type , typename detail::env_value_type<Arg4>::type , typename detail::env_value_type<Arg5>::type , typename detail::env_value_type<Arg6>::type , typename detail::env_value_type<Arg7>::type>(
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
            typename detail::tuple_cat_result<
                typename util::decay<T0>::type, typename util::decay<T1>::type
            >::type
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
